"""MiniLab model: MiniMind-style decoder with swappable math pieces."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Cfg:
    vocab: int = 6400
    d: int = 384
    layers: int = 6
    heads: int = 6
    kv_heads: int = 2
    seq: int = 512
    ffn_mult: float = math.pi      # MiniMind uses d*pi rounded to 64
    attn: str = "softmax"          # softmax | sigmoid | softmax_learned_temp
    qk_norm: bool = True
    norm: str = "rms"              # rms | dyt
    act: str = "swiglu"            # swiglu | geglu | relu2
    pos: str = "rope"              # rope | nope
    rope_theta: float = 1e6
    tie: bool = True


class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-6):
        super().__init__(); s.eps = eps; s.w = nn.Parameter(torch.ones(d))
    def forward(s, x):
        return s.w * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + s.eps)).type_as(x)


class DyT(nn.Module):  # Dynamic Tanh (Zhu et al. 2025): norm-free replacement
    def __init__(s, d, a0=0.5):
        super().__init__(); s.a = nn.Parameter(torch.tensor(a0)); s.w = nn.Parameter(torch.ones(d)); s.b = nn.Parameter(torch.zeros(d))
    def forward(s, x):
        return s.w * torch.tanh(s.a * x) + s.b


def Norm(c, d): return DyT(d) if c.norm == "dyt" else RMSNorm(d)


def rope_tables(hd, n, theta):
    f = 1.0 / theta ** (torch.arange(0, hd, 2).float() / hd)
    a = torch.outer(torch.arange(n).float(), f)
    return torch.cat([a.cos(), a.cos()], -1), torch.cat([a.sin(), a.sin()], -1)


def rope(x, cos, sin):  # x: B,H,T,hd
    h = x.shape[-1] // 2
    rot = torch.cat([-x[..., h:], x[..., :h]], -1)
    return (x * cos + rot * sin).type_as(x)


class Attention(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c, s.hd = c, c.d // c.heads
        s.q = nn.Linear(c.d, c.heads * s.hd, bias=False)
        s.kv = nn.Linear(c.d, 2 * c.kv_heads * s.hd, bias=False)
        s.o = nn.Linear(c.heads * s.hd, c.d, bias=False)
        if c.qk_norm: s.qn, s.kn = RMSNorm(s.hd), RMSNorm(s.hd)
        if c.attn == "softmax_learned_temp": s.logt = nn.Parameter(torch.zeros(c.heads, 1, 1))

    def forward(s, x, cos, sin):
        B, T, _ = x.shape; c = s.c
        q = s.q(x).view(B, T, c.heads, s.hd)
        k, v = s.kv(x).view(B, T, 2, c.kv_heads, s.hd).unbind(2)
        if c.qk_norm: q, k = s.qn(q), s.kn(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if c.pos == "rope": q, k = rope(q, cos, sin), rope(k, cos, sin)
        rep = c.heads // c.kv_heads
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        if c.attn == "softmax":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            sc = (q @ k.transpose(-2, -1)).float() / math.sqrt(s.hd)
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            if c.attn == "softmax_learned_temp":
                sc = sc * s.logt.exp()
                p = sc.masked_fill(~mask, float("-inf")).softmax(-1)
            else:  # sigmoid attention (Ramapuram et al. 2024), bias = -log(T)
                p = torch.sigmoid(sc - math.log(T)) * mask
            y = p.type_as(v) @ v
        return s.o(y.transpose(1, 2).reshape(B, T, -1))


class FFN(nn.Module):
    def __init__(s, c):
        super().__init__()
        h = int(math.ceil(c.d * c.ffn_mult / 64) * 64)
        s.act = c.act
        if c.act == "relu2":  # no gate: give it 1.5x width to match params
            h = int(math.ceil(h * 1.5 / 64) * 64)
            s.up, s.down = nn.Linear(c.d, h, bias=False), nn.Linear(h, c.d, bias=False)
        else:
            s.gu = nn.Linear(c.d, 2 * h, bias=False); s.down = nn.Linear(h, c.d, bias=False)

    def forward(s, x):
        if s.act == "relu2": return s.down(F.relu(s.up(x)).square())
        g, u = s.gu(x).chunk(2, -1)
        return s.down((F.silu(g) if s.act == "swiglu" else F.gelu(g, approximate="tanh")) * u)


class Block(nn.Module):
    def __init__(s, c):
        super().__init__(); s.n1, s.att, s.n2, s.ffn = Norm(c, c.d), Attention(c), Norm(c, c.d), FFN(c)
    def forward(s, x, cos, sin):
        x = x + s.att(s.n1(x), cos, sin)
        return x + s.ffn(s.n2(x))


class LM(nn.Module):
    def __init__(s, c: Cfg):
        super().__init__()
        s.c = c
        s.emb = nn.Embedding(c.vocab, c.d)
        s.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        s.norm = Norm(c, c.d)
        s.head = nn.Linear(c.d, c.vocab, bias=False)
        if c.tie: s.head.weight = s.emb.weight
        cos, sin = rope_tables(c.d // c.heads, c.seq, c.rope_theta)
        s.register_buffer("cos", cos, persistent=False); s.register_buffer("sin", sin, persistent=False)
        s.apply(s._init)

    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)

    def forward(s, idx):
        T = idx.shape[1]
        x = s.emb(idx)
        cos, sin = s.cos[:T], s.sin[:T]
        for b in s.blocks: x = b(x, cos, sin)
        return s.head(s.norm(x))

    def n_params(s):
        return sum(p.numel() for p in s.parameters())
