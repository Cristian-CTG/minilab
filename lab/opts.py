"""Alternative optimizers: Lion (Chen et al. 2023), Muon (Jordan et al. 2024)."""
import torch


class Lion(torch.optim.Optimizer):
    def __init__(s, params, lr=3e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None: continue
                st = s.state[p]
                if not st: st["m"] = torch.zeros_like(p)
                m = st["m"]
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(torch.sign(m * b1 + p.grad * (1 - b1)), alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)


NS_DTYPE = torch.bfloat16  # train.py sets float32 when amp=fp16


def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(NS_DTYPE); X = X / (X.norm() + 1e-7)
    tr = X.size(0) > X.size(1)
    if tr: X = X.T
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return (X.T if tr else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(s, params, lr=0.02, momentum=0.95, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                st = s.state[p]
                if not st: st["buf"] = torch.zeros_like(p)
                buf = st["buf"]; buf.mul_(g["momentum"]).add_(p.grad)
                u = newton_schulz(p.grad.add(buf, alpha=g["momentum"]))
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(u, alpha=-g["lr"] * max(1, p.size(0) / p.size(1)) ** 0.5)


class Combo:
    """Muon for 2D hidden matrices, AdamW for embeddings/norms."""
    def __init__(s, opts): s.opts = opts; s.param_groups = [g for o in opts for g in o.param_groups]
    def step(s): [o.step() for o in s.opts]
    def zero_grad(s, set_to_none=True): [o.zero_grad(set_to_none=set_to_none) for o in s.opts]
    def state_dict(s): return [o.state_dict() for o in s.opts]
    def load_state_dict(s, sd): [o.load_state_dict(x) for o, x in zip(s.opts, sd)]


def muon_groups(model, lr, wd, lr_muon=0.02):
    emb = {id(model.emb.weight), id(model.head.weight)}  # embeddings and output head stay on AdamW
    hid = [p for p in model.parameters() if p.dim() == 2 and id(p) not in emb]
    rest = [p for p in model.parameters() if not (p.dim() == 2 and id(p) not in emb)]
    return Combo([Muon(hid, lr=lr_muon, weight_decay=wd),
                  torch.optim.AdamW(rest, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)])
