# Hallazgos (RTX 4050, 24,6M tokens por run)

## Fase 1: matemáticas (13M, 3 semillas, Δ pareado vs AdamW+SwiGLU)
| Variante | Δ bpb |
|---|---|
| Muon | −13,3 % |
| ReLU² | −3,8 % (con AdamW); 0 % sobre Muon |
| temp. aprendida / sigmoid / GeGLU | ruido |
| sin QK-Norm / sin RoPE | +3 % / +21 % |
| Lion / DyT | mal configurados (no justo) |

## Escalado Muon vs AdamW (2 semillas)
| Tamaño | Δ |
|---|---|
| 13M | −13,3 % |
| 29M | −4 % |
| 62M | −1,6 % |
Pendiente: barrido de LR de Muon por tamaño y más tokens.

## Fase 2: lenguaje interno (bits por frase, 6000 frases paralelas, 2 semillas)
| Variante | bits/frase |
|---|---|
| es normalizado (con pérdida) | 76,4 |
| en | 81,9 |
| es | 82,7 |
| bytes crudos | 90,7 |

## Fase 3: razonamiento (suma de 6 números de 2 cifras, acierto exacto, 0,9M params)
| Modo | s0 | s1 |
|---|---|---|
| directo | 43,8 % | 24,1 % |
| pausa (K=5) | 46,2 % | 30,2 % |
| latente ingenuo (K=5) | 46,1 % | 39,0 % |
| pasos en texto | 100 % | 99,95 % |
Pendiente: latente con currículo tipo Coconut (texto → vectores por etapas).
