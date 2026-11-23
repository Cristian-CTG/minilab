# MiniLab

Laboratorio para buscar las matemáticas y fórmulas más eficientes en LLM pequeños y escalables, basado en [MiniMind](https://github.com/jingyaogong/minimind) (Apache-2.0).

## Método
- Modelo de 13M parámetros al estilo MiniMind (6 capas, d=384, GQA, RoPE, tokenizador MiniMind de 6400 tokens).
- Se cambia **una sola pieza** cada vez frente a la línea base.
- Mismo presupuesto (1500 pasos, ~24,6M tokens) y 3 semillas por variante.
- Métrica: **bits por byte** (bpb) de validación, comparable entre tokenizadores e idiomas.
- Si una variante gana, se repite a mayor escala (26M → 64M) para comprobar que la ventaja se mantiene.

## Uso
```bash
python lab/prep.py 200000                         # tokeniza los datos
python lab/train.py nombre attn=sigmoid seed=1    # un experimento
bash lab/batch1.sh                                # tanda completa
python lab/report.py                              # results/REPORT.md + curves.png
```

## Piezas intercambiables
| clave | valores |
|---|---|
| attn | softmax, sigmoid, softmax_learned_temp |
| qk_norm | true, false |
| norm | rms, dyt |
| act | swiglu, geglu, relu2 |
| pos | rope, nope |
| opt | adamw, lion, muon |

## Hoja de ruta
1. Matemáticas de la arquitectura y del optimizador (batch 1)
2. Lenguaje interno y tokenizador (comparación en bpb)
3. Traductores humano ↔ representación interna, con prueba de ida y vuelta
4. Bucle automático: un LLM propone cambios y se conservan solo los que ganan
