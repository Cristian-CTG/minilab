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
2. Lenguaje interno: español, inglés, español normalizado y bytes crudos (métrica: bits por frase sobre un corpus paralelo, OPUS-100 en-es)
3. Razonamiento latente vs. texto, en tareas con respuesta exacta (aritmética de varios pasos, sintética):
   - **directo**: pregunta → respuesta, sin pasos intermedios
   - **CoT en texto**: el modelo escribe los pasos en lenguaje humano
   - **CoT latente** (estilo Coconut): el estado oculto final se reinyecta como siguiente entrada durante K "pensamientos" vectoriales, sin generar tokens; solo la respuesta se traduce a texto
   - **lenguaje máquina aprendido**: códigos discretos (VQ) aprendidos por un autoencoder; el codificador y el decodificador son los traductores humano ↔ máquina
   - Métricas: exactitud exacta, cómputo por respuesta, y pérdida en la ida y vuelta de los traductores
4. Bucle automático: un LLM propone cambios y se conservan solo los que ganan
