# Plan MiniLab: techo en modelo pequeño y luego escalado

**Objetivo:** exprimir el modelo pequeño (13M) hasta que ninguna pieza nueva lo mejore, y solo entonces crecer de tamaño hasta encontrar el tope de este portátil.

**Enfoque:** tres etapas con puerta de paso. No se pasa a la siguiente hasta cumplir el criterio escrito de la anterior. Todo corre por `lab/queue.sh` (pausable).

**Equipo:** RTX 4050 6 GB, PyTorch 2.11, repo `Cristian-CTG/minilab`.

---

## 1. Cálculos que se usan

| Cálculo | Fórmula | Para qué |
|---|---|---|
| Bits por byte | `bpb = L / (ln 2 · B/T)` (L = pérdida en nats/token, B = bytes, T = tokens de validación) | Calidad, comparable entre tokenizadores |
| Bits por frase | `bpf = (L / ln 2) · T/S` (S = frases) | Comparar idiomas y representaciones |
| Mejora pareada | `Δ_s = bpb_var(s) / bpb_base(s) − 1`, por semilla s | Quitar la suerte de la semilla |
| Ruido | `σ` = desviación típica del modelo base entre semillas | Saber qué diferencia es real |
| Regla de aceptación | todas las `Δ_s` del mismo signo **y** `|media Δ| > 2σ` | Decidir si una pieza entra |
| Tokens por parámetro | `r = tokens entrenados / parámetros` | Escalar de forma justa |
| Ley de escala | `bpb(N) = bpb∞ + a · N^(−α)` ajustada a 3-4 tamaños | Predecir cuánto da un modelo mayor |
| Ganancia por duplicar | `G = 1 − bpb(2N) / bpb(N)` | Detectar el tope |
| Coste | `tiempo = tokens / (tok/s)` | Presupuesto de cada tanda |
| Acierto exacto | `p ± 1,96 · sqrt(p(1−p)/n)` | Margen en tareas de respuesta exacta |
| Coste total con traductor | `bpf(interno) + bpf(restaurar original | interno)` | Saber si un lenguaje interno compensa |
| Error de ida y vuelta | `CER = distancia de edición / longitud` | Pérdida de los traductores |

**Ruido medido hoy (13M, 3 semillas):** modelo base con AdamW σ ≈ 1,4 %; con Muon σ ≈ 0,2 %. Con Muon como base se distinguen mejoras desde ~0,5 %.

**Dato que condiciona el escalado:** todos los tamaños entrenaron 24,6M tokens. Eso son 1,86 tokens/parámetro a 13M y solo 0,39 a 62M. El modelo grande estuvo infraentrenado, así que el escalado anterior no fue justo.

---

## 2. Etapa A: techo del modelo pequeño (13M)

Base de partida: Muon + SwiGLU + QK-Norm + RoPE, 1,102 bpb.

| Paso | Qué se prueba | Runs | Tiempo aprox. |
|---|---|---|---|
| A0 | Conjunto de prueba aparte (2.000 docs que nunca se usan para decidir). Añadir `lr_muon` como parámetro en `train.py` | — | 30 min de código |
| A1 | Tasa de aprendizaje justa: AdamW {1e-3, 2e-3, 3e-3, 6e-3} y Muon {0,005, 0,01, 0,02, 0,04}, 2 semillas | 16 | 2,5 h |
| A2 | Repetir los mal configurados: Lion con LR {1e-4, 3e-4, 1e-3}; DyT con alpha inicial {0,5, 1, 2} | 6 | 1 h |
| A3 | Piezas nuevas, una a una sobre la base: RoPE theta {1e4, 1e5, 1e6}; atención diferencial; z-loss; predicción multi-token; embeddings sin atar; `ffn_mult` {2, π, 4}; `kv_heads` {1, 2, 6}; 3 semillas | ~36 | 5-6 h |
| A4 | Forma a igual número de parámetros: 4 capas anchas, 6 (actual), 10 estrechas | 9 | 1,5 h |
| A5 | Entrenamiento: warmup {50, 100, 300}, weight decay {0, 0,1, 0,3}, lote {16, 32} | ~14 | 2 h |
| A6 | Combinar lo aceptado, siempre contra el mejor individual (lección ReLU²) | 6-9 | 1,5 h |
| A7 | Comprobar que el orden se mantiene entrenando 4× más (6.000 pasos) base vs campeón | 4 | 2,5 h |

**Puerta de salida de A (el "techo"):** una ronda completa A3-A5 sin ninguna pieza aceptada, o mejora acumulada de la ronda < 1 %. El campeón se evalúa una sola vez en el conjunto de prueba.

Los tiempos son estimaciones a partir de ~8 min por run; las variantes de atención no fusionada tardan el doble.

---

## 3. Etapa B: razonamiento y lenguaje interno (modelo de 0,9M-13M)

| Paso | Qué se prueba | Criterio |
|---|---|---|
| B1 | Latente con currículo tipo Coconut: empezar con pasos en texto y sustituir un paso por vectores en cada etapa | Compensa si acierta ≥ 95 % con menos pasadas que el texto |
| B2 | Dificultad creciente: sumas de 4, 6 y 8 números; multiplicación de 2 cifras | Ver dónde se rompe cada modo |
| B3 | Semillas: subir a 3 (el modo directo varió de 24 % a 44 %) | Margen ±2,2 % con n = 2.000 |
| B4 | Restaurador para español normalizado (pone tildes y puntuación) y su coste en bits | Compensa solo si `76,4 + bpf(restaurar) < 82,7` y CER < 1 % |
| B5 | Lenguaje aprendido (códigos VQ con autoencoder) | Mismo criterio de coste total que B4 |
| B6 | Tokenizador: vocabulario 2k, 6,4k, 16k | Menor bpf a igual número de parámetros |

Tiempo aprox.: 4-5 h. B1 exige reescribir el bucle de `latent.py`.

---

## 4. Etapa C: escalado hasta el tope

Solo con el campeón de A y el modelo base original, para ver si la ventaja se mantiene.

1. **Más datos:** tokenizar el archivo completo (`prep.py` con todos los documentos, ~360M tokens estimados; hoy solo hay 32,5M).
2. **Tokens por parámetro fijos, r = 2:**

| Tamaño | Tokens | Velocidad medida | Tiempo por run |
|---|---|---|---|
| 13M | 26M | 48k tok/s | 9 min |
| 29M | 58M | 28k tok/s | 35 min |
| 62M | 124M | 16,5k tok/s | 2 h 5 min |
| ~100M (si cabe en 6 GB con lote menor) | 200M | sin medir | 5-6 h estimadas |

3. **LR por tamaño:** barrido corto de 3 valores en cada tamaño antes del run largo.
4. **Ajustar la ley de escala** con los 3-4 puntos y calcular `G` en cada duplicación.

**Definición de tope:** el primero que ocurra de:
- `G < 1 %` al duplicar parámetros con r fijo.
- VRAM > 5,5 GB con lote ≥ 4.
- Más de 6 h por run.

Si el tope es de máquina y no de calidad, la curva ajustada dice cuánto ganaría una GPU alquilada antes de pagarla.

Tiempo aprox.: 12-15 h de GPU (dos noches).

---

## 5. Archivos que cambian

- `lab/train.py`: parámetro `lr_muon`, evaluación en conjunto de prueba, z-loss y multi-token.
- `lab/model.py`: atención diferencial, embeddings sin atar ya existe (`tie`).
- `lab/prep.py`: división train/val/test y archivo completo.
- `lab/latent.py`: currículo por etapas, tareas más difíciles.
- `lab/restore.py` (nuevo): restaurador de texto normalizado y cálculo de CER.
- `lab/scale_fit.py` (nuevo): ajuste de la ley de escala y tabla de `G`.
- `lab/queue.sh`: una sección por etapa.
- `results/FINDINGS.md`: una tabla por paso con veredicto.

## 6. Riesgos

- **Sobreajuste al conjunto de validación** tras decenas de comparaciones: por eso el conjunto de prueba aparte.
- **Lo que gana a 13M puede no ganar a 62M** (ya pasó con Muon): la etapa C lo comprueba, no lo supone.
- **Datos en chino:** las cifras valen, pero las respuestas del modelo no se pueden leer. Decisión pendiente abajo.
- **Calor y horas de portátil:** unas 35-40 h de GPU en total; todo pausable.

## 7. Decisión pendiente

¿El modelo final debe hablar español? Si es así, la etapa C se entrena con un corpus en español y se añade una prueba de respuestas legibles; si no, se sigue con los datos de MiniMind.
