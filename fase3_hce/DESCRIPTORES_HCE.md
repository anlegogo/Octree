# Descriptores HCE — documentación técnica

Este documento describe, para cada descriptor calculado por
`hce_extraccion.py`, qué representa, de qué elementos del **octree
adaptativo real** (`octree_real.py`) se obtiene, cómo se calcula, su
rango esperado, su sensibilidad a transformaciones geométricas y su
justificación técnica.


## Origen de los datos: exclusivamente el árbol, nunca una rejilla densa

Todos los descriptores se calculan a partir de dos fuentes, ambas
provenientes directamente de la estructura del octree — **en ningún
punto del extractor se reconstruye ni se recorre una rejilla densa
`(R, R, R)`**:

1. **Nodos y niveles del árbol** (`NodoOctree`, ver `octree_real.py`):
   usados para el grupo A (ocupación jerárquica). Se usa el conteo real
   de nodos existentes (no podados) en cada profundidad, comparado
   contra el máximo posible (`8^d`) en esa profundidad.
2. **Hojas ocupadas del árbol** (`recolectar_hojas()`): cada hoja aporta
   su **centro** (coordenada `(x, y, z)` en `[-1, 1]³`, derivada de la
   subdivisión recursiva del árbol, no de un índice de celda) y su
   **coherencia de normales** (`normal_coherencia`: magnitud del
   promedio de las normales de los puntos muestreados que cayeron en
   esa hoja, calculada **antes** de normalizar ese promedio a vector
   unitario — ver docstring de `NodoOctree` en `octree_real.py`). Se
   usan para los grupos B y C.

El extractor tiene dos rutas de entrada equivalentes (verificadas
numéricamente idénticas, ver `hce_extraccion.py::__main__` y
`validar_extraccion_hce.py::verificar_recorrido_produccion`):

| Ruta | Función | Fuente de los nodos/hojas |
|---|---|---|
| Árbol en memoria | `extraer_descriptores_hce(raiz, L)` | Recorre el objeto `NodoOctree` recién construido, vía `ocupacion_por_nivel_arbol()` y `recolectar_hojas()` |
| Disco (producción) | `extraer_descriptores_hce_desde_npz(ruta)` | Carga `centros_hoja`/`coherencias_hoja` del `.npz` disperso (`cargar_octree_disperso`); el conteo de nodos por nivel se deriva matemáticamente de esos mismos centros (`ocupacion_por_nivel_desde_hojas`), sin reconstruir el árbol completo ni una rejilla |

En ambos casos, la fuente son los **nodos y hojas del octree**, nunca
una rejilla densa materializada.

## Convención de profundidad

Raíz = `L=0` (un único nodo, cubre todo el espacio `[-1,1]³`). Hoja de
32³ → `L=5`. Hoja de 64³ → `L=6` (consistente con `octree_real.py`).

**`ocupacion_L0` (raíz) se excluye del vector de entrada al
clasificador.** La raíz de cualquier octree no vacío cubre todo el
espacio normalizado y por tanto siempre está ocupada al 100 % — es una
constante sin capacidad discriminante entre clases (verificado:
`n_valores_unicos = 1` sobre los 10 modelos de la muestra controlada,
ver sección de validación). Se sigue calculando y registrando como
**dato de verificación** (`ocupacion_L0_verificacion()`,
`datos_arbol.ocupacion_L0_raiz_verificacion` en los JSON de validación),
pero no forma parte del vector de features.

Total de valores del vector: **`L + 12`** (ocupación de `L=1..L`, es
decir `L` valores, más 10 momentos geométricos, más 2 estadísticas de
coherencia).

| Resolución | `L` (hoja) | Total de descriptores del vector |
|---|---|---|
| 32³ | 5 | **17** |
| 64³ | 6 | **18** |

---

## Tabla de descriptores

Posiciones 0-indexadas según produce `nombres_features(profundidad_max)`
(mismo orden en `extraer_descriptores_hce` y
`extraer_descriptores_hce_desde_npz`). `L` = profundidad de la hoja
(5 para 32³, 6 para 64³).

### Grupo A — Ocupación jerárquica por nivel (posiciones `0 .. L-1`)

| Posición | Nombre | Fórmula | Información del octree utilizada | Interpretación geométrica | Rango esperado | Sensibilidad | Justificación |
|---|---|---|---|---|---|---|---|
| `0 .. L-1` | `ocupacion_L1` … `ocupacion_L{L}` | `100 × (nodos existentes en profundidad d) / 8^d`, para `d = 1..L`. Un nodo a profundidad `d` "existe" si al menos una hoja ocupada desciende de él (`contar_nodos_por_profundidad`, o equivalente `ocupacion_por_nivel_desde_hojas` contando celdas únicas de resolución `2^d` que contienen al menos un centro de hoja) | Nodos del árbol en cada profundidad `d=1..L` (internos y hojas, solo los no podados) | Densidad de subdivisión de la jerarquía en cada escala: qué fracción del espacio a esa resolución contiene geometría. Decrece típicamente con `d` porque una superficie 2D ocupa una fracción cada vez menor de un espacio subdividido en más celdas (`8^d` crece exponencialmente, la superficie ocupada crece solo cuadráticamente) | `(0, 100]`; en la práctica decreciente en `d` para superficies tipo malla (no garantizado monótono estricto, pero observado así en los 10 modelos de la muestra: `L1=100 → L5≈1.5–8.6 %` en R32) | **Traslación**: el árbol se construye sobre la nube ya centrada por `normalizar_malla`; sin ese centrado previo, trasladar el objeto reubica puntos entre octantes y cambia el valor — el descriptor en sí no es invariante a traslación. **Escala**: sensible — depende de cuántas celdas de tamaño `2^-d` caben en el volumen ocupado; la normalización a `[-1,1]³` es lo que hace comparables los valores entre modelos de tamaño físico distinto. **Rotación**: sensible — la subdivisión es axis-aligned (bits por eje X/Y/Z), así que rotar el objeto redistribuye puntos entre octantes de forma no trivial y cambia el patrón de ocupación por nivel | Análisis de ocupación multi-resolución sobre estructuras jerárquicas (octrees/quadtrees) es una técnica estándar de caracterización de forma y complejidad de superficie relacionada con dimensión de conteo de cajas (*box-counting dimension*); aquí se usa de forma discreta y por nivel en vez de una única pendiente fractal, para preservar información de escala |

### Grupo B — Momentos geométricos globales sobre los centros de las hojas ocupadas (posiciones `L .. L+9`)

Calculados por `momentos_geometricos(coords)` sobre el array `(N, 3)`
de centros de hoja, tratados como una nube de puntos discreta que
aproxima la superficie del objeto.

| Posición | Nombre | Fórmula | Información del octree utilizada | Interpretación geométrica | Rango esperado | Sensibilidad | Justificación |
|---|---|---|---|---|---|---|---|
| `L`, `L+1`, `L+2` | `centroide_x`, `centroide_y`, `centroide_z` | `mean(centro[:, eje])` | Centros de las hojas ocupadas | Posición promedio de la geometría dentro del cubo `[-1,1]³` — "centro de masa" discreto tal como lo ve el árbol | `[-1, 1]` (típicamente cercano a 0 porque `normalizar_malla` centra la malla antes de construir el árbol; observado en la muestra: `\|centroide\| < 0.3` en los 10 modelos) | **Traslación**: sensible por definición — es literalmente la posición del objeto; se mantiene cercano a 0 solo porque el preprocesamiento centra la malla antes de esta etapa, no porque el descriptor sea invariante. **Escala**: sensible, escala linealmente con el tamaño del objeto en el cubo normalizado. **Rotación**: sensible, rota como cualquier vector de posición | Momento de orden 1 (centro de masa) de una nube de puntos, base estándar de descriptores de forma basados en momentos (ver Osada et al., *Shape Distributions*, 2002; y la literatura clásica de momentos geométricos para reconocimiento de formas) |
| `L+3`, `L+4`, `L+5` | `varianza_x`, `varianza_y`, `varianza_z` | `var(centro[:, eje] − centroide[eje])` | Centros de las hojas ocupadas | Dispersión de la geometría respecto al centroide, por eje — qué tan "extendido" está el objeto en cada dirección (ej. una silla es más alta que ancha → mayor varianza en el eje vertical) | `[0, ~1]` en el cubo normalizado (observado: `0.008–0.40` en la muestra) | **Traslación**: invariante (se resta la media). **Escala**: sensible, escala cuadráticamente con el factor de escala. **Rotación**: sensible por eje individual — la varianza total (traza del tensor de covarianza) es invariante a rotación, pero la varianza de un eje específico cambia si el objeto rota porque mezcla componentes de otros ejes | Momento de segundo orden (varianza por eje) sobre el eje canónico fijo del octree; equivalente a la diagonal del tensor de covarianza usado en análisis de forma basado en PCA/momentos |
| `L+6` | `dispersion_radial` | `mean(‖centro − centroide‖)` | Centros de las hojas ocupadas | Dispersión general respecto al centroide sin distinguir eje — un solo número que resume el "radio" típico ocupado por el objeto | `[0, ~1.7]` (máximo teórico ≈ diagonal del semicubo; observado: `0.42–0.84` en la muestra) | **Traslación**: invariante (usa `centro − centroide`). **Escala**: sensible, escala linealmente. **Rotación**: invariante — es una norma euclidiana de vectores centrados, y la norma no depende de la orientación de los ejes | Análogo discreto del radio de giro (*radius of gyration*) usado en descriptores de forma y en distribuciones de distancia tipo D1 de Osada et al. (2002) |
| `L+7`, `L+8`, `L+9` | `skew_x`, `skew_y`, `skew_z` | `mean((centro−centroide)³) / (std(centro−centroide)³ + 1e-12)` por eje | Centros de las hojas ocupadas | Asimetría de la distribución espacial de las hojas respecto al centroide, por eje — si la geometría se concentra más hacia un lado que hacia el otro en esa dirección (ej. respaldo de una silla desplaza masa hacia un lado del eje) | Sin cota teórica estricta, pero acotado en la práctica por el tamaño finito de la muestra; observado: `-2.75` a `0.80` en la muestra de 10 modelos | **Traslación**: invariante (usa diferencias centradas). **Escala**: invariante — es un momento estandarizado (adimensional), el numerador y denominador escalan igual (`s³`) y se cancelan. **Rotación**: sensible por eje — al igual que la varianza por eje, mezcla componentes de otros ejes al rotar | Momento estandarizado de tercer orden (skewness), medida clásica de asimetría en estadística descriptiva y en descriptores de forma basados en momentos de orden superior |

### Grupo C — Estadísticas de coherencia de normales (posiciones `L+10`, `L+11`)

Calculadas por `estadisticas_coherencia_normales(coherencias)` sobre el
array `(N,)` de `normal_coherencia` de cada hoja: la magnitud del
promedio de normales de los puntos de esa hoja **antes** de normalizar
a vector unitario (ver corrección documentada en `octree_real.py`).

| Posición | Nombre | Fórmula | Información del octree utilizada | Interpretación geométrica | Rango esperado | Sensibilidad | Justificación |
|---|---|---|---|---|---|---|---|
| `L+10` | `normal_coherencia_media` | `mean(coherencias)`, donde `coherencias[i] = ‖mean(normales de los puntos en la hoja i)‖` (magnitud **pre**-normalización) | `normal_coherencia` de cada hoja ocupada | Coherencia promedio de la orientación de superficie: cercano a 1 si, en promedio, las normales dentro de cada hoja apuntan casi todas en la misma dirección (superficie plana/lisa); cercano a 0 si tienden a cancelarse (superficie rugosa, muy curva, o una hoja que abarca una arista/esquina) | `[0, 1]` (validado explícitamente por `_validar_arrays_serializados` en `octree_real.py`; observado: `0.21–0.91` en la muestra) | **Traslación**: invariante — depende solo de direcciones de normales, no de posición. **Escala**: invariante — escalar la geometría no cambia la dirección de sus normales. **Rotación**: el estadístico en sí (longitud resultante de un conjunto de vectores unitarios) es invariante a rotación *si el conjunto de puntos por hoja no cambia*; pero como las fronteras de las hojas son axis-aligned, rotar el objeto cambia qué puntos caen en cada hoja, y por tanto altera indirectamente el valor calculado a través de la partición del árbol, no del cálculo de coherencia en sí | Longitud resultante media (*mean resultant length*) de estadística direccional, medida estándar de concentración angular de un conjunto de vectores unitarios (ver Mardia & Jupp, *Directional Statistics*); aquí aplicada por hoja a las normales de superficie muestreadas |
| `L+11` | `normal_coherencia_varianza` | `var(coherencias)` | `normal_coherencia` de cada hoja ocupada | Variabilidad de esa coherencia entre hojas — qué tan pareja o irregular es la superficie capturada por el árbol en distintas regiones (alta varianza = mezcla de zonas muy planas con zonas muy curvas/con aristas) | `[0, ~0.25]` (varianza de una magnitud acotada en `[0,1]`; observado: `0.053–0.170` en la muestra) | Igual que el anterior: invariante a traslación y escala; indirectamente sensible a rotación solo por el cambio en la composición punto-a-hoja inducido por particiones axis-aligned | Varianza de una medida de estadística direccional a través de las hojas del árbol; cuantifica heterogeneidad local de curvatura/rugosidad sin requerir estimación explícita de curvatura diferencial |

**Total: `L + 3 + 3 + 1 + 3 + 2 = L + 12`** → 17 (32³) o 18 (64³).

---

## Referencia cruzada con el código

| Grupo | Función en `hce_extraccion.py` | Función en `octree_real.py` que provee los datos |
|---|---|---|
| Ocupación por nivel (excluye `L0`) | `ocupacion_por_nivel_arbol()[1:]` / `ocupacion_por_nivel_desde_hojas()[1:]` (se recibe directamente, el índice 0 se descarta antes de concatenar) | `ocupacion_por_nivel_arbol()` (desde árbol) / `ocupacion_por_nivel_desde_hojas()` (desde `.npz`) |
| Momentos geométricos | `momentos_geometricos(coords)` | `recolectar_hojas()` → `[h.centro for h in hojas]` (desde árbol) / `centros_hoja` (desde `.npz`) |
| Estadísticas de coherencia | `estadisticas_coherencia_normales(coherencias)` | `recolectar_hojas()` → `[h.normal_coherencia for h in hojas]` (desde árbol) / `coherencias_hoja` (desde `.npz`) |
| Ocupación de la raíz (verificación, NO en el vector) | `ocupacion_L0_verificacion()` | índice 0 de `ocupacion_por_nivel_arbol()` / `ocupacion_por_nivel_desde_hojas()` |

El orden exacto de los 17/18 valores en el vector final lo produce
`nombres_features(profundidad_max)`, que debe usarse como referencia de
índices al interpretar `feature_importances_` del Random Forest o los
coeficientes/vectores de soporte del SVM.

---

## Validación empírica

Ver `validar_extraccion_hce.py` (validación individual, usada para
`chair_0001`: `validacion_hce_chair_0001_R32.json` /
`validacion_hce_chair_0001_R64.json`). La validación agregada y sus
estadísticas se generan de forma reproducible con
`validar_muestra_controlada_hce.py` sobre los
**10 modelos de la muestra controlada** (`airplane`, `car`, `chair`,
`sofa`, `table`, cada uno con 1 modelo de `train` y 1 de `test`) en
`validacion_hce_muestra_controlada.json`, separado por `R32`/`R64`.

Comando de referencia:

```bash
python fase3_hce/validar_muestra_controlada_hce.py \
  --dataset-root Dataset/ModelNet40 \
  --salida fase3_hce/validacion_hce_muestra_controlada.json \
  --categorias airplane car chair sofa table \
  --muestra-por-categoria-split 1 \
  --resoluciones 32 64 \
  --n-puntos 20000 \
  --semilla 42 \
  --auditoria-train-npz-root data
```

El último argumento audita los NPZ ya generados del split `train`; no
reconstruye ModelNet40 ni entrena clasificadores. Las pruebas unitarias
se ejecutan con `python -m pytest` y cubren dimensión, orden, valores
finitos, reproducibilidad, coherencia y equivalencia árbol–NPZ.

Para cada uno de los 10 modelos, en cada resolución, se verifica:

1. **Dimensión del vector**: 17 (32³) / 18 (64³) en los 20 casos (10
   modelos × 2 resoluciones).
2. **Orden de los descriptores**: el primero es siempre `ocupacion_L1`
   (nunca `ocupacion_L0`), verificado contra `nombres_features()`.
3. **Reproducibilidad**: el árbol se reconstruye desde cero dos veces
   de forma independiente (misma nube de puntos) y el vector resultante
   es bit-idéntico (`np.array_equal`) en los 20 casos.
4. **Ausencia de NaN/Inf**: `0` valores indefinidos en los 20 casos.
5. **Equivalencia árbol vs. `.npz`**: se construye el árbol, se guarda a
   disco con `guardar_octree_disperso`, se recarga con
   `reconstruir_octree_desde_npz`, y se compara nodo por nodo (estructura
   padre-hijo, conteo por nivel, número de hojas) además del vector de
   descriptores extraído por las dos rutas (`extraer_descriptores_hce`
   vs. `extraer_descriptores_hce_desde_npz`) — coinciden exactamente en
   los 20 casos.

### Resultado preliminar actualmente versionado

Las siguientes estadísticas corresponden a la entrega preliminar de 10
modelos (`n_puntos=20000`, `seed=42`). Deben regenerarse con el comando
anterior; el JSON producido por el script será la evidencia autoritativa.

**R32 (17 descriptores, `L=5`):**

| Descriptor | Mín | Máx | Media | Desv. Est. | Mediana | Únicos | NaN | Inf |
|---|---|---|---|---|---|---|---|---|
| `ocupacion_L1` | 100.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 1 | 0 | 0 |
| `ocupacion_L2` | 25.0000 | 84.3750 | 42.3438 | 17.4056 | 37.5000 | 8 | 0 | 0 |
| `ocupacion_L3` | 8.7891 | 32.8125 | 17.7344 | 7.0550 | 16.4062 | 10 | 0 | 0 |
| `ocupacion_L4` | 2.8076 | 18.0176 | 8.9404 | 4.5961 | 7.2510 | 10 | 0 | 0 |
| `ocupacion_L5` | 1.5198 | 8.5632 | 4.6567 | 2.3984 | 4.4464 | 10 | 0 | 0 |
| `centroide_x` | -0.1063 | 0.0757 | -0.0056 | 0.0531 | -0.0024 | 10 | 0 | 0 |
| `centroide_y` | -0.1608 | 0.1289 | -0.0169 | 0.0740 | -0.0107 | 10 | 0 | 0 |
| `centroide_z` | -0.0799 | 0.2868 | 0.0148 | 0.1063 | -0.0210 | 10 | 0 | 0 |
| `varianza_x` | 0.0356 | 0.3864 | 0.1502 | 0.1025 | 0.1325 | 10 | 0 | 0 |
| `varianza_y` | 0.0615 | 0.4026 | 0.2141 | 0.1118 | 0.1933 | 10 | 0 | 0 |
| `varianza_z` | 0.0091 | 0.2600 | 0.0771 | 0.0778 | 0.0366 | 10 | 0 | 0 |
| `dispersion_radial` | 0.4385 | 0.8300 | 0.6099 | 0.1245 | 0.5978 | 10 | 0 | 0 |
| `skew_x` | -0.4500 | 0.0748 | -0.0771 | 0.1395 | -0.0265 | 10 | 0 | 0 |
| `skew_y` | -1.2186 | 0.0023 | -0.2174 | 0.3611 | -0.0412 | 10 | 0 | 0 |
| `skew_z` | -2.7480 | 0.7347 | -0.1991 | 0.9871 | 0.0782 | 10 | 0 | 0 |
| `normal_coherencia_media` | 0.2087 | 0.8053 | 0.3764 | 0.1741 | 0.3134 | 10 | 0 | 0 |
| `normal_coherencia_varianza` | 0.0534 | 0.1604 | 0.0820 | 0.0334 | 0.0673 | 10 | 0 | 0 |

**R64 (18 descriptores, `L=6`):**

| Descriptor | Mín | Máx | Media | Desv. Est. | Mediana | Únicos | NaN | Inf |
|---|---|---|---|---|---|---|---|---|
| `ocupacion_L1` | 100.0000 | 100.0000 | 100.0000 | 0.0000 | 100.0000 | 1 | 0 | 0 |
| `ocupacion_L2` | 25.0000 | 84.3750 | 42.3438 | 17.4056 | 37.5000 | 8 | 0 | 0 |
| `ocupacion_L3` | 8.7891 | 32.8125 | 17.7344 | 7.0550 | 16.4062 | 10 | 0 | 0 |
| `ocupacion_L4` | 2.8076 | 18.0176 | 8.9404 | 4.5961 | 7.2510 | 10 | 0 | 0 |
| `ocupacion_L5` | 1.5198 | 8.5632 | 4.6567 | 2.3984 | 4.4464 | 10 | 0 | 0 |
| `ocupacion_L6` | 0.7957 | 3.4779 | 2.1267 | 1.0100 | 2.2150 | 10 | 0 | 0 |
| `centroide_x` | -0.1039 | 0.0782 | -0.0112 | 0.0506 | -0.0013 | 10 | 0 | 0 |
| `centroide_y` | -0.1511 | 0.1300 | -0.0106 | 0.0726 | -0.0008 | 10 | 0 | 0 |
| `centroide_z` | -0.0968 | 0.2889 | 0.0190 | 0.1099 | -0.0244 | 10 | 0 | 0 |
| `varianza_x` | 0.0341 | 0.3921 | 0.1488 | 0.1037 | 0.1312 | 10 | 0 | 0 |
| `varianza_y` | 0.0633 | 0.4013 | 0.2109 | 0.1141 | 0.1797 | 10 | 0 | 0 |
| `varianza_z` | 0.0078 | 0.2470 | 0.0724 | 0.0741 | 0.0340 | 10 | 0 | 0 |
| `dispersion_radial` | 0.4215 | 0.8377 | 0.6018 | 0.1292 | 0.5929 | 10 | 0 | 0 |
| `skew_x` | -0.4929 | 0.1316 | -0.0437 | 0.1598 | -0.0055 | 10 | 0 | 0 |
| `skew_y` | -1.2475 | 0.0054 | -0.2311 | 0.3655 | -0.0712 | 10 | 0 | 0 |
| `skew_z` | -2.7867 | 0.7966 | -0.1693 | 1.0355 | 0.0690 | 10 | 0 | 0 |
| `normal_coherencia_media` | 0.3642 | 0.9110 | 0.5791 | 0.1636 | 0.5840 | 10 | 0 | 0 |
| `normal_coherencia_varianza` | 0.0556 | 0.1700 | 0.1231 | 0.0335 | 0.1372 | 10 | 0 | 0 |

`ocupacion_L1` es constante (1 valor único, `100.0`) en esta muestra
reducida y en ambas resoluciones. Esto es una **alerta diagnóstica**, no
una decisión definitiva basada en los 10 modelos: cinco pertenecen a
`test`, y ese split no puede intervenir en la selección de
características. Antes de entrenar, debe auditarse `ocupacion_L1` sobre
todos los NPZ de `train`. Si permanece constante, se excluirá y las
dimensiones finales pasarán a 16 (32³) y 17 (64³); si presenta
variabilidad, se conservará con la evidencia correspondiente. Hasta
resolver esta auditoría, las dimensiones 17/18 son provisionales y no
deben iniciarse SVM ni Bosque Aleatorio.
