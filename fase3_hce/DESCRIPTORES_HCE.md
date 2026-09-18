# Descriptores HCE — documentación técnica

Este documento describe, para cada descriptor calculado por
`hce_extraccion.py`, qué representa, de qué elementos del **octree
adaptativo real** se obtiene, y cómo se calcula.

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
   **normal promedio** (vector unitario `(nx, ny, nz)`, promedio de las
   normales de los puntos muestreados que cayeron en esa hoja durante
   la construcción). Se usan para los grupos B y C.

El extractor tiene dos rutas de entrada equivalentes (verificadas
numéricamente idénticas, ver `hce_extraccion.py::__main__`):

| Ruta | Función | Fuente de los nodos/hojas |
|---|---|---|
| Árbol en memoria | `extraer_descriptores_hce(raiz, L)` | Recorre el objeto `NodoOctree` recién construido |
| Disco (producción) | `extraer_descriptores_hce_desde_npz(ruta)` | Carga `centros_hoja`/`normales_hoja` del `.npz` disperso; el conteo de nodos por nivel se deriva matemáticamente de esos mismos centros (`ocupacion_por_nivel_desde_hojas`), sin reconstruir el árbol completo ni una rejilla |

En ambos casos, la fuente son los **nodos y hojas del octree**, nunca
una rejilla densa materializada.

## Convención de profundidad

Raíz = `L=0` (un único nodo, cubre todo el espacio). Hoja de 32³ →
`L=5`. Hoja de 64³ → `L=6`. Total de valores: `(L+1) + 12`.

| Resolución | `L` (hoja) | Total de descriptores |
|---|---|---|
| 32³ | 5 | **18** |
| 64³ | 6 | **19** |

---

## Tabla de descriptores

| Nombre | Información que representa | Elementos del octree utilizados | Procedimiento de cálculo | Cantidad de valores |
|---|---|---|---|---|
| `ocupacion_L0` … `ocupacion_L{L}` | Densidad de subdivisión del árbol en cada profundidad — qué tan "llena" está la jerarquía a esa escala, de la raíz (siempre 100%) hasta la hoja (típicamente 1–3%) | **Nodos del árbol** en cada profundidad `d` (internos y hojas, solo los que existen, es decir no podados) | Para cada `d = 0..L`: `100 × (nodos existentes en profundidad d) / 8^d`. Un nodo a profundidad `d` "existe" si y solo si al menos una hoja ocupada desciende de él (`contar_nodos_por_profundidad`, o equivalente `ocupacion_por_nivel_desde_hojas` contando celdas únicas de resolución `2^d` ocupadas por al menos un centro de hoja) | `L + 1` (6 para 32³, 7 para 64³) |
| `centroide_x`, `centroide_y`, `centroide_z` | Posición promedio de la geometría dentro del cubo `[-1,1]³` — el "centro de masa" discreto del objeto, tal como lo ve el árbol | **Centros de las hojas ocupadas** (coordenada de cada hoja, derivada de su posición en la subdivisión recursiva) | Media aritmética de los centros de todas las hojas ocupadas, por eje: `mean(centro[:, eje])` | 3 |
| `varianza_x`, `varianza_y`, `varianza_z` | Dispersión de la geometría respecto al centroide, por eje — qué tan "extendido" está el objeto en cada dirección | Centros de las hojas ocupadas | Varianza de `(centro − centroide)` por eje: `var(centro[:, eje] − centroide[eje])` | 3 |
| `dispersion_radial` | Dispersión general de la geometría respecto al centroide, sin distinguir eje — un solo número que resume el "radio" típico ocupado por el objeto | Centros de las hojas ocupadas | Distancia euclidiana de cada centro de hoja al centroide, promediada: `mean(‖centro − centroide‖)` | 1 |
| `skew_x`, `skew_y`, `skew_z` | Asimetría de la distribución espacial de las hojas respecto al centroide, por eje — si la geometría se concentra más hacia un lado que hacia el otro en esa dirección | Centros de las hojas ocupadas | Tercer momento estandarizado por eje: `mean((centro−centroide)³) / std(centro−centroide)³`, con `std` derivado de la varianza ya calculada | 3 |
| `normal_norma_media` | Coherencia promedio de la orientación de superficie capturada por las hojas — cercano a 1 si las normales promedio de cada hoja están bien definidas (poca cancelación interna de direcciones opuestas dentro de la hoja) | **Normal promedio de cada hoja ocupada** (vector `(nx, ny, nz)`, calculado durante la construcción del árbol como el promedio normalizado de las normales de los puntos de superficie que cayeron en esa hoja) | Norma euclidiana de cada vector normal promedio, promediada sobre todas las hojas: `mean(‖normal‖)` | 1 |
| `normal_norma_varianza` | Variabilidad de esa coherencia de orientación entre hojas — qué tan pareja o irregular es la superficie capturada por el árbol en distintas regiones | Normal promedio de cada hoja ocupada | Varianza de las normas calculadas en el descriptor anterior: `var(‖normal‖)` | 1 |

**Total: `(L+1) + 3 + 3 + 1 + 3 + 1 + 1 = (L+1) + 12`** → 18 (32³) o 19 (64³).

---

## Referencia cruzada con el código

| Grupo | Función en `hce_extraccion.py` | Función en `octree_real.py` que provee los datos |
|---|---|---|
| Ocupación por nivel | — (se recibe directamente) | `ocupacion_por_nivel_arbol()` (desde árbol) / `ocupacion_por_nivel_desde_hojas()` (desde `.npz`) |
| Momentos geométricos | `momentos_geometricos(coords)` | `recolectar_hojas()` → `[h.centro for h in hojas]` (desde árbol) / `centros_hoja` (desde `.npz`) |
| Estadísticas de normal | `estadisticas_normales(normales)` | `recolectar_hojas()` → `[h.normal_promedio for h in hojas]` (desde árbol) / `normales_hoja` (desde `.npz`) |

El orden exacto de los 18/19 valores en el vector final lo produce
`nombres_features(profundidad_max)`, que debe usarse como referencia de
índices al interpretar `feature_importances_` del Random Forest o los
coeficientes/vectores de soporte del SVM.
