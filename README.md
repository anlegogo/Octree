# Octree: Clasificación 3D en ModelNet40 con Enfoques Clásicos y Profundos

Trabajo de grado que compara un enfoque clásico basado en descriptores manuales
extraídos de una estructura de octree (HCE + SVM/Random Forest) contra una red
convolucional jerárquica (Net5-Octree), evaluando ambos en dos resoluciones
espaciales equivalentes: **32³** y **64³**, sobre el conjunto de datos
**ModelNet40**.

Repositorio: [https://github.com/ricardoal94/Octree](https://github.com/ricardoal94/Octree)

---

## Tabla de contenido

1. [Estructura del repositorio](#estructura-del-repositorio)
2. [Iteración metodológica: enfoque descartado (PointNet)](#iteración-metodológica-enfoque-descartado-pointnet)
3. [Condiciones de ejecución de los resultados reportados](#condiciones-de-ejecución-de-los-resultados-reportados)
4. [Requisitos e instalación](#requisitos-e-instalación)
5. [Dataset ModelNet40](#dataset-modelnet40)
6. [Reproducibilidad: semillas y configuración](#reproducibilidad-semillas-y-configuración)
7. [Fase 1 — Configuración y partición](#fase-1--configuración-y-partición)
8. [Fase 2 — Construcción del octree real](#fase-2--construcción-del-octree-real)
9. [Fase 3a — Enfoque clásico (HCE + SVM + Random Forest)](#fase-3a--enfoque-clásico-hce--svm--random-forest)
10. [Fase 3b — Enfoque profundo (Net5-Octree)](#fase-3b--enfoque-profundo-net5-octree)
11. [Fase 4 — Comparación y visualización](#fase-4--comparación-y-visualización)
12. [Experimento adicional — Img2Voxel](#experimento-adicional--img2voxel)
13. [Herramientas de análisis y figuras](#herramientas-de-análisis-y-figuras)
14. [Especificaciones técnicas exactas](#especificaciones-técnicas-exactas)
15. [Registros y resultados](#registros-y-resultados)

---

## Estructura del repositorio

```
Octree/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── fase1_modelnet40/            # Configuración global y partición del dataset
│   ├── config.yaml
│   ├── fase1_setup.py
│   └── verificar_reproducibilidad.py
│
├── fase2_octree/                # Pipeline .off -> octree REAL (32³ y 64³)
│   ├── octree.py                # Lectura, normalización, muestreo, rejilla densa independiente
│   ├── octree_real.py           # Arbol real: NodoOctree, poda, serialización jerárquica
│   ├── preprocesar_octrees.py   # Procesa el dataset completo -> formato disperso
│   └── visualizar_octree_3d.py
│
├── fase3_hce/                   # Enfoque clásico: descriptores + SVM/RF
│   ├── hce_extraccion.py        # Extrae descriptores desde el árbol o el .npz disperso
│   ├── fase3_hce_entrenamiento.py
│   ├── visualizar_resultados_hce.py
│   └── visualizar_features_3d.py
│
├── fase3_net5/                  # Enfoque profundo: Net5-Octree (3D-CNN)
│   ├── net5_modelo.py
│   ├── net5_dataset.py          # Carga disperso, materializa denso al vuelo
│   ├── fase3_net5_entrenamiento.py
│   ├── visualizar_comparativa.py
│   └── visualizar_matriz_confusion_net5.py
│
├── fase4_comparacion/           # Comparación cualitativa e interpretabilidad
│   ├── comparar_predicciones.py
│   └── gradcam_3d.py
│
├── fase5_img2voxel/             # Experimento adicional (fuera de la metodología principal)
│   ├── renderizar_vistas.py
│   ├── img2voxel_modelo.py
│   ├── img2voxel_dataset.py
│   ├── entrenar_img2voxel.py
│   ├── visualizar_reconstruccion.py
│   └── visualizar_costo_img2voxel.py
│
├── explorado_descartado/         # Iteración metodológica previa (ver sección dedicada)
│   └── fase2_modelnet40_pointnet/
│       ├── dataset.py
│       └── modelo.py
│
├── Scripts_Analisis/              # Herramientas de medición y figuras del capítulo de metodología
│   ├── generar_figuras_metodologia.py
│   ├── medir_tiempo_memoria.py    # Tiempo y memoria de un archivo individual (--off)
│   ├── medir_metricas_off.py      # Lo mismo sobre el dataset completo o un archivo (--off)
│   ├── tabla_resumen_octree.py    # Tabla resumen por clase / train / test / conjunto
│   ├── generar_imagenes_json.py   # Convierte los JSON de medición en tablas SVG/PNG
│   └── resumen_metricas_completo.py
│
├── Dataset/                      # (NO versionado) ModelNet40 descargado localmente
├── data/                         # (NO versionado) Octrees dispersos y renders preprocesados
├── checkpoints/                  # (NO versionado) Pesos de modelos entrenados
├── logs/                         # (parcialmente versionado) Historiales de entrenamiento
└── Registros/                    # (versionado) CSV/JSON de métricas y resultados finales
    ├── fase1/
    ├── fase2_octree/
    ├── fase3_hce/
    ├── fase3_net5/
    └── fase5_img2voxel/
```

> **Nota:** las carpetas `Dataset/`, `data/` y `checkpoints/` están excluidas del
> control de versiones (ver `.gitignore`) porque contienen el dataset original
> (que no debe redistribuirse) y pesos de modelos regenerables mediante los
> scripts de entrenamiento. La carpeta `Registros/` sí se versiona y contiene
> únicamente los archivos ligeros de métricas (CSV/JSON) necesarios para
> analizar los resultados sin tener que re-entrenar nada.
>
> **Corrección importante:** si tu copia local tiene una carpeta llamada
> `Scripts_Analisis/Tabla_Resumen_Octree/` u otra subcarpeta similar creada
> manualmente, bórrala. Todos los scripts de `Scripts_Analisis/` leen y
> escriben en `resultados/` (o en su propia carpeta, según el script) de
> forma automática con rutas relativas a la raíz del proyecto — no debe
> haber una carpeta de resultados paralela.

---

## Iteración metodológica: enfoque descartado (PointNet)

> ⚠️ **Este código NO forma parte de la metodología final de la tesis.** Se
> conserva únicamente por transparencia respecto al proceso de desarrollo y
> no debe usarse para reproducir los resultados reportados en el documento.

Antes de adoptar la representación de octree, la primera aproximación al
problema implementó una red tipo **PointNet** (con módulos T-Net de
alineación espacial) que clasificaba directamente **nubes de puntos**
muestreadas sobre la superficie de cada malla, sin ningún paso de
voxelización ni construcción de octree.

Esa implementación se encuentra archivada en
`explorado_descartado/fase2_modelnet40_pointnet/` y no se recomienda
ejecutarla como parte del flujo de reproducción del proyecto; se incluye
exclusivamente como evidencia documental del proceso iterativo de diseño
metodológico.

---

## Condiciones de ejecución de los resultados reportados

Esta sección documenta el entorno exacto y los comandos utilizados para
generar los resultados citados en el documento de tesis.

### Equipo y sistema

| Campo | Valor |
|---|---|
| Equipo (CPU) | AMD Ryzen 7 5700X (8 núcleos / 16 hilos) |
| Equipo (GPU) | NVIDIA GeForce RTX 5070, 12 GB VRAM |
| Sistema operativo | Windows 11 (PowerShell) |
| Versión de Python | 3.12 |
| Versión de PyTorch / CUDA | 2.11.0+cu128 |

Para completar automáticamente los campos de hardware y versiones, correr:

```bash
python -c "import platform, torch; print('SO:', platform.platform()); print('Python:', platform.python_version()); print('PyTorch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Esta misma información queda registrada automáticamente en
`logs/experiment_log.json` al correr `fase1_setup.py`.

### Parámetros de ejecución del pipeline de octree

| Parámetro | Valor | Dónde se fija |
|---|---|---|
| Puntos muestreados por objeto | **20 000** | `N_PUNTOS_MUESTREO` en `preprocesar_octrees.py` y `medir_metricas_off.py` |
| Repeticiones (mediciones de tiempo) | 10 | `--repeticiones` en `medir_tiempo_memoria.py` |
| Trabajadores paralelos (`N_PROCESOS`) | 10 | `N_PROCESOS` en `preprocesar_octrees.py` y `medir_metricas_off.py` |
| Semilla global | 42 | Ver [Reproducibilidad](#reproducibilidad-semillas-y-configuración) |

### Comandos utilizados para la corrida oficial

```bash
# 1. Configuracion y particion
cd fase1_modelnet40
python fase1_setup.py

# 2. Preprocesamiento del dataset completo (formato disperso, ambas resoluciones)
cd ../fase2_octree
python preprocesar_octrees.py

# 3. Metricas de tiempo y memoria del pipeline de octree (dataset completo)
cd ../Scripts_Analisis
python medir_metricas_off.py --split ambos
python tabla_resumen_octree.py --split ambos

# 4. Entrenamiento HCE (SVM + Random Forest)
cd ../fase3_hce
python fase3_hce_entrenamiento.py --resolucion 32
python fase3_hce_entrenamiento.py --resolucion 64

# 5. Entrenamiento Net5-Octree
cd ../fase3_net5
python fase3_net5_entrenamiento.py --resolucion 32
python fase3_net5_entrenamiento.py --resolucion 64 --batch_size 8
```

> Completar/ajustar esta lista con cualquier flag adicional realmente usado
> (por ejemplo `--n_muestras` si se corrió sobre un subconjunto antes de la
> corrida completa) para que sea trazable exactamente qué comando produjo
> qué archivo en `Registros/`.

### Commit del código

| Campo | Valor |
|---|---|
| Commit (hash corto) | `50ce26008` |
| Rama | `main` |
| Fecha de la corrida | `22026-09-16 23:31:52 -0500` |

Para obtener el hash exacto del commit vigente al momento de correr los
scripts:

```bash
git rev-parse --short HEAD
git log -1 --format="%H %ci"    # hash completo + fecha del commit
```

**Recomendación:** anotar este hash inmediatamente después de correr la
corrida oficial (antes de hacer cualquier commit adicional), y citarlo
explícitamente en el capítulo de resultados de la tesis. Si se hacen
cambios al código después de la corrida oficial, este hash permite
recuperar exactamente la versión que generó los números reportados
(`git checkout <hash>`).

---

## Requisitos e instalación

- Python 3.10 – 3.12
- Sistema operativo probado: Windows 11 (PowerShell)
- GPU NVIDIA opcional pero recomendada para las Fases 3b y 5

```bash
# 1. Clonar el repositorio
git clone https://github.com/ricardoal94/Octree.git
cd Octree

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Instalación de PyTorch con soporte GPU (opcional pero recomendado)

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verificar la instalación:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Dataset ModelNet40

### Descarga

El dataset **no se distribuye en este repositorio**. Debe descargarse
manualmente desde la fuente oficial:

- **Fuente oficial:** [https://modelnet.cs.princeton.edu](https://modelnet.cs.princeton.edu)
- **Archivo esperado:** `ModelNet40.zip` (mallas en formato `.off`, partición oficial)

### Ubicación esperada

```
Octree/
└── Dataset/
    └── ModelNet40/
        ├── airplane/
        │   ├── train/
        │   │   ├── airplane_0001.off
        │   │   └── ...
        │   └── test/
        │       └── ...
        ├── bathtub/
        ├── ...
        └── xbox/
```

### Estructura esperada por el código

- 40 subcarpetas de clase.
- Cada clase contiene `train/` y `test/`, siguiendo la partición oficial
  (9 843 modelos de entrenamiento y 2 468 de prueba en total).

Si la ruta local difiere de `Dataset/ModelNet40/`, actualizar la constante
`RAIZ_DATASET` al inicio de cada script que lee directamente del conjunto de
datos (`preprocesar_octrees.py`, `medir_metricas_off.py`,
`generar_figuras_metodologia.py`, `renderizar_vistas.py`).

### Cómo se distinguen train y test

Se hereda directamente de la estructura de carpetas oficial
(`.../<clase>/train/` y `.../<clase>/test/`); no se recalcula ni se mezcla
en ningún punto del flujo de procesamiento. Del subconjunto `train/` se
reserva un 10 % para validación interna, manteniendo `test/` intacto para
la evaluación final.

---

## Reproducibilidad: semillas y configuración

Todas las operaciones estocásticas del proyecto usan una **semilla única y
fija: `seed = 42`**.

| Operación | Dónde se fija |
|---|---|
| Partición train-val | `fase1_setup.py::particionar_dataset` |
| Inicialización de pesos (Net5, Img2Voxel) | `set_global_seed()` |
| Random Forest | `random_state=42` |
| Muestreo de superficie | `np.random.default_rng(seed + idx)` por muestra |

`fase1_modelnet40/verificar_reproducibilidad.py` ejecuta la partición del
dataset 3 veces con la misma semilla y compara hashes SHA-256, confirmando
reproducibilidad bit a bit.

---

## Fase 1 — Configuración y partición

```bash
cd fase1_modelnet40
python fase1_setup.py
python verificar_reproducibilidad.py
```

**Salidas:** `logs/experiment_log.json`, `logs/particion_indices.npz`.

---

## Fase 2 — Construcción del octree real

> **Corrección metodológica importante:** las versiones iniciales de este
> proyecto construían una *rejilla densa* que aparentaba ser un octree
> (reservando memoria para las `R³` celdas posibles, sin poda de regiones
> vacías). Esto fue corregido: `fase2_octree/octree_real.py` implementa un
> **octree real**, con nodo raíz único, subdivisión recursiva en 8
> octantes, y **poda explícita** de las ramas sin geometría (no se crea
> ningún nodo, ni se reserva memoria, para las regiones vacías).

### Convención de profundidad

La raíz del árbol está en profundidad `L = 0` (un único nodo, cubre todo el
espacio `[-1, 1]³`). Cada subdivisión aumenta la profundidad en 1, y la
resolución equivalente en profundidad `d` es `R = 2^d`. Bajo esta
convención (matemáticamente estándar para octrees):

- Hoja de resolución **32³** → `L = 5`
- Hoja de resolución **64³** → `L = 6`

### Flujo determinista

1. **Normalización:** centrado y escalado al cubo `[-1, 1]³`.
2. **Muestreo de superficie:** muestreo *area-weighted* sobre los triángulos
   de la malla (20 000 puntos por objeto).
3. **Construcción del árbol:** subdivisión recursiva con poda
   (`fase2_octree/octree_real.py::construir_octree`), hasta la profundidad
   hoja (`L=5` o `L=6`).
4. **Codificación:** cada nodo hoja ocupado almacena ocupación binaria y el
   vector normal promedio de los puntos que caen en él.

### Formato de almacenamiento: DISPERSO, con estructura jerárquica completa

El `.npz` guardado por `guardar_octree_disperso()` **no** es un array denso.
Contiene, mediante un recorrido DFS (pre-orden) de **todos los nodos
existentes** (internos y hojas, nunca los podados):

```python
{
    "profundidades": np.ndarray,  # uint8, profundidad de cada nodo
    "mascaras":      np.ndarray,  # uint8, bits 0-7: que hijos existen
    "normales":      np.ndarray,  # float32 (N, 3), solo valido en hojas
    "etiqueta":      int,
    "profundidad_max": int,       # 5 (32^3) o 6 (64^3)
}
```

El centro y tamaño de cada nodo **no se guardan explícitamente**: se
reconstruyen de forma determinista replicando la subdivisión original
durante la carga (`reconstruir_octree_desde_npz()`). Esto preserva la
topología completa (relaciones padre-hijo reales), a diferencia de guardar
solo las hojas, que perdería la información jerárquica del árbol interno.

La materialización a un tensor denso `(4, R, R, R)` — necesaria únicamente
porque `Conv3d`/`ConvTranspose3D` de PyTorch requieren tensores densos —
ocurre **al vuelo, solo en memoria RAM**, en el `Dataset` de PyTorch
(`fase3_net5/net5_dataset.py`), nunca se persiste densa en disco.

### Comandos

```bash
cd fase2_octree

# Procesar el dataset completo (train + test, ambas resoluciones, formato disperso)
python preprocesar_octrees.py

# Test rapido del modulo del arbol (esfera sintetica, o --off para un archivo real)
python octree_real.py --off "ruta\a\archivo.off"

# Verificacion visual de una muestra individual
python visualizar_octree_3d.py --clase airplane --split train --indice 0
```

**Salidas:** `data/octrees_32/<clase>/<split>/<archivo>.npz` y
`data/octrees_64/<clase>/<split>/<archivo>.npz`, en formato disperso (ver
arriba).

**Parámetros de muestreo:**
- `N_PUNTOS_MUESTREO = 20000` puntos por objeto.
- `SEED = 42`.
- `N_PROCESOS` = ver [Condiciones de ejecución](#condiciones-de-ejecución-de-los-resultados-reportados).

---

## Fase 3a — Enfoque clásico (HCE + SVM + Random Forest)

### Descriptores manuales (HCE)

`hce_extraccion.py` calcula, a partir del árbol (ya sea reconstruido en
memoria o directamente desde el `.npz` disperso), un vector de
**18 características** (resolución 32³, `L=5`) o **19** (resolución 64³,
`L=6`):

| Grupo | Cantidad | Descripción |
|---|---|---|
| Ocupación jerárquica por nivel | `L+1` (6 o 7) | % de nodos que existen realmente en el árbol (no podados) en cada profundidad `d=0` (raíz) a `d=L` (hoja), respecto al máximo posible (`8^d`) |
| Momentos geométricos globales | 10 | Centroide (3), varianza por eje (3), dispersión radial (1), skewness por eje (3) — sobre los centros de las hojas ocupadas |
| Estadísticas del vector normal | 2 | Norma media y varianza de las normales en hojas ocupadas |

> El conteo de features incluye explícitamente el nivel raíz (`L=0`), por
> eso son `L+1` valores de ocupación, no `L`. Total: `(L+1) + 12`.

Dos formas de extracción, verificadas numéricamente equivalentes:
- `extraer_descriptores_hce(raiz, L)` — desde un árbol recién construido en memoria.
- `extraer_descriptores_hce_desde_npz(ruta)` — directamente desde el `.npz`
  disperso, sin reconstruir el árbol completo (usado en producción por
  `fase3_hce_entrenamiento.py`).

### Entrenamiento

```bash
cd fase3_hce
python fase3_hce_entrenamiento.py --resolucion 32
python fase3_hce_entrenamiento.py --resolucion 64
```

**Hiperparámetros:**
- **SVM:** kernel RBF, búsqueda en grilla sobre `C ∈ {0.1, 1, 10, 100}` y
  `gamma ∈ {"scale", 0.001, 0.01, 0.1}`, validación cruzada `cv=3`.
- **Random Forest:** `n_estimators=100`, `max_depth=None`, `random_state=42`.

**Salidas:**
- `checkpoints/hce_svm_R{32,64}.joblib`, `hce_rf_R{32,64}.joblib`, `hce_scaler_R{32,64}.joblib`
- `resultados/resumen_hce_R{32,64}.json`
- `logs/hce_features_R{32,64}.npz`

### Visualización

```bash
python visualizar_resultados_hce.py --resolucion 32
python visualizar_features_3d.py --resolucion 32 --metodo pca
```

---

## Fase 3b — Enfoque profundo (Net5-Octree)

Red convolucional 3D jerárquica. `net5_dataset.py` carga los `.npz`
dispersos y **materializa el grid denso al vuelo**, en cada `__getitem__`
(solo en RAM, nunca en disco).

```bash
cd fase3_net5
python fase3_net5_entrenamiento.py --resolucion 32
python fase3_net5_entrenamiento.py --resolucion 64 --batch_size 8
```

**Hiperparámetros:**
- Optimizador **Adam**, `lr=0.001`, `weight_decay=1e-4`.
- Scheduler `StepLR` (`step_size=20`, `gamma=0.7`).
- **Early stopping**, `patience=20`.
- `batch_size=16` por defecto.
- Sin aumento de datos (criterio de equivalencia experimental).

**Salidas:**
- `checkpoints/net5_mejor_R{32,64}.pth`
- `resultados/resumen_net5_R{32,64}.json`
- `logs/net5_historial_R{32,64}.csv` / `.json`

### Visualización

```bash
python visualizar_comparativa.py --resolucion 32
python visualizar_matriz_confusion_net5.py --resolucion 32
```

---

## Fase 4 — Comparación y visualización

```bash
cd fase4_comparacion
python comparar_predicciones.py --clase airplane --indice 0 --resolucion 32
python gradcam_3d.py --clase airplane --indice 0 --resolucion 32
```

Ambos scripts construyen el árbol real (o cargan el `.npz` disperso) y
materializan el grid denso solo para alimentar Net5; para SVM/RF usan
directamente el vector de descriptores HCE.

---

## Experimento adicional — Img2Voxel

> **Fuera del alcance de la metodología principal.** Incluido por
> transparencia y trazabilidad.

```bash
cd fase5_img2voxel
python renderizar_vistas.py --n_vistas 8 --resolucion 128
python entrenar_img2voxel.py --resolucion 64 --batch_size 16
python visualizar_reconstruccion.py --clase airplane --indice 0 --resolucion 64
python visualizar_costo_img2voxel.py --resolucion 64
```

**Salidas:** `checkpoints/img2voxel_mejor_R{32,64}.pth`,
`resultados/resumen_img2voxel_R{32,64}.json`.

---

## Herramientas de análisis y figuras

Scripts de soporte para el capítulo de metodología. No forman parte del
flujo de procesamiento experimental (que produce los resultados de
clasificación); documentan y validan el proceso geométrico y su costo.

```bash
cd Scripts_Analisis

# 6 figuras vectoriales del flujo .off -> octree
# (malla, nube de puntos, voxel 32^3, voxel 64^3, niveles L=0..5, L=0..6)
python generar_figuras_metodologia.py --off "ruta\a\chair_0001.off" --salida figuras_tesis

# Tiempo y memoria de un archivo individual: 4 magnitudes distintas
# (pico transitorio, memoria ocupada por la representacion en Python,
#  memoria de la rejilla densa equivalente, almacenamiento en disco)
python medir_tiempo_memoria.py --off "ruta\a\chair_0001.off" --repeticiones 10

# Lo mismo para UN SOLO archivo, sin necesitar el dataset completo
python medir_metricas_off.py --off "ruta\a\chair_0001.off"

# Procesa TODO el conjunto de datos: nodos, ocupacion, tiempos,
# comparacion disperso vs. denso (ambos con la MISMA compresion .npz)
python medir_metricas_off.py --split ambos

# Genera 3 resumenes: train, test, y CONJUNTO combinado
python tabla_resumen_octree.py --split ambos

# Convierte los JSON de los 3 scripts anteriores en tablas SVG/PNG
python generar_imagenes_json.py --nombre chair_0001

# Tabla comparativa final: SVM vs Random Forest vs Net5
python resumen_metricas_completo.py
```

### Comparación de tiempos de construcción: árbol vs. rejilla densa

`medir_metricas_off.py` mide el tiempo de construcción de **ambas**
representaciones de forma **verdaderamente independiente**: el árbol se
construye con `octree_real.py::construir_octree()`, y la rejilla densa se
construye **directamente desde los mismos puntos y normales**, con
`octree.py::construir_grid_octree()`, **sin pasar por el árbol en ningún
momento**. Ambas rutas comparten únicamente el costo de lectura,
normalización y muestreo (medido aparte, una sola vez).

Esto produce tres totales de tiempo por objeto, claramente diferenciados:

| Campo | Qué mide |
|---|---|
| `t_total_ms` | Solo la ruta del octree (lectura+normalización+muestreo+árbol+guardado disperso) |
| `t_total_denso_ms` | Solo la ruta densa independiente (los mismos costos compartidos+rejilla directa+guardado denso) |
| `t_total_experimento_ms` | Ambas representaciones juntas, sin duplicar los costos compartidos — **este es el valor comparable contra el tiempo real de ejecución paralela** |

El **tiempo real paralelo** (`tiempo_real_paralelo_min`, medido con
`time.time()` alrededor del `ProcessPoolExecutor`) refleja el tiempo total
de ejecución del experimento completo — el procesamiento de **ambas**
representaciones por objeto. No debe compararse directamente contra
`t_total_ms` (que solo cubre el árbol); la comparación correcta es contra
la suma de `t_total_experimento_ms` sobre todos los objetos.

### Memoria: tres magnitudes distintas, nunca deben confundirse

| Magnitud | Qué mide | Cómo se mide |
|---|---|---|
| Pico transitorio | Memoria máxima **durante** la construcción (incluye basura temporal de la recursión) | `tracemalloc`, durante la ejecución |
| Memoria ocupada por la representación en Python | Memoria real y persistente del árbol/rejilla **ya construidos** | `sys.getsizeof()` recursivo, sin doble conteo de `nbytes` |
| Almacenamiento en disco | Tamaño real del archivo `.npz` comprimido | `Path.stat().st_size` |

> **Corrección de doble conteo:** una versión anterior sumaba
> `sys.getsizeof(array) + array.nbytes`, pero `sys.getsizeof()` sobre un
> array de NumPy que posee su propio buffer **ya incluye** `nbytes`.
> Sumarlos duplicaba la cifra reportada (~1024 KiB en vez de ~512 KiB para
> 32³). Corregido en `octree_real.py::medir_memoria_real_python()`.

### Comparación de almacenamiento: siempre archivo real contra archivo real

`medir_metricas_off.py` guarda **ambas** representaciones (disperso y
denso) con la **misma compresión** (`np.savez_compressed`) y compara sus
tamaños reales de archivo — nunca un archivo comprimido contra una fórmula
teórica sin comprimir. El campo correspondiente se llama
`factor_reduccion_almacenamiento_{R}` (antes `factor_ahorro_real`, renombrado
porque corresponde específicamente a esta comparación de archivos, no a un
"ahorro" general). Por clase, este factor se calcula como el **cociente de
sumas totales** (tamaño denso total de la clase / tamaño disperso total de
la clase), no como el promedio de los factores individuales de cada objeto
(matemáticamente distinto, por la desigualdad de Jensen).

Todas las unidades de tamaño usan **KiB/MiB** (potencias de 1024), nunca
KB/MB, para evitar ambigüedad con las unidades decimales (potencias de 1000).

### Con `--split ambos`, tres resúmenes (no dos)

`tabla_resumen_octree.py --split ambos` genera **tres** archivos
(`resumen_octree_train.json`, `resumen_octree_test.json`,
`resumen_octree_conjunto.json`), no solo los dos separados — el resumen
`conjunto` combina ambos splits sin filtrar.

---

## Especificaciones técnicas exactas

### Criterio de ocupación de una celda / nodo hoja

Un nodo hoja (o celda de la rejilla densa) se considera **ocupado** si al
menos uno de los puntos muestreados sobre la superficie cae dentro de sus
límites espaciales al cuantizar `[-1,1]³` a índices enteros `[0, R)`. No se
aplica ningún umbral de densidad mínima en las Fases 2, 3a ni 3b; el
criterio es binario y determinista dado el conjunto de puntos muestreado.
En el árbol real, un nodo **interno** existe (no fue podado) si y solo si
al menos una hoja ocupada desciende de él.

En el experimento adicional Img2Voxel se aplica una dilatación morfológica
adicional (`max_pool3d`, kernel 3, 4 pasadas) **únicamente** durante el
entrenamiento del decoder, para compensar la baja ocupación de las mallas
huecas de ModelNet40. Esta dilatación **no se aplica** en las Fases 2, 3a
ni 3b.

### Tipo de dato

Todos los arrays numéricos (centros, normales, grids materializados) usan
`numpy.float32`. Las profundidades y máscaras de hijos en el formato
disperso usan `numpy.uint8`.

### Estructura guardada por objeto (formato disperso, Fase 2)

```python
{
    "profundidades":    np.ndarray,  # uint8, (M,) -- M = nodos existentes
    "mascaras":         np.ndarray,  # uint8, (M,) -- bits de hijos existentes
    "normales":         np.ndarray,  # float32, (M, 3) -- solo valido en hojas
    "etiqueta":         int,
    "profundidad_max":  int,         # 5 (32^3) o 6 (64^3)
}
```

### Nombres y rutas de archivos de salida

| Fase | Patrón de archivo | Contenido |
|---|---|---|
| 1 | `logs/experiment_log.json` | Config., hardware, versiones |
| 1 | `logs/particion_indices.npz` | Índices train/val |
| 2 | `data/octrees_{R}/<clase>/<split>/<nombre>.npz` | Árbol disperso, estructura completa |
| 2 | `resultados/metricas_off_completo.csv` | Métricas por objeto (nodos, tiempos, tamaños disperso/denso) |
| 2 | `resultados/metricas_off_resumen.csv` | Métricas por clase (factor de reducción con cociente de sumas) |
| 2 | `resultados/metricas_off_global.json` | Resumen global + tiempo real paralelo por split |
| 2 | `resultados/resumen_octree_{train,test,conjunto}.json` | Resúmenes de `tabla_resumen_octree.py` |
| 3a | `resultados/resumen_hce_R{R}.json` | Métricas SVM + RF |
| 3b | `resultados/resumen_net5_R{R}.json` | Métricas Net5 |
| 5 | `resultados/resumen_img2voxel_R{R}.json` | Métricas Img2Voxel |

### Semilla usada en cada fase

`seed = 42` en todas las fases sin excepción.

### Parámetros de muestreo

- Puntos muestreados por objeto: **20 000** (Fases 2, 3a, 3b, y Fase 5 para renderizado).
- Método: muestreo *area-weighted* con coordenadas baricéntricas uniformes.

### Parámetros de HCE

Total de features: `(L+1) + 12`, donde `L` es la profundidad hoja del
árbol (`L=5` para 32³ → 18 features; `L=6` para 64³ → 19 features).

### Hiperparámetros de SVM y Random Forest

| Modelo | Hiperparámetro | Valor / rango |
|---|---|---|
| SVM | kernel | RBF |
| SVM | `C` | `{0.1, 1, 10, 100}` |
| SVM | `gamma` | `{"scale", 0.001, 0.01, 0.1}` |
| SVM | validación cruzada | `cv=3` |
| Random Forest | `n_estimators` | 100 |
| Random Forest | `max_depth` | `None` |
| Random Forest | `random_state` | 42 |

### Criterios de equivalencia experimental

- Todas las pruebas de inferencia se ejecutan en el mismo equipo (ver
  [Condiciones de ejecución](#condiciones-de-ejecución-de-los-resultados-reportados)).
- **Sin aumento de datos** en ningún enfoque, para aislar el efecto de la
  resolución del octree.
- La exactitud se reporta sobre el conjunto de prueba oficial (2 468
  muestras), nunca sobre el conjunto de validación interno.

---

## Registros y resultados

```
Registros/
├── fase1/
│   └── experiment_log.json
├── fase2_octree/
│   ├── metricas_off_completo.csv
│   ├── metricas_off_resumen.csv
│   ├── metricas_off_global.json
│   ├── resumen_octree_train.json
│   ├── resumen_octree_test.json
│   └── resumen_octree_conjunto.json
├── fase3_hce/
│   ├── resumen_hce_R32.json
│   └── resumen_hce_R64.json
├── fase3_net5/
│   ├── resumen_net5_R32.json
│   ├── resumen_net5_R64.json
│   ├── net5_historial_R32.csv
│   └── net5_historial_R64.csv
└── fase5_img2voxel/
    ├── resumen_img2voxel_R32.json
    └── resumen_img2voxel_R64.json
```

Estos archivos son la fuente de verdad para las tablas y figuras reportadas
en el documento de tesis. Junto con la sección
[Condiciones de ejecución](#condiciones-de-ejecución-de-los-resultados-reportados)
(equipo, commit exacto, parámetros), permiten auditar y reproducir
cualquier cifra citada sin necesidad de acceso al dataset completo ni a los
pesos de los modelos.
