# Octree adaptativo — Objetivo específico 1

Este repositorio contiene el flujo reproducible para construir, persistir y
comparar dos representaciones tridimensionales de ModelNet40:

1. una rejilla densa de referencia;
2. un octree adaptativo real, con nodo raíz, hasta ocho hijos por nodo y poda
   explícita de regiones vacías.

El alcance de esta versión termina en la validación de las representaciones.
**HCE, SVM, Random Forest, Net5 e Img2Voxel están fuera del objetivo específico
1 y no forman parte de este procedimiento de aceptación.** Sus directorios se
conservan como trabajo posterior, pero no deben ejecutarse hasta aprobar esta
etapa.

## Estado de aceptación

El código implementa el formato definitivo, las pruebas y la generación de
métricas. El objetivo solo podrá declararse terminado cuando se ejecute
ModelNet40 completo —9 843 modelos de entrenamiento y 2 468 de prueba— en
32³ y 64³, sin errores, y el consolidador indique:

```text
ModelNet40 completo: SI
```

Los resultados antiguos de `resultados/` no sustituyen esa regeneración.

## Componentes relevantes

```text
Octree/
├── fase2_octree/
│   ├── octree.py                    # lectura, normalización y rejilla densa
│   ├── octree_real.py               # árbol adaptativo y formato NPZ v1
│   ├── validacion_objetivo1.py      # equivalencia y manifiestos
│   ├── preprocesar_octrees.py       # generación completa de ModelNet40
│   ├── validar_chair_0001.py        # prueba controlada independiente
│   └── visualizar_arbol_real.py     # visualización opcional con rutas CLI
├── Scripts_Analisis/
│   ├── medir_tiempo_memoria.py      # medición controlada de un modelo
│   └── medir_metricas_off.py        # verificación y consolidación final
├── tests/                            # pruebas sintéticas autocontenidas
├── requirements-objetivo1.txt
├── pytest.ini
└── README.md
```

No hay rutas absolutas en los componentes del objetivo 1. Todas las
ubicaciones se resuelven desde el repositorio o se reciben como argumentos.

## Instalación desde un clon limpio

Requisitos: Git y Python 3.10 o posterior.

### Windows PowerShell

```powershell
git clone https://github.com/ricardoal94/Octree.git
cd Octree
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-objetivo1.txt
```

### Linux o macOS

```bash
git clone https://github.com/ricardoal94/Octree.git
cd Octree
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-objetivo1.txt
```

Las dependencias de aprendizaje automático no son necesarias para este
objetivo.

## Ubicación de ModelNet40

El dataset no se versiona en Git. La estructura esperada es la partición
oficial en formato OFF:

```text
ModelNet40/
├── airplane/
│   ├── train/*.off
│   └── test/*.off
├── chair/
│   ├── train/chair_0001.off
│   └── test/*.off
└── ... 38 categorías adicionales
```

Puede colocarse en `Dataset/ModelNet40/` o indicarse mediante
`--dataset-root`. No es necesario editar ningún archivo Python.

## 1. Pruebas automatizadas

Desde la raíz del repositorio:

```bash
python -m pytest
```

Las pruebas son autocontenidas y no dependen de octrees o métricas generados
previamente. Verifican:

- correspondencia celda por celda entre rejilla densa y hojas del octree en
  32³ y 64³;
- casos sobre fronteras del dominio y planos de subdivisión;
- persistencia y reconstrucción completa de la jerarquía;
- rechazo explícito de archivos antiguos sin versión;
- inclusión de la coherencia normal en la carga binaria;
- determinismo de semillas y de la estructura serializada;
- generación integrada de NPZ, manifiesto y métricas.

## 2. Prueba controlada con `chair_0001`

Esta prueba vuelve a crear todo desde la malla original; no consume resultados
anteriores:

```bash
python fase2_octree/validar_chair_0001.py \
  --off Dataset/ModelNet40/chair/train/chair_0001.off
```

En Windows PowerShell puede escribirse en una sola línea. La salida por defecto
es:

```text
resultados/objetivo1/validacion_chair_0001.json
```

La prueba debe superar tanto 32³ como 64³ antes de procesar el dataset.

## 3. Diagnóstico con un solo modelo

Antes de la corrida completa:

```bash
python fase2_octree/preprocesar_octrees.py \
  --dataset-root Dataset/ModelNet40 \
  --limite 1 \
  --procesos 1 \
  --sobrescribir
```

El resumen indicará `Completo: NO`, porque `--limite` crea deliberadamente una
corrida corta. Esto no es un fallo.

## 4. Generación completa

```bash
python fase2_octree/preprocesar_octrees.py \
  --dataset-root Dataset/ModelNet40 \
  --output-root data \
  --resultados-dir resultados/objetivo1 \
  --resoluciones 32 64 \
  --n-puntos 20000 \
  --semilla 42 \
  --procesos 8 \
  --sobrescribir
```

El número de procesos debe ajustarse a la memoria y a los núcleos disponibles.
Omitir `--limite` es obligatorio en la ejecución final.

Para cada malla se realiza el siguiente procedimiento:

1. lectura y normalización al cubo `[-1, 1]³`;
2. muestreo de 20 000 puntos con una semilla estable por modelo;
3. construcción del octree de profundidad 5 y 6 usando la misma nube;
4. construcción independiente de la rejilla densa de 32³ y 64³;
5. comparación exacta de ocupación y comparación numérica de normales;
6. guardado del octree, recarga y reconstrucción de la jerarquía;
7. repetición de la prueba de equivalencia después de cargar;
8. escritura atómica del manifiesto y las métricas.

Una salida parcial o con errores nunca se marca como ModelNet40 completo.

## Formato definitivo del octree

La única versión admitida es:

```text
format_name    = octree_adaptativo_modelnet40
format_version = 1.0.0
```

Cada NPZ conserva:

| Campo | Tipo | Significado |
|---|---:|---|
| `profundidades` | `uint8[M]` | Profundidad de cada nodo en DFS preorden |
| `mascaras` | `uint8[M]` | Bits que indican cuáles de los ocho hijos existen |
| `normales` | `float32[M,3]` | Normal unitaria; cero en nodos internos |
| `coherencias` | `float32[M]` | Magnitud de la normal media antes de normalizar |
| `etiqueta` | `int16` | Índice de categoría ModelNet40 |
| `profundidad_max` | `uint8` | 5 para 32³ y 6 para 64³ |
| `resolucion` | `uint16` | Resolución espacial equivalente |
| `model_id` | texto | Identificador de la malla |
| `categoria` | texto | Categoría ModelNet40 |
| `split` | texto | `train` o `test` |
| `semilla_muestreo` | `int64` | Semilla específica del modelo |
| `n_puntos_muestreo` | `int64` | Número de puntos utilizados |
| `archivo_origen_sha256` | texto | Huella de la malla original |

El centro y el tamaño de cada nodo se reconstruyen de forma determinista a
partir de su ruta en el árbol. Los lectores rechazan archivos antiguos o
incompatibles; no existe una conversión silenciosa.

La carga estructural sin comprimir es de 18 bytes por nodo:

```text
1 profundidad + 1 máscara + 12 normal + 4 coherencia
```

Además se reporta la suma sin comprimir de todos los campos del NPZ, incluidos
los metadatos.

## Manifiesto por modelo

Los manifiestos se generan en:

```text
data/manifests/<categoria>/<split>/<model_id>.json
```

Cada uno registra:

- identificador, categoría, partición y SHA-256 de la malla;
- semilla base, semilla derivada, algoritmo de derivación y puntos muestreados;
- versión del formato;
- archivos NPZ, tamaños y SHA-256;
- métricas completas para 32³ y 64³;
- resultados de equivalencia antes del guardado y después de la carga.

La semilla específica se deriva mediante SHA-256 de la ruta relativa. Por
ello no depende del orden de los procesos, del sistema operativo ni del número
de trabajadores.

## Métricas reportadas por modelo y resolución

| Métrica | Definición |
|---|---|
| Nodos por nivel | Nodos existentes desde la raíz hasta la profundidad máxima |
| Nodos totales | Nodos internos más hojas ocupadas |
| Hojas ocupadas | Celdas ocupadas en el nivel hoja |
| Ocupación | `100 × hojas / R³` |
| Tiempo de construcción | Tiempo de `construir_octree`, en milisegundos |
| Memoria pico | Pico transitorio medido con `tracemalloc`, en bytes |
| Estructura Python | Grafo de objetos retenido por el árbol, sin doble conteo |
| Binario sin comprimir | Payload de todos los arrays del NPZ antes de ZIP |
| NPZ real | Tamaño del archivo comprimido en disco |
| Referencia densa | Tensor `float32` de forma `(4,R,R,R)` y su NPZ comparable |

Las magnitudes no se mezclan: memoria pico, estructura residente, payload
binario y archivo comprimido se conservan como mediciones distintas.

## 5. Consolidación y verificación final

Después de generar todos los objetos:

```bash
python Scripts_Analisis/medir_metricas_off.py \
  --manifests-dir data/manifests \
  --output-root data \
  --resultados-dir resultados/objetivo1
```

El consolidador comprueba:

- 9 843 modelos de entrenamiento y 2 468 de prueba;
- exactamente un manifiesto por modelo;
- dos resoluciones por manifiesto;
- equivalencia antes y después de cargar;
- existencia y SHA-256 de cada NPZ;
- coincidencia entre metadatos del NPZ y del manifiesto.

Produce:

```text
resultados/objetivo1/metricas_modelnet40_verificadas.csv
resultados/objetivo1/resumen_metricas_modelnet40.json
```

`--permitir-incompleto` está reservado para diagnósticos y nunca debe usarse
para declarar cumplido el objetivo.

## Criterio final de aprobación

El objetivo específico 1 puede aprobarse únicamente si:

- `python -m pytest` termina sin fallos;
- `chair_0001` supera 32³ y 64³;
- el procesamiento completo no reporta modelos fallidos;
- existen 12 311 manifiestos y 24 622 NPZ;
- todas las equivalencias son verdaderas;
- el consolidador informa `ModelNet40 completo: SI`;
- el código y los resultados consolidados quedan asociados a un commit
  identificable.

Hasta cumplir toda esta lista no deben iniciarse los experimentos de HCE,
clasificación o aprendizaje profundo.
