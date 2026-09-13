"""
preprocesar_octrees.py - Fase 2
=================================
Convierte TODO el dataset ModelNet40 (train + test) a un OCTREE REAL
(arbol con nodo raiz, subdivision recursiva y poda de ramas vacias,
ver octree_real.py) en ambas resoluciones (32^3 y 64^3), y lo guarda
en disco en FORMATO DISPERSO -- solo las hojas ocupadas, no una
rejilla densa R^3.

CAMBIO IMPORTANTE respecto a versiones anteriores: el .npz generado
ya NO contiene un array denso (4, R, R, R). Contiene unicamente los
centros y normales de las hojas ocupadas del arbol real, mas la
etiqueta de clase y la profundidad maxima. El tamaño en disco es
proporcional al numero de hojas ocupadas (tipicamente 1-3% de R^3),
no al volumen total del espacio.

Estructura de salida:
    data/octrees_32/<clase>/<split>/<archivo>.npz
    data/octrees_64/<clase>/<split>/<archivo>.npz

Cada .npz contiene (ver octree_real.py::guardar_octree_disperso):
    centros_hoja    : array (N_hojas, 3) float32
    normales_hoja   : array (N_hojas, 3) float32
    etiqueta        : int (indice de clase)
    profundidad_max : int (L de la hoja: 5 para R=32, 6 para R=64)

La materializacion a grid denso (necesaria solo para alimentar Net5,
que usa Conv3d) ocurre en el Dataset de PyTorch en el momento de cargar
cada muestra (ver fase3_net5/net5_dataset.py), nunca se persiste densa
en disco.

Uso:
    python preprocesar_octrees.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from octree import leer_off, normalizar_malla, muestrear_superficie_con_normales, profundidad_de
from octree_real import (
    construir_octree, guardar_octree_disperso, recolectar_hojas,
    contar_nodos_totales,
)

# ── Configuracion ──────────────────────────────────────────────
RAIZ_DATASET = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\Dataset\ModelNet40")
RAIZ_SALIDA  = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\data")
RESOLUCIONES = [32, 64]
N_PUNTOS_MUESTREO = 20000
SEED = 42
N_PROCESOS = 10  # Ryzen 7 5700X: 8 nucleos/16 hilos. Dejamos algo de margen para el sistema.

CLASES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf",
    "bottle", "bowl", "car", "chair", "cone",
    "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
    "laptop", "mantel", "monitor", "night_stand", "person",
    "piano", "plant", "radio", "range_hood", "sink",
    "sofa", "stairs", "stool", "table", "tent",
    "toilet", "tv_stand", "vase", "wardrobe", "xbox",
]
CLASE2IDX = {c: i for i, c in enumerate(CLASES)}


# ──────────────────────────────────────────────────────────────
# RECOLECCION DE ARCHIVOS
# ──────────────────────────────────────────────────────────────

def recolectar_archivos(raiz: Path, split: str) -> list:
    """Retorna lista de (ruta_off, etiqueta_int, nombre_archivo, clase)."""
    muestras = []
    for clase in CLASES:
        carpeta = raiz / clase / split
        if not carpeta.exists():
            continue
        etiqueta = CLASE2IDX[clase]
        for archivo in sorted(carpeta.glob("*.off")):
            muestras.append((str(archivo), etiqueta, archivo.stem, clase))
    return muestras


# ──────────────────────────────────────────────────────────────
# PROCESAMIENTO DE UNA MUESTRA (ejecutado en worker process)
# ──────────────────────────────────────────────────────────────

def procesar_una_muestra(args: tuple) -> tuple:
    """
    Procesa un archivo .off: construye el octree REAL (con poda) para
    cada resolucion y guarda su forma DISPERSA en disco.

    Retorna (nombre_archivo, exito: bool, error: str|None, stats: dict|None)
    stats contiene, por resolucion, el numero de hojas ocupadas y el
    numero total de nodos del arbol, para el resumen final de ahorro
    de memoria.
    """
    ruta_off, etiqueta, nombre, clase, split, idx_global = args

    try:
        # Leer y normalizar la malla UNA sola vez (compartido entre
        # resoluciones, ya que el muestreo de puntos es identico salvo
        # por la seed determinista por muestra)
        verts, caras = leer_off(ruta_off)
        verts = normalizar_malla(verts)

        rng = np.random.default_rng(SEED + idx_global)
        pts, normales = muestrear_superficie_con_normales(
            verts, caras, N_PUNTOS_MUESTREO, rng,
        )

        stats = {}
        for R in RESOLUCIONES:
            profundidad_max = profundidad_de(R)   # 5 para R=32, 6 para R=64

            raiz = construir_octree(pts, normales, profundidad_max=profundidad_max)

            dir_salida = RAIZ_SALIDA / f"octrees_{R}" / clase / split
            dir_salida.mkdir(parents=True, exist_ok=True)
            ruta_salida = dir_salida / f"{nombre}.npz"

            guardar_octree_disperso(raiz, str(ruta_salida), etiqueta, profundidad_max)

            n_hojas = len(recolectar_hojas(raiz))
            n_nodos = contar_nodos_totales(raiz)
            tam_archivo_kb = ruta_salida.stat().st_size / 1024

            stats[R] = {
                "n_hojas": n_hojas,
                "n_nodos_arbol": n_nodos,
                "tam_archivo_kb": round(tam_archivo_kb, 3),
            }

        return (nombre, True, None, stats)

    except Exception as e:
        return (nombre, False, str(e), None)


# ──────────────────────────────────────────────────────────────
# MAIN: procesar train y test en paralelo
# ──────────────────────────────────────────────────────────────

def procesar_split(split: str):
    print(f"\n{'='*60}")
    print(f"  Procesando split: {split.upper()}")
    print(f"{'='*60}")

    muestras = recolectar_archivos(RAIZ_DATASET, split)
    print(f"  Archivos encontrados: {len(muestras)}")

    if len(muestras) == 0:
        print(f"  ADVERTENCIA: no se encontraron archivos para '{split}'")
        return [], [], {}

    tareas = [
        (ruta, etiqueta, nombre, clase, split, i)
        for i, (ruta, etiqueta, nombre, clase) in enumerate(muestras)
    ]

    exitosos = []
    fallidos  = []
    stats_por_resolucion = {R: {"n_hojas": [], "n_nodos_arbol": [], "tam_archivo_kb": []}
                            for R in RESOLUCIONES}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_PROCESOS) as executor:
        futuros = {executor.submit(procesar_una_muestra, t): t for t in tareas}

        barra = tqdm(as_completed(futuros), total=len(futuros), ncols=80, desc=f"  {split}")
        for futuro in barra:
            nombre, exito, error, stats = futuro.result()
            if exito:
                exitosos.append(nombre)
                for R in RESOLUCIONES:
                    stats_por_resolucion[R]["n_hojas"].append(stats[R]["n_hojas"])
                    stats_por_resolucion[R]["n_nodos_arbol"].append(stats[R]["n_nodos_arbol"])
                    stats_por_resolucion[R]["tam_archivo_kb"].append(stats[R]["tam_archivo_kb"])
            else:
                fallidos.append((nombre, error))

    t1 = time.time()

    print(f"\n  Completado en {(t1-t0)/60:.1f} min")
    print(f"  Exitosos : {len(exitosos)}")
    print(f"  Fallidos : {len(fallidos)}")

    if fallidos:
        print("\n  Primeros errores:")
        for nombre, error in fallidos[:5]:
            print(f"    {nombre}: {error}")

    # Resumen de ahorro de memoria (disperso vs. formato denso anterior)
    print(f"\n  Resumen de almacenamiento disperso ({split}):")
    for R in RESOLUCIONES:
        s = stats_por_resolucion[R]
        if not s["n_hojas"]:
            continue
        n_hojas_media = np.mean(s["n_hojas"])
        tam_kb_media = np.mean(s["tam_archivo_kb"])
        tam_kb_total = np.sum(s["tam_archivo_kb"])
        mem_densa_kb = 4 * (R ** 3) * 4 / 1024   # 4 canales, float32, formato anterior
        mem_densa_total_mb = len(exitosos) * mem_densa_kb / 1024

        print(f"    R={R}^3:")
        print(f"      Hojas ocupadas (media)    : {n_hojas_media:.0f}")
        print(f"      Tamano .npz (media)       : {tam_kb_media:.2f} KB")
        print(f"      Tamano .npz (total split) : {tam_kb_total/1024:.1f} MB")
        print(f"      Memoria densa equivalente : {mem_densa_total_mb:.1f} MB "
              f"(formato anterior, ya no se usa)")
        if tam_kb_total > 0:
            print(f"      Factor de ahorro          : "
                  f"{(mem_densa_total_mb*1024)/tam_kb_total:.1f}x")

    return exitosos, fallidos, stats_por_resolucion


def main():
    print("=" * 60)
    print("  FASE 2: PREPROCESAMIENTO DE OCTREES REALES (32^3 y 64^3)")
    print("  Formato de salida: DISPERSO (solo hojas ocupadas)")
    print("=" * 60)
    print(f"  Dataset origen : {RAIZ_DATASET}")
    print(f"  Salida         : {RAIZ_SALIDA}")
    print(f"  Resoluciones   : {RESOLUCIONES}")
    print(f"  Puntos muestreo: {N_PUNTOS_MUESTREO}")
    print(f"  Procesos       : {N_PROCESOS}")

    resumen = {}
    stats_globales = {}
    for split in ["train", "test"]:
        exitosos, fallidos, stats = procesar_split(split)
        resumen[split] = {"exitosos": len(exitosos), "fallidos": len(fallidos)}
        stats_globales[split] = stats

    print("\n" + "=" * 60)
    print("  RESUMEN FINAL")
    print("=" * 60)
    for split, datos in resumen.items():
        print(f"  {split:10s}: {datos['exitosos']} exitosos, {datos['fallidos']} fallidos")
    print(f"\n  Archivos guardados en: {RAIZ_SALIDA}")
    print("  Estructura: data/octrees_<R>/<clase>/<split>/<archivo>.npz")
    print("  Formato: DISPERSO (centros_hoja, normales_hoja, etiqueta, profundidad_max)")
    print("=" * 60)

    # Guardar estadisticas de ahorro de memoria para citar en la tesis
    resumen_memoria = {}
    for split, stats in stats_globales.items():
        resumen_memoria[split] = {}
        for R, s in stats.items():
            if not s["n_hojas"]:
                continue
            resumen_memoria[split][str(R)] = {
                "n_objetos": len(s["n_hojas"]),
                "hojas_media": round(float(np.mean(s["n_hojas"])), 1),
                "hojas_std": round(float(np.std(s["n_hojas"])), 1),
                "tam_npz_medio_kb": round(float(np.mean(s["tam_archivo_kb"])), 3),
                "tam_npz_total_mb": round(float(np.sum(s["tam_archivo_kb"])) / 1024, 2),
                "mem_densa_equivalente_total_mb": round(
                    len(s["n_hojas"]) * 4 * (R**3) * 4 / 1024 / 1024, 1
                ),
            }

    ruta_json = RAIZ_SALIDA / "resumen_preprocesamiento_disperso.json"
    with open(ruta_json, "w") as f:
        json.dump(resumen_memoria, f, indent=2)
    print(f"\n  Resumen de ahorro de memoria guardado en: {ruta_json}")


if __name__ == "__main__":
    main()
