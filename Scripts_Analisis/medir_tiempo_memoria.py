"""
medir_tiempo_memoria.py
========================
Mide el tiempo de construccion y la MEMORIA REAL (via tracemalloc) de
cada etapa del pipeline .off -> octree REAL -> descriptores HCE, para
un archivo .off dado.

CORRECCION respecto a versiones anteriores: las etapas de octree ya no
construyen una rejilla densa (formato descartado tras la observacion
de Andres Gonzalez sobre que el codigo no construia un octree real).
Ahora se mide:
  1. Lectura del archivo .off
  2. Normalizacion geometrica
  3. Muestreo de superficie (area-weighted)
  4. Construccion del ARBOL REAL (con poda) -- 32^3 y 64^3
  5. Guardado en disco en formato disperso con ESTRUCTURA JERARQUICA
     COMPLETA (ver octree_real.py::guardar_octree_disperso)
  6. Extraccion de descriptores HCE directamente desde el .npz disperso
     (ver hce_extraccion.py::extraer_descriptores_hce_desde_npz)

Memoria medida:
  - Memoria PICO REAL de cada etapa (tracemalloc.get_traced_memory)
  - Memoria de las estructuras de datos resultantes (nbytes de arrays)
  - Tamaño REAL del archivo .npz en disco (no estimado)
  - Numero de nodos totales del arbol y de hojas ocupadas
  - Factor de ahorro vs. la rejilla densa equivalente (formato descartado)

Uso:
    python medir_tiempo_memoria.py --off chair_0001.off
    python medir_tiempo_memoria.py --off chair_0001.off --repeticiones 20
"""

import argparse
import json
import sys
import tempfile
import time
import tracemalloc
import numpy as np
from pathlib import Path

# ── Localizar los modulos del pipeline (fase2_octree, fase3_hce) ──
RAIZ_PROYECTO = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))
sys.path.insert(0, str(RAIZ_PROYECTO / "fase3_hce"))

from octree import leer_off, normalizar_malla, muestrear_superficie_con_normales
from octree_real import (
    construir_octree, guardar_octree_disperso, reconstruir_octree_desde_npz,
    recolectar_hojas, contar_nodos_totales, comparar_memoria, octree_a_grid_denso,
)
from hce_extraccion import extraer_descriptores_hce_desde_npz


PROFUNDIDAD_POR_RESOLUCION = {32: 5, 64: 6}


# ──────────────────────────────────────────────────────────────
# MEDICION GENERICA DE UNA ETAPA
# ──────────────────────────────────────────────────────────────

def medir_etapa(fn, *args, repeticiones: int = 10) -> tuple:
    """
    Ejecuta fn(*args) `repeticiones` veces y retorna:
      (resultado, tiempo_medio_ms, tiempo_min_ms, tiempo_max_ms, mem_pico_kib)

    La memoria PICO REAL se mide con tracemalloc en la PRIMERA
    ejecucion (tracemalloc introduce overhead, por eso no se activa en
    todas las repeticiones). El tiempo se mide con time.perf_counter
    en todas las repeticiones para estabilidad estadistica.
    """
    tracemalloc.start()
    resultado = fn(*args)
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    mem_pico_kib = pico / 1024

    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        fn(*args)
        tiempos.append((time.perf_counter() - t0) * 1000)

    return (
        resultado,
        round(float(np.mean(tiempos)), 4),
        round(float(np.min(tiempos)), 4),
        round(float(np.max(tiempos)), 4),
        round(mem_pico_kib, 2),
    )


def memoria_array_kib(arr: np.ndarray) -> float:
    return round(arr.nbytes / 1024, 2)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mide tiempo y memoria REAL del pipeline .off -> octree real -> HCE"
    )
    parser.add_argument("--off", type=str, required=True, help="Ruta al archivo .off")
    parser.add_argument("--n_puntos", type=int, default=20000,
                        help="Puntos a muestrear sobre la superficie (default: 20000, "
                             "igual al usado por preprocesar_octrees.py)")
    parser.add_argument("--repeticiones", type=int, default=10,
                        help="Repeticiones para estabilizar mediciones de tiempo (default: 10)")
    parser.add_argument("--salida", type=str, default=None,
                        help="Ruta JSON donde guardar los resultados (opcional)")
    args = parser.parse_args()

    ruta = args.off
    N    = args.n_puntos
    REP  = args.repeticiones

    print("=" * 70)
    print(f"  MEDICION DE TIEMPO Y MEMORIA REAL — {Path(ruta).name}")
    print(f"  Pipeline: .off -> octree REAL (con poda) -> descriptores HCE")
    print("=" * 70)
    print(f"  Puntos de muestreo : {N:,}")
    print(f"  Repeticiones       : {REP}")
    print()

    resultados = {}

    # ── Etapa 1: Lectura ──────────────────────────────────────
    (verts_raw, caras), t_med, t_min, t_max, mem_pico = \
        medir_etapa(leer_off, ruta, repeticiones=REP)

    n_v, n_c = len(verts_raw), len(caras) if caras is not None else 0

    resultados["lectura"] = {
        "descripcion": "Lectura del archivo .off",
        "n_vertices": n_v, "n_caras": n_c,
        "t_media_ms": t_med, "t_min_ms": t_min, "t_max_ms": t_max,
        "mem_pico_kib": mem_pico,
        "mem_vertices_kib": memoria_array_kib(verts_raw),
        "mem_caras_kib": memoria_array_kib(caras) if caras is not None else 0.0,
    }

    # ── Etapa 2: Normalizacion ────────────────────────────────
    verts_norm, t_med, t_min, t_max, mem_pico = \
        medir_etapa(normalizar_malla, verts_raw, repeticiones=REP)

    resultados["normalizacion"] = {
        "descripcion": "Normalizacion al cubo [-1,1]^3",
        "t_media_ms": t_med, "t_min_ms": t_min, "t_max_ms": t_max,
        "mem_pico_kib": mem_pico,
        "mem_salida_kib": memoria_array_kib(verts_norm),
    }

    # ── Etapa 3: Muestreo de superficie ───────────────────────
    rng_medicion = np.random.default_rng(42)
    (pts, normales), t_med, t_min, t_max, mem_pico = medir_etapa(
        muestrear_superficie_con_normales, verts_norm, caras, N, rng_medicion,
        repeticiones=REP,
    )

    resultados["muestreo"] = {
        "descripcion": f"Muestreo area-weighted ({N:,} puntos)",
        "n_puntos": N,
        "t_media_ms": t_med, "t_min_ms": t_min, "t_max_ms": t_max,
        "mem_pico_kib": mem_pico,
        "mem_nube_kib": memoria_array_kib(pts) + memoria_array_kib(normales),
    }

    # ── Etapas 4-6 por resolucion: construccion arbol, guardado, HCE ──
    for R in (32, 64):
        profundidad_max = PROFUNDIDAD_POR_RESOLUCION[R]

        # Etapa 4: construccion del arbol REAL (con poda)
        #
        # IMPORTANTE: mem_pico (tracemalloc) mide el PICO DE ASIGNACIONES
        # TRANSITORIAS durante la ejecucion de construir_octree() -- esto
        # incluye arrays temporales creados en cada nivel de la recursion
        # (mascaras de octante, copias de subconjuntos de puntos/normales
        # al particionar), que se liberan al terminar la funcion. NO es
        # la memoria ocupada por la representacion en Python del arbol
        # ya construido. Esa magnitud se mide por separado, de forma
        # recursiva desde la raiz, en mem_teorica["memoria_python_kib"]
        # (ver medir_memoria_real_python en octree_real.py).
        raiz, t_med, t_min, t_max, mem_pico_transitorio = medir_etapa(
            construir_octree, pts, normales, profundidad_max, repeticiones=REP,
        )
        n_nodos = contar_nodos_totales(raiz)
        n_hojas = len(recolectar_hojas(raiz))
        mem_teorica = comparar_memoria(raiz, R, profundidad_max)

        # Memoria ocupada por la REPRESENTACION EN PYTHON de la rejilla
        # densa equivalente, medida con el MISMO criterio que se aplico
        # al arbol -- no la formula teorica 4*R^3*4.
        #
        # CORRECCION (observacion de Andres Gonzalez): sys.getsizeof()
        # sobre un array de numpy que posee su propio buffer YA INCLUYE
        # el tamaño de los datos (equivalente a nbytes + un pequeño
        # overhead de objeto). Sumar nbytes de nuevo duplicaba esa
        # memoria (se veian ~1024 KiB en vez de ~512 KiB para 32^3, y
        # ~8192 KiB en vez de ~4096 KiB para 64^3). Se usa unicamente
        # sys.getsizeof().
        grid_denso_comparacion = octree_a_grid_denso(raiz, R)
        mem_densa_real_bytes = sys.getsizeof(grid_denso_comparacion)
        mem_densa_real_kib = round(mem_densa_real_bytes / 1024, 3)

        resultados[f"construccion_arbol_R{R}"] = {
            "descripcion": f"Construccion del octree real (con poda), R={R}^3",
            "profundidad_max": profundidad_max,
            "n_nodos_totales": n_nodos,
            "n_hojas_ocupadas": n_hojas,
            "t_media_ms": t_med, "t_min_ms": t_min, "t_max_ms": t_max,
            # Pico TRANSITORIO durante la construccion (tracemalloc);
            # NO representa el tamaño del arbol ya construido.
            "mem_pico_transitorio_kib": mem_pico_transitorio,
            # Memoria ocupada por la REPRESENTACION EN PYTHON del arbol
            # completo, medida recursivamente desde la raiz
            # (sys.getsizeof sobre cada nodo, sus hijos y sus arrays de
            # numpy, SIN sumar nbytes por separado -- ver correccion
            # de doble conteo arriba).
            "mem_python_arbol_kib": mem_teorica["memoria_python_kib"],
            # Memoria ocupada por la REPRESENTACION EN PYTHON de la
            # rejilla densa equivalente, mismo criterio que la anterior.
            "mem_python_densa_kib": mem_densa_real_kib,
            # Estimacion teorica de referencia (no medida): tamaño
            # minimo de una serializacion binaria compacta.
            "mem_binaria_estimada_kib": mem_teorica["memoria_binaria_estimada_kib"],
            "factor_ahorro_arbol_vs_densa": round(
                mem_densa_real_kib / max(mem_teorica["memoria_python_kib"], 0.001), 2
            ),
        }

        # Etapa 5: guardado en disco (estructura jerarquica completa)
        ruta_npz_tmp = str(Path(tempfile.gettempdir()) / f"medicion_{Path(ruta).stem}_R{R}.npz")

        def _guardar():
            guardar_octree_disperso(raiz, ruta_npz_tmp, etiqueta=0,
                                    profundidad_max=profundidad_max)

        _, t_med, t_min, t_max, mem_pico = medir_etapa(_guardar, repeticiones=REP)

        tam_archivo_kib = Path(ruta_npz_tmp).stat().st_size / 1024

        resultados[f"guardado_disco_R{R}"] = {
            "descripcion": f"Guardado en disco, estructura jerarquica completa, R={R}^3",
            "t_media_ms": t_med, "t_min_ms": t_min, "t_max_ms": t_max,
            "mem_pico_kib": mem_pico,
            "tam_archivo_real_kib": round(tam_archivo_kib, 3),
        }

        # Etapa 6: extraccion de descriptores HCE desde el .npz disperso
        (feats), t_med, t_min, t_max, mem_pico = medir_etapa(
            extraer_descriptores_hce_desde_npz, ruta_npz_tmp, repeticiones=REP,
        )

        resultados[f"extraccion_hce_R{R}"] = {
            "descripcion": f"Extraccion de descriptores HCE desde .npz disperso, R={R}^3",
            "n_features": len(feats),
            "t_media_ms": t_med, "t_min_ms": t_min, "t_max_ms": t_max,
            "mem_pico_kib": mem_pico,
        }

    # ── Pipeline completo (todas las etapas encadenadas) ──────
    def pipeline_completo(ruta, N):
        v, c = leer_off(ruta)
        v = normalizar_malla(v)
        rng_local = np.random.default_rng(42)
        p, n = muestrear_superficie_con_normales(v, c, N, rng_local)
        for R in (32, 64):
            pm = PROFUNDIDAD_POR_RESOLUCION[R]
            raiz_local = construir_octree(p, n, pm)
            ruta_tmp = str(Path(tempfile.gettempdir()) / f"pipeline_completo_{R}.npz")
            guardar_octree_disperso(raiz_local, ruta_tmp, 0, pm)
            extraer_descriptores_hce_desde_npz(ruta_tmp)

    _, t_med, t_min, t_max, mem_pico = medir_etapa(
        pipeline_completo, ruta, N, repeticiones=REP,
    )

    resultados["pipeline_completo"] = {
        "descripcion": "Pipeline completo: lectura + normalizacion + muestreo + "
                       "(construccion arbol + guardado + HCE) x 2 resoluciones",
        "t_media_ms": t_med, "t_min_ms": t_min, "t_max_ms": t_max,
        "mem_pico_kib": mem_pico,
        "mem_pico_mib": round(mem_pico / 1024, 3),
    }

    # ── Impresion de tabla resumen ────────────────────────────
    print(f"  Archivo : {Path(ruta).name}")
    print(f"  Vertices: {n_v:,}   Caras: {n_c:,}")
    print()

    ancho = 52
    sep = "-" * 82
    print(f"  {'ETAPA':<{ancho}} {'T.media':>9} {'Mem.pico':>10}")
    print(f"  {'':.<{ancho}} {'(ms)':>9} {'(KiB, transitorio)':>10}")
    print("  NOTA: 'Mem.pico' (tracemalloc) mide asignaciones TRANSITORIAS")
    print("  durante la operacion (arrays temporales de la recursion), NO")
    print("  el tamaño del arbol que permanece en memoria. Ver tabla de")
    print("  estructura mas abajo para el tamaño PERSISTENTE real del arbol.")
    print("  " + sep)

    orden_impresion = [
        ("1. Lectura .off", "lectura"),
        ("2. Normalizacion [-1,1]^3", "normalizacion"),
        (f"3. Muestreo superficie ({N:,} pts)", "muestreo"),
        ("4a. Construccion arbol real 32^3", "construccion_arbol_R32"),
        ("4b. Construccion arbol real 64^3", "construccion_arbol_R64"),
        ("5a. Guardado disco (estructura) 32^3", "guardado_disco_R32"),
        ("5b. Guardado disco (estructura) 64^3", "guardado_disco_R64"),
        ("6a. Extraccion HCE desde .npz 32^3", "extraccion_hce_R32"),
        ("6b. Extraccion HCE desde .npz 64^3", "extraccion_hce_R64"),
    ]

    for nombre, clave in orden_impresion:
        d = resultados[clave]
        mem_mostrar = d.get("mem_pico_kib", d.get("mem_pico_transitorio_kib", 0.0))
        print(f"  {nombre:<{ancho}} {d['t_media_ms']:>9.3f} {mem_mostrar:>10.1f}")

    print("  " + sep)
    d = resultados["pipeline_completo"]
    print(f"  {'PIPELINE COMPLETO (ambas resoluciones)':<{ancho}} "
          f"{d['t_media_ms']:>9.3f} {d['mem_pico_kib']:>10.1f}")

    # ── Tabla de estructura del arbol: 4 magnitudes claramente separadas ──
    print()
    print(f"  ESTRUCTURA DEL ARBOL — CUATRO MAGNITUDES DISTINTAS, MISMO CRITERIO")
    print("  " + sep)
    print(f"  {'Resolucion':<11}{'Nodos':>8}{'Hojas':>9}{'Transit.(KiB)':>15}"
          f"{'Python(KiB)':>13}{'Densa(KiB)':>13}{'Archivo(KiB)':>14}{'Ahorro':>9}")
    print("  " + sep)
    for R in (32, 64):
        c = resultados[f"construccion_arbol_R{R}"]
        g = resultados[f"guardado_disco_R{R}"]
        print(f"  {f'{R}^3':<11}{c['n_nodos_totales']:>8,}{c['n_hojas_ocupadas']:>9,}"
              f"{c['mem_pico_transitorio_kib']:>15.1f}{c['mem_python_arbol_kib']:>13.2f}"
              f"{c['mem_python_densa_kib']:>13.1f}{g['tam_archivo_real_kib']:>14.2f}"
              f"{c['factor_ahorro_arbol_vs_densa']:>8.1f}x")

    print()
    print("  Columnas:")
    print("    Transit. = pico TRANSITORIO durante construir_octree() (tracemalloc);")
    print("               incluye arrays temporales de la recursion, NO es el")
    print("               tamaño del arbol ya construido.")
    print("    Python   = memoria ocupada por la REPRESENTACION EN PYTHON del")
    print("               arbol completo, medida recursivamente desde la raiz")
    print("               (sys.getsizeof sobre cada nodo, sus 8 hijos y sus")
    print("               arrays de numpy -- sin doble conteo de nbytes).")
    print("    Densa    = memoria ocupada por la REPRESENTACION EN PYTHON de la")
    print("               rejilla equivalente materializada, mismo criterio")
    print("               que 'Python' -- no una formula teorica.")
    print("    Archivo  = tamaño REAL del .npz comprimido en disco.")
    print("    Ahorro   = factor Densa/Python (representacion en Python).")

    print()
    print("=" * 70)

    # ── Guardar JSON ──────────────────────────────────────────
    resultados["archivo"] = Path(ruta).name
    resultados["n_puntos_muestreo"] = N
    resultados["repeticiones"] = REP

    ruta_json = Path(args.salida) if args.salida else Path(f"metricas_{Path(ruta).stem}.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"  Resultados guardados en: {ruta_json.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
