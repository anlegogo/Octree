"""
medir_metricas_off.py
======================
Procesa TODOS los archivos .off de ModelNet40 y calcula las metricas
de estructura del octree REAL, tiempo de procesamiento y almacenamiento
real en disco, para la tabla del capitulo de metodologia.

CORRECCION respecto a versiones anteriores: ya no se voxeliza a una
rejilla densa de un solo canal para "estimar" memoria. Se construye
el octree REAL (con poda, ver octree_real.py) para cada objeto, se
guarda en disco con su estructura jerarquica completa, y se miden
directamente:
  - Numero de nodos totales del arbol (internos + hojas)
  - Numero de hojas ocupadas
  - Tamaño REAL del archivo .npz en disco (no estimado)
  - Porcentaje de ocupacion en el nivel hoja

Metricas calculadas por objeto:
  - Numero de vertices y caras de la malla original
  - Tiempo de lectura + normalizacion + muestreo (ms)
  - Tiempo de construccion del arbol real, 32^3 y 64^3 (ms)
  - Nodos totales y hojas ocupadas del arbol, 32^3 y 64^3
  - Porcentaje de ocupacion en la hoja, 32^3 y 64^3
  - Tamaño REAL del archivo .npz guardado (KiB), 32^3 y 64^3

Salida:
  - resultados/metricas_off_completo.csv   (todas las muestras)
  - resultados/metricas_off_resumen.csv    (estadisticas por clase)
  - resultados/metricas_off_global.json    (estadisticas globales)

Uso:
    python medir_metricas_off.py
    python medir_metricas_off.py --n_muestras 100   # muestra rapida
    python medir_metricas_off.py --split train
"""

import sys
import csv
import json
import time
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

RAIZ_PROYECTO = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))

from octree import (
    leer_off, normalizar_malla, muestrear_superficie_con_normales,
    construir_grid_octree,
)
from octree_real import (
    construir_octree, guardar_octree_disperso, recolectar_hojas,
    contar_nodos_totales,
)

RAIZ_DATASET   = RAIZ_PROYECTO / "Dataset" / "ModelNet40"
DIR_RESULTADOS = RAIZ_PROYECTO / "resultados"
DIR_TEMP_NPZ   = RAIZ_PROYECTO / "data" / "_temp_medicion"  # npz temporales de medicion
N_PUNTOS_MUESTREO = 20000
N_PROCESOS = 10
SEED = 42
PROFUNDIDAD_POR_RESOLUCION = {32: 5, 64: 6}
RESOLUCIONES = [32, 64]

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


# ──────────────────────────────────────────────────────────────
# PROCESAMIENTO DE UN ARCHIVO
# ──────────────────────────────────────────────────────────────

def procesar_archivo(args: tuple) -> dict:
    ruta, clase, split = args
    nombre = Path(ruta).stem

    try:
        # Lectura + normalizacion + muestreo (una sola vez, compartido
        # entre resoluciones)
        t0 = time.perf_counter()
        verts_raw, caras = leer_off(ruta)
        verts = normalizar_malla(verts_raw)
        t_preproceso_ms = (time.perf_counter() - t0) * 1000

        n_v = len(verts_raw)
        n_c = len(caras) if caras is not None else 0

        rng = np.random.default_rng(SEED)
        t0 = time.perf_counter()
        pts, normales = muestrear_superficie_con_normales(
            verts, caras, N_PUNTOS_MUESTREO, rng,
        )
        t_muestreo_ms = (time.perf_counter() - t0) * 1000

        fila = {
            "nombre": nombre, "clase": clase, "split": split,
            "n_vertices": n_v, "n_caras": n_c,
            "t_preproceso_ms": round(t_preproceso_ms, 3),
            "t_muestreo_ms": round(t_muestreo_ms, 3),
            "error": None,
        }

        for R in RESOLUCIONES:
            profundidad_max = PROFUNDIDAD_POR_RESOLUCION[R]
            dir_split = DIR_TEMP_NPZ / f"R{R}" / clase / split
            dir_split.mkdir(parents=True, exist_ok=True)

            # ══════════════════════════════════════════════════════
            # RUTA A: OCTREE (arbol real, con poda)
            # ══════════════════════════════════════════════════════
            t0 = time.perf_counter()
            raiz = construir_octree(pts, normales, profundidad_max=profundidad_max)
            t_arbol_ms = (time.perf_counter() - t0) * 1000

            n_nodos = contar_nodos_totales(raiz)
            hojas = recolectar_hojas(raiz)
            n_hojas = len(hojas)
            pct_ocup_hoja = 100.0 * n_hojas / (R ** 3)

            ruta_npz_disperso = dir_split / f"{nombre}_disperso.npz"
            t0 = time.perf_counter()
            guardar_octree_disperso(raiz, str(ruta_npz_disperso), etiqueta=0,
                                    profundidad_max=profundidad_max)
            t_guardado_ms = (time.perf_counter() - t0) * 1000

            tam_disperso_kib = ruta_npz_disperso.stat().st_size / 1024
            ruta_npz_disperso.unlink()

            # ══════════════════════════════════════════════════════
            # RUTA B: REJILLA DENSA, CONSTRUCCION INDEPENDIENTE
            # ══════════════════════════════════════════════════════
            # CORRECCION (observacion de Andres Gonzalez): la version
            # anterior materializaba el grid denso A PARTIR DEL ARBOL YA
            # CONSTRUIDO (octree_a_grid_denso(raiz, R)), por lo que su
            # tiempo total incluía -sin quererlo- el costo de construir
            # el octree primero. Eso no representa el tiempo de
            # construccion independiente de una voxelizacion densa.
            #
            # Ahora la rejilla densa se construye DIRECTAMENTE desde los
            # MISMOS puntos y normales ya muestreados (pts, normales),
            # usando construir_grid_octree() de octree.py, SIN construir
            # el arbol en esta ruta. Ambas rutas parten del mismo
            # muestreo (costo compartido, medido aparte en
            # t_preproceso_ms y t_muestreo_ms), pero divergen desde ahi:
            # una construye un arbol con poda, la otra cuantiza
            # directamente a una rejilla completa.
            t0 = time.perf_counter()
            grid_denso = construir_grid_octree(pts, normales, R)
            t_denso_directo_ms = (time.perf_counter() - t0) * 1000

            ruta_npz_denso = dir_split / f"{nombre}_denso.npz"
            t0 = time.perf_counter()
            np.savez_compressed(ruta_npz_denso, grid=grid_denso, etiqueta=0)
            t_guardado_denso_ms = (time.perf_counter() - t0) * 1000

            tam_denso_kib = ruta_npz_denso.stat().st_size / 1024
            ruta_npz_denso.unlink()

            factor_ahorro_real = tam_denso_kib / max(tam_disperso_kib, 0.001)

            fila[f"t_arbol_{R}_ms"]           = round(t_arbol_ms, 3)
            fila[f"t_guardado_{R}_ms"]        = round(t_guardado_ms, 3)
            fila[f"t_denso_directo_{R}_ms"]   = round(t_denso_directo_ms, 3)
            fila[f"t_guardado_denso_{R}_ms"]  = round(t_guardado_denso_ms, 3)
            fila[f"nodos_totales_{R}"]        = n_nodos
            fila[f"hojas_ocupadas_{R}"]       = n_hojas
            fila[f"ocup_hoja_pct_{R}"]        = round(pct_ocup_hoja, 4)
            fila[f"tam_npz_disperso_kib_{R}"] = round(tam_disperso_kib, 3)
            fila[f"tam_npz_denso_kib_{R}"]    = round(tam_denso_kib, 3)
            fila[f"factor_reduccion_almacenamiento_{R}"]  = round(factor_ahorro_real, 3)

        # CORRECCION (observacion de Andres Gonzalez): t_total_ms
        # corresponde UNICAMENTE al tiempo de procesamiento del octree
        # (lectura + normalizacion + muestreo + construccion del arbol +
        # guardado disperso). NO incluye la construccion ni el guardado
        # de la representacion densa -- eso se mide por separado en
        # t_total_denso_ms, calculado ahora con una ruta de construccion
        # verdaderamente INDEPENDIENTE (construir_grid_octree() sobre
        # los puntos, SIN pasar por el arbol). Los costos compartidos
        # (lectura+normalizacion+muestreo) se cuentan UNA sola vez en
        # cada total -- no se duplican al sumarlos por separado.
        fila["t_total_ms"] = round(
            t_preproceso_ms + t_muestreo_ms
            + sum(fila[f"t_arbol_{R}_ms"] for R in RESOLUCIONES)
            + sum(fila[f"t_guardado_{R}_ms"] for R in RESOLUCIONES), 3,
        )

        # RUTA DENSA INDEPENDIENTE: ya NO incluye construccion del
        # arbol (t_arbol_R_ms). Solo lectura+normalizacion+muestreo
        # (compartidos) + construccion directa de la rejilla + guardado.
        fila["t_total_denso_ms"] = round(
            t_preproceso_ms + t_muestreo_ms
            + sum(fila[f"t_denso_directo_{R}_ms"] for R in RESOLUCIONES)
            + sum(fila[f"t_guardado_denso_{R}_ms"] for R in RESOLUCIONES), 3,
        )

        # NUEVO: tiempo total del EXPERIMENTO por objeto -- lo que
        # realmente hace procesar_archivo() (ambas representaciones,
        # compartiendo lectura+normalizacion+muestreo UNA sola vez).
        # Sumado sobre todos los objetos, este es el valor comparable
        # contra el tiempo REAL paralelo (que mide exactamente este
        # mismo trabajo, pero ejecutado en paralelo entre procesos).
        # NO debe compararse el tiempo real paralelo contra t_total_ms
        # solo (eso subestima el trabajo real, que incluye ambas rutas).
        fila["t_total_experimento_ms"] = round(
            t_preproceso_ms + t_muestreo_ms
            + sum(fila[f"t_arbol_{R}_ms"] for R in RESOLUCIONES)
            + sum(fila[f"t_guardado_{R}_ms"] for R in RESOLUCIONES)
            + sum(fila[f"t_denso_directo_{R}_ms"] for R in RESOLUCIONES)
            + sum(fila[f"t_guardado_denso_{R}_ms"] for R in RESOLUCIONES), 3,
        )

        return fila

    except Exception as e:
        return {"nombre": nombre, "clase": clase, "split": split, "error": str(e)}


# ──────────────────────────────────────────────────────────────
# RECOLECCION Y PROCESAMIENTO MASIVO
# ──────────────────────────────────────────────────────────────

def recolectar(split: str, n_muestras: int = None) -> list:
    tareas = []
    for clase in CLASES:
        carpeta = RAIZ_DATASET / clase / split
        if not carpeta.exists():
            continue
        archivos = sorted(carpeta.glob("*.off"))
        if n_muestras:
            archivos = archivos[:max(1, n_muestras // len(CLASES))]
        for f in archivos:
            tareas.append((str(f), clase, split))
    return tareas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test", "ambos"])
    parser.add_argument("--n_muestras", type=int, default=None)
    parser.add_argument("--off", type=str, default=None,
                        help="Procesar UN SOLO archivo .off directamente "
                             "(sin necesitar la estructura completa del "
                             "dataset). Util para pruebas rapidas, ej. "
                             "chair_0001.off.")
    parser.add_argument("--salida", type=str, default=None,
                        help="Ruta JSON de salida cuando se usa --off "
                             "(opcional; por defecto usa el nombre del archivo)")
    args = parser.parse_args()

    DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    DIR_TEMP_NPZ.mkdir(parents=True, exist_ok=True)

    # Confirmacion explicita de la carpeta REAL donde se lee/escribe,
    # para evitar discrepancias de carpeta entre este script y
    # tabla_resumen_octree.py (ambos deben usar la MISMA carpeta).
    print(f"[Carpeta de resultados] {DIR_RESULTADOS.resolve()}")

    # ── Modo archivo unico: bypass completo de la estructura de dataset ──
    if args.off:
        print("=" * 70)
        print("  METRICAS DE PROCESAMIENTO .off -> Octree REAL (archivo unico)")
        print("=" * 70)
        print(f"  Archivo: {args.off}\n")

        nombre_stem = Path(args.off).stem
        resultado = procesar_archivo((args.off, "sin_clase", "individual"))

        if resultado.get("error"):
            print(f"[ERROR] {resultado['error']}")
            return

        print(f"  Vertices: {resultado['n_vertices']:,}   "
              f"Caras: {resultado['n_caras']:,}")
        print(f"  Lectura+normalizacion : {resultado['t_preproceso_ms']:.3f} ms")
        print(f"  Muestreo superficie   : {resultado['t_muestreo_ms']:.3f} ms\n")

        for R in RESOLUCIONES:
            print(f"  --- Resolucion {R}^3 ---")
            print(f"    Nodos totales      : {resultado[f'nodos_totales_{R}']:,}")
            print(f"    Hojas ocupadas     : {resultado[f'hojas_ocupadas_{R}']:,}")
            print(f"    Ocupacion hoja     : {resultado[f'ocup_hoja_pct_{R}']:.3f}%")
            print(f"    Archivo disperso   : {resultado[f'tam_npz_disperso_kib_{R}']:.3f} KiB")
            print(f"    Archivo denso      : {resultado[f'tam_npz_denso_kib_{R}']:.3f} KiB "
                  f"(misma compresion)")
            print(f"    Factor de reduccion: {resultado[f'factor_reduccion_almacenamiento_{R}']:.3f}x")
            print()
            print(f"    Tiempo construccion ARBOL (con poda) : "
                  f"{resultado[f't_arbol_{R}_ms']:.3f} ms")
            print(f"    Tiempo construccion DENSA (directa)  : "
                  f"{resultado[f't_denso_directo_{R}_ms']:.3f} ms")
            t_arbol = resultado[f't_arbol_{R}_ms']
            t_denso = resultado[f't_denso_directo_{R}_ms']
            if t_denso > 0:
                print(f"    >>> El arbol tarda {t_arbol / t_denso:.1f}x mas que la "
                      f"rejilla densa directa <<<")
            print(f"    Tiempo guardado disperso             : "
                  f"{resultado[f't_guardado_{R}_ms']:.3f} ms")
            print(f"    Tiempo guardado denso                : "
                  f"{resultado[f't_guardado_denso_{R}_ms']:.3f} ms\n")

        print(f"  --- Totales por objeto (costos compartidos contados una vez) ---")
        print(f"    t_total_ms             (solo octree) : {resultado['t_total_ms']:.3f} ms")
        print(f"    t_total_denso_ms       (solo denso)  : {resultado['t_total_denso_ms']:.3f} ms")
        print(f"    t_total_experimento_ms (ambas juntas): "
              f"{resultado['t_total_experimento_ms']:.3f} ms")
        print(f"    (este ultimo es el comparable contra el tiempo real paralelo)\n")

        ruta_salida = Path(args.salida) if args.salida else \
                     DIR_RESULTADOS / f"metricas_off_{nombre_stem}.json"
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        print(f"  Resultados guardados en: {ruta_salida.resolve()}")

        import shutil
        shutil.rmtree(DIR_TEMP_NPZ, ignore_errors=True)
        return

    # ── Modo dataset completo (comportamiento original) ──
    splits = ["train", "test"] if args.split == "ambos" else [args.split]

    print("=" * 70)
    print("  METRICAS DE PROCESAMIENTO .off -> Octree REAL")
    print("  (nodos, hojas, tamaño real de archivo, tiempos)")
    print("=" * 70)

    todos_resultados = []
    # Tiempo REAL de ejecucion paralela por split (wall-clock, no la
    # suma de tiempos individuales). Se guarda para distinguirlo del
    # "tiempo secuencial acumulado estimado" que se calcula sumando
    # t_total_ms de cada objeto (ver punto 3 de la revision de Andres).
    tiempo_real_paralelo_por_split = {}

    for split in splits:
        tareas = recolectar(split, args.n_muestras)
        print(f"\n[{split.upper()}] {len(tareas)} archivos a procesar...")

        t_inicio = time.time()
        with ProcessPoolExecutor(max_workers=N_PROCESOS) as executor:
            futuros = {executor.submit(procesar_archivo, t): t for t in tareas}
            barra = tqdm(as_completed(futuros), total=len(futuros), ncols=80)
            for futuro in barra:
                todos_resultados.append(futuro.result())
        t_total = time.time() - t_inicio
        tiempo_real_paralelo_por_split[split] = round(t_total / 60, 3)  # minutos

        n_ok  = sum(1 for r in todos_resultados if r.get("error") is None and r["split"] == split)
        n_err = sum(1 for r in todos_resultados if r.get("error") is not None and r["split"] == split)
        print(f"  Completado en {t_total:.1f}s | OK: {n_ok} | Errores: {n_err}")

    # ── Guardar CSV completo ──
    validos = [r for r in todos_resultados if r.get("error") is None]
    campos = ["nombre", "clase", "split", "n_vertices", "n_caras",
              "t_preproceso_ms", "t_muestreo_ms",
              "t_total_ms", "t_total_denso_ms", "t_total_experimento_ms"]
    for R in RESOLUCIONES:
        campos += [f"t_arbol_{R}_ms", f"t_guardado_{R}_ms",
                  f"t_denso_directo_{R}_ms", f"t_guardado_denso_{R}_ms",
                  f"nodos_totales_{R}", f"hojas_ocupadas_{R}", f"ocup_hoja_pct_{R}",
                  f"tam_npz_disperso_kib_{R}", f"tam_npz_denso_kib_{R}",
                  f"factor_reduccion_almacenamiento_{R}"]

    csv_completo = DIR_RESULTADOS / "metricas_off_completo.csv"
    with open(csv_completo, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for r in validos:
            w.writerow({k: r.get(k, "") for k in campos})
    print(f"\n[CSV] Guardado: {csv_completo.name} ({len(validos)} filas)")

    # ── Resumen por clase ──
    from collections import defaultdict
    por_clase = defaultdict(list)
    for r in validos:
        por_clase[r["clase"]].append(r)

    csv_resumen = DIR_RESULTADOS / "metricas_off_resumen.csv"
    campos_resumen = ["clase", "n", "t_total_media_ms"]
    for R in RESOLUCIONES:
        campos_resumen += [f"nodos_media_{R}", f"hojas_media_{R}", f"ocup_pct_media_{R}",
                           f"tam_disperso_kib_media_{R}", f"tam_denso_kib_media_{R}",
                           f"factor_reduccion_almacenamiento_{R}"]

    with open(csv_resumen, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos_resumen)
        w.writeheader()
        for clase, muestras in por_clase.items():
            fila = {
                "clase": clase, "n": len(muestras),
                "t_total_media_ms": round(np.mean([m["t_total_ms"] for m in muestras]), 2),
            }
            for R in RESOLUCIONES:
                fila[f"nodos_media_{R}"] = round(np.mean([m[f"nodos_totales_{R}"] for m in muestras]), 1)
                fila[f"hojas_media_{R}"] = round(np.mean([m[f"hojas_ocupadas_{R}"] for m in muestras]), 1)
                fila[f"ocup_pct_media_{R}"] = round(np.mean([m[f"ocup_hoja_pct_{R}"] for m in muestras]), 4)
                fila[f"tam_disperso_kib_media_{R}"] = round(
                    np.mean([m[f"tam_npz_disperso_kib_{R}"] for m in muestras]), 3)
                fila[f"tam_denso_kib_media_{R}"] = round(
                    np.mean([m[f"tam_npz_denso_kib_{R}"] for m in muestras]), 3)
                # CORRECCION (observacion de Andres Gonzalez): el factor
                # de reduccion por clase se calcula como el COCIENTE DE
                # SUMAS (tamaño denso total de la clase / tamaño disperso
                # total de la clase), NO como el promedio de los factores
                # individuales de cada objeto (que es matematicamente
                # distinto por la desigualdad de Jensen).
                suma_disperso_clase = sum(m[f"tam_npz_disperso_kib_{R}"] for m in muestras)
                suma_denso_clase    = sum(m[f"tam_npz_denso_kib_{R}"] for m in muestras)
                fila[f"factor_reduccion_almacenamiento_{R}"] = round(
                    suma_denso_clase / max(suma_disperso_clase, 0.001), 3
                )
            w.writerow(fila)
    print(f"[CSV] Guardado: {csv_resumen.name}")

    # ── Resumen global ──
    # IMPORTANTE: la comparacion de almacenamiento ya NO usa una formula
    # teorica para la rejilla densa. Ambas representaciones (octree
    # disperso y rejilla densa) se guardaron en disco con la MISMA
    # compresion (np.savez_compressed) y se comparan sus tamaños reales
    # de archivo, en igualdad de condiciones.
    resumen_global = {"n_total": len(validos), "n_errores": len(todos_resultados) - len(validos)}
    resumen_global["t_total_ms"] = {
        "media": round(float(np.mean([r["t_total_ms"] for r in validos])), 4),
        "std":   round(float(np.std([r["t_total_ms"] for r in validos])), 4),
        "min":   round(float(np.min([r["t_total_ms"] for r in validos])), 4),
        "max":   round(float(np.max([r["t_total_ms"] for r in validos])), 4),
    }
    resumen_global["t_total_denso_ms"] = {
        "media": round(float(np.mean([r["t_total_denso_ms"] for r in validos])), 4),
        "std":   round(float(np.std([r["t_total_denso_ms"] for r in validos])), 4),
        "min":   round(float(np.min([r["t_total_denso_ms"] for r in validos])), 4),
        "max":   round(float(np.max([r["t_total_denso_ms"] for r in validos])), 4),
    }
    # t_total_experimento_ms: tiempo por objeto de AMBAS representaciones
    # combinadas (lo que realmente hace procesar_archivo()). Este es el
    # valor comparable contra el tiempo REAL paralelo -- ver nota mas
    # abajo (observacion de Andres Gonzalez).
    resumen_global["t_total_experimento_ms"] = {
        "media": round(float(np.mean([r["t_total_experimento_ms"] for r in validos])), 4),
        "std":   round(float(np.std([r["t_total_experimento_ms"] for r in validos])), 4),
        "min":   round(float(np.min([r["t_total_experimento_ms"] for r in validos])), 4),
        "max":   round(float(np.max([r["t_total_experimento_ms"] for r in validos])), 4),
    }
    resumen_global["t_secuencial_acumulado_estimado_experimento_min"] = round(
        len(validos) * resumen_global["t_total_experimento_ms"]["media"] / 60000, 3
    )
    # Tiempo REAL de ejecucion paralela (wall-clock), medido por split
    # con time.time() alrededor del ProcessPoolExecutor. Distinto del
    # "tiempo secuencial acumulado estimado" (suma de t_total_ms de
    # cada objeto, como si se hubieran procesado uno tras otro).
    resumen_global["tiempo_real_paralelo_min"] = tiempo_real_paralelo_por_split
    resumen_global["tiempo_real_paralelo_total_min"] = round(
        sum(tiempo_real_paralelo_por_split.values()), 3
    )
    for R in RESOLUCIONES:
        nodos = [r[f"nodos_totales_{R}"] for r in validos]
        hojas = [r[f"hojas_ocupadas_{R}"] for r in validos]
        ocup  = [r[f"ocup_hoja_pct_{R}"] for r in validos]
        tam_disperso = [r[f"tam_npz_disperso_kib_{R}"] for r in validos]
        tam_denso    = [r[f"tam_npz_denso_kib_{R}"] for r in validos]

        tam_disperso_total_mib = float(np.sum(tam_disperso)) / 1024
        tam_denso_total_mib    = float(np.sum(tam_denso)) / 1024

        resumen_global[f"R{R}"] = {
            "nodos_media": round(float(np.mean(nodos)), 1),
            "nodos_std":   round(float(np.std(nodos)), 1),
            "hojas_media": round(float(np.mean(hojas)), 1),
            "hojas_std":   round(float(np.std(hojas)), 1),
            "ocup_pct_media": round(float(np.mean(ocup)), 4),
            "ocup_pct_std":   round(float(np.std(ocup)), 4),
            # Ambos archivos guardados con la MISMA compresion, tamaños
            # REALES medidos en disco (no formulas teoricas)
            "tam_disperso_kib_media": round(float(np.mean(tam_disperso)), 3),
            "tam_denso_kib_media":    round(float(np.mean(tam_denso)), 3),
            "tam_disperso_total_dataset_mib": round(tam_disperso_total_mib, 2),
            "tam_denso_total_dataset_mib":    round(tam_denso_total_mib, 2),
            "factor_reduccion_almacenamiento": round(
                tam_denso_total_mib / max(tam_disperso_total_mib, 0.001), 3
            ),
        }

    ruta_json = DIR_RESULTADOS / "metricas_off_global.json"
    with open(ruta_json, "w") as f:
        json.dump(resumen_global, f, indent=2)
    print(f"[JSON] Guardado: {ruta_json.name}")

    # ── Resumen en consola ──
    print("\n" + "=" * 70)
    print("  RESUMEN GLOBAL (octree real, medido)")
    print("=" * 70)
    print(f"  Objetos procesados : {resumen_global['n_total']:,}")
    print(f"  Errores            : {resumen_global['n_errores']}")
    print(f"  Tiempo octree/objeto (solo octree)   : "
          f"media={resumen_global['t_total_ms']['media']:.2f} ms, "
          f"max={resumen_global['t_total_ms']['max']:.2f} ms")
    print(f"  Tiempo experimento/objeto (ambas rep): "
          f"media={resumen_global['t_total_experimento_ms']['media']:.2f} ms, "
          f"max={resumen_global['t_total_experimento_ms']['max']:.2f} ms")
    print()
    print(f"  CORRECCION (Andres Gonzalez): el tiempo REAL paralelo (wall-clock)")
    print(f"  mide el tiempo TOTAL DE EJECUCION DEL EXPERIMENTO COMPLETO, es")
    print(f"  decir, el procesamiento de AMBAS representaciones (octree +")
    print(f"  rejilla densa) por objeto. NO debe compararse directamente")
    print(f"  contra 't_total_ms' (que solo cubre el octree) -- la comparacion")
    print(f"  correcta es contra la suma de 't_total_experimento_ms'.")
    print()
    print(f"  Tiempo REAL paralelo (wall-clock), por split:")
    for sp, min_reales in tiempo_real_paralelo_por_split.items():
        print(f"    {sp}: {min_reales:.2f} min")
    print(f"  Tiempo REAL paralelo total: "
          f"{resumen_global['tiempo_real_paralelo_total_min']:.2f} min")

    for R in RESOLUCIONES:
        g = resumen_global[f"R{R}"]
        print(f"\n  --- Resolucion {R}^3 ---")
        print(f"    Nodos totales (media)   : {g['nodos_media']:.0f} (±{g['nodos_std']:.0f})")
        print(f"    Hojas ocupadas (media)  : {g['hojas_media']:.0f} (±{g['hojas_std']:.0f})")
        print(f"    Ocupacion hoja (media)  : {g['ocup_pct_media']:.3f}% (±{g['ocup_pct_std']:.3f}%)")
        print(f"    Archivo disperso (media): {g['tam_disperso_kib_media']:.3f} KiB (comprimido)")
        print(f"    Archivo denso (media)   : {g['tam_denso_kib_media']:.3f} KiB (comprimido, misma compresion)")
        print(f"    Conjunto de datos disperso (real) : {g['tam_disperso_total_dataset_mib']:.2f} MiB")
        print(f"    Conjunto de datos denso (real)    : {g['tam_denso_total_dataset_mib']:.2f} MiB")
        print(f"    >>> Factor de reduccion del almacenamiento comprimido: {g['factor_reduccion_almacenamiento']:.2f}x <<<")

    print("=" * 70)

    # Limpiar carpeta temporal
    import shutil
    shutil.rmtree(DIR_TEMP_NPZ, ignore_errors=True)


if __name__ == "__main__":
    main()
