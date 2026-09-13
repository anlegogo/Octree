"""
tabla_resumen_octree.py
========================
Lee el CSV completo generado por medir_metricas_off.py (formato del
octree REAL, con nodos/hojas/tamaño de archivo medidos, no estimados)
y produce:

  1. Tabla resumen por clase (consola) con:
       - n modelos procesados
       - ocupacion media en el nivel hoja, 32^3 y 64^3
       - nodos totales medios del arbol, 32^3 y 64^3
       - hojas ocupadas medias, 32^3 y 64^3
       - tiempo medio de procesamiento por objeto
       - almacenamiento REAL (tamaño de archivo .npz medido, no estimado)

  2. Tabla resumen global (train + test) citable en el documento

  3. Figura PNG/SVG de la tabla lista para incluir en la tesis

  4. JSON con los valores para citar en el texto

CORRECCION respecto a versiones anteriores: las columnas del CSV ya
no reportan memoria estimada de una rejilla densa (1 o 4 canales) ni
comparan un .npz disperso comprimido contra una formula teorica sin
comprimir. Ahora se comparan los tamaños REALES de dos archivos .npz
guardados con la MISMA compresion: el octree disperso (estructura
jerarquica completa) y la rejilla densa equivalente materializada del
mismo arbol (ver octree_real.py y medir_metricas_off.py). Todas las
unidades de tamaño usan KiB/MiB (potencias de 1024), no KB/MB.

Uso:
    python tabla_resumen_octree.py
    python tabla_resumen_octree.py --split train
    python tabla_resumen_octree.py --split test
"""

import csv
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

DIR_RESULTADOS = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\resultados")

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
# CARGA DEL CSV
# ──────────────────────────────────────────────────────────────

def cargar_csv(split_filtro: str = None) -> list:
    ruta = DIR_RESULTADOS / "metricas_off_completo.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro {ruta.name}.\n"
            f"Corre primero: python medir_metricas_off.py"
        )
    filas = []
    with open(ruta, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            if split_filtro and fila["split"] != split_filtro:
                continue
            filas.append(fila)
    return filas


# ──────────────────────────────────────────────────────────────
# CALCULO DE ESTADISTICAS
# ──────────────────────────────────────────────────────────────

def calcular_estadisticas(filas: list) -> tuple:
    """
    Retorna (stats_por_clase, stats_global), usando las columnas REALES
    (medidas) del CSV del octree real: nodos_totales_R, hojas_ocupadas_R,
    ocup_hoja_pct_R, tam_npz_disperso_kib_R, tam_npz_denso_kib_R.

    IMPORTANTE: tam_npz_denso_kib_R es el tamaño REAL de la rejilla
    densa equivalente, guardada con la MISMA compresion que el octree
    disperso (ver medir_metricas_off.py) -- no una formula teorica sin
    comprimir. El factor de ahorro se calcula entre ambos archivos
    reales, en igualdad de condiciones.
    """
    por_clase = defaultdict(list)
    for fila in filas:
        por_clase[fila["clase"]].append(fila)

    stats_clase = {}
    for clase in CLASES:
        muestras = por_clase.get(clase, [])
        if not muestras:
            continue
        n = len(muestras)

        def media(campo):
            vals = [float(m[campo]) for m in muestras if m.get(campo)]
            return round(np.mean(vals), 4) if vals else 0.0

        def suma(campo):
            vals = [float(m[campo]) for m in muestras if m.get(campo)]
            return round(sum(vals), 4) if vals else 0.0

        entrada = {"n": n, "t_total_media_ms": media("t_total_ms")}
        for R in RESOLUCIONES:
            entrada[f"nodos_media_{R}"] = media(f"nodos_totales_{R}")
            entrada[f"hojas_media_{R}"] = media(f"hojas_ocupadas_{R}")
            entrada[f"ocup_pct_media_{R}"] = media(f"ocup_hoja_pct_{R}")
            entrada[f"tam_disperso_kib_media_{R}"] = media(f"tam_npz_disperso_kib_{R}")
            entrada[f"tam_denso_kib_media_{R}"] = media(f"tam_npz_denso_kib_{R}")
            entrada[f"tam_disperso_total_mib_{R}"] = round(
                suma(f"tam_npz_disperso_kib_{R}") / 1024, 4)
            entrada[f"factor_ahorro_media_{R}"] = media(f"factor_ahorro_real_{R}")

        stats_clase[clase] = entrada

    # Estadisticas globales
    n_total = len(filas)

    def media_global(campo):
        vals = [float(f[campo]) for f in filas if f.get(campo)]
        return round(np.mean(vals), 4) if vals else 0.0

    def std_global(campo):
        vals = [float(f[campo]) for f in filas if f.get(campo)]
        return round(np.std(vals), 4) if vals else 0.0

    def suma_global(campo):
        vals = [float(f[campo]) for f in filas if f.get(campo)]
        return round(float(np.sum(vals)), 4) if vals else 0.0

    stats_global = {
        "n_total": n_total,
        "n_clases": len(stats_clase),
        "t_total_media_ms": media_global("t_total_ms"),
        "t_total_std_ms": std_global("t_total_ms"),
        "t_total_dataset_min": round(n_total * media_global("t_total_ms") / 60000, 2),
    }

    for R in RESOLUCIONES:
        tam_disperso_total_kib = suma_global(f"tam_npz_disperso_kib_{R}")
        tam_denso_total_kib    = suma_global(f"tam_npz_denso_kib_{R}")

        stats_global[f"R{R}"] = {
            "nodos_media": media_global(f"nodos_totales_{R}"),
            "nodos_std":   std_global(f"nodos_totales_{R}"),
            "hojas_media": media_global(f"hojas_ocupadas_{R}"),
            "hojas_std":   std_global(f"hojas_ocupadas_{R}"),
            "ocup_pct_media": media_global(f"ocup_hoja_pct_{R}"),
            "ocup_pct_std":   std_global(f"ocup_hoja_pct_{R}"),
            # Ambos tamaños son REALES (archivos .npz comprimidos con
            # el mismo metodo), no formulas teoricas.
            "tam_disperso_kib_media": media_global(f"tam_npz_disperso_kib_{R}"),
            "tam_denso_kib_media":    media_global(f"tam_npz_denso_kib_{R}"),
            "tam_disperso_total_dataset_mib": round(tam_disperso_total_kib / 1024, 2),
            "tam_denso_total_dataset_mib":    round(tam_denso_total_kib / 1024, 2),
            "factor_ahorro_real": round(
                tam_denso_total_kib / max(tam_disperso_total_kib, 0.001), 3
            ),
        }

    return stats_clase, stats_global


# ──────────────────────────────────────────────────────────────
# IMPRESION EN CONSOLA
# ──────────────────────────────────────────────────────────────

def imprimir_tabla_consola(stats_clase: dict, stats_global: dict, split: str):
    print("\n" + "=" * 100)
    print(f"  TABLA RESUMEN DEL OCTREE REAL — ModelNet40  ({split.upper()})")
    print(f"  Archivos DISPERSO y DENSO guardados con la MISMA compresion (.npz)")
    print("=" * 100)

    enc = ["Clase", "N", "Ocup.32³(%)", "Ocup.64³(%)", "Nodos 32³", "Nodos 64³",
           "T.media(ms)", "Disp32³(KiB)", "Den32³(KiB)", "Disp64³(KiB)", "Den64³(KiB)"]
    anchos = [14, 5, 12, 12, 10, 10, 13, 13, 12, 13, 12]
    header = "  " + "  ".join(f"{h:<{w}}" for h, w in zip(enc, anchos))
    print(header)
    print("  " + "-" * 98)

    for clase in CLASES:
        if clase not in stats_clase:
            continue
        s = stats_clase[clase]
        vals = [
            clase, str(s["n"]),
            f"{s['ocup_pct_media_32']:.3f}",
            f"{s['ocup_pct_media_64']:.3f}",
            f"{s['nodos_media_32']:.0f}",
            f"{s['nodos_media_64']:.0f}",
            f"{s['t_total_media_ms']:.2f}",
            f"{s['tam_disperso_kib_media_32']:.3f}",
            f"{s['tam_denso_kib_media_32']:.3f}",
            f"{s['tam_disperso_kib_media_64']:.3f}",
            f"{s['tam_denso_kib_media_64']:.3f}",
        ]
        print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(vals, anchos)))

    print("  " + "=" * 98)
    g = stats_global
    vals_g = [
        "GLOBAL", str(g["n_total"]),
        f"{g['R32']['ocup_pct_media']:.3f}±{g['R32']['ocup_pct_std']:.3f}",
        f"{g['R64']['ocup_pct_media']:.3f}±{g['R64']['ocup_pct_std']:.3f}",
        f"{g['R32']['nodos_media']:.0f}",
        f"{g['R64']['nodos_media']:.0f}",
        f"{g['t_total_media_ms']:.2f}±{g['t_total_std_ms']:.2f}",
        f"{g['R32']['tam_disperso_kib_media']:.3f}",
        f"{g['R32']['tam_denso_kib_media']:.3f}",
        f"{g['R64']['tam_disperso_kib_media']:.3f}",
        f"{g['R64']['tam_denso_kib_media']:.3f}",
    ]
    print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(vals_g, anchos)))
    print("=" * 100)

    print(f"\n  Resumen citables en el documento (octree real, medido):")
    print(f"  - Objetos procesados            : {g['n_total']:,}")
    for R in RESOLUCIONES:
        gr = g[f"R{R}"]
        print(f"\n  Resolucion {R}^3:")
        print(f"    - Nodos totales (media)        : {gr['nodos_media']:.0f} (±{gr['nodos_std']:.0f})")
        print(f"    - Hojas ocupadas (media)       : {gr['hojas_media']:.0f} (±{gr['hojas_std']:.0f})")
        print(f"    - Ocupacion hoja (media)       : {gr['ocup_pct_media']:.3f}% (±{gr['ocup_pct_std']:.3f}%)")
        print(f"    - Archivo disperso (media)     : {gr['tam_disperso_kib_media']:.3f} KiB")
        print(f"    - Archivo denso (media)        : {gr['tam_denso_kib_media']:.3f} KiB "
              f"(misma compresion)")
        print(f"    - Dataset disperso (real)      : {gr['tam_disperso_total_dataset_mib']:.2f} MiB")
        print(f"    - Dataset denso (real)         : {gr['tam_denso_total_dataset_mib']:.2f} MiB")
        print(f"    - >>> Factor de ahorro REAL (ambos comprimidos): "
              f"{gr['factor_ahorro_real']:.2f}x <<<")
    print(f"\n  - Tiempo medio por objeto        : {g['t_total_media_ms']:.2f} ms "
          f"(±{g['t_total_std_ms']:.2f} ms)")
    print(f"  - Tiempo total dataset           : ~{g['t_total_dataset_min']:.1f} min")


# ──────────────────────────────────────────────────────────────
# FIGURA: tabla imagen para la tesis
# ──────────────────────────────────────────────────────────────

def graficar_tabla_resumen(stats_clase: dict, stats_global: dict, split: str):
    columnas = [
        "Clase", "N", "Ocup.\n32³ (%)", "Ocup.\n64³ (%)",
        "Nodos\n32³", "Nodos\n64³", "T. media\n(ms)",
        "Disperso\n32³ (KiB)", "Denso\n32³ (KiB)",
        "Disperso\n64³ (KiB)", "Denso\n64³ (KiB)",
    ]

    clases_presentes = [c for c in CLASES if c in stats_clase]
    celdas = []
    for clase in clases_presentes:
        s = stats_clase[clase]
        celdas.append([
            clase, str(s["n"]),
            f"{s['ocup_pct_media_32']:.3f}",
            f"{s['ocup_pct_media_64']:.3f}",
            f"{s['nodos_media_32']:.0f}",
            f"{s['nodos_media_64']:.0f}",
            f"{s['t_total_media_ms']:.2f}",
            f"{s['tam_disperso_kib_media_32']:.3f}",
            f"{s['tam_denso_kib_media_32']:.3f}",
            f"{s['tam_disperso_kib_media_64']:.3f}",
            f"{s['tam_denso_kib_media_64']:.3f}",
        ])

    g = stats_global
    celdas.append([
        "TOTAL / MEDIA", str(g["n_total"]),
        f"{g['R32']['ocup_pct_media']:.3f}",
        f"{g['R64']['ocup_pct_media']:.3f}",
        f"{g['R32']['nodos_media']:.0f}",
        f"{g['R64']['nodos_media']:.0f}",
        f"{g['t_total_media_ms']:.2f}",
        f"{g['R32']['tam_disperso_kib_media']:.3f}",
        f"{g['R32']['tam_denso_kib_media']:.3f}",
        f"{g['R64']['tam_disperso_kib_media']:.3f}",
        f"{g['R64']['tam_denso_kib_media']:.3f}",
    ])

    alto = max(8, 0.42 * len(celdas) + 1.5)
    fig, ax = plt.subplots(figsize=(20, alto))
    ax.axis("off")

    tabla = ax.table(cellText=celdas, colLabels=columnas, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8.5)
    tabla.scale(1.05, 1.65)

    for (row, col), cell in tabla.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == len(celdas):
            cell.set_facecolor("#CFD8DC")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")

    ax.set_title(
        f"Tabla de estructura del octree real y costo del pipeline — "
        f"ModelNet40 ({split})\n"
        f"Disperso y Denso guardados con la MISMA compresion (.npz); "
        f"tamaños REALES medidos, no estimados. "
        f"Ahorro real: {g['R32']['factor_ahorro_real']:.2f}x (32³), "
        f"{g['R64']['factor_ahorro_real']:.2f}x (64³)",
        fontsize=10.5, pad=18, fontweight="bold",
    )

    sufijo = f"_{split}"
    for ext in ("png", "svg"):
        salida = DIR_RESULTADOS / f"tabla_resumen_octree{sufijo}.{ext}"
        plt.savefig(salida, dpi=150 if ext == "png" else None,
                    format=ext, bbox_inches="tight")
        print(f"[Guardado] {salida.name}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="ambos",
                        choices=["train", "test", "ambos"])
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "ambos" else [args.split]

    for split in splits:
        print(f"\nCargando CSV para split={split}...")
        try:
            filas = cargar_csv(split_filtro=None if split == "ambos" else split)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return

        if not filas:
            print(f"  [Aviso] No hay datos para split={split}")
            continue

        print(f"  Filas cargadas: {len(filas):,}")

        stats_clase, stats_global = calcular_estadisticas(filas)
        imprimir_tabla_consola(stats_clase, stats_global, split)

        json_salida = DIR_RESULTADOS / f"resumen_octree_{split}.json"
        with open(json_salida, "w", encoding="utf-8") as f:
            json.dump({
                "split": split, "global": stats_global, "por_clase": stats_clase,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n[JSON] Guardado: {json_salida.name}")

        print("\nGenerando tabla imagen...")
        graficar_tabla_resumen(stats_clase, stats_global, split)

    print(f"\nTodo guardado en: {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
