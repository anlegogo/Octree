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
no reportan memoria estimada de una rejilla densa (1 o 4 canales),
sino el numero real de nodos del arbol y el tamaño REAL medido del
archivo .npz guardado con estructura jerarquica completa (ver
octree_real.py y medir_metricas_off.py).

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
    ocup_hoja_pct_R, tam_npz_kb_R.
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
            entrada[f"tam_npz_kb_media_{R}"] = media(f"tam_npz_kb_{R}")
            entrada[f"tam_npz_total_mb_{R}"] = round(suma(f"tam_npz_kb_{R}") / 1024, 4)

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
        tam_total_kb = suma_global(f"tam_npz_kb_{R}")
        # Memoria densa equivalente (4 canales float32, formato descartado)
        mem_densa_total_mb = round(n_total * 4 * (R**3) * 4 / 1024 / 1024, 1)

        stats_global[f"R{R}"] = {
            "nodos_media": media_global(f"nodos_totales_{R}"),
            "nodos_std":   std_global(f"nodos_totales_{R}"),
            "hojas_media": media_global(f"hojas_ocupadas_{R}"),
            "hojas_std":   std_global(f"hojas_ocupadas_{R}"),
            "ocup_pct_media": media_global(f"ocup_hoja_pct_{R}"),
            "ocup_pct_std":   std_global(f"ocup_hoja_pct_{R}"),
            "tam_npz_kb_media": media_global(f"tam_npz_kb_{R}"),
            "tam_npz_total_dataset_mb": round(tam_total_kb / 1024, 2),
            "mem_densa_equivalente_total_mb": mem_densa_total_mb,
            "factor_ahorro_real": round(
                mem_densa_total_mb * 1024 / max(tam_total_kb, 0.001), 1
            ),
        }

    return stats_clase, stats_global


# ──────────────────────────────────────────────────────────────
# IMPRESION EN CONSOLA
# ──────────────────────────────────────────────────────────────

def imprimir_tabla_consola(stats_clase: dict, stats_global: dict, split: str):
    print("\n" + "=" * 100)
    print(f"  TABLA RESUMEN DEL OCTREE REAL — ModelNet40  ({split.upper()})")
    print("=" * 100)

    enc = ["Clase", "N", "Ocup.32³(%)", "Ocup.64³(%)",
           "Nodos 32³", "Nodos 64³", "T.media(ms)", "Archivo32³(KB)", "Archivo64³(KB)"]
    anchos = [14, 5, 12, 12, 10, 10, 13, 15, 15]
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
            f"{s['tam_npz_kb_media_32']:.3f}",
            f"{s['tam_npz_kb_media_64']:.3f}",
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
        f"{g['R32']['tam_npz_kb_media']:.3f}",
        f"{g['R64']['tam_npz_kb_media']:.3f}",
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
        print(f"    - Tamaño .npz (media)          : {gr['tam_npz_kb_media']:.3f} KB")
        print(f"    - Almacenamiento dataset (real): {gr['tam_npz_total_dataset_mb']:.2f} MB")
        print(f"    - Densa equivalente (descartada): {gr['mem_densa_equivalente_total_mb']:.1f} MB")
        print(f"    - >>> Factor de ahorro REAL     : {gr['factor_ahorro_real']:.1f}x <<<")
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
        "Archivo\n32³ (KB)", "Archivo\n64³ (KB)",
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
            f"{s['tam_npz_kb_media_32']:.3f}",
            f"{s['tam_npz_kb_media_64']:.3f}",
        ])

    g = stats_global
    celdas.append([
        "TOTAL / MEDIA", str(g["n_total"]),
        f"{g['R32']['ocup_pct_media']:.3f}",
        f"{g['R64']['ocup_pct_media']:.3f}",
        f"{g['R32']['nodos_media']:.0f}",
        f"{g['R64']['nodos_media']:.0f}",
        f"{g['t_total_media_ms']:.2f}",
        f"{g['R32']['tam_npz_kb_media']:.3f}",
        f"{g['R64']['tam_npz_kb_media']:.3f}",
    ])

    alto = max(8, 0.42 * len(celdas) + 1.5)
    fig, ax = plt.subplots(figsize=(18, alto))
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
        f"Nodos y tamaño de archivo REALES (medidos), no estimados. "
        f"Ahorro vs. rejilla densa: {g['R32']['factor_ahorro_real']:.0f}x (32³), "
        f"{g['R64']['factor_ahorro_real']:.0f}x (64³)",
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
