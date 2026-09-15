"""
generar_imagenes_json.py
===========================
Convierte los 3 archivos JSON generados por octree_real.py --off,
medir_metricas_off.py --off, y medir_tiempo_memoria.py --off, en
tablas imagen (PNG + SVG), listas para incluir en el documento de tesis.

Funciona con CUALQUIER objeto, no solo chair_0001 -- basta con que los
3 JSON existan siguiendo la convencion de nombres por defecto de cada
script, o se pueden pasar rutas explicitas.

Convencion de nombres por defecto (segun --nombre NOMBRE):
    octree_real_NOMBRE.json      (de octree_real.py --off)
    metricas_off_NOMBRE.json     (de medir_metricas_off.py --off)
    metricas_NOMBRE.json         (de medir_tiempo_memoria.py --off)

Uso:
    # Busca los 3 JSON con el nombre por defecto en las carpetas usuales
    python generar_imagenes_json.py --nombre chair_0001

    # Especificando rutas explicitas (si estan en otro lugar)
    python generar_imagenes_json.py --nombre chair_0001 \
        --octree-real "ruta/octree_real_chair_0001.json" \
        --metricas-off "ruta/metricas_off_chair_0001.json" \
        --tiempo-memoria "ruta/metricas_chair_0001.json" \
        --salida "ruta/de/salida"
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

RAIZ_PROYECTO = Path(__file__).parent.parent

# Ubicaciones por defecto donde cada script suele guardar su JSON
RUTA_DEFECTO_OCTREE_REAL     = RAIZ_PROYECTO / "fase2_octree"
RUTA_DEFECTO_METRICAS_OFF    = RAIZ_PROYECTO / "resultados"
RUTA_DEFECTO_TIEMPO_MEMORIA  = RAIZ_PROYECTO / "Scripts_Analisis"


# ──────────────────────────────────────────────────────────────
# UTILIDAD DE DIBUJO DE TABLA (comun a las 3 figuras)
# ──────────────────────────────────────────────────────────────

def guardar_tabla(columnas, filas, titulo, ruta_salida_base, alto_minimo=3):
    """Dibuja una tabla con encabezado oscuro y filas alternadas, y la
    guarda en PNG + SVG con el mismo nombre base.

    alto_minimo permite subir la altura base para tablas con pocas filas
    pero encabezados de varias lineas (evita solapamiento visual)."""
    alto = max(alto_minimo, 0.5 * len(filas) + 1.5)
    fig, ax = plt.subplots(figsize=(max(12, 1.6 * len(columnas)), alto))
    ax.axis("off")

    tabla = ax.table(cellText=filas, colLabels=columnas, loc="center", cellLoc="center")
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9.5)
    tabla.scale(1.05, 1.8)

    for (row, col), cell in tabla.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")

    ax.set_title(titulo, fontsize=12, pad=15, fontweight="bold")
    plt.tight_layout()

    ruta_png = Path(f"{ruta_salida_base}.png")
    ruta_svg = Path(f"{ruta_salida_base}.svg")
    plt.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.savefig(ruta_svg, format="svg", bbox_inches="tight")
    print(f"  [Guardado] {ruta_png.name}  +  {ruta_svg.name}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# TABLA 1: octree_real_<nombre>.json
# ──────────────────────────────────────────────────────────────

def tabla_octree_real(ruta_json: Path, nombre: str, dir_salida: Path):
    if not ruta_json.exists():
        print(f"  [Omitido] No existe: {ruta_json}")
        return

    d = json.load(open(ruta_json))

    columnas = ["Resolución", "Prof. (L)", "Nodos totales", "Hojas ocupadas",
               "Python (KiB)", "Densa (KiB)", "Binaria est. (KiB)", "Ahorro"]
    filas = []
    for R in [32, 64]:
        clave_R = f"R{R}"
        if clave_R not in d:
            continue
        r = d[clave_R]
        filas.append([
            f"{R}³", str(r["profundidad_max"]), f"{r['n_nodos_totales']:,}",
            f"{r['n_hojas_ocupadas']:,}", f"{r['memoria_python_kib']:.2f}",
            f"{r['memoria_densa_kib']:.2f}", f"{r['memoria_binaria_estimada_kib']:.2f}",
            f"{r['factor_ahorro_python_vs_densa']:.2f}x",
        ])

    if not filas:
        print(f"  [Omitido] {ruta_json.name} no tiene datos de resolucion")
        return

    guardar_tabla(
        columnas, filas,
        f"octree_real.py — {nombre}\nEstructura del árbol y memoria "
        f"ocupada por la representación en Python",
        dir_salida / f"tabla_octree_real_{nombre}",
    )


# ──────────────────────────────────────────────────────────────
# TABLA 2: metricas_off_<nombre>.json
# ──────────────────────────────────────────────────────────────

def tabla_metricas_off(ruta_json: Path, nombre: str, dir_salida: Path):
    if not ruta_json.exists():
        print(f"  [Omitido] No existe: {ruta_json}")
        return

    d = json.load(open(ruta_json))

    columnas = ["Resolución", "Nodos", "Hojas", "Ocup. hoja (%)",
               "Disperso (KiB)", "Denso (KiB)", "Factor ahorro"]
    filas = []
    for R in [32, 64]:
        if f"nodos_totales_{R}" not in d:
            continue
        filas.append([
            f"{R}³", f"{d[f'nodos_totales_{R}']:,}", f"{d[f'hojas_ocupadas_{R}']:,}",
            f"{d[f'ocup_hoja_pct_{R}']:.3f}",
            f"{d[f'tam_npz_disperso_kib_{R}']:.3f}",
            f"{d[f'tam_npz_denso_kib_{R}']:.3f}",
            f"{d[f'factor_reduccion_almacenamiento_{R}']:.3f}x",
        ])

    if not filas:
        print(f"  [Omitido] {ruta_json.name} no tiene datos de resolucion")
        return

    subtitulo = ""
    if "n_vertices" in d:
        subtitulo = (f"({d['n_vertices']:,} vértices, {d['n_caras']:,} caras, "
                     f"{d['t_total_ms']:.1f} ms total)\n")

    guardar_tabla(
        columnas, filas,
        f"medir_metricas_off.py — {nombre}\n{subtitulo}"
        f"Comparación disperso vs. denso, MISMA compresión",
        dir_salida / f"tabla_metricas_off_{nombre}",
    )


def tabla_metricas_off_tiempos(ruta_json: Path, nombre: str, dir_salida: Path):
    """
    Segunda tabla: comparacion de TIEMPOS de construccion entre el
    arbol (con poda) y la rejilla densa construida de forma
    independiente (construir_grid_octree(), sin pasar por el arbol).
    """
    if not ruta_json.exists():
        print(f"  [Omitido] No existe: {ruta_json}")
        return

    d = json.load(open(ruta_json))

    if f"t_arbol_32_ms" not in d:
        print(f"  [Omitido] {ruta_json.name} no tiene campos de tiempo "
             f"(version antigua del JSON, volver a generar)")
        return

    columnas = ["Resolución", "T. árbol\n(ms)", "T. denso directo\n(ms)",
               "Árbol es\nX veces más lento", "T. guardado\ndisperso (ms)",
               "T. guardado\ndenso (ms)"]
    filas = []
    for R in [32, 64]:
        if f"t_arbol_{R}_ms" not in d:
            continue
        t_arbol = d[f"t_arbol_{R}_ms"]
        t_denso = d[f"t_denso_directo_{R}_ms"]
        factor = t_arbol / t_denso if t_denso > 0 else 0
        filas.append([
            f"{R}³", f"{t_arbol:.3f}", f"{t_denso:.3f}", f"{factor:.1f}x",
            f"{d[f't_guardado_{R}_ms']:.3f}", f"{d[f't_guardado_denso_{R}_ms']:.3f}",
        ])

    if not filas:
        return

    guardar_tabla(
        columnas, filas,
        f"medir_metricas_off.py — {nombre}\n"
        f"Comparación de tiempos de construcción: ÁRBOL (con poda) vs. "
        f"REJILLA DENSA (construcción independiente,\nsin pasar por el árbol — "
        f"construir_grid_octree() aplicado directamente a los puntos)",
        dir_salida / f"tabla_metricas_off_tiempos_{nombre}",
    )

    # Tabla adicional con los 3 totales por objeto (no depende de R)
    if "t_total_experimento_ms" in d:
        columnas_tot = ["Total octree\n(t_total_ms)", "Total denso\n(t_total_denso_ms)",
                        "Total experimento\n(t_total_experimento_ms)"]
        filas_tot = [[
            f"{d['t_total_ms']:.3f} ms",
            f"{d['t_total_denso_ms']:.3f} ms",
            f"{d['t_total_experimento_ms']:.3f} ms",
        ]]
        guardar_tabla(
            columnas_tot, filas_tot,
            f"medir_metricas_off.py — {nombre}\n"
            f"Totales de tiempo por objeto — 'experimento' es el comparable "
            f"contra el tiempo REAL paralelo\n(incluye ambas representaciones, "
            f"sin duplicar los costos compartidos de lectura/muestreo)",
            dir_salida / f"tabla_metricas_off_totales_{nombre}",
            alto_minimo=4.5,
        )


# ──────────────────────────────────────────────────────────────
# TABLA 3: metricas_<nombre>.json (medir_tiempo_memoria.py)
# ──────────────────────────────────────────────────────────────

def tabla_tiempo_memoria(ruta_json: Path, nombre: str, dir_salida: Path):
    if not ruta_json.exists():
        print(f"  [Omitido] No existe: {ruta_json}")
        return

    d = json.load(open(ruta_json))

    columnas = ["Resolución", "Nodos", "Hojas", "Transitorio\n(KiB)",
               "Python\n(KiB)", "Densa\n(KiB)", "Archivo\ndisco (KiB)", "Ahorro"]
    filas = []
    for R in [32, 64]:
        clave = f"construccion_arbol_R{R}"
        clave_disco = f"guardado_disco_R{R}"
        if clave not in d or clave_disco not in d:
            continue
        c = d[clave]
        g = d[clave_disco]
        filas.append([
            f"{R}³", f"{c['n_nodos_totales']:,}", f"{c['n_hojas_ocupadas']:,}",
            f"{c['mem_pico_transitorio_kib']:.1f}",
            f"{c['mem_python_arbol_kib']:.2f}",
            f"{c['mem_python_densa_kib']:.1f}",
            f"{g['tam_archivo_real_kib']:.2f}",
            f"{c['factor_ahorro_arbol_vs_densa']:.1f}x",
        ])

    if not filas:
        print(f"  [Omitido] {ruta_json.name} no tiene datos de resolucion")
        return

    guardar_tabla(
        columnas, filas,
        f"medir_tiempo_memoria.py — {nombre}\n"
        f"Cuatro magnitudes distintas, mismo criterio de medición",
        dir_salida / f"tabla_tiempo_memoria_{nombre}",
    )


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convierte los JSON de octree_real.py, medir_metricas_off.py "
                    "y medir_tiempo_memoria.py en tablas imagen (PNG + SVG)."
    )
    parser.add_argument("--nombre", type=str, required=True,
                        help="Nombre base del objeto (ej: chair_0001), usado "
                             "para localizar los JSON con nombres por defecto")
    parser.add_argument("--octree-real", type=str, default=None,
                        help="Ruta explicita a octree_real_<nombre>.json (opcional)")
    parser.add_argument("--metricas-off", type=str, default=None,
                        help="Ruta explicita a metricas_off_<nombre>.json (opcional)")
    parser.add_argument("--tiempo-memoria", type=str, default=None,
                        help="Ruta explicita a metricas_<nombre>.json (opcional)")
    parser.add_argument("--salida", type=str, default=".",
                        help="Carpeta donde guardar las imagenes (default: carpeta actual)")
    args = parser.parse_args()

    nombre = args.nombre
    dir_salida = Path(args.salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    ruta_octree_real = Path(args.octree_real) if args.octree_real else \
                       RUTA_DEFECTO_OCTREE_REAL / f"octree_real_{nombre}.json"
    ruta_metricas_off = Path(args.metricas_off) if args.metricas_off else \
                        RUTA_DEFECTO_METRICAS_OFF / f"metricas_off_{nombre}.json"
    ruta_tiempo_memoria = Path(args.tiempo_memoria) if args.tiempo_memoria else \
                          RUTA_DEFECTO_TIEMPO_MEMORIA / f"metricas_{nombre}.json"

    print("=" * 60)
    print(f"  GENERANDO IMAGENES DE TABLAS — {nombre}")
    print("=" * 60)

    print(f"\n[1/3] octree_real.py -> {ruta_octree_real}")
    tabla_octree_real(ruta_octree_real, nombre, dir_salida)

    print(f"\n[2/3] medir_metricas_off.py -> {ruta_metricas_off}")
    tabla_metricas_off(ruta_metricas_off, nombre, dir_salida)
    tabla_metricas_off_tiempos(ruta_metricas_off, nombre, dir_salida)

    print(f"\n[3/3] medir_tiempo_memoria.py -> {ruta_tiempo_memoria}")
    tabla_tiempo_memoria(ruta_tiempo_memoria, nombre, dir_salida)

    print(f"\nImagenes guardadas en: {dir_salida.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
