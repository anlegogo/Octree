"""
visualizar_matriz_confusion_net5.py
=====================================
Genera las matrices de confusion de Net5-Octree para las resoluciones
32^3 y 64^3, a partir de los JSON guardados por fase3_net5_entrenamiento.py.

Figuras generadas:
  - matriz_confusion_net5_R32.png  /  .svg
  - matriz_confusion_net5_R64.png  /  .svg
  - metricas_por_clase_net5_R32.png  /  .svg
  - metricas_por_clase_net5_R64.png  /  .svg

Uso:
    python visualizar_matriz_confusion_net5.py
    python visualizar_matriz_confusion_net5.py --resolucion 32
    python visualizar_matriz_confusion_net5.py --resolucion 64
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DIR_RESULTADOS = Path(r"C:\Users\ricar\Documents\Codigos\Tesis\resultados")

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
# CARGA
# ──────────────────────────────────────────────────────────────

def cargar_resumen(R: int) -> dict:
    ruta = DIR_RESULTADOS / f"resumen_net5_R{R}.json"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontro {ruta.name}. "
            f"Asegurate de haber entrenado Net5 con la version actualizada "
            f"del script que guarda la matriz de confusion."
        )
    with open(ruta) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────
# MATRIZ DE CONFUSION
# ──────────────────────────────────────────────────────────────

def graficar_matriz_confusion(resumen: dict, R: int):
    """
    Genera un mapa de calor de la matriz de confusion normalizada por fila
    (normalizacion por clase real, para que el valor en la diagonal sea
    la tasa de acierto por clase, comparable entre clases con distinto
    numero de muestras).
    """
    matriz = np.array(resumen["matriz_confusion"])
    matriz_norm = matriz.astype(float) / matriz.sum(axis=1, keepdims=True).clip(min=1)

    test_acc  = resumen["test_acc"] * 100
    mejor_ep  = resumen["mejor_epoca"]
    val_acc   = resumen["mejor_val_acc"] * 100

    fig, ax = plt.subplots(figsize=(13, 11))

    im = ax.imshow(matriz_norm, cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03,
                 label="Tasa de clasificacion (por clase real)")

    ax.set_xticks(range(len(CLASES)))
    ax.set_yticks(range(len(CLASES)))
    ax.set_xticklabels(CLASES, rotation=90, fontsize=6.5)
    ax.set_yticklabels(CLASES, fontsize=6.5)
    ax.set_xlabel("Clase predicha", fontsize=11)
    ax.set_ylabel("Clase real", fontsize=11)

    # Marcar la diagonal con un borde para resaltarla
    for i in range(len(CLASES)):
        rect = plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            linewidth=0.6, edgecolor="black", facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title(
        f"Matriz de Confusión — Net5-Octree  {R}³\n"
        f"Test Acc: {test_acc:.2f}%  |  "
        f"Val Acc: {val_acc:.2f}%  |  "
        f"Mejor época: {mejor_ep}",
        fontsize=12, pad=12,
    )

    plt.tight_layout()

    for ext in ("png", "svg"):
        salida = DIR_RESULTADOS / f"matriz_confusion_net5_R{R}.{ext}"
        plt.savefig(salida, dpi=150 if ext == "png" else None,
                    format=ext, bbox_inches="tight")
        print(f"[Guardado] {salida.name}")

    plt.close()


# ──────────────────────────────────────────────────────────────
# METRICAS POR CLASE (Precision / Recall / F1)
# ──────────────────────────────────────────────────────────────

def graficar_metricas_por_clase(resumen: dict, R: int):
    """
    Grafica precision, recall y F1-score por clase, ordenadas de mayor
    a menor F1 para identificar rapidamente las clases mas dificiles.
    """
    reporte = resumen.get("reporte_clasificacion", {})
    if not reporte:
        print(f"  [Aviso] No hay reporte de clasificacion en resumen_net5_R{R}.json")
        return

    # Extraer metricas por clase en el mismo orden que CLASES
    precision = [reporte.get(c, {}).get("precision", 0) for c in CLASES]
    recall    = [reporte.get(c, {}).get("recall", 0)    for c in CLASES]
    f1        = [reporte.get(c, {}).get("f1-score", 0)  for c in CLASES]

    # Ordenar por F1 descendente para mejor legibilidad
    orden = np.argsort(f1)[::-1]
    clases_ord    = [CLASES[i] for i in orden]
    precision_ord = [precision[i] for i in orden]
    recall_ord    = [recall[i]    for i in orden]
    f1_ord        = [f1[i]        for i in orden]

    x     = np.arange(len(CLASES))
    ancho = 0.27

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.bar(x - ancho, precision_ord, ancho, label="Precisión",  color="#2196F3", alpha=0.85)
    ax.bar(x,          recall_ord,    ancho, label="Recall",     color="#FF9800", alpha=0.85)
    ax.bar(x + ancho,  f1_ord,        ancho, label="F1-score",   color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(clases_ord, rotation=90, fontsize=8)
    ax.set_ylabel("Puntuación", fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.axhline(y=resumen["test_acc"], color="red", linestyle="--",
               linewidth=1.2, alpha=0.7,
               label=f"Test Acc global ({resumen['test_acc']*100:.2f}%)")
    ax.set_title(
        f"Precisión / Recall / F1 por clase — Net5-Octree {R}³\n"
        f"(ordenado por F1 descendente)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    for ext in ("png", "svg"):
        salida = DIR_RESULTADOS / f"metricas_por_clase_net5_R{R}.{ext}"
        plt.savefig(salida, dpi=150 if ext == "png" else None,
                    format=ext, bbox_inches="tight")
        print(f"[Guardado] {salida.name}")

    plt.close()

    # Imprimir las 5 clases con menor F1 (las mas dificiles)
    print(f"\n  Top 5 clases con menor F1 (R={R}^3):")
    for i in range(-1, -6, -1):
        idx = orden[i]
        print(f"    {CLASES[idx]:<15}: F1={f1[idx]:.3f}  "
              f"P={precision[idx]:.3f}  R={recall[idx]:.3f}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolucion", type=int, default=0, choices=[0, 32, 64],
                        help="0 = ambas (default), 32 o 64")
    args = parser.parse_args()

    resoluciones = [32, 64] if args.resolucion == 0 else [args.resolucion]

    for R in resoluciones:
        print(f"\n{'='*55}")
        print(f"  Net5-Octree — Resolucion {R}^3")
        print(f"{'='*55}")

        try:
            resumen = cargar_resumen(R)
        except FileNotFoundError as e:
            print(f"  [ERROR] {e}")
            continue

        print(f"  Test Acc   : {resumen['test_acc']*100:.2f}%")
        print(f"  Val  Acc   : {resumen['mejor_val_acc']*100:.2f}%")
        print(f"  Mejor epoca: {resumen['mejor_epoca']}")

        print(f"\n  Generando matriz de confusion...")
        graficar_matriz_confusion(resumen, R)

        print(f"\n  Generando metricas por clase...")
        graficar_metricas_por_clase(resumen, R)

    print(f"\nFiguras guardadas en: {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()
