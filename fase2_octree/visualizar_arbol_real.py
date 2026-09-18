"""
Visualiza el octree REAL como cajas de subdivision (wireframe), a
diferencia de las figuras anteriores que mostraban una rejilla densa
uniforme. Aqui cada caja dibujada es un nodo que REALMENTE EXISTE en
el arbol (no fue podado), y su tamano varia segun la profundidad --
la caracteristica visual distintiva de un octree adaptativo real.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from octree import leer_off, normalizar_malla, muestrear_superficie_con_normales
from octree_real import construir_octree, NodoOctree


def obtener_nodos_en_profundidad(raiz: NodoOctree, profundidad_objetivo: int) -> list:
    """Recolecta (centro, tamano) de todos los nodos EXISTENTES (no
    podados) en una profundidad especifica del arbol."""
    nodos = []

    def _rec(nodo):
        if nodo is None:
            return
        if nodo.profundidad == profundidad_objetivo:
            nodos.append((nodo.centro, nodo.tamano))
            return  # no seguir bajando mas alla de la profundidad pedida
        if not nodo.es_hoja:
            for hijo in nodo.hijos:
                _rec(hijo)
        # Si es hoja pero a menor profundidad que la pedida, no hay
        # nodos mas profundos en esta rama (ya no se subdividio mas).

    _rec(raiz)
    return nodos


def dibujar_caja_wireframe(ax, centro, tamano, color="steelblue", alpha=0.6, lw=0.6):
    """Dibuja las 12 aristas de un cubo centrado en `centro` con lado `tamano`."""
    h = tamano / 2.0
    # 8 vertices del cubo
    signos = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
              (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    verts = [centro + h*np.array(s) for s in signos]

    # 12 aristas (pares de indices de vertices)
    aristas_idx = [
        (0,1),(1,2),(2,3),(3,0),   # cara inferior
        (4,5),(5,6),(6,7),(7,4),   # cara superior
        (0,4),(1,5),(2,6),(3,7),   # verticales
    ]
    segmentos = [[verts[i], verts[j]] for i, j in aristas_idx]

    coleccion = Line3DCollection(segmentos, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection3d(coleccion)


def graficar_niveles_arbol_real(raiz, profundidad_max, R_hoja, salida_base):
    """Genera una figura multi-panel mostrando las cajas REALES del
    arbol en cada nivel, ilustrando la poda (menos cajas, de tamano
    variable, concentradas donde hay geometria)."""

    niveles_a_mostrar = list(range(1, profundidad_max + 1))  # omitir L=0 (trivial, 1 caja)
    n_paneles = len(niveles_a_mostrar)

    fig = plt.figure(figsize=(4 * n_paneles, 4.5))

    for col, d in enumerate(niveles_a_mostrar):
        ax = fig.add_subplot(1, n_paneles, col + 1, projection="3d")
        nodos = obtener_nodos_en_profundidad(raiz, d)

        for centro, tamano in nodos:
            dibujar_caja_wireframe(ax, centro, tamano)

        max_posible = 8 ** d
        pct = 100 * len(nodos) / max_posible

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"$L={d}$\n{len(nodos)} / {max_posible} nodos ({pct:.1f}%)",
                    fontsize=10)
        ax.view_init(elev=22, azim=-55)
        ax.set_box_aspect([1, 1, 1])

    fig.text(0.5, -0.02,
             f"Figura: Octree REAL — cajas de subdivision existentes por nivel "
             f"(sin poda no se muestran; hoja en $L={profundidad_max}$, "
             f"resolucion equivalente ${R_hoja}^3$)",
             ha="center", va="top", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{salida_base}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{salida_base}.svg", format="svg", bbox_inches="tight")
    print(f"[Guardado] {salida_base}.png / .svg")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--salida-dir", type=Path, default=Path("figuras_octree"))
    parser.add_argument("--n-puntos", type=int, default=20000)
    parser.add_argument("--semilla", type=int, default=42)
    args = parser.parse_args()

    if not args.off.is_file():
        raise FileNotFoundError(args.off)
    args.salida_dir.mkdir(parents=True, exist_ok=True)

    verts, caras = leer_off(str(args.off))
    verts = normalizar_malla(verts)
    rng = np.random.default_rng(args.semilla)
    pts, norms = muestrear_superficie_con_normales(
        verts, caras, args.n_puntos, rng,
    )

    for resolucion, profundidad in ((32, 5), (64, 6)):
        raiz = construir_octree(pts, norms, profundidad_max=profundidad)
        salida = args.salida_dir / f"octree_real_niveles_R{resolucion}"
        graficar_niveles_arbol_real(
            raiz, profundidad, resolucion, str(salida),
        )
    print("\nListo.")


if __name__ == "__main__":
    main()
