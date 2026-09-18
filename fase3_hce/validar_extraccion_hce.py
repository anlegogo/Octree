"""
validar_extraccion_hce.py
============================
Organiza y valida el codigo que recibe un octree (arbol real, con poda)
y produce un vector de caracteristicas HCE de dimension fija.

Para el objeto indicado (por defecto chair_0001.off), en cada resolucion
(32^3 y 64^3), genera un archivo JSON con:
  - Nombre y valor de cada descriptor (en el orden exacto del vector)
  - Dimension total del vector
  - Datos basicos del arbol usados en el calculo (nodos, hojas, ocupacion)
  - Resultado de 3 verificaciones automaticas:
      1. Sin valores indefinidos (NaN / Inf) en ningun descriptor
      2. Reproducibilidad: la extraccion se corre 2 veces de forma
         independiente (arbol reconstruido desde cero ambas veces) y
         se exige que el vector resultante sea bit-identico
      3. Convencion de niveles: el primer descriptor de ocupacion debe
         ser 'ocupacion_L0' (raiz), y el ultimo 'ocupacion_L{profundidad_max}'
         (hoja) -- confirma que la raiz esta en L=0, no en otra convencion

No se usa una rejilla densa en ningun punto: la extraccion opera sobre
el arbol real (NodoOctree) o el .npz disperso con estructura jerarquica
completa (ver hce_extraccion.py y octree_real.py).

Uso:
    python validar_extraccion_hce.py --off chair_0001.off
    python validar_extraccion_hce.py --off chair_0001.off --salida validacion_hce/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAIZ_PROYECTO = Path(__file__).parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))
sys.path.insert(0, str(RAIZ_PROYECTO / "fase3_hce"))

from octree import leer_off, normalizar_malla, muestrear_superficie_con_normales
from octree_real import (
    construir_octree, recolectar_hojas, contar_nodos_totales,
    contar_nodos_por_profundidad,
)
from hce_extraccion import extraer_descriptores_hce, nombres_features

PROFUNDIDAD_POR_RESOLUCION = {32: 5, 64: 6}


def extraer_con_datos_de_arbol(pts: np.ndarray, normales: np.ndarray,
                               profundidad_max: int) -> tuple:
    """
    Construye el arbol REAL desde la nube de puntos y extrae el vector
    de descriptores, retornando tambien los datos basicos del arbol
    usados en el calculo (no solo el vector final).
    """
    raiz = construir_octree(pts, normales, profundidad_max=profundidad_max)

    feats = extraer_descriptores_hce(raiz, profundidad_max)

    n_nodos_totales = contar_nodos_totales(raiz)
    n_hojas = len(recolectar_hojas(raiz))
    conteo_por_nivel = contar_nodos_por_profundidad(raiz, profundidad_max)

    datos_arbol = {
        "profundidad_max": profundidad_max,
        "n_nodos_totales": n_nodos_totales,
        "n_hojas_ocupadas": n_hojas,
        "nodos_por_nivel": {
            f"L{d}": int(conteo_por_nivel[d]) for d in range(profundidad_max + 1)
        },
        "pct_ocupacion_hoja": round(100 * n_hojas / (2 ** profundidad_max) ** 3, 4),
    }

    return feats, datos_arbol


def validar_resolucion(pts: np.ndarray, normales: np.ndarray, R: int) -> dict:
    """
    Ejecuta la extraccion para una resolucion, con las 3 verificaciones
    automaticas, y arma el diccionario final a guardar como JSON.
    """
    L = PROFUNDIDAD_POR_RESOLUCION[R]
    nombres = nombres_features(L)

    # ── Extraccion 1: para el resultado final reportado ──
    feats_1, datos_arbol = extraer_con_datos_de_arbol(pts, normales, L)

    # ── Extraccion 2: arbol reconstruido DESDE CERO, para verificar
    #    reproducibilidad (no reutiliza ningun objeto de la primera) ──
    feats_2, _ = extraer_con_datos_de_arbol(pts, normales, L)

    # ── Verificacion 1: sin valores indefinidos ──
    tiene_nan = bool(np.any(np.isnan(feats_1)))
    tiene_inf = bool(np.any(np.isinf(feats_1)))
    sin_indefinidos = (not tiene_nan) and (not tiene_inf)

    # ── Verificacion 2: reproducibilidad (bit-identico entre corridas) ──
    es_reproducible = bool(np.array_equal(feats_1, feats_2))

    # ── Verificacion 3: convencion de niveles, raiz en L=0 ──
    primer_descriptor_es_raiz = nombres[0] == "ocupacion_L0"
    ultimo_descriptor_ocupacion = f"ocupacion_L{L}"
    hoja_presente = ultimo_descriptor_ocupacion in nombres
    convencion_correcta = primer_descriptor_es_raiz and hoja_presente

    todas_las_verificaciones_ok = (
        sin_indefinidos and es_reproducible and convencion_correcta
    )

    resultado = {
        "objeto": None,  # se completa en main()
        "resolucion": R,
        "profundidad_max": L,
        "dimension_vector": len(feats_1),
        "descriptores": [
            {"nombre": n, "valor": float(v)} for n, v in zip(nombres, feats_1)
        ],
        "datos_arbol": datos_arbol,
        "verificaciones": {
            "sin_valores_indefinidos": sin_indefinidos,
            "detalle_indefinidos": {
                "tiene_nan": tiene_nan,
                "tiene_inf": tiene_inf,
            },
            "reproducible_entre_corridas": es_reproducible,
            "convencion_raiz_L0_correcta": convencion_correcta,
            "detalle_convencion": {
                "primer_descriptor": nombres[0],
                "esperado": "ocupacion_L0",
                "ultimo_descriptor_ocupacion_hoja": ultimo_descriptor_ocupacion,
                "presente_en_vector": hoja_presente,
            },
            "TODAS_LAS_VERIFICACIONES_OK": todas_las_verificaciones_ok,
        },
    }

    return resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", type=str, required=True)
    parser.add_argument("--n_puntos", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--salida", type=str, default=".")
    args = parser.parse_args()

    dir_salida = Path(args.salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    nombre_objeto = Path(args.off).stem

    print("=" * 70)
    print(f"  VALIDACION DE EXTRACCION HCE — {nombre_objeto}")
    print("=" * 70)

    verts, caras = leer_off(args.off)
    verts = normalizar_malla(verts)
    rng = np.random.default_rng(args.seed)
    pts, normales = muestrear_superficie_con_normales(verts, caras, args.n_puntos, rng)
    print(f"\nNube de puntos: {len(pts)} (seed={args.seed})")

    todo_ok_global = True

    for R in (32, 64):
        print(f"\n--- Resolucion {R}^3 ---")
        resultado = validar_resolucion(pts, normales, R)
        resultado["objeto"] = nombre_objeto

        v = resultado["verificaciones"]
        print(f"  Dimension del vector          : {resultado['dimension_vector']}")
        print(f"  Sin valores indefinidos (NaN/Inf): {v['sin_valores_indefinidos']}")
        print(f"  Reproducible entre corridas    : {v['reproducible_entre_corridas']}")
        print(f"  Convencion raiz L=0 correcta   : {v['convencion_raiz_L0_correcta']}")
        print(f"  >>> TODAS LAS VERIFICACIONES OK: {v['TODAS_LAS_VERIFICACIONES_OK']} <<<")

        todo_ok_global = todo_ok_global and v["TODAS_LAS_VERIFICACIONES_OK"]

        ruta_salida = dir_salida / f"validacion_hce_{nombre_objeto}_R{R}.json"
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        print(f"  [Guardado] {ruta_salida}")

    print("\n" + "=" * 70)
    if todo_ok_global:
        print("  RESULTADO GLOBAL: TODAS LAS VERIFICACIONES PASARON (32^3 y 64^3)")
    else:
        print("  RESULTADO GLOBAL: HAY VERIFICACIONES QUE FALLARON -- revisar JSON")
    print("=" * 70)

    return 0 if todo_ok_global else 1


if __name__ == "__main__":
    sys.exit(main())
