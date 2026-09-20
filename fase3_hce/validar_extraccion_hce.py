"""
validar_extraccion_hce.py
============================
Organiza y valida el codigo que recibe un octree (arbol real, con poda)
y produce un vector de caracteristicas HCE de dimension fija.

Para el objeto indicado (por defecto chair_0001.off), en cada resolucion
(32^3 y 64^3), genera un archivo JSON con:
  - Nombre y valor de cada descriptor DEL VECTOR (17 para 32^3, 18 para
    64^3 -- ocupacion_L0/raiz excluida por ser constante, ver mas abajo)
  - Dimension total del vector
  - Datos basicos del arbol usados en el calculo (nodos, hojas, ocupacion
    por nivel COMPLETA incluyendo L0 como dato de verificacion)
  - Configuracion de reproduccion: semilla, numero de puntos muestreados,
    resolucion, profundidad
  - Resultado de 4 verificaciones automaticas:
      1. Sin valores indefinidos (NaN / Inf) en ningun descriptor
      2. Reproducibilidad: la extraccion se corre 2 veces de forma
         independiente (arbol reconstruido desde cero ambas veces) y
         se exige que el vector resultante sea bit-identico
      3. Convencion de niveles: la raiz (L=0) tiene 100% de ocupacion
         (verificado, no incluido en el vector) y el primer descriptor
         del vector es 'ocupacion_L1'
      4. Recorrido completo de produccion (construir -> guardar .npz ->
         cargar -> comparar): estructura padre-hijo nodo por nodo,
         nodos por nivel, numero de hojas, y vector de descriptores
         final, todos deben coincidir exactamente entre el arbol
         original en memoria y el reconstruido desde disco.

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
import tempfile
from pathlib import Path

import numpy as np

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))
sys.path.insert(0, str(RAIZ_PROYECTO / "fase3_hce"))

from octree import leer_off, normalizar_malla, muestrear_superficie_con_normales
from octree_real import (
    construir_octree, recolectar_hojas, contar_nodos_totales,
    contar_nodos_por_profundidad, guardar_octree_disperso,
    reconstruir_octree_desde_npz,
)
from hce_extraccion import (
    extraer_descriptores_hce, extraer_descriptores_hce_desde_npz,
    nombres_features, ocupacion_L0_verificacion,
)

PROFUNDIDAD_POR_RESOLUCION = {32: 5, 64: 6}


# ──────────────────────────────────────────────────────────────
# Comparacion exhaustiva de arboles, nodo por nodo (para el punto 4)
# ──────────────────────────────────────────────────────────────

def comparar_arboles_exhaustivo(nodo_a, nodo_b, ruta="raiz") -> list:
    """Compara recursivamente DOS arboles nodo por nodo, incluyendo
    normal_coherencia. Retorna lista de discrepancias (vacia si son
    identicos)."""
    errores = []
    if nodo_a is None and nodo_b is None:
        return errores
    if (nodo_a is None) != (nodo_b is None):
        errores.append(f"{ruta}: uno es None y el otro no")
        return errores

    if nodo_a.profundidad != nodo_b.profundidad:
        errores.append(f"{ruta}: profundidad {nodo_a.profundidad} != {nodo_b.profundidad}")
    if nodo_a.es_hoja != nodo_b.es_hoja:
        errores.append(f"{ruta}: es_hoja {nodo_a.es_hoja} != {nodo_b.es_hoja}")
    if not np.allclose(nodo_a.centro, nodo_b.centro, atol=1e-5):
        errores.append(f"{ruta}: centro {nodo_a.centro} != {nodo_b.centro}")

    if nodo_a.es_hoja and nodo_b.es_hoja:
        if (nodo_a.normal_promedio is None) != (nodo_b.normal_promedio is None):
            errores.append(f"{ruta}: presencia de normal_promedio distinta")
        elif nodo_a.normal_promedio is not None:
            if not np.allclose(nodo_a.normal_promedio, nodo_b.normal_promedio, atol=1e-5):
                errores.append(f"{ruta}: normal_promedio distinto")
        if (nodo_a.normal_coherencia is None) != (nodo_b.normal_coherencia is None):
            errores.append(f"{ruta}: presencia de normal_coherencia distinta")
        elif nodo_a.normal_coherencia is not None:
            if abs(nodo_a.normal_coherencia - nodo_b.normal_coherencia) > 1e-5:
                errores.append(
                    f"{ruta}: normal_coherencia {nodo_a.normal_coherencia} "
                    f"!= {nodo_b.normal_coherencia}"
                )

    if not nodo_a.es_hoja and not nodo_b.es_hoja:
        for i in range(8):
            errores.extend(comparar_arboles_exhaustivo(
                nodo_a.hijos[i], nodo_b.hijos[i], f"{ruta}.hijo[{i}]",
            ))

    return errores


def verificar_recorrido_produccion(pts: np.ndarray, normales: np.ndarray,
                                   R: int, etiqueta: int = 0) -> dict:
    """
    PUNTO 3 (observacion de Andres Gonzalez): construye el octree,
    lo guarda en .npz, lo carga de nuevo, y verifica que se conservan:
      - la estructura padre-hijo completa (comparacion nodo por nodo)
      - el numero de nodos por nivel
      - el numero de hojas
      - el vector de descriptores HCE final
    """
    L = PROFUNDIDAD_POR_RESOLUCION[R]

    # 1. Construir en memoria
    raiz_original = construir_octree(pts, normales, profundidad_max=L)

    # 2-4. Guardar, cargar y comparar usando un directorio temporal unico.
    # Esto evita colisiones entre validaciones ejecutadas en paralelo.
    with tempfile.TemporaryDirectory(prefix=f"verif_hce_R{R}_") as dir_tmp:
        ruta_tmp = Path(dir_tmp) / "octree.npz"
        guardar_octree_disperso(
            raiz_original, str(ruta_tmp), etiqueta=etiqueta,
            profundidad_max=L,
        )

        raiz_cargada, etiqueta_cargada, profundidad_cargada = (
            reconstruir_octree_desde_npz(str(ruta_tmp))
        )
        errores_estructura = comparar_arboles_exhaustivo(
            raiz_original, raiz_cargada,
        )

        conteo_original = contar_nodos_por_profundidad(raiz_original, L)
        conteo_cargado = contar_nodos_por_profundidad(raiz_cargada, L)
        nodos_por_nivel_coinciden = bool(
            np.array_equal(conteo_original, conteo_cargado)
        )

        n_hojas_original = len(recolectar_hojas(raiz_original))
        n_hojas_cargada = len(recolectar_hojas(raiz_cargada))
        hojas_coinciden = n_hojas_original == n_hojas_cargada

        feats_memoria = extraer_descriptores_hce(raiz_original, L)
        feats_desde_npz = extraer_descriptores_hce_desde_npz(str(ruta_tmp))
        vector_coincide = bool(
            np.array_equal(feats_memoria, feats_desde_npz)
        )
        tam_archivo_kib = round(ruta_tmp.stat().st_size / 1024, 4)

    return {
        "sin_errores_estructura_padre_hijo": len(errores_estructura) == 0,
        "n_discrepancias_estructura": len(errores_estructura),
        "primeras_discrepancias": errores_estructura[:5],
        "nodos_por_nivel_coinciden": nodos_por_nivel_coinciden,
        "n_hojas_original": n_hojas_original,
        "n_hojas_desde_disco": n_hojas_cargada,
        "n_hojas_coinciden": hojas_coinciden,
        "vector_descriptores_coincide": vector_coincide,
        "etiqueta_preservada": etiqueta_cargada == etiqueta,
        "profundidad_max_preservada": profundidad_cargada == L,
        "tam_archivo_kib": tam_archivo_kib,
        "RECORRIDO_PRODUCCION_OK": (
            len(errores_estructura) == 0 and nodos_por_nivel_coinciden
            and hojas_coinciden and vector_coincide
            and etiqueta_cargada == etiqueta and profundidad_cargada == L
        ),
    }


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
    ocupacion_raiz = ocupacion_L0_verificacion(raiz, profundidad_max, desde_arbol=True)

    datos_arbol = {
        "profundidad_max": profundidad_max,
        "n_nodos_totales": n_nodos_totales,
        "n_hojas_ocupadas": n_hojas,
        "nodos_por_nivel": {
            f"L{d}": int(conteo_por_nivel[d]) for d in range(profundidad_max + 1)
        },
        "pct_ocupacion_hoja": round(100 * n_hojas / (2 ** profundidad_max) ** 3, 4),
        "ocupacion_L0_raiz_verificacion": ocupacion_raiz,
    }

    return feats, datos_arbol, raiz


def validar_resolucion(pts: np.ndarray, normales: np.ndarray, R: int,
                       seed: int, n_puntos: int, nombre_objeto: str) -> dict:
    """
    Ejecuta la extraccion para una resolucion, con las 4 verificaciones
    automaticas, y arma el diccionario final a guardar como JSON.
    """
    L = PROFUNDIDAD_POR_RESOLUCION[R]
    nombres = nombres_features(L)

    # ── Extraccion 1: para el resultado final reportado ──
    feats_1, datos_arbol, _ = extraer_con_datos_de_arbol(pts, normales, L)

    # ── Extraccion 2: arbol reconstruido DESDE CERO, para verificar
    #    reproducibilidad (no reutiliza ningun objeto de la primera) ──
    feats_2, _, _ = extraer_con_datos_de_arbol(pts, normales, L)

    # ── Verificacion 1: sin valores indefinidos ──
    tiene_nan = bool(np.any(np.isnan(feats_1)))
    tiene_inf = bool(np.any(np.isinf(feats_1)))
    sin_indefinidos = (not tiene_nan) and (not tiene_inf)

    # ── Verificacion 2: reproducibilidad (bit-identico entre corridas) ──
    es_reproducible = bool(np.array_equal(feats_1, feats_2))

    # ── Verificacion 3: convencion de niveles, raiz en L=0 ──
    # La raiz (L=0) debe tener 100% de ocupacion (constante, por eso se
    # excluye del vector); el primer descriptor del VECTOR debe ser L=1.
    raiz_es_100 = datos_arbol["ocupacion_L0_raiz_verificacion"] == 100.0
    primer_descriptor_es_L1 = nombres[0] == "ocupacion_L1"
    convencion_correcta = raiz_es_100 and primer_descriptor_es_L1

    dimension_esperada = L + 12
    dimension_correcta = (
        len(feats_1) == dimension_esperada == len(nombres)
        and len(set(nombres)) == len(nombres)
    )

    # ── Verificacion 4: recorrido completo de produccion ──
    recorrido = verificar_recorrido_produccion(pts, normales, R, etiqueta=0)

    todas_las_verificaciones_ok = (
        sin_indefinidos and es_reproducible and convencion_correcta
        and dimension_correcta
        and recorrido["RECORRIDO_PRODUCCION_OK"]
    )

    resultado = {
        "objeto": nombre_objeto,
        "resolucion": R,
        "profundidad_max": L,
        "configuracion_reproduccion": {
            "seed": seed,
            "n_puntos_muestreados": n_puntos,
            "metodo_muestreo": "area-weighted sobre triangulos de la malla",
        },
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
            "dimension_y_orden_correctos": dimension_correcta,
            "dimension_esperada": dimension_esperada,
            "convencion_raiz_L0_correcta": convencion_correcta,
            "detalle_convencion": {
                "ocupacion_raiz_L0": datos_arbol["ocupacion_L0_raiz_verificacion"],
                "esperado_raiz": 100.0,
                "primer_descriptor_del_vector": nombres[0],
                "esperado_primer_descriptor": "ocupacion_L1",
                "nota": "L0 (raiz) se verifica pero NO forma parte del vector "
                       "de entrada al clasificador, por ser constante.",
            },
            "recorrido_produccion": recorrido,
            "TODAS_LAS_VERIFICACIONES_OK": todas_las_verificaciones_ok,
        },
    }

    return resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--n-puntos", "--n_puntos", dest="n_puntos",
                        type=int, default=20000)
    parser.add_argument("--semilla", "--seed", dest="seed",
                        type=int, default=42)
    parser.add_argument("--salida", type=Path, default=Path("."))
    args = parser.parse_args()

    if args.n_puntos <= 0:
        raise ValueError("--n-puntos debe ser positivo")
    if not args.off.is_file():
        raise FileNotFoundError(f"No se encontro la malla: {args.off}")

    dir_salida = args.salida
    dir_salida.mkdir(parents=True, exist_ok=True)
    nombre_objeto = args.off.stem

    print("=" * 70)
    print(f"  VALIDACION DE EXTRACCION HCE — {nombre_objeto}")
    print("=" * 70)

    verts, caras = leer_off(str(args.off))
    verts = normalizar_malla(verts)
    rng = np.random.default_rng(args.seed)
    pts, normales = muestrear_superficie_con_normales(verts, caras, args.n_puntos, rng)
    print(f"\nNube de puntos: {len(pts)} (seed={args.seed})")

    todo_ok_global = True

    for R in (32, 64):
        print(f"\n--- Resolucion {R}^3 ---")
        resultado = validar_resolucion(
            pts, normales, R, args.seed, args.n_puntos, nombre_objeto,
        )

        v = resultado["verificaciones"]
        rp = v["recorrido_produccion"]
        print(f"  Dimension del vector             : {resultado['dimension_vector']}")
        print(f"  Sin valores indefinidos (NaN/Inf): {v['sin_valores_indefinidos']}")
        print(f"  Reproducible entre corridas       : {v['reproducible_entre_corridas']}")
        print(f"  Convencion raiz L=0 correcta      : {v['convencion_raiz_L0_correcta']}")
        print(f"  Recorrido de produccion completo  : {rp['RECORRIDO_PRODUCCION_OK']}")
        print(f"    - Estructura padre-hijo intacta : {rp['sin_errores_estructura_padre_hijo']}")
        print(f"    - Nodos por nivel coinciden     : {rp['nodos_por_nivel_coinciden']}")
        print(f"    - Hojas coinciden ({rp['n_hojas_original']} == {rp['n_hojas_desde_disco']})       : {rp['n_hojas_coinciden']}")
        print(f"    - Vector desde disco coincide   : {rp['vector_descriptores_coincide']}")
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
