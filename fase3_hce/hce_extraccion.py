"""
hce_extraccion.py - Fase 3 (Enfoque clasico: HCE)
====================================================
Hand-Crafted Extraction: calcula descriptores manuales a partir de un
OCTREE REAL (arbol con nodo raiz, subdivision recursiva y poda de ramas
vacias -- ver octree_real.py), sin usar deep learning.

Dos formas de uso:
  1. extraer_descriptores_hce(raiz, profundidad_max) -- cuando se tiene
     el arbol NodoOctree recien construido (ej. pruebas, visualizacion).
  2. extraer_descriptores_hce_desde_npz(ruta_npz) -- cuando se carga
     el archivo disperso ya persistido por preprocesar_octrees.py
     (uso normal en el entrenamiento de SVM/Random Forest). Esta
     variante NO reconstruye el arbol completo: calcula la ocupacion
     por nivel directamente desde los centros de las hojas guardadas
     (ver octree_real.py::ocupacion_por_nivel_desde_hojas, verificada
     numericamente equivalente a recorrer el arbol).

Convencion de profundidad (raiz = L=0, un solo nodo, R=2^L en cada
nivel, consistente con octree_real.py):
    32^3 -> L=5 (hoja)
    64^3 -> L=6 (hoja)

CORRECCIONES (observaciones de Andres Gonzalez):

  1. Coherencia de normales, NO norma de un vector ya normalizado.
     octree_real.py::construir_octree() normaliza normal_promedio a
     vector UNITARIO antes de guardarlo (se usa asi para materializar
     el grid denso con una direccion consistente). Calcular la norma
     de ese vector ya normalizado da ~1 o exactamente 0 -- no aporta
     informacion real sobre la superficie. La medida util es
     normal_coherencia: la MAGNITUD del promedio de normales ANTES de
     normalizar (equivalente a la "longitud resultante media" de
     estadistica direccional). Se guarda por hoja en el arbol y se
     persiste en el .npz disperso (ver octree_real.py).

  2. ocupacion_L0 (la raiz) EXCLUIDA del vector de features. La raiz de
     cualquier octree no vacio siempre tiene 100% de ocupacion (por
     definicion: si el objeto tiene geometria, la raiz -que cubre todo
     el espacio- esta ocupada). Es una constante sin capacidad
     discriminante entre clases, y se retira del vector que ve el
     clasificador (SVM/Random Forest). Se mantiene disponible como dato
     de verificacion (ver datos_arbol en validar_extraccion_hce.py),
     no como entrada del modelo.

Descriptores extraidos:

  A) Ocupacion jerarquica por nivel, EXCLUYENDO la raiz (L=0): % de
     nodos que REALMENTE EXISTEN en el arbol (no podados) en cada
     profundidad d=1 .. L (hoja), respecto al maximo posible en ese
     nivel (8^d). Produce L valores (no L+1).

  B) Momentos geometricos globales (sobre los CENTROS de las hojas
     ocupadas, tratados como una nube de puntos discreta):
     centroide (3), varianza por eje (3), dispersion radial (1),
     skewness por eje (3) = 10 valores.

  C) Estadisticas de COHERENCIA de normales (sobre las hojas ocupadas,
     usando la magnitud PRE-normalizacion, no la norma del vector ya
     normalizado): coherencia media (1) y varianza de la coherencia
     (1) = 2 valores.

Total de features: L + 10 + 2 = L + 12
  Para R=32 (L=5): 5 + 12 = 17 features
  Para R=64 (L=6): 6 + 12 = 18 features
"""

import sys
import tempfile
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "fase2_octree"))
from octree_real import (
    NodoOctree, recolectar_hojas, ocupacion_por_nivel_arbol,
    ocupacion_por_nivel_desde_hojas, cargar_octree_disperso,
)


# ──────────────────────────────────────────────────────────────
# B. MOMENTOS GEOMETRICOS GLOBALES (sobre arrays crudos de centros)
# ──────────────────────────────────────────────────────────────

def momentos_geometricos(coords: np.ndarray) -> np.ndarray:
    """
    Calcula momentos geometricos sobre un array (N, 3) de coordenadas
    (centros de hojas ocupadas del octree real).

    Retorna 10 valores:
        [cx, cy, cz, vx, vy, vz, dispersion_radial, skew_x, skew_y, skew_z]
    """
    if len(coords) == 0:
        return np.zeros(10, dtype=np.float32)

    centroide = coords.mean(axis=0)
    diff = coords - centroide
    varianza = diff.var(axis=0)

    dist_radial = np.linalg.norm(diff, axis=1)
    dispersion_radial = dist_radial.mean()

    std = np.sqrt(np.clip(varianza, 1e-12, None))
    skew = np.mean(diff ** 3, axis=0) / (std ** 3 + 1e-12)

    features = np.concatenate([
        centroide, varianza, [dispersion_radial], skew,
    ]).astype(np.float32)

    return features


# ──────────────────────────────────────────────────────────────
# C. ESTADISTICAS DE COHERENCIA DE NORMALES
# (magnitud PRE-normalizacion, no la norma del vector unitario)
# ──────────────────────────────────────────────────────────────

def estadisticas_coherencia_normales(coherencias: np.ndarray) -> np.ndarray:
    """
    A partir de un array (N,) con la magnitud del promedio de normales
    de cada hoja ANTES de normalizar (normal_coherencia, en [0, 1]),
    calcula la media y varianza de esa coherencia.

    CORRECCION: antes se calculaba la norma de normal_promedio (vector
    YA normalizado a longitud unitaria), lo que daba siempre ~1 o
    exactamente 0 -- sin informacion real. Este descriptor usa la
    magnitud real pre-normalizacion, que refleja que tan alineadas
    estaban las normales de los puntos dentro de cada hoja (cercano a
    1 = superficie plana/coherente; cercano a 0 = superficie rugosa,
    curva, o hoja que abarca una arista/esquina).

    Retorna array de 2 valores: [coherencia_media, coherencia_varianza]
    """
    if len(coherencias) == 0:
        return np.zeros(2, dtype=np.float32)

    return np.array(
        [coherencias.mean(), coherencias.var()], dtype=np.float32,
    )


# ──────────────────────────────────────────────────────────────
# EXTRACTOR 1: desde un arbol NodoOctree recien construido
# ──────────────────────────────────────────────────────────────

def extraer_descriptores_hce(raiz: NodoOctree, profundidad_max: int) -> np.ndarray:
    """
    Extraccion HCE a partir de un arbol NodoOctree ya construido en
    memoria (uso tipico: pruebas, visualizacion, scripts que acaban
    de llamar a construir_octree() y no pasan por disco).

    El vector retornado EXCLUYE ocupacion_L0 (raiz, siempre 100%, ver
    docstring del modulo).
    """
    hojas = recolectar_hojas(raiz)
    centros = np.array([h.centro for h in hojas], dtype=np.float32) if hojas else np.zeros((0,3), dtype=np.float32)
    coherencias = np.array(
        [h.normal_coherencia if h.normal_coherencia is not None else 0.0 for h in hojas],
        dtype=np.float32,
    ) if hojas else np.zeros(0, dtype=np.float32)

    # Ocupacion por nivel completa (L+1 valores, indice 0 = raiz),
    # luego se descarta el indice 0 antes de concatenar al vector final.
    feats_nivel_completo = ocupacion_por_nivel_arbol(raiz, profundidad_max)
    feats_nivel = feats_nivel_completo[1:]   # excluye L=0 (raiz)

    feats_mom = momentos_geometricos(centros)
    feats_coherencia = estadisticas_coherencia_normales(coherencias)

    return np.concatenate([feats_nivel, feats_mom, feats_coherencia]).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# EXTRACTOR 2: directamente desde el .npz disperso persistido
# (uso normal en fase3_hce_entrenamiento.py -- NO reconstruye el arbol)
# ──────────────────────────────────────────────────────────────

def extraer_descriptores_hce_desde_npz(ruta_npz: str) -> np.ndarray:
    """
    Extraccion HCE directamente desde el archivo disperso guardado por
    preprocesar_octrees.py, SIN reconstruir el arbol completo. La
    ocupacion por nivel se calcula desde los centros de hoja (ver
    octree_real.py::ocupacion_por_nivel_desde_hojas, verificada
    numericamente equivalente a recorrer el arbol real).

    El vector retornado EXCLUYE ocupacion_L0 (raiz, siempre 100%).

    Este es el metodo usado en produccion por fase3_hce_entrenamiento.py.
    """
    d = cargar_octree_disperso(ruta_npz)
    centros = d["centros_hoja"]
    coherencias = d["coherencias_hoja"]
    profundidad_max = d["profundidad_max"]

    feats_nivel_completo = ocupacion_por_nivel_desde_hojas(centros, profundidad_max)
    feats_nivel = feats_nivel_completo[1:]   # excluye L=0 (raiz)

    feats_mom = momentos_geometricos(centros)
    feats_coherencia = estadisticas_coherencia_normales(coherencias)

    return np.concatenate([feats_nivel, feats_mom, feats_coherencia]).astype(np.float32)


def nombres_features(profundidad_max: int) -> list:
    """
    Nombres descriptivos de cada feature, mismo orden que producen
    extraer_descriptores_hce() / extraer_descriptores_hce_desde_npz().

    indice 0 -> L=1 ... indice profundidad_max-1 -> L=profundidad_max (hoja).
    L=0 (raiz) NO aparece: se excluye del vector del clasificador por
    ser constante (100% siempre). Ver ocupacion_L0_verificacion() para
    obtenerlo como dato de verificacion, no de entrada del modelo.
    """
    nombres = [f"ocupacion_L{d}" for d in range(1, profundidad_max + 1)]
    nombres += ["centroide_x", "centroide_y", "centroide_z",
                "varianza_x", "varianza_y", "varianza_z",
                "dispersion_radial",
                "skew_x", "skew_y", "skew_z"]
    nombres += ["normal_coherencia_media", "normal_coherencia_varianza"]
    return nombres


def ocupacion_L0_verificacion(raiz_o_centros, profundidad_max: int,
                              desde_arbol: bool = True) -> float:
    """
    Retorna la ocupacion del nivel raiz (L=0) como DATO DE
    VERIFICACION -- debe ser siempre 100.0 para cualquier objeto no
    vacio. NO se incluye en el vector de features del clasificador
    (ver nombres_features()); esta funcion existe solo para que los
    scripts de validacion puedan confirmar y registrar este valor
    constante en el JSON de auditoria.
    """
    if desde_arbol:
        return float(ocupacion_por_nivel_arbol(raiz_o_centros, profundidad_max)[0])
    else:
        return float(ocupacion_por_nivel_desde_hojas(raiz_o_centros, profundidad_max)[0])


# ──────────────────────────────────────────────────────────────
# TEST RAPIDO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from octree_real import construir_octree, guardar_octree_disperso

    print("=" * 60)
    print("  TEST: extraccion HCE -- arbol en memoria vs .npz disperso")
    print("  (ocupacion_L0 excluida, coherencia pre-normalizacion)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_pts = 5000
    theta = rng.uniform(0, np.pi, n_pts)
    phi = rng.uniform(0, 2 * np.pi, n_pts)
    radio = 0.7
    x = radio * np.sin(theta) * np.cos(phi)
    y = radio * np.sin(theta) * np.sin(phi)
    z = radio * np.cos(theta)
    puntos = np.stack([x, y, z], axis=1).astype(np.float32)
    normales = puntos / radio

    for R, L in [(32, 5), (64, 6)]:
        print(f"\n--- Resolucion {R}^3 (L={L}) ---")
        raiz = construir_octree(puntos, normales, profundidad_max=L)

        feats_memoria = extraer_descriptores_hce(raiz, L)
        nombres = nombres_features(L)

        ruta_tmp = str(Path(tempfile.gettempdir()) / f"test_hce_R{R}.npz")
        guardar_octree_disperso(raiz, ruta_tmp, etiqueta=0, profundidad_max=L)
        feats_disco = extraer_descriptores_hce_desde_npz(ruta_tmp)

        print(f"  Dimension del vector: {len(feats_memoria)}  (esperado {L + 12})")
        print(f"  Primer descriptor   : {nombres[0]}  (debe ser 'ocupacion_L1', NO L0)")
        print(f"  Features (memoria) == Features (disco): "
              f"{np.allclose(feats_memoria, feats_disco)}")
        assert np.allclose(feats_memoria, feats_disco), "Inconsistencia memoria vs disco"
        assert len(feats_memoria) == L + 12
        assert nombres[0] == "ocupacion_L1", "ocupacion_L0 no deberia estar en el vector"

        ocup_raiz = ocupacion_L0_verificacion(raiz, L, desde_arbol=True)
        print(f"  ocupacion_L0 (verificacion, fuera del vector): {ocup_raiz}%  (debe ser 100.0)")
        assert ocup_raiz == 100.0

    print("\n  Test completado: ambos extractores dan resultados identicos,")
    print("  ocupacion_L0 excluida correctamente, coherencia funcional.")
