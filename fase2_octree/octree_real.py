"""
octree_real.py
================
Implementacion de un octree REAL: arbol con nodo raiz unico, subdivision
recursiva en 8 octantes, y poda explicita de ramas vacias (no se crea
ningun nodo para regiones sin geometria).

Esto reemplaza la aproximacion anterior (construir_grid_octree en
octree.py), que en realidad era una rejilla densa multicanal, NO un
octree: reservaba memoria para las R^3 celdas sin importar si estaban
ocupadas o no.

Convencion de profundidad (matematicamente estandar para octrees):
    - Raiz: profundidad d=0, UN SOLO NODO, cubre todo el espacio [-1,1]^3.
    - Cada subdivision reparte el nodo en 8 octantes (hijos).
    - A profundidad d, la resolucion equivalente (si todo estuviera
      subdividido) es R = 2^d.
    - Hoja de 32^3  -> d=5.
    - Hoja de 64^3  -> d=6.

IMPORTANTE: esta convencion (root=d=0 como UN nodo, R=2^d) es la
definicion matematica estandar de un octree y es la que se usa para
la LOGICA DE CONSTRUCCION del arbol. Es distinta de la convencion
"documental" L=0->2^3 acordada anteriormente para las FIGURAS de la
rejilla densa (que no era un arbol real). Con un arbol real construido
de verdad, se recomienda usar esta convencion matematica estandar
(root=1 nodo) en el documento, en lugar de la convencion anterior, ya
que ahora si existe una raiz unica que puede mostrarse como tal.
"""

import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# ESTRUCTURA DEL NODO
# ──────────────────────────────────────────────────────────────

class NodoOctree:
    """
    Nodo de un octree real.

    Atributos
    ---------
    centro       : np.ndarray (3,) -- centro del cubo que representa este nodo
    tamano       : float -- longitud de arista del cubo
    profundidad  : int -- 0 para la raiz, aumenta 1 por cada subdivision
    ocupado      : bool -- True si este nodo (o algun descendiente) contiene puntos
    es_hoja      : bool -- True si no tiene hijos (por profundidad maxima o por
                   no requerir mas subdivision)
    hijos        : list de 8 elementos, cada uno NodoOctree o None.
                   None significa rama PODADA (no se crea el nodo, no se
                   reserva memoria para esa region vacia).
    normal_promedio : np.ndarray (3,) o None -- solo definido en hojas ocupadas
    n_puntos     : int -- cantidad de puntos de la nube que caen en este nodo
                   (diagnostico / verificacion, no se persiste en disco)
    """

    __slots__ = ("centro", "tamano", "profundidad", "ocupado", "es_hoja",
                "hijos", "normal_promedio", "n_puntos")

    def __init__(self, centro: np.ndarray, tamano: float, profundidad: int):
        self.centro = centro
        self.tamano = tamano
        self.profundidad = profundidad
        self.ocupado = False
        self.es_hoja = True
        self.hijos = [None] * 8
        self.normal_promedio = None
        self.n_puntos = 0


# ──────────────────────────────────────────────────────────────
# CONSTRUCCION RECURSIVA CON PODA
# ──────────────────────────────────────────────────────────────

def _octante_de_puntos(puntos: np.ndarray, centro: np.ndarray) -> np.ndarray:
    """
    Determina a que octante (0-7) pertenece cada punto, segun su
    posicion relativa al centro del nodo actual en cada eje.
    Codificacion de bits: bit0=eje X, bit1=eje Y, bit2=eje Z.
    """
    bits_x = (puntos[:, 0] >= centro[0]).astype(np.int64)
    bits_y = (puntos[:, 1] >= centro[1]).astype(np.int64)
    bits_z = (puntos[:, 2] >= centro[2]).astype(np.int64)
    return bits_x + bits_y * 2 + bits_z * 4


def _centro_hijo(centro: np.ndarray, tamano: float, octante: int) -> np.ndarray:
    """Calcula el centro del octante hijo indicado (0-7)."""
    cuarto = tamano / 4.0
    dx = cuarto if (octante & 1) else -cuarto
    dy = cuarto if (octante & 2) else -cuarto
    dz = cuarto if (octante & 4) else -cuarto
    return centro + np.array([dx, dy, dz], dtype=np.float32)


def construir_octree(
    puntos: np.ndarray,
    normales: np.ndarray,
    profundidad_max: int,
    centro: np.ndarray = None,
    tamano: float = 2.0,
    profundidad: int = 0,
) -> NodoOctree:
    """
    Construye recursivamente un octree real a partir de una nube de
    puntos ya normalizada al cubo [-1,1]^3.

    PODA: si un octante no contiene ningun punto, su hijo correspondiente
    se deja como None -- NO se crea el nodo, NO se reserva memoria, y NO
    se continua la recursion en esa rama. Esta es la propiedad central
    que distingue a un octree real de una rejilla densa.

    Parametros
    ----------
    puntos          : (N, 3) nube de puntos normalizada
    normales        : (N, 3) normal de superficie de cada punto
    profundidad_max : profundidad de la hoja (5 para resolucion 32,
                      6 para resolucion 64; ver convencion en el docstring
                      del modulo)
    centro, tamano, profundidad : usados internamente por la recursion,
                      no deben pasarse en la llamada inicial

    Retorna
    -------
    NodoOctree raiz (profundidad=0)
    """
    if centro is None:
        centro = np.zeros(3, dtype=np.float32)

    nodo = NodoOctree(centro, tamano, profundidad)
    n = len(puntos)
    nodo.n_puntos = n

    if n == 0:
        nodo.ocupado = False
        nodo.es_hoja = True
        return nodo

    nodo.ocupado = True

    if profundidad >= profundidad_max:
        nodo.es_hoja = True
        normal_prom = normales.mean(axis=0)
        norma = np.linalg.norm(normal_prom)
        nodo.normal_promedio = (
            (normal_prom / norma).astype(np.float32)
            if norma > 1e-12 else np.zeros(3, dtype=np.float32)
        )
        return nodo

    nodo.es_hoja = False
    octantes = _octante_de_puntos(puntos, centro)

    for i in range(8):
        mask = octantes == i
        if not mask.any():
            nodo.hijos[i] = None   # PODA: no se crea nodo ni se recursa
            continue
        centro_hijo = _centro_hijo(centro, tamano, i)
        nodo.hijos[i] = construir_octree(
            puntos[mask], normales[mask], profundidad_max,
            centro=centro_hijo, tamano=tamano / 2.0, profundidad=profundidad + 1,
        )

    return nodo


# ──────────────────────────────────────────────────────────────
# RECORRIDO DEL ARBOL
# ──────────────────────────────────────────────────────────────

def recolectar_hojas(raiz: NodoOctree) -> list:
    """Retorna la lista de nodos hoja OCUPADOS del arbol (recorrido DFS)."""
    hojas = []

    def _rec(nodo):
        if nodo is None:
            return
        if nodo.es_hoja:
            if nodo.ocupado:
                hojas.append(nodo)
        else:
            for hijo in nodo.hijos:
                _rec(hijo)

    _rec(raiz)
    return hojas


def contar_nodos_por_profundidad(raiz: NodoOctree, profundidad_max: int) -> np.ndarray:
    """
    Cuenta cuantos nodos REALMENTE EXISTEN (no podados) en cada
    profundidad del arbol, de 0 (raiz) a profundidad_max (hojas).
    """
    conteo = np.zeros(profundidad_max + 1, dtype=np.int64)

    def _rec(nodo):
        if nodo is None:
            return
        conteo[nodo.profundidad] += 1
        if not nodo.es_hoja:
            for hijo in nodo.hijos:
                _rec(hijo)

    _rec(raiz)
    return conteo


def ocupacion_por_nivel_arbol(raiz: NodoOctree, profundidad_max: int) -> np.ndarray:
    """
    Calcula el % de ocupacion en cada nivel del arbol: nodos que
    REALMENTE EXISTEN (no podados) respecto al maximo posible en ese
    nivel (8^profundidad). Version basada en el arbol real -- reemplaza
    la funcion anterior que operaba via max-pool sobre una rejilla densa.

    Retorna un array de tamaño (profundidad_max + 1): indice 0 = raiz,
    indice profundidad_max = nivel hoja.
    """
    conteo = contar_nodos_por_profundidad(raiz, profundidad_max)
    porcentajes = np.zeros(profundidad_max + 1, dtype=np.float32)
    for d in range(profundidad_max + 1):
        total_posible = 8 ** d
        porcentajes[d] = 100.0 * conteo[d] / total_posible
    return porcentajes


def contar_nodos_totales(raiz: NodoOctree) -> int:
    """Cuenta el total de nodos reales existentes en el arbol."""
    total = [0]

    def _rec(nodo):
        if nodo is None:
            return
        total[0] += 1
        if not nodo.es_hoja:
            for hijo in nodo.hijos:
                _rec(hijo)

    _rec(raiz)
    return total[0]


# ──────────────────────────────────────────────────────────────
# MATERIALIZACION A GRID DENSO (solo para alimentar Net5-Octree,
# que por diseño de Conv3d/ConvTranspose3d requiere tensores densos)
# ──────────────────────────────────────────────────────────────

def octree_a_grid_denso(raiz: NodoOctree, resolucion: int) -> np.ndarray:
    """
    Convierte el arbol disperso a un grid denso (4, R, R, R), SOLO en
    el momento de alimentar la red neuronal. El almacenamiento en disco
    (ver guardar_octree_disperso) NUNCA usa este formato denso.
    """
    grid = np.zeros((4, resolucion, resolucion, resolucion), dtype=np.float32)
    hojas = recolectar_hojas(raiz)

    if len(hojas) == 0:
        return grid

    centros = np.array([h.centro for h in hojas], dtype=np.float32)
    normales = np.array([h.normal_promedio for h in hojas], dtype=np.float32)

    idx = np.clip(((centros + 1.0) * 0.5 * resolucion).astype(np.int64),
                 0, resolucion - 1)

    grid[0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    grid[1, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 0]
    grid[2, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 1]
    grid[3, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 2]

    return grid


# ──────────────────────────────────────────────────────────────
# SERIALIZACION DISPERSA (ahorro de memoria real)
# ──────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────
# SERIALIZACION CON ESTRUCTURA JERARQUICA COMPLETA
# ──────────────────────────────────────────────────────────────
#
# CORRECCION (observacion de Andres Gonzalez): la version anterior de
# esta serializacion guardaba UNICAMENTE los centros y normales de las
# hojas ocupadas. Aunque eso permite recalcular estadisticas de
# ocupacion por nivel (ver ocupacion_por_nivel_desde_hojas, verificada
# matematicamente equivalente), NO preserva la estructura del arbol:
# al cargar el archivo se recuperaba una simple lista de celdas, sin
# relaciones padre-hijo ni informacion de que nodos internos existen.
#
# La version corregida serializa TODOS los nodos existentes del arbol
# (internos y hojas, nunca los podados) mediante un recorrido DFS
# (pre-orden), guardando para cada nodo:
#   - profundidad (uint8)
#   - mascara de hijos (uint8, bits 0-7): bit i =1 si hijos[i] existe.
#     Un nodo hoja tiene mascara=0 (por construccion, ver
#     construir_octree: un nodo solo dejar de subdividirse si es hoja).
#   - normal promedio (3 x float32): solo tiene significado en hojas;
#     se guarda como ceros en nodos internos.
#
# El centro y tamaño de cada nodo NO se guardan explicitamente: se
# reconstruyen de forma deterministica durante la carga, replicando la
# misma logica de subdivision (_centro_hijo) que uso construir_octree(),
# siguiendo el orden exacto de descenso indicado por las mascaras de
# hijos. Esto reduce el tamaño del archivo sin perder informacion.
#
# Con este formato, cargar_octree_disperso() reconstruye el NodoOctree
# completo (con todas las relaciones padre-hijo reales) y luego deriva
# de el las hojas -- por lo que el resto del pipeline (net5_dataset.py,
# hce_extraccion.py) sigue funcionando sin cambios, ahora respaldado
# por una persistencia que si preserva la jerarquia real.

def _mascara_hijos(nodo: NodoOctree) -> int:
    """Codifica en un entero de 8 bits cuales de los 8 hijos existen."""
    mascara = 0
    if not nodo.es_hoja:
        for i in range(8):
            if nodo.hijos[i] is not None:
                mascara |= (1 << i)
    return mascara


def _serializar_dfs(raiz: NodoOctree) -> tuple:
    """
    Recorre el arbol en pre-orden (DFS) y produce 3 arrays paralelos,
    uno por cada nodo EXISTENTE (internos y hojas, nunca podados):
        profundidades : (M,) uint8
        mascaras      : (M,) uint8 -- 0 para hojas
        normales      : (M, 3) float32 -- solo valida si mascara==0
    """
    profundidades = []
    mascaras = []
    normales = []

    def _rec(nodo):
        if nodo is None:
            return
        profundidades.append(nodo.profundidad)
        m = _mascara_hijos(nodo)
        mascaras.append(m)
        if nodo.es_hoja:
            normales.append(nodo.normal_promedio
                           if nodo.normal_promedio is not None
                           else np.zeros(3, dtype=np.float32))
        else:
            normales.append(np.zeros(3, dtype=np.float32))
            for i in range(8):
                if nodo.hijos[i] is not None:
                    _rec(nodo.hijos[i])

    _rec(raiz)

    return (
        np.array(profundidades, dtype=np.uint8),
        np.array(mascaras, dtype=np.uint8),
        np.array(normales, dtype=np.float32),
    )


def _reconstruir_dfs(profundidades: np.ndarray, mascaras: np.ndarray,
                     normales: np.ndarray) -> NodoOctree:
    """
    Reconstruye el arbol NodoOctree completo (topologia identica al
    original: mismos nodos, mismas relaciones padre-hijo, mismas
    profundidades) a partir de los 3 arrays paralelos producidos por
    _serializar_dfs(). El centro y tamaño de cada nodo se derivan
    deterministicamente de la ruta de descenso, replicando la misma
    formula usada durante la construccion original (_centro_hijo).
    """
    puntero = [0]   # indice mutable de lectura sobre los arrays planos

    def _rec(centro, tamano, profundidad_esperada):
        i = puntero[0]
        profundidad = int(profundidades[i])
        assert profundidad == profundidad_esperada, (
            "Estructura serializada inconsistente: profundidad inesperada"
        )
        mascara = int(mascaras[i])
        normal = normales[i]
        puntero[0] += 1

        nodo = NodoOctree(centro, tamano, profundidad)
        nodo.ocupado = True

        if mascara == 0:
            # Nodo hoja (por construccion, un nodo interno siempre
            # tiene al menos un hijo -- ver construir_octree)
            nodo.es_hoja = True
            nodo.normal_promedio = normal
        else:
            nodo.es_hoja = False
            for bit in range(8):
                if mascara & (1 << bit):
                    centro_hijo = _centro_hijo(centro, tamano, bit)
                    nodo.hijos[bit] = _rec(
                        centro_hijo, tamano / 2.0, profundidad_esperada + 1,
                    )
                else:
                    nodo.hijos[bit] = None
        return nodo

    raiz = _rec(centro=np.zeros(3, dtype=np.float32), tamano=2.0, profundidad_esperada=0)
    assert puntero[0] == len(profundidades), (
        "Estructura serializada inconsistente: sobraron o faltaron nodos"
    )
    return raiz


def guardar_octree_disperso(raiz: NodoOctree, ruta_npz: str, etiqueta: int,
                            profundidad_max: int) -> None:
    """
    Guarda la ESTRUCTURA JERARQUICA COMPLETA del arbol (todos los nodos
    existentes, internos y hojas, con sus relaciones padre-hijo), no
    solo las hojas. El archivo sigue pesando proporcionalmente al
    numero de nodos reales del arbol (internos + hojas), tipicamente
    muy por debajo de una rejilla densa R^3 (ver comparar_memoria()).
    """
    profundidades, mascaras, normales = _serializar_dfs(raiz)

    np.savez_compressed(
        ruta_npz,
        profundidades=profundidades,
        mascaras=mascaras,
        normales=normales,
        etiqueta=etiqueta,
        profundidad_max=profundidad_max,
    )


def reconstruir_octree_desde_npz(ruta_npz: str) -> tuple:
    """
    Carga el archivo y reconstruye el arbol NodoOctree COMPLETO, con
    la topologia identica al arbol original (mismos nodos internos,
    mismas relaciones padre-hijo, mismas hojas con sus normales).

    Retorna (raiz: NodoOctree, etiqueta: int, profundidad_max: int).
    """
    data = np.load(ruta_npz)
    raiz = _reconstruir_dfs(
        data["profundidades"], data["mascaras"], data["normales"],
    )
    etiqueta = int(data["etiqueta"])
    profundidad_max = int(data["profundidad_max"])
    return raiz, etiqueta, profundidad_max


def cargar_octree_disperso(ruta_npz: str) -> dict:
    """
    Interfaz de compatibilidad con el resto del pipeline (net5_dataset.py,
    hce_extraccion.py): reconstruye el arbol completo desde el archivo
    (ver reconstruir_octree_desde_npz) y deriva de el las hojas ocupadas,
    devolviendo el mismo diccionario que la version anterior. La
    diferencia es que ahora el .npz en disco SI contiene la jerarquia
    completa; las hojas se derivan del arbol reconstruido, no se leen
    directamente de un array plano de hojas.
    """
    raiz, etiqueta, profundidad_max = reconstruir_octree_desde_npz(ruta_npz)
    hojas = recolectar_hojas(raiz)

    n = len(hojas)
    centros = np.zeros((n, 3), dtype=np.float32)
    normales = np.zeros((n, 3), dtype=np.float32)
    for i, h in enumerate(hojas):
        centros[i] = h.centro
        normales[i] = h.normal_promedio

    return {
        "centros_hoja": centros,
        "normales_hoja": normales,
        "etiqueta": etiqueta,
        "profundidad_max": profundidad_max,
    }


def cargar_y_materializar(ruta_npz: str, resolucion: int) -> tuple:
    """
    Carga un octree disperso desde disco (reconstruyendo su jerarquia
    completa) y lo materializa a un grid denso (4, R, R, R) SOLO en
    memoria RAM, en el momento de usarlo (p. ej. para alimentar
    Net5-Octree). El archivo en disco permanece disperso con su
    estructura jerarquica completa; nunca se guarda un grid denso.

    Retorna (grid_denso, etiqueta).
    """
    d = cargar_octree_disperso(ruta_npz)
    centros = d["centros_hoja"]
    normales = d["normales_hoja"]

    grid = np.zeros((4, resolucion, resolucion, resolucion), dtype=np.float32)
    if len(centros) > 0:
        idx = np.clip(((centros + 1.0) * 0.5 * resolucion).astype(np.int64),
                     0, resolucion - 1)
        grid[0, idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
        grid[1, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 0]
        grid[2, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 1]
        grid[3, idx[:, 0], idx[:, 1], idx[:, 2]] = normales[:, 2]

    return grid, d["etiqueta"]


def ocupacion_por_nivel_desde_hojas(centros_hoja: np.ndarray, profundidad_max: int) -> np.ndarray:
    """
    Calcula el % de ocupacion por nivel DIRECTAMENTE desde los centros
    de las hojas persistidas en disco (ver guardar_octree_disperso),
    sin necesidad de reconstruir el arbol completo.

    Equivalencia matematica (verificada numericamente): en este octree,
    un nodo a profundidad d existe (no fue podado) si y solo si al
    menos una hoja ocupada cae dentro de su region. Como todas las
    hojas ocupadas estan exactamente a profundidad_max, basta con
    contar cuantas celdas UNICAS de resolucion 2^d contienen al menos
    un centro de hoja -- esto da identicamente el mismo resultado que
    contar_nodos_por_profundidad() sobre el arbol reconstruido.

    Util quirurgicamente para extraer features HCE (hce_extraccion.py)
    directamente desde el .npz disperso, sin reconstruir NodoOctree.

    Parametros
    ----------
    centros_hoja    : (N, 3) centros de las hojas ocupadas
    profundidad_max : L de la hoja (5 para R=32, 6 para R=64)

    Retorna
    -------
    array de tamaño (profundidad_max + 1): indice 0 = raiz, indice
    profundidad_max = hoja.
    """
    porcentajes = np.zeros(profundidad_max + 1, dtype=np.float32)
    if len(centros_hoja) == 0:
        return porcentajes

    for d in range(profundidad_max + 1):
        R_d = 2 ** d
        idx = np.clip(((centros_hoja + 1.0) * 0.5 * R_d).astype(np.int64),
                     0, R_d - 1)
        celdas_unicas = np.unique(idx, axis=0)
        porcentajes[d] = 100.0 * len(celdas_unicas) / (8 ** d)

    return porcentajes


# ──────────────────────────────────────────────────────────────
# MEDICION REAL DE MEMORIA (objetos Python) vs REJILLA DENSA
# ──────────────────────────────────────────────────────────────
#
# CORRECCION (observacion de Andres Gonzalez): la version anterior de
# comparar_memoria() calculaba la memoria como "24 bytes por hoja"
# (3 floats de centro + 3 floats de normal), IGNORANDO por completo:
#   - los nodos INTERNOS del arbol (solo contaba hojas)
#   - la lista de 8 punteros a hijos de cada nodo
#   - la sobrecarga real de los objetos Python (incluso con __slots__,
#     cada instancia y cada array de numpy tiene overhead propio)
#
# La version corregida MIDE (no estima) la memoria real en RAM
# recorriendo el arbol completo con sys.getsizeof() sobre cada objeto
# NodoOctree, su lista de hijos, y sus arrays de numpy (centro y
# normal). Esto captura el costo real de TODOS los nodos existentes,
# internos y hojas, tal como estan efectivamente representados en
# memoria durante la ejecucion.

import sys


def medir_memoria_real_python(raiz: NodoOctree) -> dict:
    """
    Mide la memoria REAL en RAM ocupada por el arbol de objetos Python,
    recorriendo TODOS los nodos existentes (internos y hojas, nunca los
    podados) y sumando sys.getsizeof() de:
      - el objeto NodoOctree en si (usa __slots__, sin __dict__)
      - la lista de 8 elementos hijos (aunque varios sean None)
      - el array numpy del centro (3 floats)
      - el array numpy de la normal promedio (solo en hojas ocupadas)

    A diferencia de una formula teorica simplificada, esto es una
    MEDICION real del costo en memoria de cada objeto tal como Python
    los representa, incluyendo su overhead individual.

    Retorna
    -------
    dict con n_nodos_medidos, memoria_real_bytes, memoria_real_kib
    (KiB = kibibytes, 1024 bytes; ver nota de unidades en el modulo)
    """
    n_nodos = [0]
    total_bytes = [0]

    def _rec(nodo):
        if nodo is None:
            return
        n_nodos[0] += 1

        total_bytes[0] += sys.getsizeof(nodo)          # objeto NodoOctree
        total_bytes[0] += sys.getsizeof(nodo.hijos)     # lista de 8 punteros
        total_bytes[0] += sys.getsizeof(nodo.centro) + nodo.centro.nbytes

        if nodo.normal_promedio is not None:
            total_bytes[0] += (
                sys.getsizeof(nodo.normal_promedio) + nodo.normal_promedio.nbytes
            )

        if not nodo.es_hoja:
            for hijo in nodo.hijos:
                _rec(hijo)

    _rec(raiz)

    return {
        "n_nodos_medidos": n_nodos[0],
        "memoria_real_bytes": total_bytes[0],
        "memoria_real_kib": round(total_bytes[0] / 1024, 3),
    }


def comparar_memoria(raiz: NodoOctree, resolucion: int, profundidad_max: int) -> dict:
    """
    Compara tres magnitudes de memoria, claramente diferenciadas:

    1. memoria_ram_real_kib: memoria REAL medida (en KiB, 1024 bytes) de los objetos Python
       del arbol completo (internos + hojas), via sys.getsizeof()
       recorriendo cada nodo -- ver medir_memoria_real_python().

    2. memoria_binaria_estimada_kib: estimacion TEORICA (en KiB) (no medida) del
       tamaño minimo si se serializara el arbol en un formato binario
       compacto (1 byte profundidad + 1 byte mascara de hijos + 12
       bytes de normal por CADA nodo existente, internos y hojas). Esta
       cifra NO es memoria de objetos Python; es una cota inferior de
       referencia para comparar contra el archivo .npz real (que ademas
       incluye compresion y overhead del formato .npz).

    3. memoria_densa_kib: memoria de la rejilla densa equivalente (en KiB)
       (formato descartado tras la observacion de Andres Gonzalez):
       4 canales x R^3 celdas x 4 bytes/celda (float32).

    Retorna un dict con las tres magnitudes y los factores de ahorro
    correspondientes (RAM real vs. densa, y binario teorico vs. densa).
    """
    n_hojas = len(recolectar_hojas(raiz))
    n_nodos_totales = contar_nodos_totales(raiz)

    medicion_ram = medir_memoria_real_python(raiz)

    # Estimacion teorica de una serializacion binaria minima (NO es
    # memoria de objetos Python): 1 byte profundidad + 1 byte mascara
    # de hijos + 12 bytes de normal (3 floats), por CADA nodo existente
    # (internos y hojas), igual al formato usado por
    # guardar_octree_disperso() antes de la compresion .npz.
    bytes_por_nodo_binario = 1 + 1 + 12
    memoria_binaria_estimada_bytes = n_nodos_totales * bytes_por_nodo_binario

    # Memoria densa equivalente (formato anterior, descartado):
    # 4 canales x R^3 celdas x 4 bytes/celda (float32)
    memoria_densa_bytes = 4 * (resolucion ** 3) * 4

    return {
        "n_hojas_ocupadas": n_hojas,
        "n_nodos_totales_arbol": n_nodos_totales,
        "memoria_ram_real_kib": medicion_ram["memoria_real_kib"],
        "memoria_binaria_estimada_kib": round(memoria_binaria_estimada_bytes / 1024, 3),
        "memoria_densa_kib": round(memoria_densa_bytes / 1024, 3),
        "factor_ahorro_ram_vs_densa": round(
            memoria_densa_bytes / max(medicion_ram["memoria_real_bytes"], 1), 2
        ),
        "factor_ahorro_binario_vs_densa": round(
            memoria_densa_bytes / max(memoria_binaria_estimada_bytes, 1), 2
        ),
        "pct_ocupacion_hoja": round(100 * n_hojas / (resolucion ** 3), 4),
    }


# ──────────────────────────────────────────────────────────────
# TEST RAPIDO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys as sys_module
    from pathlib import Path as PathModule

    parser = argparse.ArgumentParser(
        description="Test de octree_real.py: poda de ramas vacias y "
                    "comparativa de memoria. Sin --off, usa una esfera "
                    "sintetica; con --off, usa la malla real indicada."
    )
    parser.add_argument("--off", type=str, default=None,
                        help="Ruta a un archivo .off real (opcional)")
    parser.add_argument("--salida", type=str, default=None,
                        help="Ruta JSON donde guardar los resultados (opcional)")
    args = parser.parse_args()

    print("=" * 60)
    print("  TEST: Octree real con poda de ramas vacias")
    print("=" * 60)

    nombre_objeto = "esfera_sintetica"

    if args.off:
        # Usar una malla real: reutiliza el pipeline de octree.py para
        # leer, normalizar y muestrear la superficie con normales.
        sys_module.path.insert(0, str(PathModule(__file__).parent))
        from octree import (
            leer_off, normalizar_malla, muestrear_superficie_con_normales,
        )

        print(f"\n  Archivo: {args.off}")
        verts, caras = leer_off(args.off)
        verts = normalizar_malla(verts)
        rng = np.random.default_rng(42)
        puntos, normales = muestrear_superficie_con_normales(verts, caras, 20000, rng)
        nombre_objeto = PathModule(args.off).stem
        print(f"  Vertices: {len(verts)}   Puntos muestreados: {len(puntos)}")
    else:
        print("\n  (Sin --off: usando esfera sintetica de prueba)")
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

    resultados_json = {"archivo": nombre_objeto}

    for R, prof_max in [(32, 5), (64, 6)]:
        print(f"\n--- Resolucion {R}^3 (profundidad hoja d={prof_max}) ---")

        raiz = construir_octree(puntos, normales, profundidad_max=prof_max)

        n_hojas = len(recolectar_hojas(raiz))
        n_nodos = contar_nodos_totales(raiz)
        ocup_por_nivel = ocupacion_por_nivel_arbol(raiz, prof_max)

        print(f"  Nodos totales en el arbol : {n_nodos:,}")
        print(f"  Hojas ocupadas            : {n_hojas:,}")
        print(f"  Ocupacion por nivel (%)   : {np.round(ocup_por_nivel, 2)}")

        mem = comparar_memoria(raiz, R, prof_max)
        print(f"  Memoria RAM real (arbol)  : {mem['memoria_ram_real_kib']:.2f} KiB")
        print(f"  Memoria densa (referencia): {mem['memoria_densa_kib']:.2f} KiB")
        print(f"  Factor de ahorro (RAM)    : {mem['factor_ahorro_ram_vs_densa']:.1f}x")

        grid = octree_a_grid_denso(raiz, R)
        n_ocup_grid = int(grid[0].sum())
        assert n_ocup_grid == n_hojas, "Inconsistencia hojas vs grid materializado"
        print(f"  Verificacion grid denso   : {n_ocup_grid} celdas (coincide con hojas) OK")

        resultados_json[f"R{R}"] = {
            "profundidad_max": prof_max,
            "n_nodos_totales": n_nodos,
            "n_hojas_ocupadas": n_hojas,
            "ocupacion_por_nivel_pct": [round(float(v), 4) for v in ocup_por_nivel],
            "memoria_ram_real_kib": mem["memoria_ram_real_kib"],
            "memoria_binaria_estimada_kib": mem["memoria_binaria_estimada_kib"],
            "memoria_densa_kib": mem["memoria_densa_kib"],
            "factor_ahorro_ram_vs_densa": mem["factor_ahorro_ram_vs_densa"],
            "factor_ahorro_binario_vs_densa": mem["factor_ahorro_binario_vs_densa"],
        }

    print("\n" + "=" * 60)
    print("  Test completado: la poda de ramas vacias funciona")
    print("  correctamente y el arbol es un octree real.")
    print("=" * 60)

    if args.off:
        ruta_salida = PathModule(args.salida) if args.salida else \
                     PathModule(f"octree_real_{nombre_objeto}.json")
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(resultados_json, f, indent=2, ensure_ascii=False)
        print(f"\n  Resultados guardados en: {ruta_salida.resolve()}")
