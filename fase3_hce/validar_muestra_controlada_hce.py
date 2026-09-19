"""Valida de forma reproducible la extraccion HCE sobre una muestra controlada.

El script selecciona deterministamente un modelo de ``train`` y uno de
``test`` para cada categoria solicitada, construye los octrees de 32^3 y
64^3, verifica las dos rutas de extraccion (arbol y NPZ) y consolida las
estadisticas descriptivas por caracteristica.

La decision de conservar o excluir una caracteristica se apoya solo en
``train``. De forma opcional, ``--auditoria-train-npz-root`` recorre los NPZ
ya preprocesados del split de entrenamiento sin regenerar ModelNet40.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))
sys.path.insert(0, str(RAIZ_PROYECTO / "fase3_hce"))

from octree import (  # noqa: E402
    leer_off,
    muestrear_superficie_con_normales,
    normalizar_malla,
)
from preprocesar_octrees import (  # noqa: E402
    CLASES_MODELNET40,
    CONTEOS_OFICIALES,
    recolectar_modelos,
    seleccionar_muestra_controlada,
)
from validacion_objetivo1 import (  # noqa: E402
    guardar_json_atomico,
    semilla_estable_modelo,
)
from hce_extraccion import (  # noqa: E402
    extraer_descriptores_hce_desde_npz,
    nombres_features,
)
from validar_extraccion_hce import (  # noqa: E402
    PROFUNDIDAD_POR_RESOLUCION,
    validar_resolucion,
)


CATEGORIAS_CONTROL = ["airplane", "car", "chair", "sofa", "table"]


def ruta_portable(ruta: Path) -> str:
    """Evita registrar rutas absolutas dependientes del equipo."""
    ruta = ruta.resolve()
    try:
        return ruta.relative_to(RAIZ_PROYECTO.resolve()).as_posix()
    except ValueError:
        return ruta.name


def estadisticas_features(matriz: np.ndarray, nombres: list[str]) -> list[dict]:
    """Calcula estadisticas descriptivas y conteos de integridad por columna."""
    valores = np.asarray(matriz, dtype=np.float64)
    if valores.ndim != 2 or valores.shape[1] != len(nombres):
        raise ValueError(
            "La matriz debe ser bidimensional y coincidir con los nombres"
        )

    salida = []
    for indice, nombre in enumerate(nombres):
        columna = valores[:, indice]
        n_nan = int(np.isnan(columna).sum())
        n_inf = int(np.isinf(columna).sum())
        finitos = columna[np.isfinite(columna)]
        unicos = np.unique(finitos)

        if finitos.size:
            minimo = float(np.min(finitos))
            maximo = float(np.max(finitos))
            media = float(np.mean(finitos))
            mediana = float(np.median(finitos))
            desviacion = float(np.std(finitos, ddof=0))
        else:
            minimo = maximo = media = mediana = desviacion = None

        salida.append({
            "nombre": nombre,
            "min": minimo,
            "max": maximo,
            "media": media,
            "mediana": mediana,
            "desviacion_estandar": desviacion,
            "n_valores_unicos": int(unicos.size),
            "n_nan": n_nan,
            "n_inf": n_inf,
            "es_constante_finita": bool(
                finitos.size > 0 and unicos.size == 1 and n_nan == 0 and n_inf == 0
            ),
        })
    return salida


def resumir_resultados(resultados: list[dict], profundidad: int) -> dict:
    """Consolida verificaciones y estadisticas de una resolucion."""
    nombres = nombres_features(profundidad)
    matriz = np.asarray([
        [descriptor["valor"] for descriptor in resultado["descriptores"]]
        for resultado in resultados
    ], dtype=np.float64)

    indices_train = [
        indice for indice, resultado in enumerate(resultados)
        if resultado["split"] == "train"
    ]
    matriz_train = matriz[indices_train]
    estadisticas_todos = estadisticas_features(matriz, nombres)
    estadisticas_train = estadisticas_features(matriz_train, nombres)

    return {
        "n_modelos": int(len(resultados)),
        "n_modelos_train": int(len(indices_train)),
        "dimension_vector": int(matriz.shape[1]),
        "orden_descriptores": nombres,
        "todas_las_verificaciones_ok": all(
            resultado["verificaciones"]["TODAS_LAS_VERIFICACIONES_OK"]
            for resultado in resultados
        ),
        "estadisticas_muestra_completa": estadisticas_todos,
        "estadisticas_solo_train": estadisticas_train,
        "caracteristicas_constantes_solo_train": [
            item["nombre"] for item in estadisticas_train
            if item["es_constante_finita"]
        ],
    }


def diagnostico_ocupacion_l1(
    estadisticas: list[dict], n_modelos: int, auditoria_completa: bool,
) -> dict:
    """Emite una recomendacion sin utilizar observaciones de test."""
    l1 = next(item for item in estadisticas if item["nombre"] == "ocupacion_L1")
    if not auditoria_completa:
        estado = "evidencia_insuficiente"
        recomendacion = (
            "Ejecutar la auditoria sobre todos los NPZ de train antes de "
            "conservar o excluir ocupacion_L1."
        )
    elif l1["es_constante_finita"]:
        estado = "constante_en_train_completo"
        recomendacion = (
            "Excluir ocupacion_L1 del vector final y actualizar dimensiones, "
            "documentacion y pruebas antes del entrenamiento."
        )
    else:
        estado = "variable_en_train_completo"
        recomendacion = "Conservar ocupacion_L1 y documentar esta evidencia."

    return {
        "estado": estado,
        "n_modelos_train_evaluados": int(n_modelos),
        "estadisticas_ocupacion_L1": l1,
        "recomendacion": recomendacion,
        "nota_metodologica": (
            "La decision se toma exclusivamente con train; test no participa "
            "en la seleccion de caracteristicas."
        ),
    }


def auditar_npz_train(
    data_root: Path, resoluciones: list[int], limite: int | None = None,
) -> dict:
    """Audita descriptores sobre NPZ de train ya existentes."""
    auditoria = {
        "data_root": ruta_portable(data_root),
        "limite_tecnico": limite,
        "resoluciones": {},
    }
    for resolucion in resoluciones:
        profundidad = PROFUNDIDAD_POR_RESOLUCION[resolucion]
        rutas = sorted(
            (data_root / f"octrees_{resolucion}").glob("*/train/*.npz")
        )
        if limite is not None:
            rutas = rutas[:limite]
        if not rutas:
            raise FileNotFoundError(
                f"No se encontraron NPZ de train para R{resolucion} en {data_root}"
            )

        nombres = nombres_features(profundidad)
        matriz = np.asarray([
            extraer_descriptores_hce_desde_npz(str(ruta)) for ruta in rutas
        ], dtype=np.float64)
        if matriz.shape[1] != len(nombres):
            raise AssertionError("Dimension HCE inconsistente en la auditoria train")

        estadisticas = estadisticas_features(matriz, nombres)
        completa = limite is None and len(rutas) == CONTEOS_OFICIALES["train"]
        auditoria["resoluciones"][f"R{resolucion}"] = {
            "n_modelos_train": len(rutas),
            "auditoria_train_completa": completa,
            "estadisticas": estadisticas,
            "caracteristicas_constantes": [
                item["nombre"] for item in estadisticas
                if item["es_constante_finita"]
            ],
            "diagnostico_ocupacion_L1": diagnostico_ocupacion_l1(
                estadisticas, len(rutas), completa,
            ),
        }
    return auditoria


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Valida HCE sobre la muestra controlada de ModelNet40",
    )
    parser.add_argument(
        "--dataset-root", type=Path,
        default=RAIZ_PROYECTO / "Dataset" / "ModelNet40",
    )
    parser.add_argument(
        "--salida", type=Path,
        default=RAIZ_PROYECTO / "fase3_hce"
        / "validacion_hce_muestra_controlada.json",
    )
    parser.add_argument(
        "--categorias", nargs="+", choices=CLASES_MODELNET40,
        default=CATEGORIAS_CONTROL,
    )
    parser.add_argument("--muestra-por-categoria-split", type=int, default=1)
    parser.add_argument(
        "--resoluciones", nargs="+", type=int, choices=[32, 64],
        default=[32, 64],
    )
    parser.add_argument("--n-puntos", type=int, default=20000)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument(
        "--auditoria-train-npz-root", type=Path, default=None,
        help="Raiz que contiene octrees_32/ y octrees_64/ ya generados",
    )
    parser.add_argument(
        "--limite-auditoria-train", type=int, default=None,
        help="Limite tecnico; si se usa, la auditoria no se considera completa",
    )
    return parser


def main() -> int:
    args = construir_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"No se encontro ModelNet40 en {dataset_root}; use --dataset-root"
        )
    if args.n_puntos <= 0 or args.muestra_por_categoria_split <= 0:
        raise ValueError("--n-puntos y --muestra-por-categoria-split deben ser positivos")
    if args.limite_auditoria_train is not None and args.limite_auditoria_train <= 0:
        raise ValueError("--limite-auditoria-train debe ser positivo")

    modelos = recolectar_modelos(dataset_root, ["train", "test"])
    categorias = set(args.categorias)
    modelos = [modelo for modelo in modelos if modelo["categoria"] in categorias]
    modelos = seleccionar_muestra_controlada(
        modelos, args.muestra_por_categoria_split,
    )
    modelos.sort(key=lambda item: (
        item["categoria"], item["split"], item["model_id"],
    ))

    esperados = len(args.categorias) * 2 * args.muestra_por_categoria_split
    if len(modelos) != esperados:
        raise RuntimeError(
            f"Muestra incompleta: se esperaban {esperados} modelos y se hallaron "
            f"{len(modelos)}"
        )

    resultados = {f"R{resolucion}": [] for resolucion in args.resoluciones}
    for numero, modelo in enumerate(modelos, start=1):
        print(f"[{numero}/{len(modelos)}] {modelo['ruta_relativa']}")
        vertices, caras = leer_off(modelo["ruta"])
        vertices = normalizar_malla(vertices)
        semilla_modelo = semilla_estable_modelo(
            args.semilla, modelo["ruta_relativa"],
        )

        rng_a = np.random.default_rng(semilla_modelo)
        rng_b = np.random.default_rng(semilla_modelo)
        puntos, normales = muestrear_superficie_con_normales(
            vertices, caras, args.n_puntos, rng_a,
        )
        puntos_b, normales_b = muestrear_superficie_con_normales(
            vertices, caras, args.n_puntos, rng_b,
        )
        muestreo_reproducible = bool(
            np.array_equal(puntos, puntos_b)
            and np.array_equal(normales, normales_b)
        )
        if not muestreo_reproducible:
            raise AssertionError(
                f"Muestreo no reproducible para {modelo['ruta_relativa']}"
            )

        for resolucion in args.resoluciones:
            resultado = validar_resolucion(
                puntos, normales, resolucion, semilla_modelo,
                args.n_puntos, modelo["ruta_relativa"].removesuffix(".off"),
            )
            resultado["categoria"] = modelo["categoria"]
            resultado["split"] = modelo["split"]
            resultado["ruta_origen_relativa"] = modelo["ruta_relativa"]
            resultado["configuracion_reproduccion"]["semilla_base"] = args.semilla
            resultado["verificaciones"][
                "muestreo_reproducible_misma_semilla"
            ] = muestreo_reproducible
            resultado["verificaciones"]["TODAS_LAS_VERIFICACIONES_OK"] = bool(
                resultado["verificaciones"]["TODAS_LAS_VERIFICACIONES_OK"]
                and muestreo_reproducible
            )
            resultados[f"R{resolucion}"].append(resultado)

    resumen = {
        clave: resumir_resultados(
            detalle, PROFUNDIDAD_POR_RESOLUCION[int(clave[1:])],
        )
        for clave, detalle in resultados.items()
    }

    salida = {
        "esquema_validacion": "hce-muestra-controlada-v1",
        "dataset_root": ruta_portable(dataset_root),
        "categorias": args.categorias,
        "splits": ["train", "test"],
        "muestra_por_categoria_split": args.muestra_por_categoria_split,
        "semilla_base": args.semilla,
        "n_puntos": args.n_puntos,
        "resoluciones": args.resoluciones,
        "modelos": [modelo["ruta_relativa"] for modelo in modelos],
        "resultados_detallados": resultados,
        "resumen_por_resolucion": resumen,
        "nota_seleccion_features": (
            "Las estadisticas de test son solo descriptivas. La decision de "
            "conservar o excluir caracteristicas debe basarse exclusivamente "
            "en la auditoria del split train."
        ),
    }

    if args.auditoria_train_npz_root is None:
        salida["auditoria_train_npz"] = {
            "estado": "no_ejecutada",
            "instruccion": (
                "Ejecute de nuevo con --auditoria-train-npz-root para decidir "
                "sobre ocupacion_L1 usando todos los NPZ de train."
            ),
        }
    else:
        salida["auditoria_train_npz"] = auditar_npz_train(
            args.auditoria_train_npz_root.resolve(), args.resoluciones,
            args.limite_auditoria_train,
        )

    guardar_json_atomico(salida, args.salida.resolve())
    todo_ok = all(
        datos["todas_las_verificaciones_ok"] for datos in resumen.values()
    )
    print(f"Resultado: {'OK' if todo_ok else 'FALLO'}")
    print(f"JSON: {args.salida.resolve()}")
    return 0 if todo_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
