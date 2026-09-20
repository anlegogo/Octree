"""Audita HCE sobre el split train completo y genera su contrato de features.

Este paso se ejecuta despues de regenerar y verificar los octrees definitivos
de ModelNet40. Solo usa ``train``: el conjunto ``test`` permanece intacto hasta
la evaluacion final de los clasificadores.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np


RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))
sys.path.insert(0, str(RAIZ_PROYECTO / "fase3_hce"))

from contrato_hce import (  # noqa: E402
    construir_contrato_hce,
    estadisticas_features,
)
from hce_extraccion import (  # noqa: E402
    extraer_descriptores_hce_desde_datos,
    nombres_features,
)
from octree_real import cargar_octree_disperso  # noqa: E402
from preprocesar_octrees import (  # noqa: E402
    CLASES_MODELNET40,
    CLASE_A_INDICE,
    CONTEOS_OFICIALES,
)
from validacion_objetivo1 import guardar_json_atomico  # noqa: E402


PROFUNDIDAD_POR_RESOLUCION = {32: 5, 64: 6}


def recolectar_npz_train(
    data_root: Path,
    resolucion: int,
    categorias: list[str] | tuple[str, ...] = tuple(CLASES_MODELNET40),
) -> list[tuple[Path, str]]:
    """Enumera NPZ de train en orden canonico categoria/modelo."""
    raiz = data_root / f"octrees_{resolucion}"
    archivos = []
    for categoria in categorias:
        archivos.extend(
            (ruta, categoria)
            for ruta in sorted((raiz / categoria / "train").glob("*.npz"))
        )
    return archivos


def _validar_metadatos(
    datos: dict,
    ruta: Path,
    categoria_carpeta: str,
    resolucion: int,
    profundidad: int,
) -> list[str]:
    metadatos = datos["metadatos"]
    errores = []
    esperados = {
        "model_id": ruta.stem,
        "categoria": categoria_carpeta,
        "split": "train",
        "resolucion": resolucion,
        "profundidad_max": profundidad,
        "etiqueta": CLASE_A_INDICE[categoria_carpeta],
    }
    observados = {
        "model_id": metadatos["model_id"],
        "categoria": metadatos["categoria"],
        "split": metadatos["split"],
        "resolucion": metadatos["resolucion"],
        "profundidad_max": metadatos["profundidad_max"],
        "etiqueta": datos["etiqueta"],
    }
    for campo, esperado in esperados.items():
        if observados[campo] != esperado:
            errores.append(
                f"{campo}: {observados[campo]!r} != {esperado!r}"
            )
    return errores


def auditar_resolucion(
    data_root: Path,
    resolucion: int,
    *,
    limite: int | None = None,
    conteo_esperado: int = CONTEOS_OFICIALES["train"],
    categorias_esperadas: list[str] | tuple[str, ...] = tuple(CLASES_MODELNET40),
) -> dict:
    """Audita archivos, metadatos y columnas HCE de una resolucion."""
    if resolucion not in PROFUNDIDAD_POR_RESOLUCION:
        raise ValueError(f"Resolucion HCE no soportada: {resolucion}")
    if limite is not None and limite <= 0:
        raise ValueError("El limite debe ser positivo")

    data_root = data_root.resolve()
    profundidad = PROFUNDIDAD_POR_RESOLUCION[resolucion]
    nombres = nombres_features(profundidad)
    archivos_totales = recolectar_npz_train(
        data_root, resolucion, categorias_esperadas,
    )
    archivos = archivos_totales[:limite] if limite is not None else archivos_totales

    features_validas = []
    ids_vistos: set[tuple[str, str]] = set()
    errores: list[dict] = []
    conteos_archivos = Counter(categoria for _, categoria in archivos_totales)
    conteos_validos = Counter()

    for ruta, categoria in archivos:
        ruta_relativa = ruta.relative_to(data_root).as_posix()
        try:
            datos = cargar_octree_disperso(str(ruta))
            errores_archivo = _validar_metadatos(
                datos, ruta, categoria, resolucion, profundidad,
            )
            clave = (categoria, datos["metadatos"]["model_id"])
            if clave in ids_vistos:
                errores_archivo.append(f"model_id duplicado: {clave}")
            else:
                ids_vistos.add(clave)

            if errores_archivo:
                raise ValueError("; ".join(errores_archivo))

            features = extraer_descriptores_hce_desde_datos(datos)
            if features.shape != (len(nombres),):
                raise ValueError(
                    f"dimension {features.shape} != ({len(nombres)},)"
                )
            features_validas.append(features)
            conteos_validos[categoria] += 1
        except Exception as exc:  # conserva evidencia de todos los fallos
            errores.append({
                "archivo": ruta_relativa,
                "error": f"{type(exc).__name__}: {exc}",
            })

    if features_validas:
        matriz = np.asarray(features_validas, dtype=np.float64)
    else:
        matriz = np.empty((0, len(nombres)), dtype=np.float64)
    estadisticas = estadisticas_features(matriz, nombres)

    categorias_sin_archivos = [
        categoria for categoria in categorias_esperadas
        if conteos_archivos[categoria] == 0
    ]
    auditoria_completa = bool(
        limite is None
        and len(archivos_totales) == conteo_esperado
        and len(features_validas) == conteo_esperado
        and not errores
        and not categorias_sin_archivos
    )
    contrato = construir_contrato_hce(
        resolucion=resolucion,
        profundidad_max=profundidad,
        estadisticas=estadisticas,
        n_modelos_train=len(features_validas),
        conteo_train_esperado=conteo_esperado,
        auditoria_train_completa=auditoria_completa,
        n_errores=len(errores),
    )

    return {
        "esquema_auditoria": "hce-train-audit-v1",
        "resolucion": resolucion,
        "profundidad_max": profundidad,
        "split": "train",
        "data_root": data_root.name,
        "limite_tecnico": limite,
        "n_archivos_encontrados": len(archivos_totales),
        "n_archivos_auditados": len(archivos),
        "n_archivos_validos": len(features_validas),
        "conteo_train_esperado": conteo_esperado,
        "auditoria_train_completa": auditoria_completa,
        "conteos_por_categoria": {
            categoria: int(conteos_archivos[categoria])
            for categoria in categorias_esperadas
        },
        "conteos_validos_por_categoria": {
            categoria: int(conteos_validos[categoria])
            for categoria in categorias_esperadas
        },
        "categorias_sin_archivos": categorias_sin_archivos,
        "n_errores": len(errores),
        "errores": errores[:100],
        "errores_truncados": max(0, len(errores) - 100),
        "orden_base": nombres,
        "estadisticas_train": estadisticas,
        "contrato_features": contrato,
    }


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audita descriptores HCE sobre ModelNet40/train y genera el "
            "contrato de columnas para SVM y Random Forest"
        ),
    )
    parser.add_argument(
        "--data-root", type=Path, default=RAIZ_PROYECTO / "data",
        help="Raiz que contiene octrees_32/ y octrees_64/",
    )
    parser.add_argument(
        "--resultados-dir", type=Path,
        default=RAIZ_PROYECTO / "resultados" / "objetivo2",
    )
    parser.add_argument(
        "--resoluciones", nargs="+", type=int, choices=[32, 64],
        default=[32, 64],
    )
    parser.add_argument(
        "--limite", type=int, default=None,
        help="Limite tecnico para una prueba parcial; no autoriza entrenamiento",
    )
    parser.add_argument(
        "--permitir-incompleto", action="store_true",
        help="Permite finalizar una auditoria parcial sin autorizar entrenamiento",
    )
    return parser


def main() -> int:
    args = construir_parser().parse_args()
    if args.limite is not None and not args.permitir_incompleto:
        raise ValueError("--limite requiere --permitir-incompleto")
    if not args.data_root.resolve().is_dir():
        raise FileNotFoundError(
            f"No se encontro la raiz de octrees: {args.data_root.resolve()}"
        )

    resultados_dir = args.resultados_dir.resolve()
    resultados_dir.mkdir(parents=True, exist_ok=True)
    todas_autorizadas = True
    for resolucion in args.resoluciones:
        auditoria = auditar_resolucion(
            args.data_root, resolucion, limite=args.limite,
        )
        contrato = auditoria["contrato_features"]
        ruta_auditoria = resultados_dir / f"auditoria_hce_train_R{resolucion}.json"
        ruta_contrato = resultados_dir / f"contrato_hce_R{resolucion}.json"
        guardar_json_atomico(auditoria, ruta_auditoria)
        guardar_json_atomico(contrato, ruta_contrato)

        autorizado = bool(contrato["autorizado_para_entrenamiento"])
        todas_autorizadas = todas_autorizadas and autorizado
        print(
            f"R{resolucion}: {auditoria['n_archivos_validos']}/"
            f"{auditoria['conteo_train_esperado']} validos; "
            f"entrenamiento={'AUTORIZADO' if autorizado else 'BLOQUEADO'}"
        )
        print(f"  Auditoria: {ruta_auditoria}")
        print(f"  Contrato : {ruta_contrato}")

    if todas_autorizadas or args.permitir_incompleto:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
