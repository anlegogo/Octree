"""Contrato versionado para seleccionar las caracteristicas HCE del modelo.

El extractor conserva un vector base estable de 17 valores para R32 y 18 para
R64. La decision de retirar columnas constantes se toma exclusivamente con el
split oficial de entrenamiento completo y queda registrada en un contrato
JSON. El conjunto de prueba no interviene en esta decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hce_extraccion import HCE_RAW_SCHEMA_VERSION, nombres_features


HCE_CONTRACT_NAME = "hce-feature-contract"
HCE_CONTRACT_VERSION = "1.0.0"


def estadisticas_features(
    matriz: np.ndarray, nombres: list[str],
) -> list[dict]:
    """Calcula estadisticas e integridad de cada columna, sin usar test."""
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
                finitos.size > 0
                and unicos.size == 1
                and n_nan == 0
                and n_inf == 0
            ),
        })
    return salida


def diagnostico_ocupacion_l1(
    estadisticas: list[dict], n_modelos: int, auditoria_completa: bool,
) -> dict:
    """Emite la decision sobre L1 usando exclusivamente train."""
    l1 = next(
        item for item in estadisticas if item["nombre"] == "ocupacion_L1"
    )
    if not auditoria_completa:
        estado = "evidencia_insuficiente"
        recomendacion = (
            "Ejecutar la auditoria sobre todos los NPZ de train antes de "
            "conservar o excluir ocupacion_L1."
        )
    elif l1["es_constante_finita"]:
        estado = "constante_en_train_completo"
        recomendacion = "Excluir ocupacion_L1 del vector de los clasificadores."
    else:
        estado = "variable_en_train_completo"
        recomendacion = "Conservar ocupacion_L1 en el vector de los clasificadores."

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


def construir_contrato_hce(
    *,
    resolucion: int,
    profundidad_max: int,
    estadisticas: list[dict],
    n_modelos_train: int,
    conteo_train_esperado: int,
    auditoria_train_completa: bool,
    n_errores: int = 0,
) -> dict:
    """Construye el contrato de columnas que consumiran SVM y RF."""
    nombres_base = nombres_features(profundidad_max)
    nombres_estadisticas = [item["nombre"] for item in estadisticas]
    if nombres_estadisticas != nombres_base:
        raise ValueError("Las estadisticas no respetan el esquema HCE base")

    sin_no_finitos = all(
        item["n_nan"] == 0 and item["n_inf"] == 0
        for item in estadisticas
    )
    indices_excluidos = [
        indice for indice, item in enumerate(estadisticas)
        if item["es_constante_finita"]
    ]
    indices_incluidos = [
        indice for indice in range(len(nombres_base))
        if indice not in indices_excluidos
    ]
    completo = bool(
        auditoria_train_completa
        and n_modelos_train == conteo_train_esperado
        and n_errores == 0
    )
    autorizado = bool(completo and sin_no_finitos and indices_incluidos)

    return {
        "contract_name": HCE_CONTRACT_NAME,
        "contract_version": HCE_CONTRACT_VERSION,
        "hce_raw_schema_version": HCE_RAW_SCHEMA_VERSION,
        "resolucion": int(resolucion),
        "profundidad_max": int(profundidad_max),
        "fuente_seleccion": "ModelNet40/train",
        "n_modelos_train_auditados": int(n_modelos_train),
        "conteo_train_esperado": int(conteo_train_esperado),
        "auditoria_train_completa": completo,
        "sin_valores_no_finitos": sin_no_finitos,
        "dimension_base": len(nombres_base),
        "orden_base": nombres_base,
        "indices_incluidos": indices_incluidos,
        "orden_modelo": [nombres_base[i] for i in indices_incluidos],
        "dimension_modelo": len(indices_incluidos),
        "caracteristicas_excluidas": [
            {
                "indice_base": indice,
                "nombre": nombres_base[indice],
                "motivo": "constante_en_train_completo"
                if completo else "constante_en_auditoria_incompleta",
            }
            for indice in indices_excluidos
        ],
        "autorizado_para_entrenamiento": autorizado,
        "diagnostico_ocupacion_L1": diagnostico_ocupacion_l1(
            estadisticas, n_modelos_train, completo,
        ),
    }


def validar_contrato_hce(
    contrato: dict,
    *,
    resolucion: int | None = None,
    exigir_autorizado: bool = True,
) -> None:
    """Valida version, orden e indices de un contrato antes de usarlo."""
    if contrato.get("contract_name") != HCE_CONTRACT_NAME:
        raise ValueError("Contrato HCE no reconocido")
    if contrato.get("contract_version") != HCE_CONTRACT_VERSION:
        raise ValueError("Version de contrato HCE no compatible")
    if contrato.get("hce_raw_schema_version") != HCE_RAW_SCHEMA_VERSION:
        raise ValueError("El contrato no corresponde al esquema HCE vigente")

    resolucion_contrato = int(contrato["resolucion"])
    profundidad = int(contrato["profundidad_max"])
    if resolucion is not None and resolucion_contrato != resolucion:
        raise ValueError(
            f"Contrato R{resolucion_contrato} incompatible con R{resolucion}"
        )
    if resolucion_contrato != 2 ** profundidad:
        raise ValueError("Resolucion y profundidad del contrato son incompatibles")

    nombres_base = nombres_features(profundidad)
    if contrato.get("orden_base") != nombres_base:
        raise ValueError("El orden base del contrato no coincide con el extractor")
    if int(contrato.get("dimension_base", -1)) != len(nombres_base):
        raise ValueError("Dimension base invalida en el contrato")

    indices = [int(i) for i in contrato.get("indices_incluidos", [])]
    if indices != sorted(set(indices)):
        raise ValueError("Los indices incluidos deben ser unicos y ordenados")
    if any(i < 0 or i >= len(nombres_base) for i in indices):
        raise ValueError("El contrato contiene indices fuera de rango")
    nombres_modelo = [nombres_base[i] for i in indices]
    if contrato.get("orden_modelo") != nombres_modelo:
        raise ValueError("El orden del modelo no coincide con los indices incluidos")
    if int(contrato.get("dimension_modelo", -1)) != len(indices):
        raise ValueError("Dimension de modelo invalida en el contrato")
    if exigir_autorizado and not contrato.get("autorizado_para_entrenamiento"):
        raise RuntimeError(
            "El contrato HCE no autoriza entrenamiento: falta una auditoria "
            "completa y valida del split train."
        )


def aplicar_contrato_hce(
    features: np.ndarray,
    contrato: dict,
    *,
    exigir_autorizado: bool = True,
) -> np.ndarray:
    """Selecciona columnas del vector o matriz base segun el contrato."""
    validar_contrato_hce(
        contrato, exigir_autorizado=exigir_autorizado,
    )
    valores = np.asarray(features)
    if valores.ndim not in (1, 2):
        raise ValueError("Las features deben ser un vector o una matriz")
    if valores.shape[-1] != contrato["dimension_base"]:
        raise ValueError("La dimension de entrada no coincide con el contrato")
    seleccion = valores[..., contrato["indices_incluidos"]]
    if not np.isfinite(seleccion).all():
        raise ValueError("Las features seleccionadas contienen NaN o Inf")
    return seleccion.astype(np.float32, copy=False)


def cargar_contrato_hce(
    ruta: str | Path,
    *,
    resolucion: int | None = None,
    exigir_autorizado: bool = True,
) -> dict:
    """Carga y valida un contrato JSON."""
    with Path(ruta).open("r", encoding="utf-8") as archivo:
        contrato = json.load(archivo)
    validar_contrato_hce(
        contrato, resolucion=resolucion,
        exigir_autorizado=exigir_autorizado,
    )
    return contrato
