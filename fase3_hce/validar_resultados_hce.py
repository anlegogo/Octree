"""Valida la evidencia final de los entrenamientos HCE de R32 y R64.

El validador cruza cada resumen con el contrato de características aprobado y
comprueba la integridad de conteos, métricas, reportes y matrices de confusión.
No carga modelos ni vuelve a ejecutar entrenamiento.

Uso:
    python fase3_hce/validar_resultados_hce.py \
      --resultados-dir resultados/objetivo2 --resoluciones 32 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))
sys.path.insert(0, str(RAIZ_PROYECTO / "fase3_hce"))

from contrato_hce import validar_contrato_hce  # noqa: E402
from preprocesar_octrees import CLASES_MODELNET40  # noqa: E402


RESULTADOS_SCHEMA_NAME = "hce-training-results"
RESULTADOS_SCHEMA_VERSION = "1.0.0"
VALIDACION_SCHEMA_NAME = "hce-training-results-validation"
VALIDACION_SCHEMA_VERSION = "1.0.0"
CONTEO_TRAIN_ESPERADO = 9843
CONTEO_TEST_ESPERADO = 2468
SEED_ESPERADA = 42
VAL_SPLIT_ESPERADO = 0.10
PROFUNDIDAD_POR_RESOLUCION = {32: 5, 64: 6}


def _es_numero_finito(valor: Any) -> bool:
    return (
        isinstance(valor, (int, float))
        and not isinstance(valor, bool)
        and math.isfinite(float(valor))
    )


def _buscar_no_finitos(valor: Any, ruta: str = "resultado") -> list[str]:
    errores: list[str] = []
    if isinstance(valor, dict):
        for clave, elemento in valor.items():
            errores.extend(_buscar_no_finitos(elemento, f"{ruta}.{clave}"))
    elif isinstance(valor, list):
        for indice, elemento in enumerate(valor):
            errores.extend(_buscar_no_finitos(elemento, f"{ruta}[{indice}]"))
    elif isinstance(valor, float) and not math.isfinite(valor):
        errores.append(f"{ruta} contiene un valor no finito")
    return errores


def _agregar_igual(
    errores: list[str], observado: Any, esperado: Any, campo: str,
) -> None:
    if observado != esperado:
        errores.append(
            f"{campo}={observado!r}; se esperaba {esperado!r}"
        )


def _validar_metrica(
    errores: list[str], valor: Any, campo: str, minimo: float = 0.0,
    maximo: float | None = None,
) -> None:
    if not _es_numero_finito(valor):
        errores.append(f"{campo} debe ser un número finito")
        return
    if float(valor) < minimo:
        errores.append(f"{campo} debe ser >= {minimo}")
    if maximo is not None and float(valor) > maximo:
        errores.append(f"{campo} debe ser <= {maximo}")


def _validar_reporte_clasificacion(
    reporte: Any, n_test: int, prefijo: str,
) -> list[str]:
    errores: list[str] = []
    if not isinstance(reporte, dict):
        return [f"{prefijo}.reporte_clasificacion debe ser un objeto"]

    soportes = []
    for clase in CLASES_MODELNET40:
        metricas = reporte.get(clase)
        ruta = f"{prefijo}.reporte_clasificacion.{clase}"
        if not isinstance(metricas, dict):
            errores.append(f"{ruta} no está presente")
            continue
        for nombre in ("precision", "recall", "f1-score"):
            _validar_metrica(
                errores, metricas.get(nombre), f"{ruta}.{nombre}", 0.0, 1.0,
            )
        soporte = metricas.get("support")
        if (
            not _es_numero_finito(soporte)
            or float(soporte) < 0
            or not float(soporte).is_integer()
        ):
            errores.append(f"{ruta}.support debe ser un entero no negativo")
        else:
            soportes.append(int(soporte))

    if len(soportes) == len(CLASES_MODELNET40) and sum(soportes) != n_test:
        errores.append(
            f"{prefijo}.reporte_clasificacion suma {sum(soportes)} muestras; "
            f"se esperaban {n_test}"
        )

    _validar_metrica(
        errores, reporte.get("accuracy"),
        f"{prefijo}.reporte_clasificacion.accuracy", 0.0, 1.0,
    )
    for promedio in ("macro avg", "weighted avg"):
        metricas = reporte.get(promedio)
        ruta = f"{prefijo}.reporte_clasificacion.{promedio}"
        if not isinstance(metricas, dict):
            errores.append(f"{ruta} no está presente")
            continue
        for nombre in ("precision", "recall", "f1-score"):
            _validar_metrica(
                errores, metricas.get(nombre), f"{ruta}.{nombre}", 0.0, 1.0,
            )
        soporte = metricas.get("support")
        if (
            not _es_numero_finito(soporte)
            or not float(soporte).is_integer()
            or int(soporte) != n_test
        ):
            errores.append(f"{ruta}.support debe ser {n_test}")

    return errores


def _validar_matriz_confusion(
    matriz: Any, n_test: int, prefijo: str,
) -> tuple[list[str], int | None]:
    errores: list[str] = []
    n_clases = len(CLASES_MODELNET40)
    if not isinstance(matriz, list) or len(matriz) != n_clases:
        return [f"{prefijo}.matriz_confusion debe tener {n_clases} filas"], None

    total = 0
    diagonal = 0
    for fila_indice, fila in enumerate(matriz):
        if not isinstance(fila, list) or len(fila) != n_clases:
            errores.append(
                f"{prefijo}.matriz_confusion[{fila_indice}] debe tener "
                f"{n_clases} columnas"
            )
            continue
        suma_fila = 0
        for columna_indice, valor in enumerate(fila):
            if (
                not isinstance(valor, int)
                or isinstance(valor, bool)
                or valor < 0
            ):
                errores.append(
                    f"{prefijo}.matriz_confusion[{fila_indice}]"
                    f"[{columna_indice}] debe ser un entero no negativo"
                )
                continue
            suma_fila += valor
            if fila_indice == columna_indice:
                diagonal += valor
        if suma_fila == 0:
            errores.append(
                f"{prefijo}.matriz_confusion[{fila_indice}] no contiene "
                "muestras de su clase"
            )
        total += suma_fila

    if total != n_test:
        errores.append(
            f"{prefijo}.matriz_confusion suma {total}; se esperaban {n_test}"
        )
    return errores, diagonal if total == n_test else None


def _validar_modelo(
    resultado: dict, nombre: str, n_test: int, n_features: int,
    orden_features: list[str],
) -> list[str]:
    errores: list[str] = []
    datos = resultado.get(nombre)
    if not isinstance(datos, dict):
        return [f"{nombre} no está presente o no es un objeto"]

    _validar_metrica(errores, datos.get("val_acc"), f"{nombre}.val_acc", 0, 1)
    _validar_metrica(errores, datos.get("test_acc"), f"{nombre}.test_acc", 0, 1)
    _validar_metrica(
        errores, datos.get("tamano_modelo_mb"),
        f"{nombre}.tamano_modelo_mb", 0,
    )
    _validar_metrica(
        errores, datos.get("tiempo_inferencia_total_s"),
        f"{nombre}.tiempo_inferencia_total_s", 0,
    )
    _validar_metrica(
        errores, datos.get("tiempo_inferencia_promedio_ms"),
        f"{nombre}.tiempo_inferencia_promedio_ms", 0,
    )

    matriz = datos.get("matriz_confusion")
    errores_matriz, diagonal = _validar_matriz_confusion(matriz, n_test, nombre)
    errores.extend(errores_matriz)
    errores.extend(_validar_reporte_clasificacion(
        datos.get("reporte_clasificacion"), n_test, nombre,
    ))

    test_acc = datos.get("test_acc")
    reporte = datos.get("reporte_clasificacion")
    if diagonal is not None and _es_numero_finito(test_acc):
        accuracy_matriz = diagonal / n_test
        if not math.isclose(float(test_acc), accuracy_matriz, abs_tol=1e-12):
            errores.append(
                f"{nombre}.test_acc no coincide con la matriz de confusión"
            )
    if diagonal is not None and isinstance(reporte, dict):
        for indice, clase in enumerate(CLASES_MODELNET40):
            metricas = reporte.get(clase)
            if not isinstance(metricas, dict):
                continue
            soporte = metricas.get("support")
            if _es_numero_finito(soporte):
                soporte_matriz = sum(matriz[indice])
                if float(soporte) != soporte_matriz:
                    errores.append(
                        f"{nombre}.reporte_clasificacion.{clase}.support no "
                        "coincide con la matriz de confusión"
                    )
    if (
        isinstance(reporte, dict)
        and _es_numero_finito(reporte.get("accuracy"))
        and _es_numero_finito(test_acc)
        and not math.isclose(
            float(test_acc), float(reporte["accuracy"]), abs_tol=1e-12,
        )
    ):
        errores.append(
            f"{nombre}.test_acc no coincide con reporte_clasificacion.accuracy"
        )

    total_s = datos.get("tiempo_inferencia_total_s")
    promedio_ms = datos.get("tiempo_inferencia_promedio_ms")
    if _es_numero_finito(total_s) and _es_numero_finito(promedio_ms):
        esperado_ms = float(total_s) * 1000 / n_test
        if not math.isclose(float(promedio_ms), esperado_ms, rel_tol=1e-9):
            errores.append(
                f"{nombre}.tiempo_inferencia_promedio_ms no coincide con "
                "el tiempo total y n_test"
            )

    if nombre == "svm":
        _validar_metrica(
            errores, datos.get("tiempo_busqueda_hiperparam_s"),
            "svm.tiempo_busqueda_hiperparam_s", 0,
        )
        parametros = datos.get("mejores_hiperparametros")
        if not isinstance(parametros, dict):
            errores.append("svm.mejores_hiperparametros debe ser un objeto")
        else:
            valor_c = parametros.get("C")
            if valor_c not in (0.1, 1, 10, 100):
                errores.append("svm.C no pertenece a la grilla autorizada")
            gamma = parametros.get("gamma")
            if gamma not in ("scale", 0.001, 0.01, 0.1):
                errores.append("svm.gamma no pertenece a la grilla autorizada")

    if nombre == "random_forest":
        _validar_metrica(
            errores, datos.get("tiempo_entrenamiento_s"),
            "random_forest.tiempo_entrenamiento_s", 0,
        )
        n_estimators = datos.get("n_estimators")
        if n_estimators != 100:
            errores.append("random_forest.n_estimators debe ser 100")
        max_depth = datos.get("max_depth")
        if max_depth is not None:
            errores.append("random_forest.max_depth debe ser null")

        importancias = datos.get("feature_importances")
        if not isinstance(importancias, list) or len(importancias) != n_features:
            errores.append(
                "random_forest.feature_importances debe contener una entrada "
                "por característica"
            )
        else:
            entradas_validas = all(
                isinstance(item, dict)
                and isinstance(item.get("feature"), str)
                and _es_numero_finito(item.get("importancia"))
                for item in importancias
            )
            if not entradas_validas:
                errores.append(
                    "random_forest.feature_importances contiene entradas inválidas"
                )
                return errores

            nombres = [item["feature"] for item in importancias]
            valores = [item["importancia"] for item in importancias]
            if sorted(nombres) != sorted(orden_features):
                errores.append(
                    "random_forest.feature_importances no coincide con las "
                    "características del contrato"
                )
            if not all(float(v) >= 0 for v in valores):
                errores.append(
                    "random_forest.feature_importances contiene valores inválidos"
                )
            elif not math.isclose(sum(map(float, valores)), 1.0, abs_tol=1e-6):
                errores.append("random_forest.feature_importances no suma 1")
            elif any(
                float(valores[i]) < float(valores[i + 1])
                for i in range(len(valores) - 1)
            ):
                errores.append(
                    "random_forest.feature_importances debe estar ordenado "
                    "de mayor a menor"
                )

    return errores


def validar_resultado_hce(
    resultado: dict, contrato: dict, resolucion: int,
) -> list[str]:
    """Retorna todas las inconsistencias detectadas en un resumen HCE."""
    errores = _buscar_no_finitos(resultado)
    try:
        validar_contrato_hce(
            contrato, resolucion=resolucion, exigir_autorizado=True,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        errores.append(f"Contrato HCE inválido: {exc}")
        return errores

    _agregar_igual(
        errores, contrato.get("fuente_seleccion"), "ModelNet40/train",
        "contrato.fuente_seleccion",
    )
    _agregar_igual(
        errores, contrato.get("conteo_train_esperado"),
        CONTEO_TRAIN_ESPERADO, "contrato.conteo_train_esperado",
    )
    _agregar_igual(
        errores, contrato.get("n_modelos_train_auditados"),
        CONTEO_TRAIN_ESPERADO, "contrato.n_modelos_train_auditados",
    )
    _agregar_igual(
        errores, contrato.get("auditoria_train_completa"), True,
        "contrato.auditoria_train_completa",
    )
    _agregar_igual(
        errores, contrato.get("sin_valores_no_finitos"), True,
        "contrato.sin_valores_no_finitos",
    )

    _agregar_igual(
        errores, resultado.get("schema_name"), RESULTADOS_SCHEMA_NAME,
        "schema_name",
    )
    _agregar_igual(
        errores, resultado.get("schema_version"), RESULTADOS_SCHEMA_VERSION,
        "schema_version",
    )
    _agregar_igual(errores, resultado.get("resolucion"), resolucion, "resolucion")
    _agregar_igual(
        errores, resultado.get("profundidad_octree"),
        PROFUNDIDAD_POR_RESOLUCION[resolucion], "profundidad_octree",
    )
    _agregar_igual(
        errores, resultado.get("contract_version"),
        contrato["contract_version"], "contract_version",
    )
    ruta_contrato = resultado.get("contrato_features")
    nombre_contrato = (
        ruta_contrato.replace("\\", "/").rsplit("/", 1)[-1]
        if isinstance(ruta_contrato, str) else None
    )
    _agregar_igual(
        errores, nombre_contrato, f"contrato_hce_R{resolucion}.json",
        "contrato_features",
    )
    _agregar_igual(
        errores, resultado.get("features_excluidas"),
        contrato.get("caracteristicas_excluidas"), "features_excluidas",
    )
    orden_features = contrato["orden_modelo"]
    n_features = int(contrato["dimension_modelo"])
    _agregar_igual(
        errores, resultado.get("orden_features_modelo"),
        orden_features, "orden_features_modelo",
    )
    _agregar_igual(
        errores, resultado.get("n_features"), n_features, "n_features",
    )

    n_train_total = CONTEO_TRAIN_ESPERADO
    _agregar_igual(
        errores, resultado.get("n_train_total"),
        n_train_total, "n_train_total",
    )
    n_train = resultado.get("n_train")
    n_val = resultado.get("n_val")
    n_val_esperado = math.ceil(n_train_total * VAL_SPLIT_ESPERADO)
    n_train_esperado = n_train_total - n_val_esperado
    if not isinstance(n_train, int) or isinstance(n_train, bool) or n_train <= 0:
        errores.append("n_train debe ser un entero positivo")
    if not isinstance(n_val, int) or isinstance(n_val, bool) or n_val <= 0:
        errores.append("n_val debe ser un entero positivo")
    if isinstance(n_train, int) and not isinstance(n_train, bool):
        _agregar_igual(errores, n_train, n_train_esperado, "n_train")
    if isinstance(n_val, int) and not isinstance(n_val, bool):
        _agregar_igual(errores, n_val, n_val_esperado, "n_val")
    if (
        isinstance(n_train, int) and not isinstance(n_train, bool)
        and isinstance(n_val, int) and not isinstance(n_val, bool)
        and n_train + n_val != n_train_total
    ):
        errores.append(
            f"n_train + n_val={n_train + n_val}; se esperaban {n_train_total}"
        )

    _agregar_igual(
        errores, resultado.get("n_test"), CONTEO_TEST_ESPERADO, "n_test",
    )
    _agregar_igual(errores, resultado.get("seed"), SEED_ESPERADA, "seed")

    for nombre in ("svm", "random_forest"):
        errores.extend(_validar_modelo(
            resultado, nombre, CONTEO_TEST_ESPERADO,
            n_features, orden_features,
        ))
    return errores


def cargar_json_estricto(ruta: Path) -> dict:
    """Carga JSON y rechaza constantes no estándar como NaN e Infinity."""
    def rechazar_constante(valor: str) -> None:
        raise ValueError(f"Constante JSON no válida: {valor}")

    with ruta.open("r", encoding="utf-8") as archivo:
        datos = json.load(archivo, parse_constant=rechazar_constante)
    if not isinstance(datos, dict):
        raise ValueError("La raíz del JSON debe ser un objeto")
    return datos


def validar_archivos(
    resultados_dir: Path, resoluciones: list[int],
) -> dict:
    """Valida los archivos requeridos y construye un informe consolidado."""
    validaciones = []
    for resolucion in resoluciones:
        ruta_resultado = resultados_dir / f"resumen_hce_R{resolucion}.json"
        ruta_contrato = resultados_dir / f"contrato_hce_R{resolucion}.json"
        errores: list[str] = []
        try:
            contrato = cargar_json_estricto(ruta_contrato)
        except (OSError, ValueError) as exc:
            contrato = None
            errores.append(f"No se pudo cargar {ruta_contrato.name}: {exc}")
        try:
            resultado = cargar_json_estricto(ruta_resultado)
        except (OSError, ValueError) as exc:
            resultado = None
            errores.append(f"No se pudo cargar {ruta_resultado.name}: {exc}")

        if contrato is not None and resultado is not None:
            errores.extend(validar_resultado_hce(
                resultado, contrato, resolucion,
            ))
        validaciones.append({
            "resolucion": resolucion,
            "archivo_resultado": ruta_resultado.name,
            "archivo_contrato": ruta_contrato.name,
            "valido": not errores,
            "errores": errores,
        })

    return {
        "schema_name": VALIDACION_SCHEMA_NAME,
        "schema_version": VALIDACION_SCHEMA_VERSION,
        "resoluciones_solicitadas": resoluciones,
        "validacion_completa": all(item["valido"] for item in validaciones),
        "validaciones": validaciones,
    }


def guardar_json_atomico(ruta: Path, contenido: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(
        prefix=f".{ruta.name}.", suffix=".tmp", dir=ruta.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as archivo:
            json.dump(contenido, archivo, indent=2, ensure_ascii=False)
            archivo.write("\n")
        os.replace(temporal, ruta)
    except BaseException:
        try:
            os.unlink(temporal)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida los resultados finales de HCE para SVM y RF",
    )
    parser.add_argument(
        "--resultados-dir", type=Path,
        default=RAIZ_PROYECTO / "resultados" / "objetivo2",
    )
    parser.add_argument(
        "--resoluciones", type=int, nargs="+", default=[32, 64],
        choices=sorted(PROFUNDIDAD_POR_RESOLUCION),
    )
    parser.add_argument(
        "--salida", type=Path, default=None,
        help=(
            "Informe consolidado. Por defecto: "
            "<resultados-dir>/validacion_entrenamientos_hce.json"
        ),
    )
    args = parser.parse_args()

    resoluciones = list(dict.fromkeys(args.resoluciones))
    informe = validar_archivos(args.resultados_dir, resoluciones)
    ruta_salida = args.salida or (
        args.resultados_dir / "validacion_entrenamientos_hce.json"
    )
    guardar_json_atomico(ruta_salida, informe)

    for validacion in informe["validaciones"]:
        estado = "VÁLIDO" if validacion["valido"] else "INVÁLIDO"
        print(f"R{validacion['resolucion']}: {estado}")
        for error in validacion["errores"]:
            print(f"  - {error}")
    print(f"Informe: {ruta_salida}")
    return 0 if informe["validacion_completa"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
