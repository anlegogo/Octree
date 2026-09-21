"""
fase3_hce_entrenamiento.py
============================
Pipeline completo del enfoque clasico HCE:
  1. Cargar los octrees dispersos precomputados (.npz de la Fase 2)
  2. Extraer descriptores hand-crafted (ocupacion por nivel + momentos)
  3. Aplicar el contrato de caracteristicas generado solo con train
  4. Entrenar SVM (kernel RBF) y Random Forest
  5. Evaluar sobre el conjunto de test oficial y guardar resultados

Restriccion metodologica: NO se usa aumento de datos (criterio de
equivalencia experimental, seccion 6.6).

Uso:
    python fase3_hce/fase3_hce_entrenamiento.py --resolucion 32 --data-root data
    python fase3_hce/fase3_hce_entrenamiento.py --resolucion 64 --data-root data
"""

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - solo afecta la barra visual
    def tqdm(iterable, **_kwargs):
        return iterable

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
import joblib

# ── Configuracion ──────────────────────────────────────────────
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ_PROYECTO / "fase2_octree"))
sys.path.insert(0, str(RAIZ_PROYECTO / "fase3_hce"))

from contrato_hce import aplicar_contrato_hce, cargar_contrato_hce  # noqa: E402
from hce_extraccion import extraer_descriptores_hce_desde_datos  # noqa: E402
from octree_real import cargar_octree_disperso  # noqa: E402
from preprocesar_octrees import (  # noqa: E402
    CLASES_MODELNET40,
    CLASE_A_INDICE,
)

SEED = 42
RESULTADOS_SCHEMA_NAME = "hce-training-results"
RESULTADOS_SCHEMA_VERSION = "1.0.0"
CONTEO_TEST_ESPERADO = 2468

PROFUNDIDAD_POR_RESOLUCION = {32: 5, 64: 6}

CLASES = CLASES_MODELNET40


def ruta_portable(ruta: Path) -> str:
    """Evita registrar rutas absolutas dependientes del equipo."""
    ruta = ruta.resolve()
    try:
        return ruta.relative_to(RAIZ_PROYECTO.resolve()).as_posix()
    except ValueError:
        return ruta.name


# ──────────────────────────────────────────────────────────────
# 1. CARGA Y EXTRACCION DE FEATURES
# ──────────────────────────────────────────────────────────────

def recolectar_npz(raiz_resolucion: Path, split: str) -> list:
    """Lista todos los .npz de un split, retorna (ruta, clase)."""
    archivos = []
    for clase in CLASES:
        carpeta = raiz_resolucion / clase / split
        if not carpeta.exists():
            continue
        for archivo in sorted(carpeta.glob("*.npz")):
            archivos.append((archivo, clase))
    return archivos


def extraer_features_split(
    raiz_resolucion: Path,
    split: str,
    profundidad: int,
    contrato: dict,
) -> tuple:
    """
    Carga todos los .npz de un split y extrae sus descriptores HCE.

    Retorna
    -------
    X : array (N, n_features)
    y : array (N,) etiquetas enteras
    """
    archivos = recolectar_npz(raiz_resolucion, split)
    print(f"  [{split}] Archivos encontrados: {len(archivos)}")
    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron NPZ para '{split}' en {raiz_resolucion}"
        )

    X = []
    y = []

    for ruta, clase in tqdm(archivos, desc=f"  Extrayendo {split}", ncols=80):
        # La carga valida formato, version, topologia y metadatos. El vector
        # base se calcula una sola vez y luego se aplica el contrato generado
        # exclusivamente con el split train completo.
        datos = cargar_octree_disperso(str(ruta))
        metadatos = datos["metadatos"]
        esperados = {
            "categoria": clase,
            "split": split,
            "resolucion": 2 ** profundidad,
            "profundidad_max": profundidad,
            "etiqueta": CLASE_A_INDICE[clase],
        }
        observados = {
            "categoria": metadatos["categoria"],
            "split": metadatos["split"],
            "resolucion": metadatos["resolucion"],
            "profundidad_max": metadatos["profundidad_max"],
            "etiqueta": datos["etiqueta"],
        }
        discrepancias = [
            f"{campo}={observados[campo]!r}, esperado={esperado!r}"
            for campo, esperado in esperados.items()
            if observados[campo] != esperado
        ]
        if discrepancias:
            raise ValueError(
                f"Metadatos incompatibles en {ruta}: " + "; ".join(discrepancias)
            )

        feats_base = extraer_descriptores_hce_desde_datos(datos)
        feats = aplicar_contrato_hce(feats_base, contrato)
        dimension_esperada = int(contrato["dimension_modelo"])
        if len(feats) != dimension_esperada:
            raise ValueError(
                f"Dimension HCE inesperada en {ruta}: {len(feats)} != "
                f"{dimension_esperada}"
            )

        X.append(feats)
        y.append(datos["etiqueta"])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ──────────────────────────────────────────────────────────────
# 2. PARTICION TRAIN / VAL (10% reservado, igual que en Fase 1)
# ──────────────────────────────────────────────────────────────

def particionar_train_val(X: np.ndarray, y: np.ndarray, val_split: float = 0.10,
                          seed: int = SEED) -> tuple:
    """Reserva validacion de forma reproducible y estratificada por clase."""
    if not 0.0 < val_split < 1.0:
        raise ValueError("val_split debe estar entre 0 y 1")
    indices = np.arange(len(X))
    idx_train, idx_val = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )

    return (X[idx_train], y[idx_train], X[idx_val], y[idx_val])


# ──────────────────────────────────────────────────────────────
# 3. ENTRENAMIENTO: SVM (RBF) con busqueda de hiperparametros
# ──────────────────────────────────────────────────────────────

def entrenar_svm(X_train, y_train, X_val, y_val, seed=SEED):
    """
    Entrena SVM con kernel RBF, ajustando C y gamma via grid search.

    CORRECCION (observacion de rigor metodologico): el escalado
    (StandardScaler) se incluye DENTRO de un Pipeline pasado a
    GridSearchCV, en vez de ajustarse una sola vez sobre todo X_train
    antes de la busqueda. De esa forma, cada fold de la validacion
    cruzada (cv=3) ajusta su propio scaler solo con los datos de
    entrenamiento de ESE fold, sin que la porcion de validacion interna
    del fold influya en las estadisticas de escalado (fuga de datos
    leve pero real, evitada asi).
    """
    print("\n[SVM] Iniciando busqueda de hiperparametros (C, gamma)...")

    param_grid = {
        "svm__C":     [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01, 0.1],
    }

    t0 = time.time()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", random_state=seed, cache_size=1000)),
    ])

    # GridSearchCV con cv=3 sobre train; el Pipeline reajusta el scaler
    # en cada fold de forma independiente. X_train aqui debe ser el
    # array SIN escalar (el Pipeline se encarga del escalado interno).
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    mejor_pipeline = grid.best_estimator_
    t1 = time.time()

    # X_val tampoco debe venir pre-escalado: el Pipeline aplica el
    # scaler ajustado (con TODO X_train, ya con los mejores hiperparametros)
    # automaticamente al predecir.
    val_acc = accuracy_score(y_val, mejor_pipeline.predict(X_val))

    # best_params_ usa el prefijo "svm__" por el Pipeline; se limpia
    # para reportarlo igual que antes (C, gamma sueltos)
    mejores_params_limpios = {
        k.replace("svm__", ""): v for k, v in grid.best_params_.items()
    }

    print(f"\n[SVM] Mejores hiperparametros: {mejores_params_limpios}")
    print(f"[SVM] Val accuracy           : {val_acc*100:.2f}%")
    print(f"[SVM] Tiempo de busqueda      : {(t1-t0)/60:.1f} min")

    return mejor_pipeline, mejores_params_limpios, val_acc, (t1 - t0)


# ──────────────────────────────────────────────────────────────
# 4. ENTRENAMIENTO: Random Forest
# ──────────────────────────────────────────────────────────────

def entrenar_random_forest(X_train, y_train, X_val, y_val,
                           n_estimators=100, max_depth=None, seed=SEED):
    """Entrena Random Forest con los hiperparametros base de la metodologia."""
    print(f"\n[RandomForest] Entrenando con n_estimators={n_estimators}, "
          f"max_depth={max_depth}...")

    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    t1 = time.time()

    val_acc = accuracy_score(y_val, rf.predict(X_val))

    print(f"[RandomForest] Val accuracy : {val_acc*100:.2f}%")
    print(f"[RandomForest] Tiempo        : {t1-t0:.1f}s")

    return rf, val_acc, (t1 - t0)


# ──────────────────────────────────────────────────────────────
# 5. EVALUACION FINAL EN TEST
# ──────────────────────────────────────────────────────────────

def evaluar_modelo(modelo, X_test, y_test, nombre_modelo: str) -> dict:
    """Evalua un modelo entrenado sobre el conjunto de test oficial."""
    t0 = time.time()
    y_pred = modelo.predict(X_test)
    t1 = time.time()

    tiempo_inferencia_total = t1 - t0
    tiempo_inferencia_promedio = tiempo_inferencia_total / len(X_test)

    test_acc = accuracy_score(y_test, y_pred)
    etiquetas = np.arange(len(CLASES))
    reporte = classification_report(
        y_test, y_pred, labels=etiquetas, target_names=CLASES,
        output_dict=True, zero_division=0,
    )
    matriz_confusion = confusion_matrix(y_test, y_pred, labels=etiquetas)

    print(f"\n[{nombre_modelo}] Test accuracy            : {test_acc*100:.2f}%")
    print(f"[{nombre_modelo}] Tiempo inferencia (total)  : {tiempo_inferencia_total:.3f}s")
    print(f"[{nombre_modelo}] Tiempo inferencia (1 muestra): "
          f"{tiempo_inferencia_promedio*1000:.3f} ms")

    return {
        "test_acc": float(test_acc),
        "tiempo_inferencia_total_s": float(tiempo_inferencia_total),
        "tiempo_inferencia_promedio_ms": float(tiempo_inferencia_promedio * 1000),
        "reporte_clasificacion": reporte,
        "matriz_confusion": matriz_confusion.tolist(),
    }


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento HCE (SVM + Random Forest)")
    parser.add_argument("--resolucion", type=int, default=32, choices=[32, 64])
    parser.add_argument(
        "--data-root", type=Path, default=RAIZ_PROYECTO / "data",
        help="Raiz que contiene octrees_32/ y octrees_64/",
    )
    parser.add_argument(
        "--logs-dir", type=Path, default=RAIZ_PROYECTO / "logs",
    )
    parser.add_argument(
        "--checkpoints-dir", type=Path,
        default=RAIZ_PROYECTO / "checkpoints",
    )
    parser.add_argument(
        "--resultados-dir", type=Path,
        default=RAIZ_PROYECTO / "resultados",
    )
    parser.add_argument(
        "--contrato-features", type=Path, default=None,
        help=(
            "Contrato generado por auditar_train_hce.py. Por defecto usa "
            "resultados/objetivo2/contrato_hce_R<resolucion>.json"
        ),
    )
    args = parser.parse_args()

    R = args.resolucion
    L = PROFUNDIDAD_POR_RESOLUCION[R]
    ruta_contrato = (
        args.contrato_features.resolve()
        if args.contrato_features is not None
        else RAIZ_PROYECTO / "resultados" / "objetivo2"
        / f"contrato_hce_R{R}.json"
    )
    contrato = cargar_contrato_hce(
        ruta_contrato, resolucion=R, exigir_autorizado=True,
    )
    nombres = contrato["orden_modelo"]

    print("=" * 60)
    print(f"  FASE 3: ENFOQUE CLASICO (HCE) — Resolucion {R}^3")
    print("=" * 60)

    raiz_resolucion = args.data_root.resolve() / f"octrees_{R}"
    dir_logs = args.logs_dir.resolve()
    dir_ckpt = args.checkpoints_dir.resolve()
    dir_resultados = args.resultados_dir.resolve()
    if not raiz_resolucion.is_dir():
        raise FileNotFoundError(
            f"No se encontro {raiz_resolucion}; indique la raiz con --data-root"
        )
    dir_logs.mkdir(parents=True, exist_ok=True)
    dir_ckpt.mkdir(parents=True, exist_ok=True)
    dir_resultados.mkdir(parents=True, exist_ok=True)

    # 1. Extraer y auditar train antes de tocar el conjunto oficial de test.
    print("\n[1/4] Extrayendo y auditando descriptores HCE de train...")
    X_train_full, y_train_full = extraer_features_split(
        raiz_resolucion, "train", L, contrato,
    )

    conteo_train_esperado = int(contrato["conteo_train_esperado"])
    if len(X_train_full) != conteo_train_esperado:
        raise RuntimeError(
            f"Train incompleto: {len(X_train_full)} muestras; "
            f"se esperaban {conteo_train_esperado}."
        )
    if not np.isfinite(X_train_full).all():
        raise RuntimeError("Train contiene características NaN o Inf")

    # Salvaguarda metodologica: no iniciar clasificadores mientras el vector
    # contenga caracteristicas constantes en train (por ejemplo, ocupacion_L1).
    constantes = [
        nombre for indice, nombre in enumerate(nombres)
        if np.unique(X_train_full[:, indice]).size == 1
    ]
    if constantes:
        raise RuntimeError(
            "Se detectaron caracteristicas constantes en train: "
            f"{constantes}. Resuelva y documente su exclusion antes de entrenar."
        )

    # Test solo se carga cuando el vector supero la auditoria de train.
    X_test, y_test = extraer_features_split(
        raiz_resolucion, "test", L, contrato,
    )
    if len(X_test) != CONTEO_TEST_ESPERADO:
        raise RuntimeError(
            f"Test incompleto: {len(X_test)} muestras; "
            f"se esperaban {CONTEO_TEST_ESPERADO}."
        )
    if not np.isfinite(X_test).all():
        raise RuntimeError("Test contiene características NaN o Inf")
    print(f"\n  X_train_full : {X_train_full.shape}")
    print(f"  X_test       : {X_test.shape}")

    # Guardar features extraidas (cache para no recalcular)
    np.savez_compressed(
        dir_logs / f"hce_features_R{R}.npz",
        X_train=X_train_full, y_train=y_train_full,
        X_test=X_test, y_test=y_test,
        nombres_features=np.asarray(nombres),
        contract_version=np.asarray(contrato["contract_version"]),
    )

    # 2. Particionar train/val (10%, seed=42, igual que Fase 1)
    print("\n[2/4] Particionando train/val estratificado (10%, seed=42)...")
    X_train, y_train, X_val, y_val = particionar_train_val(X_train_full, y_train_full)
    print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]}")

    # 3. Entrenar ambos clasificadores
    print("\n[3/4] Entrenando clasificadores...")

    # X_train SIN escalar: el Pipeline interno de entrenar_svm() aplica
    # el StandardScaler correctamente dentro de cada fold de GridSearchCV.
    svm_modelo, svm_params, svm_val_acc, svm_tiempo = entrenar_svm(
        X_train, y_train, X_val, y_val,
    )
    rf_modelo, rf_val_acc, rf_tiempo = entrenar_random_forest(
        X_train, y_train, X_val, y_val,   # RF no necesita escalado
        n_estimators=100, max_depth=None,
    )

    # 4. Evaluacion final en test
    print("\n[4/4] Evaluando en conjunto de test oficial...")
    resultados_svm = evaluar_modelo(svm_modelo, X_test, y_test, "SVM")
    resultados_rf  = evaluar_modelo(rf_modelo,  X_test, y_test, "RandomForest")

    # Tamaño de los modelos guardados (MB)
    ruta_svm = dir_ckpt / f"hce_svm_R{R}.joblib"
    ruta_rf  = dir_ckpt / f"hce_rf_R{R}.joblib"
    joblib.dump(svm_modelo, ruta_svm)
    joblib.dump(rf_modelo,  ruta_rf)

    tam_svm_mb = ruta_svm.stat().st_size / 1e6
    tam_rf_mb  = ruta_rf.stat().st_size / 1e6

    # Feature importances del Random Forest (interpretabilidad)
    importancias = sorted(
        zip(nombres, rf_modelo.feature_importances_),
        key=lambda t: -t[1],
    )

    print("\n[Random Forest] Top 5 features mas importantes:")
    for nombre, imp in importancias[:5]:
        print(f"    {nombre:25s}: {imp:.4f}")

    # Guardar resumen completo
    resumen = {
        "schema_name": RESULTADOS_SCHEMA_NAME,
        "schema_version": RESULTADOS_SCHEMA_VERSION,
        "resolucion": R,
        "profundidad_octree": L,
        "contrato_features": ruta_portable(ruta_contrato),
        "contract_version": contrato["contract_version"],
        "features_excluidas": contrato["caracteristicas_excluidas"],
        "orden_features_modelo": nombres,
        "n_features": X_train.shape[1],
        "n_train_total": int(X_train_full.shape[0]),
        "n_train": int(X_train.shape[0]),
        "n_val":   int(X_val.shape[0]),
        "n_test":  int(X_test.shape[0]),
        "seed": SEED,
        "svm": {
            "mejores_hiperparametros": svm_params,
            "val_acc": svm_val_acc,
            "tiempo_busqueda_hiperparam_s": svm_tiempo,
            "tamano_modelo_mb": tam_svm_mb,
            **resultados_svm,
        },
        "random_forest": {
            "n_estimators": 100,
            "max_depth": None,
            "val_acc": rf_val_acc,
            "tiempo_entrenamiento_s": rf_tiempo,
            "tamano_modelo_mb": tam_rf_mb,
            "feature_importances": [
                {"feature": n, "importancia": float(i)} for n, i in importancias
            ],
            **resultados_rf,
        },
    }

    ruta_resumen = dir_resultados / f"resumen_hce_R{R}.json"
    with open(ruta_resumen, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")

    print("\n" + "=" * 60)
    print(f"  RESUMEN FINAL — HCE Resolucion {R}^3")
    print("=" * 60)
    print(f"  SVM           Test Acc : {resultados_svm['test_acc']*100:.2f}%  "
          f"({tam_svm_mb:.2f} MB)")
    print(f"  RandomForest  Test Acc : {resultados_rf['test_acc']*100:.2f}%  "
          f"({tam_rf_mb:.2f} MB)")
    print(f"  Resultados guardados   : {ruta_resumen}")
    print("=" * 60)

    return resumen


if __name__ == "__main__":
    main()
