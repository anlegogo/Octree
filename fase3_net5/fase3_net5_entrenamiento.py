"""Entrenamiento de Net5-Octree y de su referencia densa diagnostica.

El backend predeterminado ``octree_native`` opera directamente sobre el
grid-octree. ``dense_reference`` se conserva solo como diagnostico y sus
resultados nunca son oficiales.

Caracteristicas:
  - CPU o GPU via PyTorch
  - Early Stopping basado en val accuracy (segun metodologia, sec. 6.3)
  - Optimizador Adam + CrossEntropy
  - Log CSV y JSON por epoca
  - Checkpoint del mejor modelo
  - Medicion de VRAM, tiempo de entrenamiento e inferencia (Fase 4)
  - Sin aumento de datos (criterio de equivalencia, sec. 6.6)

Uso:
    python fase3_net5/fase3_net5_entrenamiento.py --resolucion 32 \
      --backend dense_reference --tag _smoke --limite_train 80 \
      --limite_val 40 --limite_test 80
"""

import sys
import csv
import json
import time
import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "fase1_modelnet40"))
sys.path.insert(0, str(Path(__file__).parent))

from fase1_setup import set_global_seed
from net5_dataset import CLASES, crear_dataloaders_densos_referencia
from net5_dataset_octree import crear_dataloaders_octree
from net5_modelo import crear_modelo, crear_modelo_denso_referencia, get_device
from particion_objetivo2 import (
    cargar_o_crear_particion,
    listar_muestras_octree,
    seleccionar_subconjunto_balanceado,
)

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
SEED = 42


def _sincronizar_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _ruta_portable(ruta: Path) -> str:
    ruta = ruta.resolve()
    try:
        return ruta.relative_to(RAIZ_PROYECTO.resolve()).as_posix()
    except ValueError:
        return ruta.name


def _estado_git() -> dict:
    """Identifica exactamente el codigo usado por una corrida."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=RAIZ_PROYECTO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=RAIZ_PROYECTO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


# ──────────────────────────────────────────────────────────────
# ENTRENAMIENTO DE UNA EPOCA
# ──────────────────────────────────────────────────────────────

def entrenar_epoca(modelo, loader, criterio, optimizador, device):
    modelo.train()
    perdida_total, correctos, total = 0.0, 0, 0

    barra = tqdm(loader, desc="  Train", leave=False, ncols=80)
    for grids, etiquetas in barra:
        grids     = grids.to(device, non_blocking=True)
        etiquetas = etiquetas.to(device, non_blocking=True)

        optimizador.zero_grad()
        logits  = modelo(grids)
        perdida = criterio(logits, etiquetas)
        perdida.backward()
        optimizador.step()

        perdida_total += perdida.item() * grids.size(0)
        correctos     += (logits.argmax(1) == etiquetas).sum().item()
        total         += grids.size(0)
        barra.set_postfix(loss=f"{perdida.item():.4f}")

    return perdida_total / total, correctos / total


# ──────────────────────────────────────────────────────────────
# EVALUACION
# ──────────────────────────────────────────────────────────────

def evaluar(modelo, loader, criterio, device, desc="Val"):
    modelo.eval()
    perdida_total, correctos, total = 0.0, 0, 0

    with torch.no_grad():
        barra = tqdm(loader, desc=f"  {desc}", leave=False, ncols=80)
        for grids, etiquetas in barra:
            grids     = grids.to(device, non_blocking=True)
            etiquetas = etiquetas.to(device, non_blocking=True)
            logits    = modelo(grids)
            perdida   = criterio(logits, etiquetas)
            perdida_total += perdida.item() * grids.size(0)
            correctos     += (logits.argmax(1) == etiquetas).sum().item()
            total         += grids.size(0)

    return perdida_total / total, correctos / total


# ──────────────────────────────────────────────────────────────
# MEDIR TIEMPO DE INFERENCIA (Fase 4 de la metodologia)
# ──────────────────────────────────────────────────────────────

def medir_tiempo_inferencia(modelo, loader, device, n_repeticiones: int = 3) -> dict:
    """
    Mide el tiempo promedio de clasificar una sola muestra sobre el
    conjunto de test completo, promediando N repeticiones.
    Cumple con la Fase 4: "tiempo promedio requerido para clasificar
    una sola muestra, calculado sobre el total del conjunto de prueba".
    """
    modelo.eval()
    tiempos = []
    total_ultima_repeticion = 0

    for _ in range(n_repeticiones):
        tiempo_modelo = 0.0
        total = 0
        with torch.no_grad():
            for grids, _ in loader:
                grids = grids.to(device, non_blocking=True)
                _sincronizar_cuda(device)
                t0 = time.perf_counter()
                modelo(grids)
                _sincronizar_cuda(device)
                tiempo_modelo += time.perf_counter() - t0
                total += grids.size(0)
        tiempos.append(tiempo_modelo)
        total_ultima_repeticion = total

    tiempo_total_prom = np.mean(tiempos)
    tiempo_por_muestra_ms = (tiempo_total_prom / total_ultima_repeticion) * 1000

    return {
        "tiempo_inferencia_total_s":       round(float(tiempo_total_prom), 4),
        "tiempo_inferencia_promedio_ms":    round(float(tiempo_por_muestra_ms), 4),
        "n_muestras_test":                 int(total_ultima_repeticion),
        "protocolo_tiempo": "solo forward; excluye carga y transferencia CPU-GPU",
        "repeticiones_inferencia": int(n_repeticiones),
    }


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolucion", type=int, default=32, choices=[32, 64])
    parser.add_argument(
        "--backend", choices=["octree_native", "dense_reference"],
        default="octree_native",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping: epocas sin mejora antes de parar")
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Por defecto: 1 para octree_native y 16 para dense_reference",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--limite_train", type=int, default=None,
                        help="Usar N muestras balanceadas de train "
                             "(smoke test rapido, no afecta el uso normal)")
    parser.add_argument("--limite_val", type=int, default=None,
                        help="Usar N muestras balanceadas de val")
    parser.add_argument("--limite_test", type=int, default=None,
                        help="Usar N muestras balanceadas de test")
    parser.add_argument("--tag", type=str, default="",
                        help="Sufijo para checkpoint/logs/resultados, para no "
                             "sobreescribir una corrida completa (ej. '_smoke')")
    parser.add_argument("--data-root", type=Path, default=RAIZ_PROYECTO / "data")
    parser.add_argument(
        "--particion-manifest", type=Path,
        default=RAIZ_PROYECTO / "logs" / "particion_objetivo2_modelos.json",
    )
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=RAIZ_PROYECTO / "checkpoints",
    )
    parser.add_argument("--logs-dir", type=Path, default=RAIZ_PROYECTO / "logs")
    parser.add_argument(
        "--resultados-dir", type=Path,
        default=RAIZ_PROYECTO / "resultados" / "objetivo3",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    R = args.resolucion
    if args.batch_size is None:
        args.batch_size = 1 if args.backend == "octree_native" else 16

    if args.backend == "dense_reference" and not args.tag:
        raise ValueError(
            "La referencia densa es solo diagnostica y requiere --tag "
            "(por ejemplo, --tag _smoke_dense)."
        )

    print("=" * 60)
    nombre_backend = (
        "NET5-OCTREE NATIVO"
        if args.backend == "octree_native"
        else "DIAGNOSTICO DENSO TABLA 5"
    )
    print(f"  {nombre_backend} — Resolucion {R}^3")
    print("=" * 60)

    set_global_seed(SEED)
    device = get_device()

    for d in (args.checkpoints_dir, args.logs_dir, args.resultados_dir):
        d.mkdir(parents=True, exist_ok=True)

    raiz_resolucion = args.data_root / f"octrees_{R}"
    idx_train, idx_val, idx_test, particion = cargar_o_crear_particion(
        raiz_resolucion=raiz_resolucion,
        ruta_manifest=args.particion_manifest,
        seed=SEED,
        val_split=0.10,
    )
    _, etiquetas_train_total, _ = listar_muestras_octree(
        raiz_resolucion, "train",
    )
    _, etiquetas_test, _ = listar_muestras_octree(raiz_resolucion, "test")
    idx_train = seleccionar_subconjunto_balanceado(
        idx_train, etiquetas_train_total, args.limite_train, SEED,
    )
    idx_val = seleccionar_subconjunto_balanceado(
        idx_val, etiquetas_train_total, args.limite_val, SEED,
    )
    idx_test = seleccionar_subconjunto_balanceado(
        idx_test, etiquetas_test, args.limite_test, SEED,
    )

    argumentos_loader = {
        "raiz_resolucion": str(raiz_resolucion),
        "resolucion": R,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": SEED,
    }
    if args.backend == "octree_native":
        loader_train, loader_val, loader_test = crear_dataloaders_octree(
            **argumentos_loader,
        )
        modelo, device = crear_modelo(resolucion=R, device=device)
    else:
        loader_train, loader_val, loader_test = (
            crear_dataloaders_densos_referencia(**argumentos_loader)
        )
        modelo, device = crear_modelo_denso_referencia(
            resolucion=R, device=device,
        )
    criterio    = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.StepLR(optimizador, step_size=20, gamma=0.7)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Archivos de log
    prefijo = "net5_octree" if args.backend == "octree_native" else "dense_tabla5"
    csv_path = args.logs_dir / f"{prefijo}_historial_R{R}{args.tag}.csv"
    json_path = args.logs_dir / f"{prefijo}_historial_R{R}{args.tag}.json"
    ckpt_path = args.checkpoints_dir / f"{prefijo}_mejor_R{R}{args.tag}.pth"

    historial = []
    mejor_val_acc = float("-inf")
    epocas_sin_mejora = 0

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(
            ["epoca", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "tiempo_s"]
        )

    print(f"\n  Epocas max   : {args.epochs}")
    print(f"  Early stop   : patience={args.patience}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR inicial   : {args.lr}")
    print(f"  Train batches: {len(loader_train)}")
    print(f"  Val   batches: {len(loader_val)}\n")

    _sincronizar_cuda(device)
    t_inicio = time.perf_counter()

    for epoca in range(1, args.epochs + 1):
        _sincronizar_cuda(device)
        t_ep = time.perf_counter()

        train_loss, train_acc = entrenar_epoca(modelo, loader_train, criterio, optimizador, device)
        val_loss,   val_acc   = evaluar(modelo, loader_val, criterio, device, "Val")

        scheduler.step()
        lr_actual = scheduler.get_last_lr()[0]
        _sincronizar_cuda(device)
        t_ep = time.perf_counter() - t_ep

        # Checkpoint si mejora
        es_mejor = val_acc > mejor_val_acc
        if es_mejor:
            mejor_val_acc = val_acc
            epocas_sin_mejora = 0
            torch.save({
                "epoca": epoca, "model_state": modelo.state_dict(),
                "optimizer_state": optimizador.state_dict(),
                "mejor_val_acc": mejor_val_acc,
                "resolucion": R,
            }, ckpt_path)
            marca = " <- mejor"
        else:
            epocas_sin_mejora += 1
            marca = ""

        print(
            f"  Ep {epoca:03d}/{args.epochs} | "
            f"Train {train_acc*100:.2f}% loss {train_loss:.4f} | "
            f"Val {val_acc*100:.2f}% loss {val_loss:.4f} | "
            f"LR {lr_actual:.6f} | {t_ep:.1f}s{marca}"
        )

        # VRAM
        if device.type == "cuda" and epoca == 1:
            vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
            print(f"  [GPU] VRAM pico: {vram_mb:.1f} MB")

        # Log
        fila = [epoca, round(train_loss,6), round(train_acc,6),
                round(val_loss,6), round(val_acc,6), round(lr_actual,8), round(t_ep,2)]
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(fila)

        historial.append({
            "epoca": epoca, "train_loss": round(train_loss,6),
            "train_acc": round(train_acc,6), "val_loss": round(val_loss,6),
            "val_acc": round(val_acc,6), "lr": round(lr_actual,8),
            "tiempo_s": round(t_ep,2),
        })
        with open(json_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(historial, f, indent=2, ensure_ascii=False)
            f.write("\n")

        # Early stopping
        if epocas_sin_mejora >= args.patience:
            print(f"\n  [Early Stopping] Sin mejora en {args.patience} epocas. "
                  f"Deteniendo en epoca {epoca}.")
            break

    _sincronizar_cuda(device)
    t_total = (time.perf_counter() - t_inicio) / 60

    # Evaluacion final en test con el mejor modelo
    print("\n[Test] Cargando mejor modelo y evaluando...")
    ckpt = torch.load(ckpt_path, map_location=device)
    modelo.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluar(modelo, loader_test, criterio, device, "Test")

    # Matriz de confusion y reporte por clase
    print("[Test] Generando matriz de confusion...")
    from sklearn.metrics import confusion_matrix, classification_report
    todas_pred, todas_real = [], []
    modelo.eval()
    with torch.no_grad():
        for grids, etiquetas in loader_test:
            grids = grids.to(device, non_blocking=True)
            logits = modelo(grids)
            todas_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            todas_real.extend(etiquetas.numpy().tolist())
    etiquetas_40 = list(range(len(CLASES)))
    matriz_conf = confusion_matrix(todas_real, todas_pred, labels=etiquetas_40).tolist()
    reporte_cls = classification_report(todas_real, todas_pred,
                                        labels=etiquetas_40,
                                        target_names=CLASES,
                                        output_dict=True, zero_division=0)

    # Medir tiempo de inferencia (Fase 4)
    print("[Fase 4] Midiendo tiempo de inferencia...")
    metricas_inf = medir_tiempo_inferencia(modelo, loader_test, device)

    # Tamano del modelo en MB
    tamano_mb = ckpt_path.stat().st_size / 1e6

    # VRAM pico total
    vram_pico_mb = torch.cuda.max_memory_allocated(device) / 1e6 \
                   if device.type == "cuda" else 0.0

    print("\n" + "=" * 60)
    print(f"  {nombre_backend} COMPLETADO — R={R}^3")
    print("=" * 60)
    print(f"  Tiempo total    : {t_total:.1f} min")
    print(f"  Mejor Val Acc   : {mejor_val_acc*100:.2f}%  (ep {ckpt['epoca']})")
    print(f"  Test  Acc final : {test_acc*100:.2f}%")
    print(f"  Inf/muestra     : {metricas_inf['tiempo_inferencia_promedio_ms']:.3f} ms")
    print(f"  VRAM pico       : {vram_pico_mb:.1f} MB")
    print(f"  Tamano modelo   : {tamano_mb:.2f} MB")
    print("=" * 60)

    limites_activos = any(
        limite is not None
        for limite in (args.limite_train, args.limite_val, args.limite_test)
    )
    es_nativo = args.backend == "octree_native"
    test_oficial_completo = len(loader_test.dataset) == 2468
    resultado_oficial = bool(
        es_nativo and not limites_activos and test_oficial_completo
    )
    if resultado_oficial:
        motivo_no_valido = None
    elif not es_nativo:
        motivo_no_valido = (
            "Usa nn.Conv3d sobre una rejilla densa; no implementa las "
            "operaciones OctNet sobre el grid-octree."
        )
    elif limites_activos:
        motivo_no_valido = (
            "Corrida nativa limitada; sirve como smoke test, no como "
            "evaluacion completa."
        )
    else:
        motivo_no_valido = (
            "El test no contiene las 2.468 muestras oficiales completas."
        )
    resumen = {
        "schema_name": (
            "net5-octree-native" if es_nativo else "dense-table5-diagnostic"
        ),
        "schema_version": "1.0.0",
        "backend_version": "1.0.0",
        "alcance": (
            "PARCIAL_SMOKE"
            if limites_activos
            else (
                "COMPLETO"
                if resultado_oficial
                else ("INCOMPLETO" if es_nativo else "COMPLETO_DIAGNOSTICO")
            )
        ),
        "valido_como_resultado_objetivo3": resultado_oficial,
        "motivo_no_valido": motivo_no_valido,
        "backend": args.backend,
        **_estado_git(),
        "resolucion": R,
        "profundidad_octree": 5 if R == 32 else 6,
        "parametros_entrenables": sum(
            p.numel() for p in modelo.parameters() if p.requires_grad
        ),
        "seed": SEED,
        "particion": {
            "manifest": _ruta_portable(args.particion_manifest),
            "raiz_datos": _ruta_portable(raiz_resolucion),
            "metodo": particion["metodo"],
            "n_train_total": particion["n_train_total"],
            "n_train_usado": len(loader_train.dataset),
            "n_val_usado": len(loader_val.dataset),
            "n_test_usado": len(loader_test.dataset),
        },
        "configuracion": {
            "epochs_max": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": 1e-4,
            "scheduler": "StepLR(step_size=20, gamma=0.7)",
        },
        "mejor_val_acc":  round(float(mejor_val_acc), 6),
        "mejor_epoca":    int(ckpt["epoca"]),
        "test_acc":       round(float(test_acc), 6),
        "test_loss":      round(float(test_loss), 6),
        "tiempo_total_min": round(t_total, 2),
        "tamano_modelo_mb": round(tamano_mb, 2),
        "vram_pico_mb":     round(vram_pico_mb, 1),
        "matriz_confusion":          matriz_conf,
        "reporte_clasificacion":     reporte_cls,
        **metricas_inf,
    }

    salida = args.resultados_dir / f"resumen_{prefijo}_R{R}{args.tag}.json"
    with salida.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(resumen, f, indent=2)
        f.write("\n")

    return resumen


if __name__ == "__main__":
    main()
