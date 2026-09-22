"""Dataset oficial para el backend disperso OctNet.

Cada NPZ se reconstruye como arbol global y se convierte a la rejilla de
octrees superficiales de profundidad 3. En ningun punto se crea una matriz
``(4, R, R, R)``.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "fase2_octree"))

from octree_real import reconstruir_octree_desde_npz  # noqa: E402
from grid_octree import MuestraGridOctree, convertir_a_grid_octree  # noqa: E402
from octnet_backend import LoteGridOctree  # noqa: E402
from particion_objetivo2 import (  # noqa: E402
    CLASES_MODELNET40,
    listar_muestras_octree,
)


CLASES = CLASES_MODELNET40


class OctreeNativeDataset(Dataset):
    """Carga la jerarquia preservada del Objetivo 1 para OctNet."""

    def __init__(self, raiz_resolucion: str | Path, resolucion: int,
                 split: str = "train", idx_subset: np.ndarray | None = None):
        super().__init__()
        self.resolucion = int(resolucion)
        carpeta_split = "train" if split in ("train", "val") else "test"
        rutas, etiquetas, _ = listar_muestras_octree(
            raiz_resolucion, carpeta_split,
        )
        self.muestras = rutas
        self.etiquetas = etiquetas

        if idx_subset is not None:
            indices = np.asarray(idx_subset, dtype=np.int64)
            if indices.size and (indices.min() < 0 or indices.max() >= len(rutas)):
                raise IndexError("La particion contiene indices fuera del dataset")
            self.muestras = [rutas[i] for i in indices]
            self.etiquetas = etiquetas[indices]

        print(
            f"[Dataset] Grid-octree nativo '{split}': {len(self.muestras)} "
            f"muestras (R={self.resolucion}, sin expansion densa)"
        )

    def __len__(self) -> int:
        return len(self.muestras)

    def __getitem__(self, idx: int) -> tuple[MuestraGridOctree, int]:
        raiz, etiqueta, profundidad = reconstruir_octree_desde_npz(
            str(self.muestras[idx]),
        )
        resolucion_archivo = 2 ** int(profundidad)
        if resolucion_archivo != self.resolucion:
            raise ValueError(
                f"El NPZ tiene R={resolucion_archivo}, se esperaba "
                f"R={self.resolucion}"
            )
        etiqueta_esperada = int(self.etiquetas[idx])
        if etiqueta != etiqueta_esperada:
            raise ValueError("La etiqueta del NPZ no coincide con su carpeta")
        muestra = convertir_a_grid_octree(raiz, self.resolucion)
        return muestra, etiqueta


def collate_grid_octree(
    elementos: list[tuple[MuestraGridOctree, int]],
) -> tuple[LoteGridOctree, torch.Tensor]:
    muestras, etiquetas = zip(*elementos)
    lote = LoteGridOctree.desde_muestras(muestras)
    return lote, torch.as_tensor(etiquetas, dtype=torch.long)


def _worker_init_fn(worker_id: int, seed: int = 42) -> None:
    np.random.seed(seed + worker_id)


def crear_dataloaders_octree(
    raiz_resolucion: str | Path,
    resolucion: int,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Crea los tres DataLoaders del protocolo oficial."""

    ds_train = OctreeNativeDataset(
        raiz_resolucion, resolucion, "train", idx_train,
    )
    ds_val = OctreeNativeDataset(
        raiz_resolucion, resolucion, "val", idx_val,
    )
    ds_test = OctreeNativeDataset(
        raiz_resolucion, resolucion, "test", idx_test,
    )

    generador = torch.Generator()
    generador.manual_seed(seed)
    init_fn = partial(_worker_init_fn, seed=seed)
    comunes = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_grid_octree,
    }
    loader_train = DataLoader(
        ds_train,
        shuffle=True,
        worker_init_fn=init_fn,
        generator=generador,
        **comunes,
    )
    loader_val = DataLoader(ds_val, shuffle=False, **comunes)
    loader_test = DataLoader(ds_test, shuffle=False, **comunes)
    return loader_train, loader_val, loader_test
