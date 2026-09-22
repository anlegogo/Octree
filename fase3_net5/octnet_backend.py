"""Backend PyTorch disperso para operaciones sobre grid-octrees.

Las convoluciones usan los planes geometricos de :mod:`grid_octree` y son
completamente diferenciables. No se invoca ``Conv3d`` ni se materializa el
volumen de entrada. Solo se expande la salida final de resolucion 8^3, que es
la entrada explicita de la capa totalmente conectada de la Tabla 5.

Esta implementacion prioriza trazabilidad y equivalencia matematica. Los
planes se construyen en CPU y se almacenan por lote; las multiplicaciones y
la reduccion se ejecutan en el dispositivo de los atributos (CPU o CUDA).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from grid_octree import GeometriaGridOctree, MuestraGridOctree


@dataclass
class LoteGridOctree:
    """Lote de hojas concatenadas con geometria independiente por muestra."""

    atributos: torch.Tensor
    geometrias: tuple[GeometriaGridOctree, ...]
    offsets: np.ndarray
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.atributos.ndim != 2:
            raise ValueError("Los atributos del lote deben tener forma (N, C)")
        self.offsets = np.asarray(self.offsets, dtype=np.int64)
        if self.offsets.shape != (len(self.geometrias) + 1,):
            raise ValueError("Los offsets no corresponden al numero de muestras")
        if int(self.offsets[0]) != 0 or int(self.offsets[-1]) != len(self.atributos):
            raise ValueError("Los offsets no cubren todas las hojas")
        esperados = np.asarray(
            [geometria.n_hojas for geometria in self.geometrias], dtype=np.int64,
        )
        if not np.array_equal(np.diff(self.offsets), esperados):
            raise ValueError("Los offsets no coinciden con las geometrias")
        resoluciones = {geometria.resolucion for geometria in self.geometrias}
        if len(resoluciones) != 1:
            raise ValueError("Todas las muestras de un lote deben tener igual resolucion")

    @classmethod
    def desde_muestras(cls, muestras: Iterable[MuestraGridOctree]) -> "LoteGridOctree":
        muestras = tuple(muestras)
        if not muestras:
            raise ValueError("No se puede construir un lote vacio")
        geometrias = tuple(muestra.geometria for muestra in muestras)
        cantidades = [geometria.n_hojas for geometria in geometrias]
        offsets = np.concatenate([
            np.asarray([0], dtype=np.int64),
            np.cumsum(cantidades, dtype=np.int64),
        ])
        atributos_np = np.concatenate(
            [muestra.atributos for muestra in muestras], axis=0,
        )
        return cls(
            atributos=torch.from_numpy(atributos_np.copy()),
            geometrias=geometrias,
            offsets=offsets,
        )

    @property
    def resolucion(self) -> int:
        return self.geometrias[0].resolucion

    @property
    def batch_size(self) -> int:
        return len(self.geometrias)

    def size(self, dim: int | None = None):
        forma = (self.batch_size, self.atributos.shape[1])
        return forma if dim is None else forma[dim]

    def to(self, *args, **kwargs) -> "LoteGridOctree":
        return LoteGridOctree(
            atributos=self.atributos.to(*args, **kwargs),
            geometrias=self.geometrias,
            offsets=self.offsets,
            _cache=self._cache,
        )

    def pin_memory(self) -> "LoteGridOctree":
        return LoteGridOctree(
            atributos=self.atributos.pin_memory(),
            geometrias=self.geometrias,
            offsets=self.offsets,
            _cache=self._cache,
        )

    def con_atributos(self, atributos: torch.Tensor) -> "LoteGridOctree":
        return LoteGridOctree(
            atributos=atributos,
            geometrias=self.geometrias,
            offsets=self.offsets,
            _cache=self._cache,
        )

    def _plan_convolucion_numpy(self) -> tuple[np.ndarray, ...]:
        clave = "conv_numpy"
        if clave not in self._cache:
            salidas_por_kernel = [[] for _ in range(27)]
            entradas_por_kernel = [[] for _ in range(27)]
            coeficientes_por_kernel = [[] for _ in range(27)]
            for offset, geometria in zip(self.offsets[:-1], self.geometrias):
                plan = geometria.plan_convolucion
                for kernel in range(27):
                    inicio = int(plan.offsets_kernel[kernel])
                    fin = int(plan.offsets_kernel[kernel + 1])
                    salidas_por_kernel[kernel].append(
                        plan.salida[inicio:fin] + int(offset)
                    )
                    entradas_por_kernel[kernel].append(
                        plan.entrada[inicio:fin] + int(offset)
                    )
                    coeficientes_por_kernel[kernel].append(
                        plan.coeficiente[inicio:fin]
                    )
            salidas = []
            entradas = []
            coeficientes = []
            offsets_kernel = [0]
            for kernel in range(27):
                salida_kernel = np.concatenate(salidas_por_kernel[kernel])
                entrada_kernel = np.concatenate(entradas_por_kernel[kernel])
                coeficiente_kernel = np.concatenate(
                    coeficientes_por_kernel[kernel]
                )
                salidas.append(salida_kernel)
                entradas.append(entrada_kernel)
                coeficientes.append(coeficiente_kernel)
                offsets_kernel.append(offsets_kernel[-1] + len(salida_kernel))
            self._cache[clave] = (
                np.concatenate(salidas),
                np.concatenate(entradas),
                np.concatenate(coeficientes),
                np.asarray(offsets_kernel, dtype=np.int64),
            )
        return self._cache[clave]

    def plan_convolucion_torch(self) -> tuple:
        dispositivo = self.atributos.device
        clave = ("conv_torch", dispositivo.type, dispositivo.index)
        if clave not in self._cache:
            salida, entrada, coeficiente, offsets = (
                self._plan_convolucion_numpy()
            )
            self._cache[clave] = (
                torch.as_tensor(salida, dtype=torch.long, device=dispositivo),
                torch.as_tensor(entrada, dtype=torch.long, device=dispositivo),
                torch.as_tensor(
                    coeficiente,
                    dtype=self.atributos.dtype,
                    device=dispositivo,
                ),
                offsets,
            )
        return self._cache[clave]

    def max_pool2(self) -> "LoteGridOctree":
        geometrias_salida = tuple(
            geometria.plan_pooling.geometria_salida
            for geometria in self.geometrias
        )
        cantidades_salida = [g.n_hojas for g in geometrias_salida]
        offsets_salida = np.concatenate([
            np.asarray([0], dtype=np.int64),
            np.cumsum(cantidades_salida, dtype=np.int64),
        ])
        mapeos = []
        for offset_salida, geometria in zip(
            offsets_salida[:-1], self.geometrias,
        ):
            mapeos.append(
                geometria.plan_pooling.entrada_a_salida + int(offset_salida)
            )
        mapa_np = np.concatenate(mapeos)
        mapa = torch.as_tensor(
            mapa_np, dtype=torch.long, device=self.atributos.device,
        )
        indice = mapa[:, None].expand(-1, self.atributos.shape[1])
        salida = torch.full(
            (int(offsets_salida[-1]), self.atributos.shape[1]),
            -torch.inf,
            dtype=self.atributos.dtype,
            device=self.atributos.device,
        )
        salida.scatter_reduce_(
            0, indice, self.atributos, reduce="amax", include_self=True,
        )
        return LoteGridOctree(
            atributos=salida,
            geometrias=geometrias_salida,
            offsets=offsets_salida,
        )

    def tensor_final_8(self) -> torch.Tensor:
        """Expande unicamente la salida final 8^3 para la capa FC."""

        if self.resolucion != 8:
            raise ValueError("La capa FC requiere una salida de resolucion 8^3")
        mapas = []
        for offset, geometria in zip(self.offsets[:-1], self.geometrias):
            mapas.append(geometria.indices_voxel_a_hoja() + int(offset))
        indices = torch.as_tensor(
            np.concatenate(mapas),
            dtype=torch.long,
            device=self.atributos.device,
        )
        # El mapa se genero en orden (x,y,z). Se devuelve (B,C,X,Y,Z).
        denso = self.atributos.index_select(0, indices)
        denso = denso.reshape(self.batch_size, 8, 8, 8, -1)
        return denso.permute(0, 4, 1, 2, 3).contiguous()


class ConvolucionOctree3x3(nn.Module):
    """Convolucion 3x3x3 exacta sobre hojas, sin ``torch.nn.Conv3d``."""

    def __init__(self, canales_entrada: int, canales_salida: int,
                 bias: bool = True):
        super().__init__()
        self.canales_entrada = int(canales_entrada)
        self.canales_salida = int(canales_salida)
        self.weight = nn.Parameter(torch.empty(
            canales_salida, canales_entrada, 3, 3, 3,
        ))
        if bias:
            self.bias = nn.Parameter(torch.empty(canales_salida))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(
            self.weight, mode="fan_out", nonlinearity="relu",
        )
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, lote: LoteGridOctree) -> LoteGridOctree:
        if lote.atributos.shape[1] != self.canales_entrada:
            raise ValueError(
                f"Se esperaban {self.canales_entrada} canales y llegaron "
                f"{lote.atributos.shape[1]}"
            )
        salida_idx, entrada_idx, coeficiente, offsets_kernel = (
            lote.plan_convolucion_torch()
        )
        n_hojas = lote.atributos.shape[0]
        if self.bias is None:
            salida = lote.atributos.new_zeros((n_hojas, self.canales_salida))
        else:
            salida = self.bias.unsqueeze(0).expand(n_hojas, -1).clone()

        pesos = self.weight.reshape(
            self.canales_salida, self.canales_entrada, 27,
        )
        for indice_kernel in range(27):
            inicio = int(offsets_kernel[indice_kernel])
            fin = int(offsets_kernel[indice_kernel + 1])
            indices_entrada = entrada_idx[inicio:fin]
            contribucion = (
                lote.atributos.index_select(0, indices_entrada)
                @ pesos[:, :, indice_kernel].transpose(0, 1)
            )
            contribucion = contribucion * coeficiente[inicio:fin, None]
            salida.index_add_(0, salida_idx[inicio:fin], contribucion)
        return lote.con_atributos(salida)


class MaxPoolOctree2(nn.Module):
    """Max-pooling 2x2x2 que transforma directamente la jerarquia."""

    def forward(self, lote: LoteGridOctree) -> LoteGridOctree:
        return lote.max_pool2()
