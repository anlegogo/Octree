"""
net5_modelo.py - Fase 3 (Enfoque profundo)
===========================================
Implementacion de Net5: red convolucional jerarquica que opera sobre
los grids de octree (4, R, R, R) generados en la Fase 2.

Arquitectura (segun metodologia "red convolucional jerarquica que opera
sobre los nodos del octree"):

    Entrada  : (B, 4, R, R, R)  — 4 canales: [ocupacion, nx, ny, nz]
    5 bloques Conv3D jerarquicos (con stride=2 para reduccion espacial)
    Global Average Pooling 3D
    3 capas FC con Dropout
    Salida   : (B, 40) logits

La jerarquia de stride=2 imita la estructura del octree:
    R=32: 32 -> 16 -> 8 -> 4 -> 2 -> 1  (5 niveles = profundidad L=5)
    R=64: 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1  (6 niveles = L=6)

Esto hace que cada nivel convolucional corresponda a un nivel del octree,
siendo "jerarquica" en el sentido literal de la metodologia.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# BLOQUE BASICO: Conv3D + BN + ReLU
# ──────────────────────────────────────────────────────────────

class ConvBnRelu3D(nn.Module):
    """Conv3D(stride=1) → BatchNorm3D → ReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel,
                              stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class BloqueJerarquico(nn.Module):
    """
    Bloque jerarquico del octree: dos Conv3D (la primera con stride=2
    para reducir la resolucion espacial a la mitad, igual que pasar al
    nivel padre en el octree) seguidas de una conexion residual ligera.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBnRelu3D(in_ch, out_ch, stride=2, padding=1)
        self.conv2 = ConvBnRelu3D(out_ch, out_ch, stride=1, padding=1)

        # Proyeccion residual (si los canales cambian)
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm3d(out_ch),
        ) if in_ch != out_ch else nn.Sequential(
            nn.AvgPool3d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return F.relu(x + residual)


# ──────────────────────────────────────────────────────────────
# NET5 — RED CONVOLUCIONAL JERARQUICA SOBRE OCTREE
# ──────────────────────────────────────────────────────────────

class Net5Octree(nn.Module):
    """
    Red convolucional jerarquica (Net5) que opera sobre grids de octree.

    Parametros
    ----------
    resolucion    : 32 o 64
    num_clases    : 40 para ModelNet40
    dropout       : tasa de dropout en las capas FC
    in_channels   : 4 (ocupacion + nx + ny + nz)
    """

    # Configuracion de canales por nivel del octree
    # 7 valores: canal inicial + 6 bloques (soporta hasta L=6 para R=64)
    CANALES = [4, 32, 64, 128, 256, 512, 512]

    def __init__(self, resolucion: int = 32, num_clases: int = 40,
                 dropout: float = 0.3, in_channels: int = 4):
        super().__init__()

        assert resolucion in (32, 64), "Resolucion debe ser 32 o 64"
        self.resolucion = resolucion
        n_niveles = 5 if resolucion == 32 else 6

        # Proyeccion inicial de 4 canales a 32
        self.proyeccion = ConvBnRelu3D(in_channels, 32, kernel=3, stride=1, padding=1)

        # Bloques jerarquicos (uno por nivel del octree)
        bloques = []
        canales_in = 32
        for i in range(n_niveles):
            canales_out = min(self.CANALES[i + 1], 512)
            bloques.append(BloqueJerarquico(canales_in, canales_out))
            canales_in = canales_out
        self.bloques = nn.Sequential(*bloques)

        # Global Average Pooling: colapsa la dimension espacial a (B, C, 1, 1, 1)
        self.gap = nn.AdaptiveAvgPool3d(1)

        # Clasificador FC
        self.fc1 = nn.Linear(canales_in, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_clases)

        self.bn_fc1 = nn.BatchNorm1d(512)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout)

        self._init_pesos()

    def _init_pesos(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 4, R, R, R)
        Retorna logits (B, num_clases)
        """
        x = self.proyeccion(x)    # (B, 32, R, R, R)
        x = self.bloques(x)       # (B, 512, 1, 1, 1) aprox al final

        x = self.gap(x)           # (B, C, 1, 1, 1)
        x = x.flatten(1)          # (B, C)

        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)

        return x

    def contar_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        nombre  = torch.cuda.get_device_name(0)
        memoria = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Dispositivo] GPU: {nombre} ({memoria:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[Dispositivo] CPU")
    return device


def crear_modelo(resolucion: int = 32, num_clases: int = 40,
                 dropout: float = 0.3, device: torch.device = None) -> tuple:
    if device is None:
        device = get_device()
    modelo = Net5Octree(resolucion=resolucion, num_clases=num_clases,
                        dropout=dropout).to(device)
    params = modelo.contar_parametros()
    n_niveles = 5 if resolucion == 32 else 6
    print(f"\n[Modelo] Net5-Octree creado:")
    print(f"  Resolucion    : {resolucion}^3  (L={n_niveles} niveles)")
    print(f"  Clases        : {num_clases}")
    print(f"  Dropout       : {dropout}")
    print(f"  Parametros    : {params:,}")
    print(f"  Dispositivo   : {device}")
    return modelo, device


# ──────────────────────────────────────────────────────────────
# TEST RAPIDO
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    torch.manual_seed(42)

    for R in (32, 64):
        modelo, device = crear_modelo(resolucion=R)
        x = torch.randn(2, 4, R, R, R).to(device)
        logits = modelo(x)
        print(f"  Input {x.shape} -> Output {logits.shape}\n")