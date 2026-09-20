import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FASE2 = ROOT / "fase2_octree"
FASE3_HCE = ROOT / "fase3_hce"
sys.path.insert(0, str(FASE2))
sys.path.insert(0, str(FASE3_HCE))
