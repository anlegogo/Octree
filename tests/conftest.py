import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FASE2 = ROOT / "fase2_octree"
sys.path.insert(0, str(FASE2))
