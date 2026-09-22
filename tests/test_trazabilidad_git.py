"""Pruebas de la trazabilidad previa a los experimentos."""

import subprocess
import sys
from pathlib import Path

import pytest


RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "fase3_net5"))

from trazabilidad_git import capturar_estado_git, exigir_estado_git_limpio


def _git(raiz: Path, *argumentos: str) -> None:
    subprocess.run(["git", *argumentos], cwd=raiz, check=True,
                   capture_output=True, text=True)


def test_estado_git_distingue_repositorio_limpio_y_sucio(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "pruebas@example.com")
    _git(tmp_path, "config", "user.name", "Pruebas")
    archivo = tmp_path / "control.txt"
    archivo.write_text("versionado\n", encoding="utf-8")
    _git(tmp_path, "add", "control.txt")
    _git(tmp_path, "commit", "-m", "estado inicial")

    limpio = capturar_estado_git(tmp_path)
    assert limpio["git_commit"]
    assert limpio["git_branch"]
    assert limpio["git_dirty"] is False
    assert limpio["git_status_inicio"] == []
    exigir_estado_git_limpio(limpio)

    (tmp_path / "local.txt").write_text("no versionado\n", encoding="utf-8")
    sucio = capturar_estado_git(tmp_path)
    assert sucio["git_dirty"] is True
    assert sucio["git_status_inicio"] == ["?? local.txt"]
    with pytest.raises(RuntimeError, match="cambios locales"):
        exigir_estado_git_limpio(sucio)
