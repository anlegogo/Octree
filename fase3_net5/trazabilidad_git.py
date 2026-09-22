"""Trazabilidad reproducible del codigo usado en los experimentos."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _git(raiz: Path, *argumentos: str) -> str:
    return subprocess.run(
        ["git", *argumentos],
        cwd=raiz,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def capturar_estado_git(raiz: Path) -> dict:
    """Captura commit, rama y cambios locales antes de una corrida."""

    try:
        commit = _git(raiz, "rev-parse", "HEAD")
        rama = _git(raiz, "branch", "--show-current") or None
        estado = _git(raiz, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": None,
            "git_branch": None,
            "git_dirty": None,
            "git_status_inicio": None,
        }

    entradas = estado.splitlines() if estado else []
    return {
        "git_commit": commit,
        "git_branch": rama,
        "git_dirty": bool(entradas),
        "git_status_inicio": entradas,
    }


def exigir_estado_git_limpio(estado: dict) -> None:
    """Interrumpe una corrida si no puede vincularse a un commit limpio."""

    if estado.get("git_commit") is None or estado.get("git_dirty") is None:
        raise RuntimeError(
            "No fue posible identificar el commit y el estado del repositorio."
        )
    if estado["git_dirty"]:
        detalles = "\n".join(estado.get("git_status_inicio") or [])
        raise RuntimeError(
            "El repositorio tiene cambios locales antes de la corrida. "
            "Confirme, descarte o retire esos archivos y repita la ejecucion."
            + (f"\n{detalles}" if detalles else "")
        )
