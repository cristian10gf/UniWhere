# tests/test_cli.py
"""Tests del parsing de argumentos de visualize_ace.py."""
import sys
from pathlib import Path
import pytest


def test_serie_derives_scene(tmp_path, monkeypatch):
    """--serie <X> debe derivar --scene a DATA_ROOT/X/ace."""
    import visualize_ace as m
    monkeypatch.setattr(m, "DATA_ROOT", tmp_path)
    (tmp_path / "mi-serie" / "ace").mkdir(parents=True)

    args = m.build_parser().parse_args(["--serie", "mi-serie"])
    resolved = m.resolve_paths(args)
    assert resolved.scene == tmp_path / "mi-serie" / "ace"


def test_explicit_scene_takes_precedence(tmp_path, monkeypatch):
    """--scene explícito tiene precedencia sobre --serie."""
    import visualize_ace as m
    monkeypatch.setattr(m, "DATA_ROOT", tmp_path)
    explicit_scene = tmp_path / "custom" / "ace"
    explicit_scene.mkdir(parents=True)
    (tmp_path / "mi-serie").mkdir()

    args = m.build_parser().parse_args([
        "--serie", "mi-serie",
        "--scene", str(explicit_scene),
    ])
    resolved = m.resolve_paths(args)
    assert resolved.scene == explicit_scene


def test_no_serie_no_scene_raises(tmp_path):
    """Sin --serie ni --scene debe lanzar SystemExit."""
    import visualize_ace as m
    args = m.build_parser().parse_args(["--point-cloud", str(tmp_path)])
    with pytest.raises(SystemExit):
        m.resolve_paths(args)


def test_query_images_without_model_raises(tmp_path, monkeypatch):
    """--query-images sin modelo disponible debe lanzar SystemExit."""
    import visualize_ace as m
    monkeypatch.setattr(m, "DATA_ROOT", tmp_path)
    (tmp_path / "mi-serie" / "ace").mkdir(parents=True)
    img = tmp_path / "foto.jpg"
    img.touch()

    args = m.build_parser().parse_args([
        "--serie", "mi-serie",
        "--query-images", str(img),
    ])
    resolved = m.resolve_paths(args)
    # No hay modelo en data/output/mi-serie.pt, así que model=None
    assert resolved.model is None
    with pytest.raises(SystemExit):
        m.validate_args(resolved)
