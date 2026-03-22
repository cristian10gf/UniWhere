"""
Tests de humo para QueryRelocalizer.
No requieren GPU ni modelo real — verifican la interfaz y la inicialización.
"""
import numpy as np
import pytest
from pathlib import Path


def test_query_result_fields():
    """QueryResult tiene los campos requeridos por la spec."""
    from ace_rerun.relocalization import QueryResult
    result = QueryResult(
        image_path=Path("test.jpg"),
        pose_c2w=np.eye(4),
        inlier_count=50,
        success=True,
    )
    assert result.success is True
    assert result.pose_c2w.shape == (4, 4)


def test_query_result_failure_pose_none():
    """pose_c2w puede ser None cuando success=False."""
    from ace_rerun.relocalization import QueryResult
    result = QueryResult(
        image_path=Path("bad.jpg"),
        pose_c2w=None,
        inlier_count=0,
        success=False,
    )
    assert result.success is False
    assert result.pose_c2w is None


def test_pnp_result_success(tmp_path):
    """solve_pose_pnp retorna PnPResult con el contrato correcto."""
    from ace_rerun.relocalization import solve_pose_pnp
    import torch

    # Coordenadas de escena sintéticas (H=60, W=80 output del network)
    sc = torch.zeros(3, 60, 80)
    # Poner coordenadas plausibles en un plano
    sc[0] = torch.linspace(-1, 1, 80).unsqueeze(0).expand(60, -1)
    sc[1] = torch.linspace(-1, 1, 60).unsqueeze(1).expand(-1, 80)
    sc[2] = 2.0  # Z constante

    result = solve_pose_pnp(
        scene_coordinates_3HW=sc,
        focal_length=500.0,
        pp_x=320.0,
        pp_y=240.0,
    )
    # La función debe retornar siempre un PnPResult (no lanzar excepción)
    assert hasattr(result, "pose_4x4")
    assert hasattr(result, "success")
    assert result.pose_4x4.shape == (4, 4)
