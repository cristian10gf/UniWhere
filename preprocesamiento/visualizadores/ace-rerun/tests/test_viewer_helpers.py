# tests/test_viewer_helpers.py
import numpy as np
import pytest
from ace_rerun.viewer import error_to_color, get_pinhole_params, _resize_for_rerun


def test_error_to_color_good_pose():
    color = error_to_color(0.0, 0.0)
    assert color[1] == 255  # verde
    assert color[0] == 0    # no rojo


def test_error_to_color_bad_pose():
    color = error_to_color(10.0, 1.0)  # por encima de umbrales
    assert color[0] == 255  # rojo
    assert color[1] == 0    # no verde


def test_get_pinhole_params_matrix():
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    fx, fy, cx, cy = get_pinhole_params(K)
    assert fx == 500.0
    assert cx == 320.0


def test_get_pinhole_params_scalar():
    fx, fy, cx, cy = get_pinhole_params(600.0)
    assert fx == 600.0
    assert cx == 320.0


def test_get_pinhole_params_none():
    assert get_pinhole_params(None) is None


def test_resize_for_rerun_wide_image():
    img = np.zeros((480, 1280, 3), dtype=np.uint8)
    resized = _resize_for_rerun(img, max_width=640)
    assert resized.shape[1] == 640
    assert resized.shape[0] == 240  # proporción mantenida


def test_resize_for_rerun_small_image():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    resized = _resize_for_rerun(img, max_width=640)
    assert resized.shape == (240, 320, 3)  # no se upscalea
