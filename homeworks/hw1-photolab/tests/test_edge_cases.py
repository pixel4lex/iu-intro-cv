"""Граничные случаи для работы на фотографиях другого размера и цвета."""
import itertools

import cv2
import numpy as np
import pytest

import photolab as pl


def test_sample_red_wraps_and_excludes_green():
    hsv = np.zeros((20, 40, 3), np.uint8)
    hsv[:] = (179, 220, 180)
    hsv[:, 20:] = (1, 220, 180)
    sample = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    lo, hi = pl.hsv_box_from(sample)
    assert lo[0] > hi[0]
    assert np.all(pl.color_mask(sample, lo, hi, 0, 0) == 255)
    green = np.full_like(sample, (20, 180, 20))
    assert not pl.color_mask(green, lo, hi, 0, 0).any()


def test_fill_hole_with_object_touching_frame():
    image = np.full((80, 100, 3), 80, np.uint8)
    image[10:70, :50] = (200, 40, 235)
    image[30:50, 10:30] = 255
    bg = np.zeros((17, 25, 3), np.uint8)
    comp, mask, stats = pl.chroma_key(image, bg, (140, 120, 40), (170, 255, 255), 0, 0, 0)
    assert mask[40, 20] == 255 and mask[0, 0] == 0 and mask[40, 90] == 0
    assert stats["holes_before_fill"] == 1 and stats["holes"] == 0
    assert np.array_equal(comp[40, 20], image[40, 20])
    with pytest.raises(ValueError, match="объект не найден"):
        pl.chroma_key(np.zeros_like(image), bg, (140, 120, 40), (170, 255, 255))


@pytest.mark.parametrize("angle", [0, 90, -90, 180, 270, -137])
def test_rotation_keeps_support_and_exact_quarter_turns(angle):
    h, w = 91, 143
    source = np.full((h, w, 3), 255, np.uint8)
    result = pl.rotate_keep_all(source, angle, border=(0, 0, 0))
    kept = (result[..., 0] > 127).sum()
    assert abs(kept - h * w) / (h * w) < 0.01
    if angle % 90 == 0:
        expected = (h, w) if angle % 180 == 0 else (w, h)
        assert result.shape[:2] == expected
        assert np.all(result == 255)


def test_diamond_corner_ties_and_invalid_quads():
    diamond = np.float32([[50, 0], [100, 50], [50, 100], [0, 50]])
    for indices in itertools.permutations(range(4)):
        assert np.array_equal(pl.order_corners(diamond[list(indices)]), diamond)
    for invalid in ([[0, 0], [0, 0], [20, 20], [0, 20]],
                    [[0, 0], [10, 0], [20, 0], [30, 0]],
                    [[0, 0], [100, 0], [100, 100], [80, 20]]):
        with pytest.raises(ValueError):
            pl.order_corners(invalid)


def test_balance_missing_channel_is_finite():
    image = np.zeros((30, 20, 3), np.uint8)
    image[..., 1:] = (60, 120)
    for method in ("grey-world", "white-patch"):
        result, gains = pl.white_balance(image, method)
        assert np.isfinite(gains).all() and np.all(result[..., 0] == 0)
        black, gains = pl.white_balance(np.zeros_like(image), method)
        assert not black.any() and np.isfinite(gains).all()


def test_sheet_accepts_transparency_and_rejects_empty_input():
    rgba = np.zeros((30, 40, 4), np.uint8)
    rgba[..., 2] = 255
    rgba[:, 20:, 3] = 255
    result = pl.contact_sheet([rgba], ["transparent"], cols=1, cell=(40, 30))
    assert np.array_equal(result[8, 8], [255, 255, 255])
    assert np.array_equal(result[8, 47], [0, 0, 255])
    with pytest.raises(ValueError):
        pl.contact_sheet([], [])


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_scale_and_gamma(value):
    image = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(ValueError):
        pl.resize_aa(image, value)
    with pytest.raises(ValueError):
        pl.gamma_correct(image, value)
