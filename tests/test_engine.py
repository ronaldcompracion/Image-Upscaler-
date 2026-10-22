from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.engine import RealESRGANEngine


class EngineValidationTests(unittest.TestCase):
    def test_rejects_unsupported_scale_before_loading_torch(self):
        engine = RealESRGANEngine("cpu", "Balanced", 2)
        with self.assertRaisesRegex(ValueError, "Scale must be"):
            engine.enhance(Image.new("RGB", (2, 2)), 3, 0.8)

    def test_rejects_invalid_strength(self):
        engine = RealESRGANEngine("cpu", "Balanced", 2)
        with self.assertRaisesRegex(ValueError, "strength"):
            engine.enhance(Image.new("RGB", (2, 2)), 2, 1.1)

    @patch("torch.set_num_threads")
    def test_uses_bgr_engine_input_and_reuses_loaded_model(self, _set_threads):
        class FakeUpscaler:
            tile_size = 0

            def __init__(self):
                self.input = None

            def enhance(self, image, outscale):
                self.input = image.copy()
                output = np.empty((image.shape[0] * 4, image.shape[1] * 4, 3), dtype=np.uint8)
                output[:] = (30, 20, 10)
                return output, "RGB"

        engine = RealESRGANEngine("cpu", "Balanced", 2)
        fake_upscaler = FakeUpscaler()
        created = []

        def create_upscaler(device, tile_size):
            created.append((device, tile_size))
            return fake_upscaler

        engine._create_upscaler = create_upscaler
        source = Image.new("RGB", (3, 2), (10, 20, 30))
        first = engine.enhance(source, 2, 1.0)
        second = engine.enhance(source, 4, 1.0)
        self.assertEqual(tuple(fake_upscaler.input[0, 0]), (30, 20, 10))
        self.assertEqual(first.image.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(first.image.size, (6, 4))
        self.assertEqual(second.image.size, (12, 8))
        self.assertEqual(len(created), 1)
        engine.close()


if __name__ == "__main__":
    unittest.main()
