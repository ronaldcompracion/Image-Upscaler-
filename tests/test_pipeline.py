from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, UnidentifiedImageError

from app.engine import InferenceResult
from app.pipeline import ProcessingOptions, process_image


class FakeEngine:
    def __init__(self, *_args):
        pass

    def enhance(self, image, scale, strength):
        output = image.resize((image.width * scale, image.height * scale))
        return InferenceResult(output, "CPU", 64, ())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source.png"
        Image.new("RGBA", (8, 6), (100, 50, 20, 96)).save(self.source)
        self.options = ProcessingOptions(
            output_dir=self.root / "out",
            suffix="_enhanced",
            output_format="PNG",
            quality=95,
            scale=2,
            profile="Natural",
            strength=None,
            processing_mode="Balanced",
            cpu_threads=2,
            use_cuda=False,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @patch("app.pipeline.RealESRGANEngine", FakeEngine)
    def test_processes_and_validates_output_dimensions_and_alpha(self):
        result = process_image(self.source, self.options)
        self.assertEqual((result.width, result.height), (16, 12))
        with Image.open(result.destination) as output:
            self.assertEqual(output.mode, "RGBA")
            self.assertEqual(output.size, (16, 12))
            self.assertEqual(output.getpixel((0, 0))[3], 96)
        self.assertTrue(self.source.is_file())

    def test_cancellation_before_processing_does_not_write_output(self):
        with self.assertRaises(InterruptedError):
            process_image(self.source, self.options, lambda: True)
        self.assertFalse((self.root / "out").exists())

    @patch("app.pipeline.RealESRGANEngine", FakeEngine)
    def test_restores_grayscale_output_mode(self):
        gray_source = self.root / "gray.png"
        Image.new("L", (4, 3), 90).save(gray_source)
        result = process_image(gray_source, self.options)
        with Image.open(result.destination) as output:
            self.assertEqual(output.mode, "L")
            self.assertEqual(output.size, (8, 6))

    @patch("app.pipeline.RealESRGANEngine", FakeEngine)
    def test_rejects_and_removes_corrupt_saved_output(self):
        from app.pipeline import save_image as real_save_image

        def write_corrupt_output(*args, **kwargs):
            destination = real_save_image(*args, **kwargs)
            destination.write_bytes(b"corrupt")
            return destination

        with patch("app.pipeline.save_image", side_effect=write_corrupt_output):
            with self.assertRaises(UnidentifiedImageError):
                process_image(self.source, self.options)
        self.assertEqual(list((self.root / "out").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
