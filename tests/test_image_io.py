from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.image_io import load_image, save_image


class ImageIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_loads_supported_formats(self):
        for image_format, extension in (
            ("JPEG", ".jpg"),
            ("PNG", ".png"),
            ("WEBP", ".webp"),
            ("BMP", ".bmp"),
            ("TIFF", ".tiff"),
        ):
            with self.subTest(image_format=image_format):
                source = self.root / f"input{extension}"
                Image.new("RGB", (9, 7), (20, 50, 90)).save(
                    source, format=image_format
                )
                loaded = load_image(source)
                self.assertEqual(loaded.image.size, (9, 7))
                self.assertEqual(loaded.image.mode, "RGB")

    def test_preserves_transparency_and_grayscale_on_load(self):
        transparent = self.root / "transparent.png"
        Image.new("RGBA", (5, 3), (10, 20, 30, 70)).save(transparent)
        self.assertEqual(load_image(transparent).image.mode, "RGBA")
        grayscale = self.root / "gray.png"
        Image.new("L", (4, 2), 100).save(grayscale)
        self.assertEqual(load_image(grayscale).image.mode, "L")

    def test_applies_exif_orientation(self):
        source = self.root / "oriented.jpg"
        image = Image.new("RGB", (6, 3), "white")
        exif = Image.Exif()
        exif[274] = 6
        image.save(source, exif=exif)
        self.assertEqual(load_image(source).image.size, (3, 6))

    def test_saves_with_alpha_and_avoids_overwriting(self):
        image = Image.new("RGBA", (3, 2), (10, 20, 30, 40))
        source = self.root / "original.png"
        image.save(source)
        output = self.root / "output"
        first = save_image(
            image, output, source, "_enhanced", "PNG", 95
        )
        second = save_image(
            image, output, source, "_enhanced", "PNG", 95
        )
        self.assertEqual(first.name, "original_enhanced.png")
        self.assertEqual(second.name, "original_enhanced_2.png")
        self.assertEqual(Image.open(first).getpixel((0, 0))[3], 40)
        self.assertEqual(Image.open(source).getpixel((0, 0))[3], 40)

    def test_jpeg_export_flattens_alpha(self):
        image = Image.new("RGBA", (4, 4), (20, 40, 60, 0))
        saved = save_image(
            image, self.root / "jpg", self.root / "input.png", "_enhanced", "JPEG", 90
        )
        with Image.open(saved) as result:
            self.assertEqual(result.mode, "RGB")

    def test_rejects_unknown_input(self):
        with self.assertRaisesRegex(ValueError, "Unsupported image format"):
            load_image(self.root / "unknown.raw")

    def test_rejects_path_traversal_in_output_suffix(self):
        with self.assertRaisesRegex(ValueError, "Filename suffix"):
            save_image(
                Image.new("RGB", (2, 2)),
                self.root / "output",
                self.root / "input.png",
                "..\\outside",
                "PNG",
                95,
            )


if __name__ == "__main__":
    unittest.main()
