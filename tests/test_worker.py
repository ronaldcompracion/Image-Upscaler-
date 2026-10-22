from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.engine import InferenceResult
from app.pipeline import ProcessingOptions
from app.worker import ProcessingWorker


class ReusableFakeEngine:
    instances = []
    fail_on_calls: set[int] = set()

    def __init__(self, *_args):
        self.calls = 0
        self.closed = False
        self.recoveries = 0
        self.__class__.instances.append(self)

    def enhance(self, image, scale, strength):
        self.calls += 1
        if self.calls in self.fail_on_calls:
            raise RuntimeError(f"simulated inference failure {self.calls}")
        return InferenceResult(
            image.resize((image.width * scale, image.height * scale)),
            "CPU",
            64,
            (),
        )

    def close(self):
        self.closed = True

    def recover_after_failure(self):
        self.recoveries += 1


class WorkerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.options = ProcessingOptions(
            output_dir=self.root / "output",
            suffix="_enhanced",
            output_format="PNG",
            quality=95,
            scale=2,
            profile="Natural",
            strength=None,
            processing_mode="Balanced",
            cpu_threads=1,
            use_cuda=False,
        )
        ReusableFakeEngine.instances = []
        ReusableFakeEngine.fail_on_calls = set()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _make_input(self, name: str, size: tuple[int, int]) -> Path:
        path = self.root / name
        Image.new("RGB", size, (30, 90, 150)).save(path)
        return path

    def _run_worker(self, files: list[Path], cancel: bool = False):
        worker = ProcessingWorker(files, self.options)
        results = []
        failures = []
        cancelled = []
        finished = []
        worker.item_finished.connect(lambda path, result: results.append((path, result)))
        worker.item_failed.connect(lambda path, error: failures.append((path, error)))
        worker.item_cancelled.connect(cancelled.append)
        worker.finished.connect(lambda: finished.append(True))
        if cancel:
            worker.cancel()
        with patch("app.worker.RealESRGANEngine", ReusableFakeEngine):
            worker.run()
        self.assertEqual(finished, [True])
        return results, failures, cancelled

    def test_ten_consecutive_images_reuse_engine_and_validate_outputs(self):
        inputs = [
            self._make_input(f"image_{index}.png", (index + 2, index + 3))
            for index in range(10)
        ]
        results, failures, cancelled = self._run_worker(inputs)

        self.assertEqual(len(results), 10)
        self.assertEqual(failures, [])
        self.assertEqual(cancelled, [])
        self.assertEqual(len(ReusableFakeEngine.instances), 1)
        engine = ReusableFakeEngine.instances[0]
        self.assertEqual(engine.calls, 10)
        self.assertTrue(engine.closed)
        for index, (path, result) in enumerate(results):
            self.assertEqual(path, str(inputs[index]))
            self.assertTrue(result.destination.is_file())
            with Image.open(result.destination) as saved:
                self.assertEqual(saved.size, ((index + 2) * 2, (index + 3) * 2))
                saved.verify()

    def test_repeated_same_input_uses_distinct_valid_output_paths(self):
        source = self._make_input("same.png", (5, 4))
        results, failures, _ = self._run_worker([source, source, source])

        self.assertEqual(failures, [])
        self.assertEqual(len(results), 3)
        destinations = [result.destination for _, result in results]
        self.assertEqual(len(set(destinations)), 3)
        for destination in destinations:
            with Image.open(destination) as saved:
                saved.load()
                self.assertEqual(saved.size, (10, 8))

    def test_three_separate_worker_runs_complete_successfully(self):
        source = self._make_input("separate.png", (5, 4))
        destinations = []

        for _ in range(3):
            results, failures, cancelled = self._run_worker([source])
            self.assertEqual(len(results), 1)
            self.assertEqual(failures, [])
            self.assertEqual(cancelled, [])
            destinations.append(results[0][1].destination)

        self.assertEqual(len(set(destinations)), 3)
        self.assertEqual(len(ReusableFakeEngine.instances), 3)
        self.assertTrue(all(engine.closed for engine in ReusableFakeEngine.instances))

    def test_inference_failure_does_not_block_later_images(self):
        ReusableFakeEngine.fail_on_calls = {1}
        inputs = [
            self._make_input("fails.png", (3, 4)),
            self._make_input("succeeds.png", (4, 5)),
        ]
        results, failures, _ = self._run_worker(inputs)

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], str(inputs[0]))
        self.assertIn("simulated inference failure", failures[0][1])
        self.assertEqual([path for path, _ in results], [str(inputs[1])])
        self.assertTrue(results[0][1].destination.is_file())
        self.assertEqual(ReusableFakeEngine.instances[0].calls, 2)
        self.assertEqual(ReusableFakeEngine.instances[0].recoveries, 1)

    def test_corrupt_input_does_not_block_following_valid_image(self):
        corrupt = self.root / "corrupt.png"
        corrupt.write_bytes(b"not an image")
        valid = self._make_input("valid.png", (4, 5))
        results, failures, _ = self._run_worker([corrupt, valid])

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][0], str(corrupt))
        self.assertEqual([path for path, _ in results], [str(valid)])
        self.assertTrue(results[0][1].destination.is_file())

    def test_output_save_failure_does_not_block_following_image(self):
        from app.image_io import save_image as real_save_image

        first = self._make_input("first.png", (3, 4))
        second = self._make_input("second.png", (4, 5))
        calls = []

        def fail_first_save(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise PermissionError("simulated output permission failure")
            return real_save_image(*args, **kwargs)

        with patch("app.pipeline.save_image", side_effect=fail_first_save):
            results, failures, _ = self._run_worker([first, second])

        self.assertEqual(len(failures), 1)
        self.assertIn("simulated output permission failure", failures[0][1])
        self.assertEqual([path for path, _ in results], [str(second)])
        self.assertTrue(results[0][1].destination.is_file())

    def test_model_initialization_failure_still_finishes_worker(self):
        source = self._make_input("valid.png", (3, 4))
        worker = ProcessingWorker([source], self.options)
        failures = []
        finished = []
        worker.item_failed.connect(lambda path, error: failures.append((path, error)))
        worker.finished.connect(lambda: finished.append(True))

        with patch(
            "app.worker.RealESRGANEngine",
            side_effect=RuntimeError("simulated model initialization failure"),
        ):
            worker.run()

        self.assertEqual(len(failures), 1)
        self.assertIn("simulated model initialization failure", failures[0][1])
        self.assertEqual(finished, [True])
        results, later_failures, _ = self._run_worker([source])
        self.assertEqual(len(results), 1)
        self.assertEqual(later_failures, [])

    def test_cancelled_worker_reports_all_jobs_and_releases_state(self):
        inputs = [
            self._make_input("first.png", (3, 4)),
            self._make_input("second.png", (4, 5)),
        ]
        results, failures, cancelled = self._run_worker(inputs, cancel=True)

        self.assertEqual(results, [])
        self.assertEqual(failures, [])
        self.assertEqual(cancelled, [str(path) for path in inputs])
        self.assertEqual(ReusableFakeEngine.instances, [])


if __name__ == "__main__":
    unittest.main()
