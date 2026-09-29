import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_model_inputs import ROOT, prepare


class ModelInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "data").mkdir()
        for name in ["modelling_assumptions.json", "test_registry.csv",
                     "pressure_curves_digitized.csv", "length_curves_digitized.csv"]:
            shutil.copyfile(ROOT / "data" / name, self.root / "data" / name)
        self.path = self.root / "data/modelling_assumptions.json"
        self.assumptions = json.loads(self.path.read_text(encoding="utf-8"))

    def save_assumptions(self):
        self.path.write_text(json.dumps(self.assumptions), encoding="utf-8")

    def test_rejects_curve_leakage_between_calibration_and_prediction(self):
        experiment = self.assumptions["experiments"][0]
        experiment["calibration_records"].append(experiment["prediction_records"][0])
        self.save_assumptions()
        with self.assertRaisesRegex(ValueError, "Calibration/prediction overlap"):
            prepare(self.root)

    def test_rejects_endpoint_clamping_beyond_the_digitized_trace(self):
        self.assumptions["experiments"][0]["sample_stop_mm"] = 100.0
        self.save_assumptions()
        with self.assertRaisesRegex(ValueError, "Sampling would extrapolate"):
            prepare(self.root)


if __name__ == "__main__":
    unittest.main()
