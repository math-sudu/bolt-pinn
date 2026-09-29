import copy
import json
import unittest

import numpy as np

from bolt_pinn.condition_model import decode
from scripts.run_condition_transfer import load_curves, ROOT, fit_schedule, objective_weights


class ConditionTransferTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/"configs/condition_transfer.json").read_text())
        self.theta = np.array(self.config["initial_parameters"])
        self.records = json.loads((ROOT/"data/model_inputs.json").read_text())["records"]

    def test_prediction_material_does_not_use_held_out_response(self):
        record = next(r for r in self.records if r["role"] == "prediction")
        changed = copy.deepcopy(record)
        changed.update(image_peak_kN=9999, image_peak_displacement_mm=9999,
                       reported_BQ="9999", trial_id="anonymous", role="calibration")
        self.assertEqual(decode(self.theta, record, self.config), decode(self.theta, changed, self.config))

    def test_length_series_shares_material_and_changes_geometry(self):
        records = [r for r in self.records if r["experiment_id"] == "length_transfer"]
        decoded = [decode(self.theta, r, self.config) for r in records]
        self.assertTrue(all(item[0] == decoded[0][0] for item in decoded))
        self.assertEqual(len({item[1].length_mm for item in decoded}), 5)

    def test_calibration_samples_stay_within_source_support(self):
        curves = load_curves(ROOT, "calibration")
        self.assertEqual(len(curves), 9)
        for curve in curves:
            self.assertEqual(curve["record"]["role"], "calibration")
            schedule = fit_schedule(curve)
            self.assertEqual(schedule[0], 0)
            self.assertGreaterEqual(schedule[1], curve["x"][0])
            self.assertLessEqual(schedule[-1], curve["x"][-1])
            # A constant relative error contributes the same total at every curve.
            peak = curve["record"]["image_peak_kN"]
            weights = objective_weights(schedule, peak)
            self.assertAlmostEqual(float(np.sum((weights*peak)**2)), 1.0, 12)

    def test_adapted_length_uses_geometry_and_remains_admissible(self):
        config = {**self.config, "length_model": "relaxation_contact"}
        theta = np.r_[self.theta[:-1], np.log(100), self.theta[-1]]
        records = [r for r in self.records if r["experiment_id"] == "length_transfer"]
        materials = []
        for record in records:
            material, _, _ = decode(theta, record, config)
            materials.append((record["embedment_mm"], material))
            self.assertEqual(material.contact_length, record["embedment_mm"])
            self.assertGreater(material.stiffness, material.softening_modulus)
            fixed, _, _ = decode(theta, record, {**config, "contact_evolution": False})
            self.assertEqual(material.stiffness, fixed.stiffness)
            self.assertEqual(material.critical_slip, fixed.critical_slip)
            changed = {**record, "image_peak_kN": 9999, "reported_BQ": 0, "trial_id": "unused"}
            self.assertEqual(decode(theta, record, config), decode(theta, changed, config))
        materials.sort(key=lambda item: item[0])
        self.assertLess(materials[0][1].peak, materials[-1][1].peak)
        self.assertLess(materials[0][1].critical_slip, materials[-1][1].critical_slip)

    def test_shared_shape_mapping_and_condition_roles(self):
        config = json.loads((ROOT/"configs/length_transfer_shape.json").read_text())
        theta = np.array(config["initial_parameters"])
        curves = load_curves(ROOT, "calibration", config["experiment_ids"])
        self.assertEqual({c["record"]["embedment_mm"] for c in curves}, {100, 300, 500})
        predictions = load_curves(ROOT, "prediction", config["experiment_ids"])
        self.assertEqual({c["record"]["embedment_mm"] for c in predictions}, {200, 400})
        record = curves[0]["record"]
        for length in np.linspace(100, 500, 17):
            condition = {**record, "embedment_mm": float(length)}
            material, _, observation = decode(theta, condition, config)
            self.assertGreater(material.stiffness, material.softening_modulus)
            self.assertGreater(material.softening_onset, 0)
            changed = {**condition, "image_peak_kN": 1e6, "image_peak_displacement_mm": 999,
                       "trial_id": "unused", "role": "prediction", "reported_BQ": 999}
            self.assertEqual(decode(theta, condition, config), decode(theta, changed, config))
        short = decode(theta, {**record, "embedment_mm": 100}, config)[0]
        long = decode(theta, {**record, "embedment_mm": 500}, config)[0]
        self.assertGreater(short.softening_onset, long.softening_onset)
        self.assertLess(short.mobilization_slip, long.mobilization_slip)


if __name__ == "__main__":
    unittest.main()
