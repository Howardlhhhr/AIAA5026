import unittest

import numpy as np
import torch

from protocol import (candidate_patch_distances, cls_cosine_distances, image_scores,
                      reference_orders, score_metrics)


class ProtocolTests(unittest.TestCase):
    def test_random_order_prefix_and_shared_first(self):
        cls = np.eye(12, dtype=np.float32)
        for seed in range(11, 21):
            orders = reference_orders(cls_cosine_distances(cls), seed)
            expected = np.random.default_rng(seed).permutation(12)
            np.testing.assert_array_equal(orders["random"], expected[:8])
            self.assertEqual(orders["random"][0], orders["diverse"][0])
            for order in orders.values():
                for shots in (2, 4, 8):
                    self.assertEqual(len(set(order[:shots])), shots)
                self.assertEqual(list(order[:2]), list(order[:4][:2]))
                self.assertEqual(list(order[:4]), list(order[:8][:4]))

    def test_exact_greedy_distance_and_tiebreak(self):
        angles = np.deg2rad([0, 180, 180, 90, -90, 45, -45, 135])
        cls = np.column_stack([np.cos(angles), np.sin(angles)])
        seed = next(seed for seed in range(100) if np.random.default_rng(seed).permutation(8)[0] == 0)
        distances = cls_cosine_distances(cls)
        orders = reference_orders(distances, seed)
        np.testing.assert_array_equal(orders["diverse"][:3], [0, 1, 3])
        chosen = [int(orders["diverse"][0])]
        for index in orders["diverse"][1:]:
            remaining = sorted(set(range(len(cls))) - set(chosen))
            expected = max(remaining, key=lambda candidate: min(distances[candidate, j] for j in chosen))
            self.assertEqual(index, expected)
            chosen.append(int(index))

    def test_chunked_cosine_and_reference_union(self):
        rng = np.random.default_rng(7)
        query = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(3, 4, 6)), dtype=torch.float32), dim=-1)
        bank = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(5, 4, 6)), dtype=torch.float32), dim=-1)
        actual = candidate_patch_distances(query, bank, query_block=5, ref_block=2)
        expected = 1 - np.einsum("qpk,rdk->rqpd", query.numpy(), bank.numpy()).max(axis=3)
        np.testing.assert_allclose(actual, expected, atol=2e-7)
        direct = 1 - np.einsum("qpk,dk->qpd", query.numpy(), bank[[0, 3]].numpy().reshape(-1, 6)).max(axis=2)
        np.testing.assert_allclose(actual[[0, 3]].min(axis=0), direct, atol=2e-7)

    def test_top_three(self):
        scores = image_scores(np.array([[0.1, 0.7, 0.5, 0.6], [0.1, 0.1, 0.9, 0.9]]))
        np.testing.assert_allclose(scores, [0.6, 1.9 / 3])

    def test_roc_ties_and_quantile_threshold_boundary(self):
        positive = [0.9] * 18 + [0.5] * 2
        negative = [0.5, 0.4, 1.0]
        calibration = [0.0] * 37 + [0.5, 0.5, 0.8]
        metrics = score_metrics(positive + negative, [1] * 20 + [0] * 3, calibration)
        self.assertEqual(metrics["auroc"], 39 / 60)
        self.assertEqual(metrics["roc_threshold"], 0.5)
        self.assertEqual(metrics["fpr_at_95"], 2 / 3)
        self.assertEqual(metrics["recall_at_95"], 1)
        self.assertEqual(metrics["calibration_threshold"], 0.5)
        self.assertEqual(metrics["calibrated_fpr"], 1 / 3)
        self.assertEqual(metrics["calibrated_recall"], 0.9)

    def test_linear_calibration_and_strictest_roc_threshold(self):
        metrics = score_metrics(list(range(1, 21)) + [0, 2, 3], [1] * 20 + [0] * 3, list(range(40)))
        self.assertAlmostEqual(metrics["calibration_threshold"], 37.05)
        self.assertEqual(metrics["roc_threshold"], 2)
        self.assertEqual(metrics["recall_at_95"], 0.95)
        self.assertEqual(metrics["fpr_at_95"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
