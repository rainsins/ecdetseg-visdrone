"""Numerical regression against ONNX Runtime's original GridSample kernel."""
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as H

from deploy.rewrite_grid_sample import rewrite_model


class BilinearParity(unittest.TestCase):
    def test_boundaries_batches_rectangles_and_single_pixels(self):
        rng = np.random.default_rng(42)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        for shape in [(2, 3, 7, 11), (1, 2, 1, 1), (2, 1, 1, 9), (1, 3, 9, 1)]:
            for align in (0, 1):
                with self.subTest(shape=shape, align=align):
                    grid_shape = (shape[0], 13, 7, 2)
                    x = rng.normal(size=shape).astype(np.float32)
                    grid = rng.uniform(-3, 3, grid_shape).astype(np.float32)
                    grid[:, 0, :, :] = np.array([-1, 1, 0, -1.00001, 1.00001, -1000, 1000])[:, None]
                    # Pixel centres and half-pixel borders of a rectangular input.
                    grid[:, 1, 0] = [-1 + 1 / shape[3], -1 + 1 / shape[2]]
                    node = H.make_node("GridSample", ["x", "g"], ["y"],
                                       mode="bilinear", padding_mode="zeros", align_corners=align)
                    m = H.make_model(H.make_graph([node], "sample",
                        [H.make_tensor_value_info("x", T.FLOAT, shape),
                         H.make_tensor_value_info("g", T.FLOAT, grid_shape)],
                        [H.make_tensor_value_info("y", T.FLOAT, [shape[0], shape[1], 13, 7])]),
                        opset_imports=[H.make_opsetid("", 17)], ir_version=9)
                    rewritten, records = rewrite_model(m)
                    self.assertEqual(len(records), 1)
                    self.assertFalse(any(n.op_type == "GridSample" for n in rewritten.graph.node))
                    def run(model):
                        return ort.InferenceSession(model.SerializeToString(), options,
                            providers=["CPUExecutionProvider"]).run(None, {"x": x, "g": grid})[0]
                    np.testing.assert_allclose(run(rewritten), run(m), atol=3e-6, rtol=3e-6)

    def test_rejects_unsupported_mode(self):
        m = H.make_model(H.make_graph([H.make_node("GridSample", ["x", "g"], ["y"], mode="nearest")],
            "bad", [H.make_tensor_value_info("x", T.FLOAT, [1, 1, 2, 2]),
                      H.make_tensor_value_info("g", T.FLOAT, [1, 1, 1, 2])],
            [H.make_tensor_value_info("y", T.FLOAT, [1, 1, 1, 1])]),
            opset_imports=[H.make_opsetid("", 17)], ir_version=9)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            rewrite_model(m)


if __name__ == "__main__":
    unittest.main()
