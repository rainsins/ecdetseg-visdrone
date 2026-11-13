"""Compile the real callback against SDK headers, compare with ONNX Runtime.
Set EC_RKNN_INCLUDE_DIR to the directory containing both official headers.
Mocked attribute/memory delivery does not validate the board callback ABI.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import onnxruntime as ort
from onnx import helper as H, TensorProto as T

ROOT = Path(__file__).resolve().parents[2]

class CpuGridSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sdk = os.environ.get("EC_RKNN_INCLUDE_DIR")
        if not sdk:
            raise unittest.SkipTest("set EC_RKNN_INCLUDE_DIR to official RKNN headers")
        cls.tmp = tempfile.TemporaryDirectory(prefix="ec-grid-cpu-", dir="/tmp/opencode")
        cls.addClassCleanup(cls.tmp.cleanup)
        lib = Path(cls.tmp.name) / "sample.so"
        subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            "-shared", "-fPIC", f"-I{sdk}", f"-I{ROOT / 'deploy/cpp'}",
            str(ROOT / "deploy/cpp/grid_sample_cpu.cpp"),
            str(ROOT / "deploy/cpp/tests/grid_sample_bridge.cpp"), "-o", str(lib)], check=True)
        cls.lib = ctypes.CDLL(str(lib))
        cls.call = cls.lib.sample
        cls.call.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*9
        cls.call.restype = ctypes.c_int

    def run_sample(self, x, grid, align, layout, fault=0):
        n,c,h,w = x.shape
        _,oh,ow,_ = grid.shape
        a = np.ascontiguousarray(x.transpose(0,2,3,1) if layout & 1 else x)
        b = np.ascontiguousarray(grid if layout & 2 else grid.transpose(0,3,1,2))
        y = np.empty((n,oh,ow,c) if layout & 4 else (n,c,oh,ow), dtype=np.float32)
        ret = self.call(a.ctypes.data,b.ctypes.data,y.ctypes.data,n,c,h,w,oh,ow,align,layout,fault)
        return ret, y.transpose(0,3,1,2) if layout & 4 else y

    def test_ort_parity(self):
        rng = np.random.default_rng(42)
        options = ort.SessionOptions(); options.intra_op_num_threads = 2
        for shape in [(2,3,7,11),(1,2,1,1),(2,1,1,9),(1,3,9,1),(8,24,80,80)]:
            for align in (0,1):
                x = rng.normal(size=shape).astype(np.float32)
                grid = rng.uniform(-3,3,(shape[0],13,7,2)).astype(np.float32)
                grid[:,0,:,:] = np.array([-1,1,0,-1.00001,1.00001,-1000,1000])[:,None]
                model = H.make_model(H.make_graph([H.make_node("GridSample", ["x","g"],["y"],
                    mode="bilinear",padding_mode="zeros",align_corners=align)],
                    "sample",[H.make_tensor_value_info("x",T.FLOAT,shape),
                    H.make_tensor_value_info("g",T.FLOAT,grid.shape)],
                    [H.make_tensor_value_info("y",T.FLOAT,[shape[0],shape[1],13,7])]),
                    opset_imports=[H.make_opsetid("",17)],ir_version=9)
                ref = ort.InferenceSession(model.SerializeToString(),options,
                    providers=["CPUExecutionProvider"]).run(None,{"x":x,"g":grid})[0]
                for layout in range(8):
                    with self.subTest(shape=shape,align=align,layout=layout):
                        ret,y = self.run_sample(x,grid,align,layout)
                        self.assertEqual(ret,0)
                        np.testing.assert_allclose(y,ref,rtol=3e-6,atol=3e-6)

    def test_rejects_unsupported_representation(self):
        x = np.ones((1,1,2,2),np.float32)
        grid = np.zeros((1,3,4,2),np.float32)
        for fault in range(1,8):
            with self.subTest(fault=fault):
                self.assertNotEqual(self.run_sample(x,grid,0,2,fault)[0],0)
        grid[0,0,0,0] = np.nan
        self.assertNotEqual(self.run_sample(x,grid,0,2)[0],0)

    def test_profile_counts_and_disable(self):
        self.lib.profile_enable.argtypes = [ctypes.c_int]
        self.lib.profile_read.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
        values = (ctypes.c_uint64 * 4)()
        x = np.ones((1,3,7,9),np.float32)
        grid = np.zeros((1,300,3,2),np.float32)
        self.lib.profile_enable(1)
        try:
            self.assertEqual(self.run_sample(x,grid,0,2,9)[0],0)
            self.assertNotEqual(self.run_sample(x,grid,0,2,1)[0],0)
            self.lib.profile_read(values)
            self.assertEqual(list(values)[:2],[2,1])
            self.assertGreater(values[3],0)
            self.assertGreaterEqual(values[2],values[3])
        finally:
            self.lib.profile_enable(0)
        self.assertEqual(self.run_sample(x,grid,0,2,9)[0],0)
        self.lib.profile_read(values)
        self.assertEqual(list(values),[0,0,0,0])

    def test_board_abi_and_width_two(self):
        rng = np.random.default_rng(81)
        options = ort.SessionOptions(); options.intra_op_num_threads = 2
        # Actual board scales/point counts plus standard NCHW OW=2, including
        # [N,2,2,2] ambiguity: explicit layout wins when output shape agrees.
        for n,c,h,w,oh,ow in [(8,24,s,s,300,p) for s in (80,40,20) for p in (3,6)] + [(1,3,7,9,5,2),(1,3,7,9,2,2)]:
            x = rng.normal(size=(n,c,h,w)).astype(np.float32)
            grid = rng.uniform(-1.2,1.2,(n,oh,ow,2)).astype(np.float32)
            for align in (0,1):
                model = H.make_model(H.make_graph([H.make_node("GridSample",["x","g"],["y"],
                    mode="bilinear",padding_mode="zeros",align_corners=align)],
                    "abi",[H.make_tensor_value_info("x",T.FLOAT,x.shape),
                    H.make_tensor_value_info("g",T.FLOAT,grid.shape)],
                    [H.make_tensor_value_info("y",T.FLOAT,[n,c,oh,ow])]),
                    opset_imports=[H.make_opsetid("",17)],ir_version=9)
                ref = ort.InferenceSession(model.SerializeToString(),options,
                    providers=["CPUExecutionProvider"]).run(None,{"x":x,"g":grid})[0]
                cases = [(0,8),(2,8)]
                if oh == 300: cases += [(2,9),(2,10)]
                for layout,fault in cases:
                    with self.subTest(shape=x.shape,grid=grid.shape,align=align,layout=layout,fault=fault):
                        ret,y = self.run_sample(x,grid,align,layout,fault)
                        self.assertEqual(ret,0)
                        np.testing.assert_allclose(y,ref,rtol=3e-6,atol=3e-6)
                if oh == 300:
                    self.assertNotEqual(self.run_sample(x,grid,align,2,11)[0],0)

if __name__ == "__main__":
    unittest.main()
