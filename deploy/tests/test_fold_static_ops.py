import unittest
import numpy as np
import onnx
from onnx import helper as H, TensorProto as T, numpy_helper as N
import onnxruntime as ort
from deploy.fold_static_ops import fold_model

class ConstantFolding(unittest.TestCase):
    def test_constants_fold_but_dynamic_masks_remain(self):
        nodes=[H.make_node("Flatten",["angle"],["flat"],axis=0),
               H.make_node("Reshape",["flat","shape"],["restored"]),
               H.make_node("Sin",["restored"],["s"]),
               H.make_node("Cos",["angle"],["c"]),
               H.make_node("Equal",["angle","angle"],["eq"]),
               H.make_node("Where",["eq","s","c"],["w"]),
               H.make_node("Equal",["x","angle"],["deq"]),
               H.make_node("Where",["deq","w","x"],["y"])]
        m=H.make_model(H.make_graph(nodes,"test",[H.make_tensor_value_info("x",T.FLOAT,[3])],
            [H.make_tensor_value_info("y",T.FLOAT,[3])],
            [N.from_array(np.array([0,1,2],np.float32),"angle"),
             N.from_array(np.array([3],np.int64),"shape")]),
            opset_imports=[H.make_opsetid("",17)],ir_version=9)
        result,report=fold_model(m)
        self.assertEqual(len(report["folded"]),4)
        self.assertEqual([n.op_type for n in result.graph.node],["Equal","Where"])
        opts=ort.SessionOptions();opts.intra_op_num_threads=1
        for x in [np.array([0,1,2],np.float32),np.array([3,4,5],np.float32)]:
            a,b=[ort.InferenceSession(v.SerializeToString(),opts,providers=["CPUExecutionProvider"]).run(None,{"x":x})[0]
                 for v in [m,result]]
            np.testing.assert_array_equal(a,b)
        # An initializer also exposed as input can be overridden at runtime.
        m.graph.input.append(H.make_tensor_value_info("angle",T.FLOAT,[3]))
        _,report=fold_model(m)
        self.assertEqual(report["folded"],[])

if __name__=="__main__": unittest.main()
