#!/usr/bin/env python3
"""Bounded RKNN compiler probes; activation calibration is diagnostic, not AP evidence."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def prepare(source, image, dest):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper as H, numpy_helper
    from PIL import Image
    from deploy.rewrite_grid_sample import rewrite_model
    dest.mkdir(parents=True, exist_ok=False)
    model = onnx.shape_inference.infer_shapes(onnx.load(source))
    nodes = list(model.graph.node)
    producers = {v:i for i,n in enumerate(nodes) for v in n.output}
    inits = {v.name:v for v in model.graph.initializer}
    graph_inputs = {v.name for v in model.graph.input}
    constant = set(inits) - graph_inputs
    # No control flow/random ops in the extraction path. Only use proven static
    # parents; all other boundary values are captured from the real model.
    from deploy.fold_static_ops import PURE
    for n in nodes:
        if not n.domain and n.op_type in PURE and all(not v or v in constant for v in n.input):
            constant.update(n.output)
    grids = [i for i,n in enumerate(nodes) if n.op_type == 'GridSample']
    if not grids:
        raise ValueError('source must retain GridSample')
    cross = nodes[grids[0]].name.rsplit('/',1)[0] + '/'
    layer = cross.rstrip('/').rsplit('/',1)[0] + '/'
    groups = {'sample':{grids[0]},
              'cross':{i for i,n in enumerate(nodes) if n.name.startswith(cross)},
              'decoder':{i for i,n in enumerate(nodes) if n.name.startswith(layer)}}
    specs = {}
    for scope,selected in groups.items():
        def add_constant(v):
            if v not in constant or v in inits or v not in producers:
                return
            i = producers[v]
            if i in selected:
                return
            selected.add(i)
            for parent in nodes[i].input:
                add_constant(parent)
        for i in list(selected):
            for v in nodes[i].input:
                add_constant(v)
        produced = {v for i in selected for v in nodes[i].output}
        inputs = sorted({v for i in selected for v in nodes[i].input
                         if v and v not in produced and v not in inits})
        outside = {v for i,n in enumerate(nodes) if i not in selected for v in n.input}
        outside.update(v.name for v in model.graph.output)
        outputs = sorted(produced & outside)
        specs[scope] = (selected,inputs,outputs)
    names = sorted({v for _,ins,outs in specs.values() for v in ins+outs})
    infos = {v.name:v for v in [*model.graph.input,*model.graph.value_info,*model.graph.output]}
    probe = copy.deepcopy(model)
    del probe.graph.output[:]
    probe.graph.output.extend(infos[v] for v in names)
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(probe.SerializeToString(),options,providers=['CPUExecutionProvider'])
    rgb = np.asarray(Image.open(image).convert('RGB').resize((640,640),Image.Resampling.BILINEAR),dtype=np.float32)
    x = ((rgb - np.array([123.675,116.28,103.53],np.float32)) /
         np.array([58.395,57.12,57.375],np.float32)).transpose(2,0,1)[None]
    values = dict(zip(names, session.run(names,{'images':x})))
    def info(v):
        a = values[v]
        if not np.isfinite(a).all():
            raise ValueError('nonfinite captured boundary: '+v)
        return H.make_tensor_value_info(v,H.np_dtype_to_tensor_dtype(a.dtype),a.shape)
    manifest = {'source':str(source.resolve()),'source_sha256':sha256(source),
                'calibration_image':str(image.resolve()),'calibration_image_sha256':sha256(image),
                'purpose':'compiler isolation only; one real activation sample; not accuracy calibration',
                'cases':{}}
    for scope,(selected,inputs,outputs) in specs.items():
        folder = dest / scope; folder.mkdir()
        used = {v for i in selected for v in nodes[i].input}
        sub = H.make_model(H.make_graph([nodes[i] for i in sorted(selected)],scope,
            [info(v) for v in inputs],[info(v) for v in outputs],
            [v for k,v in inits.items() if k in used]),
            opset_imports=list(model.opset_import),ir_version=model.ir_version)
        sub = onnx.shape_inference.infer_shapes(sub)
        onnx.checker.check_model(sub)
        rewritten,_ = rewrite_model(sub)
        parity = {}
        for variant,m in [('original',sub),('gather',rewritten)]:
            path = folder / (variant+'.onnx'); onnx.save(m,path)
            actual = ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider']).run(
                outputs,{v:values[v] for v in inputs})
            diffs = []
            for v,a in zip(outputs,actual):
                np.testing.assert_allclose(a,values[v],atol=2e-3,rtol=1e-4)
                diffs.append(float(np.max(np.abs(a-values[v]))))
            parity[variant] = {'nodes':len(m.graph.node),'sha256':sha256(path),'max_abs_diff':diffs}
        files = []
        for i,v in enumerate(inputs):
            p = folder / ('input_%d.npy'%i); np.save(p,values[v]); files.append(str(p.resolve()))
        (folder/'dataset.txt').write_text(' '.join(files)+'\n')
        manifest['cases'][scope] = {'inputs':[{ 'name':v,'shape':list(values[v].shape),
            'dtype':str(values[v].dtype)} for v in inputs],'outputs':outputs,'parity':parity}
    write_json(dest/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)

def worker(a):
    from rknn.api import RKNN
    import importlib.metadata
    print('toolkit',importlib.metadata.version('rknn-toolkit2'),flush=True)
    r = RKNN(verbose=True)
    def ok(ret):
        if ret != 0:
            raise RuntimeError('RKNN returned '+str(ret))
    try:
        cfg = dict(target_platform='rk3588',optimization_level=a.optimization,
                   quantized_dtype='w8a8',quantized_method='channel',
                   quantized_algorithm='normal',auto_hybrid_cos_thresh=0.9999)
        if a.target != 'default': cfg['op_target'] = {'GridSample':a.target}
        print('CONFIG',json.dumps(cfg),flush=True)
        ok(r.config(**cfg))
        print('LOAD',flush=True); ok(r.load_onnx(model=str(a.model)))
        print('BUILD',flush=True)
        # Internal activations are already normalized. No image mean/std here.
        ok(r.build(do_quantization=a.quantize,
                   dataset=str(a.dataset) if a.quantize else None,auto_hybrid=a.quantize))
        print('EXPORT',flush=True); ok(r.export_rknn(str(a.output)))
        print('EXPORTED',flush=True)
    finally:
        r.release()

def bounded(command, folder, timeout, rss_mib):
    import psutil
    folder.mkdir(parents=True,exist_ok=False)
    start = time.monotonic(); peak = 0; cpu = 0.0; reason = None
    p = None
    def kill_group():
        if p is not None:
            try: os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError: pass
    try:
        with (folder/'build.log').open('w') as log, (folder/'resources.jsonl').open('w') as metrics:
            env = dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            p = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                 env=env,start_new_session=True)
            root = psutil.Process(p.pid)
            while p.poll() is None:
                rss = 0; current_cpu = 0.0
                try: processes = [root]+root.children(recursive=True)
                except psutil.NoSuchProcess: processes = []
                for proc in processes:
                    try:
                        rss += proc.memory_info().rss
                        t = proc.cpu_times(); current_cpu += t.user+t.system
                    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
                peak = max(peak,rss); cpu = max(cpu,current_cpu)
                elapsed = time.monotonic()-start
                metrics.write(json.dumps({'elapsed_s':elapsed,'tree_rss_bytes':rss,
                    'live_tree_cpu_s':current_cpu})+'\n'); metrics.flush()
                if elapsed >= timeout: reason = 'timeout'
                elif rss > rss_mib*1024**2: reason = 'rss_limit'
                if reason:
                    kill_group(); break
                time.sleep(0.2)
            p.wait()
    except BaseException:
        reason = 'interrupted'; raise
    finally:
        kill_group()
        if p is not None: p.wait()
        report = {'command':command,'returncode':p.returncode if p else None,
            'stop_reason':reason,'elapsed_s':time.monotonic()-start,
            'sampled_peak_tree_rss_bytes':peak,'sampled_max_live_tree_cpu_s':cpu,
            'timeout_s':timeout,'rss_limit_mib':rss_mib,
            'resource_note':'200ms samples; RSS sums may double-count shared pages; not exact peak rusage'}
        write_json(folder/'result.json',report)
    return report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action',required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source',type=Path,required=True)
    prep.add_argument('--image',type=Path,required=True)
    prep.add_argument('--out',type=Path,required=True)
    run = sub.add_parser('run')
    run.add_argument('--prepared',type=Path,required=True)
    run.add_argument('--scope',choices=['sample','cross','decoder'],required=True)
    run.add_argument('--variant',choices=['original','gather'],required=True)
    run.add_argument('--out',type=Path,required=True)
    run.add_argument('--timeout',type=float,default=60)
    run.add_argument('--rss-mib',type=float,default=4096)
    work = sub.add_parser('_worker')
    for name in ['model','dataset','output']: work.add_argument('--'+name,type=Path,required=True)
    for q in [run,work]:
        q.add_argument('--optimization',type=int,choices=[0,1,2,3],default=3)
        q.add_argument('--target',choices=['default','gpu','cpu'],default='default')
        q.add_argument('--quantize',action='store_true')
    a = p.parse_args()
    if a.action == 'prepare': prepare(a.source,a.image,a.out); return 0
    if a.action == '_worker': worker(a); return 0
    if not 0 < a.timeout <= 1800 or not 0 < a.rss_mib <= 16384:
        p.error('timeout must be in (0,1800], rss-mib in (0,16384]')
    if a.variant == 'gather' and a.target != 'default':
        p.error('rewritten graph has no GridSample target to configure')
    folder = a.prepared.resolve()/a.scope
    model = folder/(a.variant+'.onnx')
    if not model.is_file(): p.error('prepare subgraphs first')
    output = a.out.resolve()/'probe.rknn'
    command = [sys.executable,'-B',str(Path(__file__).resolve()),'_worker',
        '--model',str(model),'--dataset',str(folder/'dataset.txt'),'--output',str(output),
        '--optimization',str(a.optimization),'--target',a.target]
    if a.quantize: command += ['--quantize']
    report = bounded(command,a.out,a.timeout,a.rss_mib)
    report.update(model_sha256=sha256(model),exported=output.is_file() and output.stat().st_size>0)
    report['success'] = report['returncode']==0 and report['exported'] and not report['stop_reason']
    write_json(a.out/'result.json',report)
    print(json.dumps(report,indent=2))
    return 0 if report['success'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
