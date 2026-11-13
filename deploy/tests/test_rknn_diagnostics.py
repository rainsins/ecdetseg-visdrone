import json
import logging
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

from deploy.diagnose_rknn import bounded
from deploy.convert import convert
from deploy.sim_eval import _crosscheck_conversion_report

class BoundedProbes(unittest.TestCase):
    def test_timeout_terminates_process_group(self):
        import psutil
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as tmp:
            root = Path(tmp); pidfile = root/'child.pid'
            code = ('import subprocess,sys,time; '
                    'from pathlib import Path; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                    f'Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)')
            report = bounded([sys.executable,'-c',code],root/'run',1,512)
            self.assertEqual(report['stop_reason'],'timeout')
            self.assertEqual(report['returncode'],-signal.SIGKILL)
            self.assertLess(report['elapsed_s'],5)
            pid = int(pidfile.read_text())
            try:
                child = psutil.Process(pid)
                child.wait(timeout=3)
            except psutil.NoSuchProcess:
                pass
            except psutil.TimeoutExpired:
                self.assertEqual(child.status(),psutil.STATUS_ZOMBIE)

    def test_success_and_resource_limit(self):
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as tmp:
            root=Path(tmp)
            r=bounded([sys.executable,'-c','print(123)'],root/'ok',3,512)
            self.assertEqual(r['returncode'],0)
            self.assertIsNone(r['stop_reason'])
            r=bounded([sys.executable,'-c','import time; x=bytearray(20*1024*1024); time.sleep(60)'],
                      root/'rss',5,10)
            self.assertEqual(r['stop_reason'],'rss_limit')
            self.assertEqual(r['returncode'],-signal.SIGKILL)

class RoutingConfig(unittest.TestCase):
    def test_config_reaches_toolkit_and_report(self):
        seen={}
        class FakeRKNN:
            def __init__(self,**kwargs): pass
            def config(self,**kwargs): seen.update(kwargs); return 0
            def load_onnx(self,**kwargs): return 0
            def build(self,**kwargs): return 0
            def export_rknn(self,path): Path(path).write_bytes(b'mock'); return 0
            def release(self): pass
        with tempfile.TemporaryDirectory(dir='/tmp/opencode') as tmp:
            root=Path(tmp); source=root/'in.onnx'; source.write_bytes(b'mock')
            (root/'image.png').write_bytes(b'mock')
            dataset=root/'calib.txt'; dataset.write_text('image.png\n')
            output=root/'gpu.rknn'
            with patch('deploy.convert._import_rknn',return_value=FakeRKNN), \
                 patch('deploy.convert.importlib.metadata.version',return_value='2.5.0'):
                report=convert(source,dataset,output,'rk3588',[1,2,3],[4,5,6],
                    'normal',3,[1,3,640,640],True,0.9999,False,grid_sample_target='gpu')
                self.assertEqual(seen['op_target'],{'GridSample':'gpu'})
                self.assertEqual(report['op_target'],seen['op_target'])
                with self.assertRaisesRegex(ValueError,'new output'):
                    convert(source,dataset,output,'rk3588',[1,2,3],[4,5,6],
                        'normal',3,[1,3,640,640],True,0.9999,False,grid_sample_target='gpu')
            output.with_suffix('.rknn.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError,'op_target'):
                _crosscheck_conversion_report(output,'rk3588',[1,2,3],[4,5,6],
                    'normal',3,True,0.9999,logging.getLogger('test'))

if __name__ == '__main__':
    unittest.main()
