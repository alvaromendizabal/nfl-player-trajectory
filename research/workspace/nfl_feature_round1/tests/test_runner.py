from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_round as runner
from test_origin_features import fixture

class RunnerTests(unittest.TestCase):
    def setup_data(self,root):
        repo=root/'repo';out=root/'out';out.mkdir();(repo/'data/raw/train').mkdir(parents=True)
        raws=[];labels=[];keys=[]
        for j in range(16):
            r,y=fixture(2023090100+j*100,seed=j+2)
            raws.append(r);labels.append(y);keys.append((int(r.game_id.iloc[0]),1))
        inputs=repo/'data/raw/train/input_2023_w01.csv';outputs=repo/'data/raw/train/output_2023_w01.csv'
        pd.concat(raws,ignore_index=True).to_csv(inputs,index=False);pd.concat(labels,ignore_index=True).to_csv(outputs,index=False)
        contract={'raw_files':[{'path':p.relative_to(repo).as_posix(),'sha256':runner.digest(p)} for p in [inputs,outputs]]}
        args=argparse.Namespace(repo=repo,out=out,label='smoke',plays=16)
        return args,(keys,contract,{'source_signature':runner.source_signature()})
    def test_raw_preparation_checkpoints_and_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            args,ret=self.setup_data(Path(d))
            with patch.object(runner,'check_repo',return_value=ret):runner.prepare(args)
            data=runner.load_dataset(args.out/'smoke')
            self.assertEqual(set(np.unique(data['offset'])),{0,5,10,20})
            original=data['offset']==0;self.assertFalse(data['prethrow'][original].any())
            files={p.name:(runner.digest(p),p.stat().st_mtime_ns) for p in (args.out/'smoke/plays').glob('*.npz')}
            with patch.object(runner,'check_repo',return_value=ret):runner.prepare(args)
            self.assertEqual(files,{p.name:(runner.digest(p),p.stat().st_mtime_ns) for p in (args.out/'smoke/plays').glob('*.npz')})
            summary=json.loads((args.out/'smoke/preparation_summary.json').read_text())
            self.assertEqual(summary['reused_play_checkpoints'],16)
    def test_changed_raw_file_stops_before_preparation(self):
        with tempfile.TemporaryDirectory() as d:
            args,ret=self.setup_data(Path(d));p=args.repo/'data/raw/train/input_2023_w01.csv';p.write_text(p.read_text()+'\n')
            with patch.object(runner,'check_repo',return_value=ret):
                with self.assertRaisesRegex(ValueError,'hash mismatch'):runner.prepare(args)
    def test_changed_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            args,ret=self.setup_data(Path(d))
            with patch.object(runner,'check_repo',return_value=ret):runner.prepare(args)
            p=next((args.out/'smoke/plays').glob('*.npz'))
            with p.open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint changed'):runner.load_dataset(args.out/'smoke')
    def test_report_excludes_raw_arrays_and_weights(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'research').mkdir();runner.atomic_json(out/'research/screen_summary.json',{'status':'synthetic'})
            (out/'research'/'private.npz').write_bytes(b'private');args=argparse.Namespace(out=out)
            runner.export_report(args)
            import zipfile
            with zipfile.ZipFile(out/'nfl_feature_round1_report.zip') as z:
                self.assertEqual(set(z.namelist()),{'research/screen_summary.json','CONTENTS.txt'})

    def test_parent_watchdog_stops_a_stalled_worker(self):
        import subprocess
        real_popen=subprocess.Popen
        def delayed_popen(cmd,**kwargs):
            return real_popen([sys.executable,'-c','import time; time.sleep(5)'],**kwargs)
        with tempfile.TemporaryDirectory() as d:
            with patch.object(sys,'argv',['run_round.py','report','--out',d,'--seconds','1']):
                with patch.object(runner.subprocess,'Popen',side_effect=delayed_popen):
                    with self.assertRaises(SystemExit) as result:runner.main()
            self.assertEqual(result.exception.code,124)
            receipt=json.loads((Path(d)/'last_command.json').read_text())
            self.assertEqual(receipt['status'],'stopped_budget')
            self.assertLess(receipt['elapsed_seconds'],3)
    def test_output_lock_blocks_concurrent_stage(self):
        import fcntl
        with tempfile.TemporaryDirectory() as d:
            with (Path(d)/'.round.lock').open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(sys,'argv',['run_round.py','report','--out',d]):
                    with self.assertRaises(SystemExit) as result:runner.main()
                self.assertEqual(result.exception.code,2)
