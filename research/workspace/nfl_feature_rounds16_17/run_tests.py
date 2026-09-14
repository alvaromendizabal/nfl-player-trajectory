"""User-invoked base test runner. This file has not been executed by the assistant."""
from pathlib import Path
import argparse,sys,unittest,time
from audit_io import atomic_json
from bridge import code_signature

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    kit=Path(__file__).resolve().parent;started=time.monotonic()
    suite=unittest.defaultTestLoader.discover(str(kit/'tests/base'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.wasSuccessful() and result.testsRun > 0 and not result.skipped
    receipt={'status':'tests_passed' if passed else 'tests_failed',
             'source_signature':code_signature(kit),'tests_run':result.testsRun,
             'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
             'elapsed_seconds':time.monotonic()-started,'python':sys.version.split()[0],
             'execution_owner':'user','scientific_models_fitted':0}
    atomic_json(a.out/'tests_receipt.json',receipt)
    if not passed:raise SystemExit(1)
if __name__=='__main__':main()
