# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "numpy==2.4.6", "pandas==3.0.5", "scikit-learn==1.8.0",
#   "plotly==7.0.0", "matplotlib==3.10.8", "filelock==3.32.5",
# ]
# ///
from __future__ import annotations
import argparse
from sequence_data import context,prepare
from parent_bridge import replay_parent,read
from parent_support import seal_json,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','smoke','prepare'])
    for n in ['kit','repo','out','round7','round5','round5-kit','round4','round4-kit','round3','round3-kit']:p.add_argument('--'+n,required=True,type=__import__('pathlib').Path)
    a=p.parse_args();ctx=context(a);sig=ctx['signature']
    from tree_core import event
    seal_json(a.out/'source_plan.json',{'signature':sig,'new_study':'learned-pair-history-v1','fold':2,'raw_output_reads':0})
    if a.command=='preflight':
        _,receipts=replay_parent(ctx,2)
        r={'status':'sequence_preflight_passed','signature':sig,'control_models_replayed':len(receipts),'new_fits':0,
           'populations':ctx['populations'][1],'feature_research':'open','github_updated':False}
        atomic_json(a.out/('preflight_reuse.json' if (a.out/'preflight.json').exists() else 'preflight.json'),r)
    else:
        if read(a.out,'preflight.json')['signature']!=sig:raise ValueError('Complete preflight')
        if a.command=='prepare' and read(a.out,'smoke.json')['status']!='sequence_smoke_passed':raise ValueError('Complete training smoke')
        r=prepare(a,ctx,smoke=a.command=='smoke',event=event)
    event('stage_complete',stage=a.command,status=r['status'])
if __name__=='__main__':main()
