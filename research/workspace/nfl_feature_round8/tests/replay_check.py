import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neural_study import train_arm,evaluate
from parent_support import atomic_json
out,kit=Path(sys.argv[1]),Path(sys.argv[2])
rs=[train_arm(out,kit,a,replay_only=True) for a in ['terminal','history']]
assert all(r['new_optimizer_steps']==0 for r in rs)
evaluate(out,kit)
atomic_json(out/'replay.json',{'status':'matched_encoder_replay_exact','new_optimizer_steps':0,'models_replayed':2,'arms':rs})
print('Fresh-process exact prediction replay, zero optimizer steps.')
