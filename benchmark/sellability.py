from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'; DATA=ROOT/'data'
with (DATA/'sempred_v7_stress_12k.jsonl').open(encoding='utf-8') as f:
    H = [json.loads(x) for x in f]
with (ART/'v4_hybrid.pkl').open('rb') as f:
    model, word, char = pickle.load(f)
X=[r['predicate']+' [SEP] '+r['text'] for r in H]; y=np.array([r['label'] for r in H])
start=time.perf_counter(); P=model.predict_proba(hstack([word.transform(X),char.transform(X)]).tocsr())[:,1]; elapsed=time.perf_counter()-start
pred=P>=.5
rows={
 'dataset':len(H),'accuracy':float(accuracy_score(y,pred)),'f1':float(f1_score(y,pred)),
 'auroc':float(roc_auc_score(y,P)),'brier':float(brier_score_loss(y,P)),
 'throughput_rows_per_sec':len(H)/elapsed,'seconds':elapsed}
selective={}
for margin in [0.05,0.10,0.15,0.20,0.25,0.30,0.40]:
 keep=np.abs(P-.5)>=margin
 selective[str(margin)]={'coverage':float(keep.mean()),'accuracy_on_kept':float(accuracy_score(y[keep],pred[keep])) if keep.any() else None,'errors_on_kept':int(np.sum(pred[keep]!=y[keep]))}
rows['selective']=selective
(ART/'sellability_benchmark.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
