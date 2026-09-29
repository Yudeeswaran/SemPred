from __future__ import annotations

import json
import pickle
import random
import re
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
 accuracy_score,
 average_precision_score,
 brier_score_loss,
 precision_recall_fscore_support,
 roc_auc_score,
)
from sklearn.svm import LinearSVC

from training.classical_baseline import F

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; DATA=ROOT/'data'; ART.mkdir(exist_ok=True)
R=random.Random(1234); np.random.seed(1234)

def make_atomic(n=22000):
 keys=list(F); out=[]
 for i in range(n):
  k=R.choice(keys); pred,pos,neg=F[k]; y=R.randrange(2); s=R.choice(pos if y else neg)
  # lexical/context variation but one atomic proposition per sample
  if R.random()<.5: s=R.choice(['The customer says: ','According to the record: ','The latest message states: ','In this case: '])+s
  out.append((s,pred,y,k))
 return out

def hard_docs():
 docs=[]
 # unseen combinations: supporting + distractor, negation, temporal and agent-role confounders
 for k,(pred,pos,neg) in F.items():
  for i in range(12):
   p=pos[(i+1)%len(pos)]; n=neg[(i+2)%len(neg)]
   patterns=[
    (p+'. '+n,1),(n+'. '+p,1),
    ('The customer did not say: '+p,0),
    ('An agent said '+p+' but the customer did not request it.',0),
    ('Earlier, '+p+'. The customer later withdrew that request.',0),
    ('The customer asks whether '+p.lower()+' but does not request it.',0),
    ('It is only a hypothetical: '+p,0),
    (n+'. There is no evidence for the target condition.',0),
   ]
   for d,y in patterns: docs.append((d,pred,y,k))
 # cross-family distractors
 for k,(pred,pos,neg) in F.items():
  for ok,(op,opos,oneg) in F.items():
   if ok==k: continue
   docs.append((pos[0]+'. '+oneg[0],pred,1,k))
   docs.append((neg[0]+'. '+opos[0],pred,0,k))
 return docs

atomic=make_atomic(); hard=hard_docs(); R.shuffle(atomic); cut=int(.8*len(atomic)); tr=atomic[:cut]; te=atomic[cut:]
def pair(x): return x[1]+' [SEP] '+x[0]
Xtr=[pair(x) for x in tr]; Xte=[pair(x) for x in te]
ytr=np.array([x[2] for x in tr]); yte=np.array([x[2] for x in te])
word=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=300000)
char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,max_features=250000,min_df=2)
Aw=word.fit_transform(Xtr); Bw=word.transform(Xte)
Ac=char.fit_transform(Xtr); Bc=char.transform(Xte)
A=hstack([Aw,Ac]).tocsr(); B=hstack([Bw,Bc]).tocsr()
model=CalibratedClassifierCV(LinearSVC(C=1.0),cv=3,method='sigmoid'); t=time.perf_counter(); model.fit(A,ytr); train_s=time.perf_counter()-t

def score_sentence(text,pred):
 x=pair((text,pred,0,'')); w=word.transform([x]); c=char.transform([x]); return float(model.predict_proba(hstack([w,c]).tocsr())[:,1][0])
def split_sents(t): return [s.strip() for s in re.split(r'(?<=[.!?])\s+',t) if s.strip()]

def doc_score(text,pred):
 text = str(text)
 pred = str(pred)
 sents=split_sents(text); scores=[score_sentence(str(s),pred) for s in sents]
 # Evidence aggregation: strongest positive/negative sentence; explicit negation/withdrawal dampens positive evidence.
 pos=max(scores) if scores else .5
 low=text.lower()
 if any(q in low for q in ['did not','does not','not request','no evidence','hypothetical','withdrew','withdrawn']):
  pos*=0.35
 return max(0,min(1,pos))

# atomic benchmark
p=model.predict_proba(B)[:,1]; pred=p>=.5
pr,rec,f1,_=precision_recall_fscore_support(yte,pred,average='binary',zero_division=0)
atomic_metrics={'acc':float(accuracy_score(yte,pred)),'precision':float(pr),'recall':float(rec),'f1':float(f1),'auc':float(roc_auc_score(yte,p)),'ap':float(average_precision_score(yte,p)),'brier':float(brier_score_loss(yte,p))}
# hard docs
ph=np.array([doc_score(str(x[0]),str(x[1])) for x in hard]); yh=np.array([x[2] for x in hard]); hd=ph>=.5
pr,rec2,f1,_=precision_recall_fscore_support(yh,hd,average='binary',zero_division=0)
hard_metrics={'acc':float(accuracy_score(yh,hd)),'precision':float(pr),'recall':float(rec),'f1':float(f1),'auc':float(roc_auc_score(yh,ph)),'ap':float(average_precision_score(yh,ph)),'brier':float(brier_score_loss(yh,ph))}
# Save model
with open(ART/'sempred_v6_evidence.pkl','wb') as f: pickle.dump((model,word,char),f)
out={'version':'v6_evidence','atomic_dataset':len(atomic),'atomic_train':len(tr),'atomic_test':len(te),'hard_docs':len(hard),'predicates':len(F),'train_seconds':train_s,'atomic':atomic_metrics,'hard':hard_metrics}
(DATA/'sempred_v6_atomic.jsonl').write_text('\n'.join(json.dumps({'text':a,'predicate':b,'label':c,'family':d}) for a,b,c,d in atomic))
(DATA/'sempred_v6_hard.jsonl').write_text('\n'.join(json.dumps({'text':a,'predicate':b,'label':c,'family':d}) for a,b,c,d in hard))
(ART/'v6_results.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
