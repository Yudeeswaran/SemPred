from __future__ import annotations

import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
 accuracy_score,
 average_precision_score,
 brier_score_loss,
 precision_recall_fscore_support,
 roc_auc_score,
)
from sklearn.svm import LinearSVC
from v4_robust import F, contexts, negators

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; DATA=ROOT/'data'; ART.mkdir(exist_ok=True)
R=random.Random(99); np.random.seed(99)

def adv_train_rows(n=32000):
 rows=[]; keys=list(F)
 for i in range(n):
  k=R.choice(keys); pred,pos,neg=F[k]; y=R.randrange(2); base=R.choice(pos if y else neg)
  # targeted semantic confounders for training
  mode=R.randrange(8)
  if mode==0: base='The customer says: '+base
  elif mode==1: base=base+' However, this was only hypothetical and no action was requested.' if y else base+' This statement does not imply the target action.'
  elif mode==2: base='Someone else said this: '+base if y else 'The customer explicitly says the opposite: '+base
  elif mode==3: base='Earlier, '+base+'. The current status is different.'
  elif mode==4: base='Question from customer: '+base
  elif mode==5: base=R.choice(contexts)+base+' '+R.choice(neg if y else pos)
  elif mode==6: base=R.choice(negators)+base
  rows.append((base,pred,y,k))
 return rows

def hard_rows():
 H=[]
 # Cross-family lexical distractors: append a true-looking sentence from another family.
 keys=list(F)
 for k in keys:
  pred,pos,neg=F[k]
  for other in keys:
   if other==k: continue
   _,opos,oneg=F[other]
   for y,base in [(1,pos[1]),(0,neg[1])]:
    H.append((base+'. '+R.choice(opos if R.random()<.5 else oneg),pred,y,k))
 # semantic state/role/negation transformations, with held-out combinations
 for k,(pred,pos,neg) in F.items():
  for base in pos[:3]:
   H += [(f'The customer did not say this: {base}',pred,0,k),
         (f'The customer said they would do this later: {base}',0 if False else pred,1,k),
         (f'An agent mentioned this while the customer did something else: {base}',pred,0,k)]
  for base in neg[:3]:
   H += [(f'The customer explicitly requests the opposite of this: {base}',pred,1,k),
         (f'The customer asks whether this is possible, but does not request it: {base}',pred,0,k)]
 return H

rows=adv_train_rows(); H=hard_rows()
R.shuffle(rows); cut=int(.82*len(rows)); tr=rows[:cut]; te=rows[cut:]
def pair(x): return x[1]+' [SEP] '+x[0]
Xtr=[pair(x) for x in tr]; Xte=[pair(x) for x in te]; Xh=[pair(x) for x in H]
ytr=np.array([x[2] for x in tr]); yte=np.array([x[2] for x in te]); yh=np.array([x[2] for x in H])
word=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=260000)
char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,max_features=260000,min_df=2)
Aw=word.fit_transform(Xtr); Bw=word.transform(Xte); Hw=word.transform(Xh)
Ac=char.fit_transform(Xtr); Bc=char.transform(Xte); Hc=char.transform(Xh)
A=hstack([Aw,Ac]).tocsr(); B=hstack([Bw,Bc]).tocsr(); Hx=hstack([Hw,Hc]).tocsr()
models={
'v5_logreg':LogisticRegression(max_iter=1500,C=2.5,class_weight='balanced'),
'v5_svm':CalibratedClassifierCV(LinearSVC(C=1.0),cv=3,method='sigmoid')}
res={}
for name,m in models.items():
 t=time.perf_counter(); m.fit(A,ytr); ts=time.perf_counter()-t
 p=m.predict_proba(B)[:,1]; ph=m.predict_proba(Hx)[:,1]
 def M(y,pp):
  pred=pp>=.5; pr,re,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
  return {'acc':float(accuracy_score(y,pred)),'precision':float(pr),'recall':float(re),'f1':float(f1),'auc':float(roc_auc_score(y,pp)),'ap':float(average_precision_score(y,pp)),'brier':float(brier_score_loss(y,pp))}
 res[name]={'train_seconds':ts,'test':M(yte,p),'hard':M(yh,ph)}
 with open(ART/(name+'.pkl'),'wb') as f: pickle.dump((m,word,char),f)
best=max(res,key=lambda n:(res[n]['hard']['f1'],res[n]['hard']['acc']))
out={'version':'v5','dataset_size':len(rows),'train':len(tr),'test':len(te),'hard':len(H),'predicates':len(F),'best':best,'results':res}
(DATA/'sempred_32k_adversarial.jsonl').write_text('\n'.join(json.dumps({'text':a,'predicate':b,'label':c,'family':d}) for a,b,c,d in rows))
(ART/'v5_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
