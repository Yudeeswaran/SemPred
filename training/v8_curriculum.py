import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; DATA=ROOT/'data'
# Read the V4 semantic families from its source by executing only the F block is cumbersome; reuse generated V4 data as base.
base=[]
for line in (DATA/'sempred_24k.jsonl').read_text().splitlines():
 r=json.loads(line); base.append((r['text'],r['predicate'],r['label'],r['family']))
rng=random.Random(20260927)
negators=['There is no evidence that ','It does not mean that ','The note does not say that ','Do not conclude that ','The customer explicitly denies that ','This is not a request to ','Nothing here indicates that ']
contexts=['In the support ticket, ','The latest message says: ','According to the event, ','The customer wrote: ','The record indicates: ','From the submitted request: ']
# curriculum transformations, with exact base rows plus generated hard cases
train=list(base)
for _ in range(90000):
 text,pred,y,f=rng.choice(base)
 mode=rng.randrange(7)
 if mode==0: text=rng.choice(negators)+text; y=1-y
 elif mode==1: text=rng.choice(contexts)+text
 elif mode==2: text='Earlier, '+text+'. Now the customer says nothing further about it.'
 elif mode==3: text=text+'. '+rng.choice(negators)+text
 elif mode==4: text='The report mentions that '+text.lower()+'. This mention is historical.'
 elif mode==5: text='The customer says: "'+text+'"'
 else: text=text+'. '+rng.choice([r[0] for r in base if r[2]==(1-y)])
 train.append((text,pred,y,f))
rng.shuffle(train)
# independently generated 12k stress set from same transformation grammar, but held-out base examples
stress=[]
held=base[12000:]
for i in range(12000):
 text,pred,y,f=rng.choice(held); mode=i%7
 if mode==0: text=rng.choice(negators)+text; y=1-y
 elif mode==1: text=rng.choice(contexts)+text
 elif mode==2: text='Earlier, '+text+'. Now the customer says nothing further about it.'
 elif mode==3: text=text+'. '+rng.choice(negators)+text
 elif mode==4: text='The report mentions that '+text.lower()+'. This mention is historical.'
 elif mode==5: text='The customer says: "'+text+'"'
 else: text=text+'. '+rng.choice([r[0] for r in held if r[2]==(1-y)])
 stress.append((text,pred,y,f))
def pair(r): return r[1]+' [SEP] '+r[0]
vec=__import__('sklearn').pipeline.FeatureUnion([
 ('word',TfidfVectorizer(ngram_range=(1,3),sublinear_tf=True,max_features=80000)),
 ('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,max_features=80000))])
A=vec.fit_transform([pair(r) for r in train]); S=vec.transform([pair(r) for r in stress])
clf=LogisticRegression(C=2.5,max_iter=500,class_weight='balanced',solver='liblinear'); t=time.time(); clf.fit(A,[r[2] for r in train]); train_s=time.time()-t
p=clf.predict_proba(S)[:,1]; y=np.array([r[2] for r in stress]); pr,re,f1,_=precision_recall_fscore_support(y,p>.5,average='binary',zero_division=0)
res={'train_size':len(train),'stress_size':len(stress),'train_seconds':train_s,'accuracy':accuracy_score(y,p>.5),'precision':pr,'recall':re,'f1':f1,'auroc':roc_auc_score(y,p),'ap':average_precision_score(y,p),'brier':brier_score_loss(y,p)}
with open(ART/'v8_curriculum.pkl','wb') as f: pickle.dump((clf,vec),f)
(DATA/'sempred_v8_curriculum.jsonl').write_text('\n'.join(json.dumps({'text':a,'predicate':b,'label':c,'family':d}) for a,b,c,d in train))
(DATA/'sempred_v8_stress_12k.jsonl').write_text('\n'.join(json.dumps({'text':a,'predicate':b,'label':c,'family':d}) for a,b,c,d in stress))
(ART/'v8_results.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
