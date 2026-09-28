from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts'; ART.mkdir(exist_ok=True)
DATA=ROOT/'data'; DATA.mkdir(exist_ok=True)
SEED=42
random.seed(SEED); np.random.seed(SEED)

# Predicate families. Each family deliberately includes lexical neighbors that can fool bag-of-words.
FAMILIES={
 'refund': {
  'predicate':'customer is requesting a refund',
  'pos':[
   'I want my money back for this order.', 'Please refund the amount I paid.',
   'I would like a refund for the purchase.', 'Can you return my payment?',
   'I am asking for my money back.', 'Please reimburse me for this order.'
  ],
  'neg':[
   'The customer already received a refund yesterday.', 'The customer asked whether refunds are available but did not request one.',
   'The customer wants an exchange instead of a refund.', 'The customer complained about the refund policy but made no request.',
   'The customer is waiting for a refund that was already approved.', 'The customer paid the invoice and did not ask for money back.'
  ]},
 'cancel': {
  'predicate':'customer is threatening to cancel their subscription',
  'pos':[
   'If this is not fixed, I will cancel my subscription.', 'I am going to cancel the plan if you cannot resolve this.',
   'Fix this or I will leave and cancel the service.', 'I may cancel my membership because of this problem.',
   'I am considering cancelling the subscription over this issue.'
  ],
  'neg':[
   'The customer cancelled the subscription last month.', 'The customer asked how to pause the subscription.',
   'The customer renewed the subscription for another year.', 'The customer mentioned cancellation in the help article but did not threaten to cancel.',
   'The customer wants to change plans, not cancel the service.', 'The agent cancelled the subscription on behalf of the customer.'
  ]},
 'delivery': {
  'predicate':'customer experienced a delivery problem',
  'pos':[
   'My package arrived three days late.', 'The order never arrived at my address.',
   'The courier delivered the parcel to the wrong house.', 'My delivery was delayed again.',
   'The shipment arrived damaged after a problem with delivery.'
  ],
  'neg':[
   'The package arrived on time and in good condition.', 'The customer asked when the package would arrive before the delivery date.',
   'The order is scheduled for delivery tomorrow.', 'The customer changed the delivery address before shipment.',
   'The customer praised the courier for delivering early.', 'The customer wants faster delivery next time but this delivery had no issue.'
  ]},
 'billing': {
  'predicate':'customer is reporting a billing problem',
  'pos':[
   'I was charged twice for the same purchase.', 'There is an incorrect charge on my bill.',
   'My invoice contains a fee I do not recognize.', 'The amount on my bill is wrong.',
   'I was billed after I cancelled the service.'
  ],
  'neg':[
   'The customer paid the bill in full.', 'The customer downloaded a copy of the invoice.',
   'The customer asked when the next bill will be issued.', 'The customer praised the clear billing statement.',
   'The customer changed the billing address without reporting a problem.', 'The customer asked for a receipt after paying.'
  ]},
 'fraud': {
  'predicate':'transaction appears fraudulent',
  'pos':[
   'I do not recognize this card transaction.', 'Someone used my card without my permission.',
   'This purchase was not made by me.', 'There is an unauthorized charge on my account.',
   'My card details appear to have been used by someone else.'
  ],
  'neg':[
   'The customer recognized the transaction and confirmed it was theirs.', 'The customer made the purchase from their own phone.',
   'The transaction was authorized by the account owner.', 'The customer asked for a copy of the transaction receipt.',
   'The customer recognized the merchant after checking the statement.'
  ]},
 'privacy': {
  'predicate':'text contains personally identifiable information',
  'pos':[
   'My full name is Ravi Kumar and my phone number is 9876543210.', 'Contact me at arun@example.com about the order.',
   'The customer account lists a home address and date of birth.', 'Employee ID 48392 belongs to the customer.',
   'Please send the report to Priya at priya@example.com.'
  ],
  'neg':[
   'The report contains only aggregate sales by region.', 'The document describes product categories without customer details.',
   'This page contains public company statistics.', 'The dashboard shows monthly revenue totals only.',
   'The article discusses general privacy principles.'
  ]},
 'schema': {
  'predicate':'record represents a breaking schema change',
  'pos':[
   'The API removed the customer_id field from the response.', 'Column account_status changed from string to integer.',
   'The producer renamed order_id to purchase_id without compatibility.', 'The table dropped the required timestamp column.',
   'A required field was removed from the event contract.'
  ],
  'neg':[
   'A nullable description column was added to the table.', 'The schema documentation was updated without changing fields.',
   'A new optional metadata field was introduced.', 'The pipeline added a non-breaking nullable column.',
   'The team published the same schema with a new version number.'
  ]},
 'pipeline': {
  'predicate':'record describes a failed data pipeline',
  'pos':[
   'The nightly ETL job failed during the load step.', 'The Airflow DAG stopped with a task failure.',
   'The pipeline run ended in an error state.', 'The ingestion job crashed before writing the partition.',
   'The scheduled transformation failed and produced no output.'
  ],
  'neg':[
   'The pipeline completed successfully with all tasks green.', 'The DAG is scheduled to run at midnight.',
   'The job produced the expected partition successfully.', 'The pipeline was paused intentionally for maintenance.',
   'The transformation finished without errors.'
  ]},
}

NEGATION_PREFIX=['The user did not say that ','There is no evidence that ','The note does not indicate that ','It is incorrect to conclude that ']
CONTEXT=['In the latest support message, ','According to the ticket, ','The customer wrote, ','The event record says, ','The report states, ']

def mutate(text, rng):
    # Surface variation while preserving label.
    choices=[
      lambda s: s,
      lambda s: s.lower(),
      lambda s: rng.choice(CONTEXT)+s,
      lambda s: 'Please note: '+s,
      lambda s: s.replace('customer','user').replace('Customer','User'),
      lambda s: s.replace('subscription','plan'),
      lambda s: s.replace('package','parcel'),
      lambda s: s.replace('refund','money-back request'),
    ]
    return rng.choice(choices)(text)

def make_dataset(n=16000, seed=42):
    rng=random.Random(seed)
    rows=[]
    keys=list(FAMILIES)
    for i in range(n):
        k=rng.choice(keys)
        f=FAMILIES[k]
        y=1 if rng.random()<0.5 else 0
        base=rng.choice(f['pos'] if y else f['neg'])
        # Hard composition: append distractor sentence from opposite side in ~30% of examples.
        if rng.random()<0.30:
            other=rng.choice(f['neg'] if y else f['pos'])
            base=base+' '+other
        if rng.random()<0.22:
            base=mutate(base,rng)
        rows.append({'text':base,'predicate':f['predicate'],'label':y,'family':k})
    return rows

# Held-out adversarial set with unseen surface forms, explicitly hand-written.
HARD=[]
for k,f in FAMILIES.items():
    hard_pos=[
      'I need the payment reversed because this purchase did not work for me.',
      'Unless you solve this, I am done with the service and will terminate my plan.',
      'The courier missed the promised date and the parcel still has not arrived.',
      'Why was an extra amount posted to my statement? That charge is wrong.',
      'This card activity is mine to recognize; I made the purchase myself.',
      'The record includes an individual contact address and direct phone detail.',
      'Removing a required contract field will break downstream consumers.',
      'The scheduled data workflow terminated unsuccessfully before completion.'
    ][list(FAMILIES).index(k)]
    hard_neg=[
      'A refund has already been processed and the customer is only checking its status.',
      'The customer says they will keep the subscription even if support cannot help.',
      'The parcel was delivered exactly on the promised date.',
      'The customer simply requested a duplicate copy of the invoice.',
      'The account owner explicitly confirmed the purchase.',
      'Only anonymized totals and category counts are present in this dataset.',
      'Only an optional field was added and existing consumers remain compatible.',
      'The workflow finished successfully and produced its output.'
    ][list(FAMILIES).index(k)]
    HARD.append({'text':hard_pos,'predicate':f['predicate'],'label':1,'family':k})
    HARD.append({'text':hard_neg,'predicate':f['predicate'],'label':0,'family':k})

rows=make_dataset()
# Split by row, but reserve hard benchmark entirely.
train, test=train_test_split(rows,test_size=.2,random_state=SEED,stratify=[r['family']+'_'+str(r['label']) for r in rows])

def pair(r): return r['predicate']+' [SEP] '+r['text']
Xtr=[pair(r) for r in train]; Xte=[pair(r) for r in test]
ytr=np.array([r['label'] for r in train]); yte=np.array([r['label'] for r in test])
Xhard=[pair(r) for r in HARD]; yhard=np.array([r['label'] for r in HARD])

# Three progressively stronger local baselines.
models={}
# V1 word TFIDF
word=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_df=.995,sublinear_tf=True,max_features=120000)
A=word.fit_transform(Xtr); B=word.transform(Xte); H=word.transform(Xhard)
models['v1_word_logreg']=(A,B,H,LogisticRegression(max_iter=1000,C=3.0,class_weight='balanced'))
# V2 char captures morphology/paraphrase and typos
char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,sublinear_tf=True,max_features=180000)
Ac=char.fit_transform(Xtr); Bc=char.transform(Xte); Hc=char.transform(Xhard)
models['v2_char_logreg']=(Ac,Bc,Hc,LogisticRegression(max_iter=1000,C=4.0,class_weight='balanced'))
# V3 word+char union
A3=hstack([A,Ac]).tocsr(); B3=hstack([B,Bc]).tocsr(); H3=hstack([H,Hc]).tocsr()
models['v3_word_char']=(A3,B3,H3,LogisticRegression(max_iter=1200,C=3.0,class_weight='balanced'))

results={}
for name,(a,b,h,m) in models.items():
    t=time.perf_counter(); m.fit(a,ytr); train_s=time.perf_counter()-t
    p=m.predict_proba(b)[:,1]; ph=m.predict_proba(h)[:,1]
    pred=(p>=.5).astype(int); predh=(ph>=.5).astype(int)
    pr,re,f1,_=precision_recall_fscore_support(yte,pred,average='binary',zero_division=0)
    phr,rh,f1h,_=precision_recall_fscore_support(yhard,predh,average='binary',zero_division=0)
    results[name]={
      'train_seconds':train_s,'test_accuracy':float(accuracy_score(yte,pred)),
      'test_precision':float(pr),'test_recall':float(re),'test_f1':float(f1),
      'test_auc':float(roc_auc_score(yte,p)),'test_ap':float(average_precision_score(yte,p)),
      'test_brier':float(brier_score_loss(yte,p)),
      'hard_accuracy':float(accuracy_score(yhard,predh)), 'hard_precision':float(phr),'hard_recall':float(rh),'hard_f1':float(f1h),
      'hard_auc':float(roc_auc_score(yhard,ph)), 'hard_brier':float(brier_score_loss(yhard,ph)),
    }
    # Persist model/vectorizer for best later; pickle handles scipy/sklearn.
    import pickle
    with open(ART/(name+'.pkl'),'wb') as fp: pickle.dump((m, word if name=='v1_word_logreg' else char if name=='v2_char_logreg' else (word,char)),fp)

best=max(results,key=lambda n:(results[n]['hard_f1'],results[n]['hard_accuracy'],results[n]['test_f1']))
out={'dataset_size':len(rows),'train_size':len(train),'test_size':len(test),'hard_size':len(HARD),'families':list(FAMILIES),'best_model':best,'results':results}
(DATA/'sempred_16k.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
(ART/'iterative_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
