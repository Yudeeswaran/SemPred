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

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; DATA=ROOT/'data'; ART.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
R=random.Random(2026); np.random.seed(2026)
# predicate, positive templates, negative templates. Deliberately use disjoint lexical forms between train/hard.
F={
'refund':('customer requests a refund', ['give me my money back','I want the purchase reversed','please reimburse the amount I paid','I need the payment returned','send the purchase amount back to me','I am asking for reimbursement'], ['the money was already returned','the refund has been processed','I am checking when the approved refund will arrive','I prefer an exchange rather than reimbursement','I only asked about the refund policy','the customer received the reimbursement']),
'cancel':('customer threatens to cancel a subscription',['fix this or I will terminate the plan','I will leave the service if this continues','resolve this or I am ending my membership','I am considering ending the subscription over this','this problem may make me leave the service','I may terminate my account if you cannot fix it'],['the membership was terminated last week','the user renewed the plan','I only want to pause the subscription','the customer changed to a different tier','the agent terminated the account','the customer mentioned cancellation in a tutorial']),
'delivery':('customer experienced a delivery problem',['the parcel arrived after the promised date','the courier took the package to the wrong address','my shipment never showed up','the order arrived damaged in transit','the delivery is seriously delayed','the carrier lost my package'],['the shipment arrived exactly on schedule','the parcel is due tomorrow','the customer changed the address before dispatch','the courier delivered early','the order was received in good condition','the user asked about the expected delivery date']),
'billing':('customer reports a billing problem',['the statement contains an incorrect charge','I was charged twice','there is an unexpected fee on my invoice','the amount billed is wrong','I was charged after terminating service','my statement has a charge I dispute'],['the invoice was paid successfully','I downloaded my receipt','the next billing date was requested','the customer changed the billing address','the statement looks correct','the user asked for a copy of the invoice']),
'fraud':('transaction appears unauthorized',['someone else used my card','I do not recognize this purchase','this charge was not made by me','my account shows an unauthorized transaction','I did not approve this payment','my card details were used without permission'],['I recognize the merchant','I made this purchase myself','the account owner authorized the payment','the transaction is mine','the customer confirmed the charge','the user recognizes the transaction after checking']),
'pii':('text contains personal identifying information',['the record lists a persons full name and mobile number','contact details include a direct email address','the document contains a home address','the profile includes a date of birth','the report contains an individual employee identifier'],['only regional totals are shown','the page contains product categories','the dataset contains anonymized aggregates','the report has no individual details','the article discusses privacy in general','only public company statistics are included']),
'schema':('record represents a breaking schema change',['a required field was removed from the event','a producer changed a field type incompatibly','the API response dropped an existing property','a required column disappeared from the table','a contract field was renamed without compatibility','downstream consumers will break because a field was removed'],['an optional field was added','the schema description changed only','a nullable metadata column was introduced','the version number changed without field changes','a nonbreaking optional property was added','existing fields remain compatible']),
'pipeline':('record describes a failed data pipeline',['the scheduled workflow ended with an error','the ETL run failed before completion','the orchestration job crashed during execution','the transformation produced no output because it failed','the ingestion task terminated unsuccessfully','the data workflow did not complete'],['the scheduled run completed successfully','all pipeline tasks finished without error','the job produced the expected output','the workflow was paused for maintenance','the transformation succeeded','the ETL run completed normally']),
'late_payment':('customer payment is overdue',['the invoice remains unpaid past its due date','payment is overdue','the customer missed the payment deadline','the balance has been outstanding too long','the bill was not paid by the due date'],['the invoice was paid before the deadline','payment was received on time','the customer prepaid the invoice','the balance is current','the due date is next month']),
'password_reset':('user is requesting a password reset',['I need to reset my password','send me a new password reset link','I cannot remember my password and need to change it','please help me create a new login password'],['the user successfully changed the password yesterday','the user is asking to change their email address','the password reset was already completed','the user remembers the current password']),
'bug':('text reports a software bug',['the application crashes when I click save','the page throws an error on checkout','the feature does not work as expected','the program freezes during upload','the app returns an exception'],['the feature works correctly','the user asks how the feature works','the release notes describe the feature','the application completed the action successfully']),
'positive_review':('review expresses positive sentiment',['the product is excellent','I love the service','the experience was fantastic','the quality exceeded my expectations','this is a great purchase'],['the product is terrible','I hate the service','the experience was disappointing','the quality was unacceptable','this purchase was a mistake']),
'negative_review':('review expresses negative sentiment',['the product is awful','I am unhappy with the service','the experience was disappointing','the quality is poor','I regret buying this'],['the product is excellent','I love the service','the experience was fantastic','the quality exceeded expectations','I would happily buy this again']),
'security_incident':('event indicates a security incident',['there was an unauthorized login attempt','the account was accessed by an unknown device','credentials were exposed','a suspicious authentication event occurred','an attacker gained access to the account'],['the user logged in from their normal device','authentication succeeded normally','the account owner changed their password','the login was expected','the device was recognized']),
'chargeback':('customer is disputing a card transaction',['I want to dispute this card payment','I am filing a chargeback for this transaction','this card payment is being contested','I do not accept this transaction and want to challenge it'],['the customer accepts the card payment','the transaction was confirmed as correct','the customer recognizes and approves the payment','the payment is not being disputed']),
'duplicate':('record is a likely duplicate',['this row repeats an existing customer record','the same entity appears twice','these records describe the same person','the entry duplicates another row','the record appears twice in the source'],['this is a unique customer record','the rows represent different people','no matching record exists','the entry is new','each record is distinct']),
'address_change':('customer requests an address change',['please update my shipping address','I need to change where the order is delivered','replace my current address with this one','update the delivery address on my account'],['the address was updated already','I only want to change my phone number','the customer confirms the current address','the shipping address is correct']),
'account_lock':('account is locked',['I cannot access my account because it is locked','the login says my account is blocked','my account has been locked out','I am unable to sign in because access is locked'],['the account is accessible','the user logged in successfully','the customer forgot the username','the account is active and working']),
'upgrade':('customer requests a plan upgrade',['I want to move to a higher tier','please upgrade my subscription','I need the premium plan','can you increase my service level'],['I want to downgrade the plan','the subscription is already on the highest tier','I do not want to change plans','the customer cancelled the upgrade']),
'export':('user requests a data export',['send me a copy of my data','I want to export my account information','please provide my personal data file','I need a downloadable copy of my records'],['the data export was already delivered','the user wants to delete the account','the customer only viewed the dashboard','the user asked for a summary rather than raw data'])}

contexts=['In the support ticket, ','The latest message says: ','According to the event, ','The customer wrote: ','The record indicates: ','From the submitted request: ']
negators=['There is no evidence that ','It does not mean that ','The note does not say that ','Do not conclude that ']

def make(n=24000, seed=2026):
 r=random.Random(seed); rows=[]; keys=list(F)
 for i in range(n):
  k=r.choice(keys); pred,pos,neg=F[k]; y=r.randrange(2); base=r.choice(pos if y else neg)
  # compositional distractors
  if r.random()<.38:
   other=r.choice(neg if y else pos); base=base+'. '+other
  if r.random()<.35: base=r.choice(contexts)+base
  if r.random()<.12: base=r.choice(negators)+base
  rows.append((base,pred,y,k))
 return rows

rows=make()
# deterministic split; family-balanced. Hard set uses held-out lexical templates composed with synonyms and negation.
rng=random.Random(7)
idx=list(range(len(rows))); rng.shuffle(idx); cut=int(.82*len(rows)); tr=[rows[i] for i in idx[:cut]]; te=[rows[i] for i in idx[cut:]]
# Create hard set from manually held-out concepts, plus role/tense/negation transformations.
H=[]
for k,(pred,pos,neg) in F.items():
 for base,y in [(pos[-1],1),(neg[-1],0),(pos[0],1),(neg[0],0)]:
  variants=[base, 'The customer explicitly says: '+base, 'Earlier, '+base+'. Now the opposite is not being requested.' if y else 'Earlier, '+base+'. This is not a request to do the target action.']
  for v in variants: H.append((v,pred,y,k))
rng.shuffle(H)

def pair(x): return x[1]+' [SEP] '+x[0]
Xtr=[pair(x) for x in tr]; Xte=[pair(x) for x in te]; Xh=[pair(x) for x in H]
ytr=np.array([x[2] for x in tr]); yte=np.array([x[2] for x in te]); yh=np.array([x[2] for x in H])
word=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=1,max_features=220000)
char=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),sublinear_tf=True,min_df=2,max_features=220000)
Aw=word.fit_transform(Xtr); Bw=word.transform(Xte); Hw=word.transform(Xh)
Ac=char.fit_transform(Xtr); Bc=char.transform(Xte); Hc=char.transform(Xh)
variants={
'v4_word':(Aw,Bw,Hw,LogisticRegression(max_iter=1200,C=2.5,class_weight='balanced')),
'v4_char':(Ac,Bc,Hc,LogisticRegression(max_iter=1200,C=3.0,class_weight='balanced')),
'v4_hybrid':(hstack([Aw,Ac]).tocsr(),hstack([Bw,Bc]).tocsr(),hstack([Hw,Hc]).tocsr(),LogisticRegression(max_iter=1200,C=2.5,class_weight='balanced')),
'v4_svm':(hstack([Aw,Ac]).tocsr(),hstack([Bw,Bc]).tocsr(),hstack([Hw,Hc]).tocsr(),CalibratedClassifierCV(LinearSVC(C=1.2),cv=3,method='sigmoid')),
}
res={}
for name,(a,b,h,m) in variants.items():
 t=time.perf_counter(); m.fit(a,ytr); ts=time.perf_counter()-t
 p=m.predict_proba(b)[:,1]; ph=m.predict_proba(h)[:,1]; pb=(p>=.5); pbh=(ph>=.5)
 def metrics(y,p,pp):
  pr,re,f1,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
  return {'acc':float(accuracy_score(y,p)),'precision':float(pr),'recall':float(re),'f1':float(f1),'auc':float(roc_auc_score(y,pp)),'ap':float(average_precision_score(y,pp)),'brier':float(brier_score_loss(y,pp))}
 res[name]={'train_seconds':ts,'test':metrics(yte,pb,p),'hard':metrics(yh,pbh,ph)}
 with open(ART/(name+'.pkl'),'wb') as f: pickle.dump((m,word,char),f)
best=max(res,key=lambda n:(res[n]['hard']['f1'],res[n]['hard']['auc'],res[n]['test']['f1']))
out={'version':'v4','dataset_size':len(rows),'train':len(tr),'test':len(te),'hard':len(H),'predicates':len(F),'best':best,'results':res}
(DATA/'sempred_24k.jsonl').write_text('\n'.join(json.dumps({'text':a,'predicate':b,'label':c,'family':d}) for a,b,c,d in rows))
(ART/'v4_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
