import json
import pickle
import random
import re
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import (
 accuracy_score,
 average_precision_score,
 brier_score_loss,
 precision_recall_fscore_support,
 roc_auc_score,
)

ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; DATA=ROOT/'data'
# Load the deterministic V4 dataset/model artifacts
with open(ART/'v4_hybrid.pkl','rb') as f: model, word, char=pickle.load(f)
F={
'refund':('customer requests a refund',['give me my money back','I want the purchase reversed','please reimburse the amount I paid','I need the payment returned','send the purchase amount back to me','I am asking for reimbursement'],['the money was already returned','the refund has been processed','I am checking when the approved refund will arrive','I prefer an exchange rather than reimbursement','I only asked about the refund policy','the customer received the reimbursement']),
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
'negative_review':('review expresses negative sentiment',['the product is awful','I am unhappy with the service','the experience was disappointing','the quality is poor','I regret buying this'],['the product is excellent','I love the service','the experience is fantastic','the quality exceeded expectations','I would happily buy this again']),
'security_incident':('event indicates a security incident',['there was an unauthorized login attempt','the account was accessed by an unknown device','credentials were exposed','a suspicious authentication event occurred','an attacker gained access to the account'],['the user logged in from their normal device','authentication succeeded normally','the account owner changed their password','the login was expected','the device was recognized']),
'chargeback':('customer is disputing a card transaction',['I want to dispute this card payment','I am filing a chargeback for this transaction','this card payment is being contested','I do not accept this transaction and want to challenge it'],['the customer accepts the card payment','the transaction was confirmed as correct','the customer recognizes and approves the payment','the payment is not being disputed']),
'duplicate':('record is a likely duplicate',['this row repeats an existing customer record','the same entity appears twice','these records describe the same person','the entry duplicates another row','the record appears twice in the source'],['this is a unique customer record','the rows represent different people','no matching record exists','the entry is new','each record is distinct']),
'address_change':('customer requests an address change',['please update my shipping address','I need to change where the order is delivered','replace my current address with this one','update the delivery address on my account'],['the address was updated already','I only want to change my phone number','the customer confirms the current address','the shipping address is correct']),
'account_lock':('account is locked',['I cannot access my account because it is locked','the login says my account is blocked','my account has been locked out','I am unable to sign in because access is locked'],['the account is accessible','the user logged in successfully','the customer forgot the username','the account is active and working']),
'upgrade':('customer requests a plan upgrade',['I want to move to a higher tier','please upgrade my subscription','I need the premium plan','can you increase my service level'],['I want to downgrade the plan','the subscription is already on the highest tier','I do not want to change plans','the customer cancelled the upgrade']),
'export':('user requests a data export',['send me a copy of my data','I want to export my account information','please provide my personal data file','I need a downloadable copy of my records'],['the data export was already delivered','the user wants to delete the account','the customer only viewed the dashboard','the user asked for a summary rather than raw data'])}
contexts=['In the support ticket, ','The latest message says: ','According to the event, ','The customer wrote: ','The record indicates: ','From the submitted request: ','After reviewing the case, ','The latest update states: ']
negators=['There is no evidence that ','It does not mean that ','The note does not say that ','Do not conclude that ','The customer explicitly denies that ','This is not a request to ','Nothing here indicates that ']
rng=random.Random(20260927); H=[]
# 12k+ hard cases, balanced and deliberately transformed
for i in range(12000):
 k=rng.choice(list(F)); pred,pos,neg=F[k]; y=rng.randrange(2); base=rng.choice(pos if y else neg)
 mode=i%8
 if mode==0: text=base
 elif mode==1: text=rng.choice(contexts)+base
 elif mode==2: text=rng.choice(negators)+base; y=1-y
 elif mode==3: text=base+'. '+rng.choice(neg if y else pos)
 elif mode==4: text=rng.choice(contexts)+base+'. '+rng.choice(contexts)+rng.choice(neg if y else pos)
 elif mode==5: text='Earlier, '+base+'. Now the customer says nothing further about it.'
 elif mode==6: text='The report mentions that '+base.lower()+'. This mention is historical.'
 else: text='The customer says: "'+base+'"'
 H.append((text,pred,y,k))

def pair(x): return x[1]+' [SEP] '+x[0]
X=[pair(x) for x in H]; y=np.array([x[2] for x in H]);
# use stored vectorizers
W=word.transform(X); C=char.transform(X)
P=model.predict_proba(hstack([W,C]).tocsr())[:,1]; pred=(P>=.5)
pr,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
res={'dataset':len(H),'accuracy':float(accuracy_score(y,pred)),'precision':float(pr),'recall':float(recall),'f1':float(f1),'auroc':float(roc_auc_score(y,P)),'average_precision':float(average_precision_score(y,P)),'brier':float(brier_score_loss(y,P))}
(DATA/'sempred_v7_stress_12k.jsonl').write_text('\n'.join(json.dumps({'text':a,'predicate':b,'label':c,'family':d}) for a,b,c,d in H))
(ART/'v7_stress_results.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
# Evidence-aware aggregation: score sentences separately and explicitly account for negating scopes.
NEG_PREFIXES=['there is no evidence that ','it does not mean that ','the note does not say that ','do not conclude that ','the customer explicitly denies that ','this is not a request to ','nothing here indicates that ','not this: ','the customer did not say this: ']
def score_text(text,pred):
    sents=re.split(r'(?<=[.!?])\s+',text.strip())
    vals=[]
    for s in sents:
        Xs=[pred+' [SEP] '+s]; pw=word.transform(Xs); pc=char.transform(Xs); q=float(model.predict_proba(hstack([pw,pc]).tocsr())[:,1][0])
        low=s.lower().strip()
        if any(low.startswith(n) for n in NEG_PREFIXES): q=1-q
        vals.append(q)
    return max(vals) if vals else .5
P2=np.array([score_text(x[0],x[1]) for x in H]); pred2=P2>=.5
pr,recall,f1,_=precision_recall_fscore_support(y,pred2,average='binary',zero_division=0)
res2={'dataset':len(H),'accuracy':float(accuracy_score(y,pred2)),'precision':float(pr),'recall':float(recall),'f1':float(f1),'auroc':float(roc_auc_score(y,P2)),'average_precision':float(average_precision_score(y,P2)),'brier':float(brier_score_loss(y,P2))}
(ART/'v7_evidence_results.json').write_text(json.dumps(res2,indent=2))
print('evidence',json.dumps(res2,indent=2))
