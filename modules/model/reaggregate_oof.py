"""Recompute metrics from actual saved held-out records, without training."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results/reaggregation')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parent/'reference'
rows=[];counts=[]
targets={'environment':'predicted_q50_log10','known_source_conditional':'known_source_conditional_q50_log10','operational':'operational_prediction_log10'}
for design in ['grouped10','random10','spatial','source','joint']:
 df=pd.read_csv(root/f'{design}_OOF.csv.gz',low_memory=False)
 assert df.stable_record_id.is_unique,design
 # An operational prediction uses source adjustment when available, otherwise environmental prediction.
 if 'operational_prediction_log10' not in df:
  df['operational_prediction_log10']=df['known_source_conditional_q50_log10'].where(df['known_source_conditional_q50_log10'].notna(),df['predicted_q50_log10'])
 counts.append({'design':design,'records':len(df),'sources':df.source_group.nunique(),'known_source_records':int(df.known_source_conditional_q50_log10.notna().sum())})
 for domain in ['all',*sorted(df.waterbody.unique())]:
  dd=df if domain=='all' else df.loc[df.waterbody.eq(domain)]
  for target,col in targets.items():
   good=np.isfinite(dd[col])&np.isfinite(dd.observed_log10)
   y=dd.loc[good,'observed_log10'].to_numpy();pred=dd.loc[good,col].to_numpy()
   if len(y)<2:continue
   rows.append({'design':design,'waterbody':domain,'target':target,'n':len(y),'r2':r2_score(y,pred),'rmse_log10':float(np.sqrt(mean_squared_error(y,pred))),'mae_log10':float(np.mean(abs(y-pred)))})
pd.DataFrame(rows).to_csv(a.output/'REAGGREGATED_METRICS.csv',index=False)
(a.output/'COHORTS.json').write_text(json.dumps(counts,indent=2),encoding='utf8')
print(pd.DataFrame(rows).query("waterbody=='all'").to_string(index=False))
