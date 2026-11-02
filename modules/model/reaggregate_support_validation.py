"""Recompute SI Table 22 from frozen fold-local membership and QAQC source-holdout predictions."""
from pathlib import Path
import argparse,ast,json
import numpy as np,pandas as pd
from scipy.stats import norm
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=HERE/'results/support_validation');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
tree=ast.parse((HERE/'provenance/original_aggregate_validation.py').read_text(encoding='utf8'))
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='score')
ns={'np':np,'norm':norm};exec(compile(ast.Module(body=[node],type_ignores=[]),'original_aggregate_validation.py','exec'),ns)
d=pd.read_csv(HERE/'reference/source_OOF.csv.gz',low_memory=False)
membership=pd.read_csv(HERE/'reference/SELECTION_WEIGHTS_AT_50PCT.csv.gz',low_memory=False)
membership=membership[(membership.design=='strict_source_formal')&(membership.task=='environmental_unknown_source')]
old=pd.read_csv(HERE/'reference/MAIN_TABLE_STRICT_SOURCE_50PCT.csv');rows=[]
for _,r in old.iterrows():
 m=membership[(membership.waterbody==r.waterbody)&(membership.rule==r.rule)]
 local=d[d.waterbody==r.waterbody].merge(m[['stable_record_id','selection_weight']],on='stable_record_id',validate='one_to_one')
 assert len(local)==r.total_records and local.selection_weight.notna().all()
 cluster=local.source_group.map(lambda s:'paper::'+str(s).split('paper::',1)[1].split('|',1)[0] if 'paper::' in str(s) else 'external_batch_surrogate::'+str(s))
 den=local.selection_weight.groupby(cluster).transform('sum')
 w=np.divide(local.selection_weight,den,out=np.zeros(len(local)),where=den>0)
 rows.append(dict(waterbody=r.waterbody,rule=r.rule,total_records=len(local),**ns['score'](local,local.predicted_q50_log10,local.future_observation_sd_log10,w)))
out=pd.DataFrame(rows);ref=pd.read_csv(HERE/'reference/TABLE26_QAQC_50PCT.csv')
audit={col:float(np.max(abs(out[col]-ref[col]))) for col in ['mae','rmse','wis','coverage95']}
out.to_csv(a.output/'SI_TABLE22_RECOMPUTED.csv',index=False)
(a.output/'REFERENCE_COMPARISON.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
print(audit)
