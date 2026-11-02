"""Recompute SI15 descriptive validation strata from real random10 OOF and frozen support labels."""
from pathlib import Path
import ast,argparse,json
import numpy as np,pandas as pd
from scipy.stats import norm
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=HERE/'results');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
tree=ast.parse((HERE/'original_aggregation.py').read_text(encoding='utf8'));fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='score');ns={'np':np,'norm':norm};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original_score','exec'),ns)
random=pd.read_csv(HERE/'random10_OOF.csv.gz',low_memory=False);labels=pd.read_csv(HERE/'RECORD_SUPPORT.csv.gz',usecols=['stable_record_id','caoa_status'])
r=random.merge(labels,on='stable_record_id',how='left',validate='one_to_one');assert len(r)==24129 and r.caoa_status.notna().all()
r['tier']=r.caoa_status.map({'CAOA_in':'in','CAOA_limit':'limited','CAOA_single_source':'out','CAOA_out_unmatched':'out'});assert r.tier.notna().all()
rows=[]
for domain in ['freshwater','transition','marine','all']:
 for tier in ['in','limited','out']:
  d=r[(r.domain.eq(domain) if domain!='all' else np.ones(len(r),bool))&r.tier.eq(tier)]
  rows.append(dict(domain=domain,tier=tier,**ns['score'](d,d.q_operational,d.sd_operational_a_uncapped)))
result=pd.DataFrame(rows);result.to_csv(a.output/'SI15_REAGGREGATED.csv',index=False)
ref=pd.read_csv(HERE/'REFERENCE_STRATA.csv');delta={col:float(np.max(abs(result[col]-ref[col]))) for col in ['rmse','wis','coverage95']}
table=json.loads((HERE/'si_v22_7_tables.json').read_text(encoding='utf8'))[14];checks=[]
for i,(index,row) in enumerate(result.iterrows(),1):
 old=table['rows'][i];checks.append({'row':i,'n_match':int(old[2])==int(row['n']),'rmse_match':old[3]==f'{row.rmse:.3f}','wis_match':old[4]==f'{row.wis:.3f}','coverage_match':old[5]==f'{100*row.coverage95:.2f}'})
audit={'records':len(r),'missing_labels':0,'all_display_values_match':all(all(v for k,v in c.items() if k.endswith('_match')) for c in checks),'checks':checks,'reference_max_differences':delta,'interpretation':'Descriptive stratification by complete-data frozen C-AOA labels. This is not independent fold-local support validation; SI22 supplies that separate test.'}
(a.output/'VALIDATION.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(json.dumps({'rows':len(rows),'records':len(r),'all_display_values_match':audit['all_display_values_match'],'reference_max_differences':delta}))
