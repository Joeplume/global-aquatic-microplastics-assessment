"""Recompute matched variance and field inference from completed QAQC models."""
from pathlib import Path
import os, sys, json, ast, importlib.util, hashlib
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
import numpy as np, pandas as pd
P=Path(__file__).resolve().parent; PROJECT=P.parents[2]
Q=P.parents[1]/'weighted_training'
H=PROJECT/'src/modeling/h26'

def write(name,obj): (P/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()

contract=json.loads((Q/'CONTRACT.json').read_text(encoding='utf-8'))
write('BATCH_CONTRACT.json',{'QAQC_contract_path':str(Q/'CONTRACT.json'),'QAQC_contract_sha256':digest(Q/'CONTRACT.json'),
 'source_hashes':contract['original_inputs'],'runner_sha256':digest(P/'run_validation_unit.py'),
 'scope':'Numeric-only update of V21 source documents; no existing model/source/document is overwritten.',
 'units':{'record_random':30,'strict_transfer':45,'same_weight_nonhierarchical':30,'paper_removal':180},
 'seed':260826,'epochs':180,'crossfits':{'random_grouped_removal_spatial':3,'strict_source_joint':0},
 'removal':'Existing two source-manifest orders and 30/50/80 percent; original grouped outer test records unchanged.',
 'independent_unchanged':['Supplementary Table 2 standalone gradient-boosting size/year analysis',
                           'Table 28 source-reference counts (training source catalogue geometry)',
                           'Existing fixed conventional benchmarks, explicitly not QAQC-weighted reruns.']})
write('MODEL_DEPENDENCIES.json',{'tables':{
 '2':{'action':'retain','reason':'Independent gradient-boosting temporal harmonization analysis; no H26 prediction input'},
 '3':{'action':'refit_variance','input':'QAQC grouped10 environmental residuals, original 975-record two-core'},
 '4':{'action':'replace','input':'Completed QAQC grouped10 environment and known-source conditional OOF'},
 '5':{'action':'retrain','input':'Original fivefold spatial/source/joint designs; 45 units'},
 '19':{'action':'recompute','input':'New record-random OOF and original current-domain C-AOA annotation'},
 '25':{'action':'retrain_and_recompute','input':'30 record-random +30 same-weight grouped comparison units'},
 '26':{'action':'recompute','input':'New strict-source OOF; unchanged support-ranking and same-50-percent retention rule'},
 '27':{'action':'infer','input':'Completed QAQC freshwater checkpoints; original 5 sites and 16 covariate scenarios'},
 '28':{'action':'retrain','input':'180 paper-removal units; geometric source-support counts retained'}},
 'note':'All cell and paragraph replacements will be emitted only after values are computed.'})

vpath=H/'evaluation/09_保守来源方差拆分_20260904/run_conservative_decomposition.py'
spec=importlib.util.spec_from_file_location('frozen_variance',vpath);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
core_path=PROJECT/'analysis/SI_V21_alignment_20261016_V2/runs/variance_20261016_161523/MATCHED_RECORD_LEDGER.csv'
core=pd.read_csv(core_path)
new=pd.read_csv(Q/'aggregate_results/OOF_qaqc.csv.gz').set_index('stable_record_id').loc[core.stable_record_id]
assert len(core)==975 and new.index.is_unique
core['residual_log10']=new.observed_log10.to_numpy()-new.predicted_q50_log10.to_numpy();core['all_group']='all'
core.to_csv(P/'TABLE3_MATCHED_QAQC_RESIDUALS.csv',index=False)
result=v.fit_variance_components(core)
result['source_ledger']=str(core_path);result['qaqc_oof']=str(Q/'aggregate_results/OOF_qaqc.csv.gz')
write('TABLE3_VARIANCE.json',result)
print('TABLE3',result['shares'],flush=True)

import torch
import torch.nn.functional as F
from scipy.stats import pearsonr,spearmanr
torch.set_num_threads(2)
posterior_path=PROJECT/'src/modeling/03_H26质量存量与监测优化_20261002_V1/src/posterior.py'
tree=ast.parse(posterior_path.read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FrozenH26' or
       isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LOG' for t in n.targets)]
space={'np':np,'pd':pd,'torch':torch,'F':F}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(posterior_path),'exec'),space)
Frozen=space['FrozenH26']
djk=PROJECT/'src/modeling/99_临时与待确认/danjiangkou_H26_20261012'
frame=pd.read_csv(djk/'scenario_covariates.csv');sites=pd.read_csv(djk/'supported_candidates.csv')
variants=frame.variant.drop_duplicates().tolist()
assert len(variants)==16 and len(frame)==112
medians=[];checks=[]
for fold in range(1,11):
 folder=Q/f'units/qaqc/freshwater/fold_{fold}/S0/protocol_aware/reference_surface_freshwater'
 model=Frozen.__new__(Frozen)
 model.ck=torch.load(folder/'H26_UNIT_MODEL.pt',map_location='cpu',weights_only=False)
 model.audit=json.loads((folder/'H26_UNIT_AUDIT.json').read_text(encoding='utf-8'))
 model.state=model.ck['state_dict'];model.domain='reference_surface_freshwater';model.fold=fold
 model.center=model.ck['target_center_log10'];model.scale=model.ck['target_scale_log10']
 model.cap=model.audit['fit']['total_cap_log10']/model.scale;model.branch_caps=model.audit['fit']['branch_caps_log10']
 x=model.transform(frame)
 draws=np.asarray([model.predict(x,model.parameters(26100900+fold*10000+j)) for j in range(512)])
 medians.append(np.median(draws,axis=0));checks.append({'fold':fold,'checkpoint':str(folder/'H26_UNIT_MODEL.pt'),'checkpoint_sha256':digest(folder/'H26_UNIT_MODEL.pt')})
pred=np.mean(medians,axis=0).reshape(16,7)-3
np.savez_compressed(P/'TABLE27_FOLD_MEDIANS.npz',fold_medians=np.asarray(medians))
rows=[];points=[]
for i,name in enumerate(variants):
 y=np.log10(sites.observed_items_L.to_numpy()[:5]);q=pred[i,:5];e=q-y
 rows.append({'variant':name,'n':5,'rmse_log10':float(np.sqrt(np.mean(e**2))),'mae_log10':float(abs(e).mean()),
              'bias_log10':float(e.mean()),'geometric_predicted_observed_ratio':float(10**e.mean()),
              'pearson_log10_r':float(pearsonr(y,q).statistic),'spearman_rho':float(spearmanr(y,q).statistic)})
 for j in range(5):points.append({'variant':name,'site':sites.site.iloc[j],'observed_items_L':float(sites.observed_items_L.iloc[j]),'predicted_log10_items_L':float(q[j])})
pd.DataFrame(rows).to_csv(P/'TABLE27_QAQC_METRICS.csv',index=False)
pd.DataFrame(points).to_csv(P/'TABLE27_QAQC_PREDICTIONS.csv',index=False)
write('TABLE27_INFERENCE_CONTRACT.json',{'original_scenarios':str(djk/'scenario_covariates.csv'),'scenario_sha256':digest(djk/'scenario_covariates.csv'),
 'checkpoints':checks,'posterior_draws_per_fold':512,'seed':'26100900+fold*10000+draw',
 'aggregation':'Original field-analysis rule: arithmetic mean of ten fold posterior medians, on log10 scale',
 'sites':'Original five primary positive-concentration sites; no site/variant selection based on outcome'})
print('TABLE27 complete',flush=True)
