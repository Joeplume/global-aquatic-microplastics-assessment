"""Reaggregate SI21 comparators and SI24 removal from saved real held-out records."""
from pathlib import Path
import argparse,ast,json
import numpy as np,pandas as pd
from scipy.stats import norm
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=HERE/'results');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
src=HERE/'supplementary_sources/analysis/validation_runs/model_lane/aggregate_validation.py'
tree=ast.parse(src.read_text(encoding='utf8'));node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='score');ns={'np':np,'norm':norm};exec(compile(ast.Module(body=[node],type_ignores=[]),'original_score','exec'),ns);score=ns['score']
tables=json.loads((HERE/'data/si_v22_7_tables.json').read_text(encoding='utf8'));checks=[]
# SI21: recorded random-fold baseline fits; current QAQC H26 result remains a distinct run.
random=pd.read_csv(HERE/'data/random_baselines.csv.gz');current=pd.read_csv(HERE/'data/random_qaqc.csv.gz',low_memory=False)
current['predicted_log10']=current.known_source_conditional_q50_log10.fillna(current.predicted_q50_log10);current['model']='H26'
random=pd.concat([random[random.model.ne('H26')],current[random.columns]],ignore_index=True)
random_rows=[]
for name,df in random.groupby('model'):
 assert df.stable_record_id.is_unique and len(df)==24129
 random_rows.append(dict(design='random10',model=name,population='All',**score(df,df.predicted_log10)))
random_out=pd.DataFrame(random_rows);random_out.to_csv(a.output/'SI21_RANDOM_REAGGREGATED.csv',index=False)
mapping={'非分层贝叶斯网络':'BNN_nonhier','分层模型':'H26','梯度提升':'HistGradientBoosting','线性回归':'LinearRegression','MLP':'MLP','径向基网络':'RBF_NN','RF':'RF','SVR':'SVR'}
for row in tables[20]['rows'][1:9]:
 r=random_out[random_out.model.eq(mapping[row[1]])].iloc[0]
 checks.append({'table':21,'row':row[1]+'/random','n_match':int(row[3])==int(r.n),'r2_match':row[4]==f'{r.r2:.4f}','rmse_match':row[5]==f'{r.rmse:.4f}'})
# SI21: matched QAQC neural arms; frozen grouped conventional RF controls.
group=pd.read_csv(HERE/'data/grouped_qaqc.csv.gz',low_memory=False);rf=pd.read_csv(HERE/'data/grouped_original_comparators.csv.gz',low_memory=False)
arms={'nonhier_noerror':('nonhier_q50_log10','nonhier_sd_noerror'),'nonhier_error':('nonhier_q50_log10','nonhier_sd_error'),'hier_noerror':('q_operational','hier_sd_noerror'),'hier_error':('q_operational','hier_sd_error')}
grow=[]
for population in ['All','Known source']:
 d=group if population=='All' else group[group.known_source_available]
 for arm,(q,sd) in arms.items():grow.append(dict(model=arm,population=population,**score(d,d[q],d[sd])))
 d=rf if population=='All' else rf[rf.known_source_available]
 for arm,q in [('RF_environment','rf_environment_log10'),('RF_context','rf_context_log10')]:grow.append(dict(model=arm,population=population,**score(d,d[q])))
gout=pd.DataFrame(grow);gout.to_csv(a.output/'SI21_GROUPED_REAGGREGATED.csv',index=False)
for offset,pop in [(9,'All'),(15,'Known source')]:
 for j,arm in enumerate([*arms,'RF_environment','RF_context']):
  row=tables[20]['rows'][offset+j];r=gout[(gout.model==arm)&(gout.population==pop)].iloc[0]
  check={'table':21,'row':arm+'/'+pop,'n_match':int(row[3])==int(r.n),'r2_match':row[4]==f'{r.r2:.4f}','rmse_match':row[5]==f'{r.rmse:.4f}'}
  if not arm.startswith('RF'):
   check.update(wis_match=row[6]==f'{r.wis:.4f}',coverage_match=row[7]==f'{100*r.coverage95:.4f}',width_match=row[8]==f'{r.width95:.4f}')
  checks.append(check)
# SI24: paired predictions on held-out records from the actually removed papers.
base=pd.read_csv(HERE/'data/grouped_baseline.csv.gz',low_memory=False).set_index('stable_record_id',drop=False);pairs=[];parameters=json.loads((HERE/'data/analysis_parameters.json').read_text(encoding='utf8'));recoveries=parameters['small_sample_units']
for pct in [30,50,80]:
 for rnd in [1,2]:
  for fold in range(1,11):
   removed=set(pd.read_csv(HERE/f'manifests/round_{rnd:02d}/fold_{fold:02d}_remove_{pct}.csv').paper_id_base)
   for domain in ['freshwater','transition','marine']:
    u=HERE/f'removal/r{rnd}_{pct}/{domain}/fold_{fold}'
    unit_parameters=parameters['removal_units'][f'r{rnd}_{pct}/{domain}/fold_{fold}']
    assert len(pd.read_csv(u/'HELDOUT.csv.gz',usecols=['stable_record_id'])) == unit_parameters['records']
    e=pd.read_csv(u/'HELDOUT.csv.gz',low_memory=False);paper=e.source_group.str.extract(r'(NMP\d+)',expand=False);e=e[paper.isin(removed)]
    b=base.loc[e.stable_record_id];assert np.array_equal(e.observed_log10,b.observed_log10)
    for label,df in [('baseline',b),('removal',e)]:pairs.append(dict(removal=pct,round=rnd,fold=fold,domain=domain,arm=label,**score(df,df.predicted_q50_log10,df.future_observation_sd_log10)))
pairs=pd.DataFrame(pairs);pairs.to_csv(a.output/'SI24_PAIRED_REAGGREGATED.csv',index=False)
wide=pairs.pivot(index=['removal','round','fold','domain'],columns='arm',values='wis').rename(columns={'baseline':'baseline_wis','removal':'removal_wis'}).reset_index();rows=[]
for (pct,domain),df in wide.groupby(['removal','domain']):
 assert len(df)==20
 rows.append(dict(removal=pct,domain=domain,baseline_wis=df.baseline_wis.mean(),removal_wis=df.removal_wis.mean(),worse_pairs=int((df.removal_wis>df.baseline_wis).sum()),relative_increase_pct=100*(df.removal_wis.mean()/df.baseline_wis.mean()-1)))
out=pd.DataFrame(rows);out.to_csv(a.output/'SI24_REAGGREGATED.csv',index=False)
domain_map={'淡水':'freshwater','河口':'transition','海洋':'marine'}
for row in tables[23]['rows'][1:]:
 r=out[(out.domain==domain_map[row[0]])&(out.removal==int(row[1]))].iloc[0]
 checks.append({'table':24,'row':row[0]+'/'+row[1],'baseline_WIS_match':row[3]==f'{r.baseline_wis:.4f}','removal_WIS_match':row[4]==f'{r.removal_wis:.4f}','worse_pairs_match':row[5]==str(int(r.worse_pairs))})
sens=[]
for domain in ['transition','marine']:
 folds=[r['fold'] for r in recoveries if r['domain']==domain];df=wide[(wide.domain==domain)&(wide.removal==80)];keep=df[~((df['round']==1)&df['fold'].isin(folds))]
 sens.append({'domain':domain,'all_pairs':len(df),'without_recovery_pairs':len(keep),'all_relative_increase_pct':100*(df.removal_wis.mean()/df.baseline_wis.mean()-1),'without_recovery_relative_increase_pct':100*(keep.removal_wis.mean()/keep.baseline_wis.mean()-1)})
result={'checks':checks,'all_display_checks_pass':all(all(v for k,v in c.items() if k.endswith('_match')) for c in checks),'recovery_units':recoveries,'recovery_sensitivity':sens,'calculation':'Scores from supplied held-out predictions; training settings in data/analysis_parameters.json.'}
(a.output/'VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'checks':len(checks),'all_display_checks_pass':result['all_display_checks_pass'],'recovery_units':len(recoveries)}))
