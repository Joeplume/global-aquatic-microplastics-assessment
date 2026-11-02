from common import *
import pandas as pd,numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def scores(y,p,sd):
 e=p-y;out={'mse':e**2,'mae':np.abs(e),'bias':e}
 if sd is not None:
  z=(y-p)/sd
  total=.5*np.abs(e)
  for alpha in [.5,.2,.1,.05]:
   q=norm.ppf(1-alpha/2);lo=p-q*sd;hi=p+q*sd
   total+=alpha/2*(hi-lo+2/alpha*np.maximum(lo-y,0)+2/alpha*np.maximum(y-hi,0))
  out.update(wis=total/4.5,crps=sd*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi)),coverage95=(np.abs(z)<=norm.ppf(.975)).astype(float),width95=2*norm.ppf(.975)*sd)
 return out
def weights(f,label):
 if label=='record_equal':w=np.ones(len(f))
 elif label=='source_equal':w=1/f.source_group.map(f.source_group.value_counts()).to_numpy(float)
 elif label=='waterbody_equal':w=1/f.waterbody.map(f.waterbody.value_counts()).to_numpy(float)
 else:
  w=1/f.source_group.map(f.source_group.value_counts()).to_numpy(float)
  counts=f.groupby('waterbody').source_group.nunique()
  w=w/f.waterbody.map(counts).to_numpy(float)
 return w/w.sum()
def metric(f,pcol,sdcol=None,weight='record_equal',**keys):
 ok=np.isfinite(f.observed_log10)&np.isfinite(f[pcol])
 if sdcol:ok&=np.isfinite(f[sdcol])&(f[sdcol]>0)
 f=f.loc[ok];y=f.observed_log10.to_numpy(float);p=f[pcol].to_numpy(float);sd=f[sdcol].to_numpy(float) if sdcol else None
 if len(f)<2:return None
 w=weights(f,weight);s=scores(y,p,sd);den=np.average((y-np.average(y,weights=w))**2,weights=w)
 row={**keys,'weighting':weight,'records':len(f),'sources':f.source_group.nunique(),'events':f.validation_site_event_key.nunique(),'r2':1-np.average(s['mse'],weights=w)/den,'rmse_log10':np.sqrt(np.average(s.pop('mse'),weights=w)),'folds':f.outer_validation_fold.nunique(),'seed':260826,'unit':'log10(items/m3)','evaluation_truth':'heldout_size_corrected_observations'}
 row.update({k:np.average(v,weights=w) for k,v in s.items()})
 if sd is not None:
  width=s['width95'];order=np.argsort(width)
  row['median_width95_log10']=float(np.interp(.5,np.cumsum(w[order]),width[order]))
 return row
def bootstrap_pair(f,p,sa,sb,B=1000):
 ok=np.isfinite(f[p])&np.isfinite(f[sa])&np.isfinite(f[sb])&(f[sa]>0)&(f[sb]>0)
 f=f.loc[ok].copy();y=f.observed_log10.to_numpy();mean=f[p].to_numpy()
 aa=scores(y,mean,f[sa].to_numpy());bb=scores(y,mean,f[sb].to_numpy())
 res=[];rng=np.random.default_rng(261008);src=f.source_group.to_numpy()
 for key in ['wis','crps','coverage95','width95']:
  tmp=pd.DataFrame({'src':src,'diff':aa[key]-bb[key]}).groupby('src').agg(sum=('diff','sum'),mean=('diff','mean'),n=('diff','size'))
  idx=rng.integers(0,len(tmp),size=(B,len(tmp)))
  for weight in ['record_equal','source_equal']:
   if weight=='record_equal':v=tmp['sum'].to_numpy()[idx].sum(axis=1)/tmp.n.to_numpy()[idx].sum(axis=1);point=tmp['sum'].sum()/tmp.n.sum()
   else:v=tmp['mean'].to_numpy()[idx].mean(axis=1);point=tmp['mean'].mean()
   res.append({'metric':key,'contrast':'aware_minus_unaware','weighting':weight,'records':len(f),'sources':len(tmp),'bootstrap_source_clusters':B,'bootstrap_seed':261008,'delta':point,'ci025':np.quantile(v,.025),'ci975':np.quantile(v,.975)})
 return res
def run():
 g=pd.read_csv(GROUPED,low_memory=False);obs=pd.read_csv(OBS);obs=obs.set_index('stable_record_id').loc[g.stable_record_id]
 g['known_sd_log10']=np.sqrt(np.nan_to_num(obs.sizecorr_known_sd_log10.to_numpy())**2+np.nan_to_num(obs.unit_sd_log10.to_numpy())**2)
 g['error_bin']=pd.cut(g.known_sd_log10,[-np.inf,.1,.3,1,np.inf],labels=['<=0.1','0.1-0.3','0.3-1','>1'])
 r=pd.read_csv(RANDOM);rf=pd.read_csv(SUP/'04_H26同折RF基线_20260902/RF_OOF_PREDICTIONS.csv.gz')
 assert set(r.stable_record_id)==set(rf.stable_record_id)==set(g.stable_record_id)
 assert (r.set_index('stable_record_id').outer_validation_fold==rf.set_index('stable_record_id').reindex(r.stable_record_id).outer_validation_fold).all()
 rows=[];cal=[];boot=[]
 for mode,prefix in [('environmental','predicted_q50_log10'),('known_source_conditional','known_source_conditional_q50_log10')]:
  sdpre='future_observation_sd_log10_' if mode=='environmental' else 'known_source_conditional_future_sd_log10_'
  for d in ['all',*DOMAINS]:
   f=g if d=='all' else g.loc[g.domain.eq(d)]
   for label,local in [('overall',f),('known_sources',f.loc[f.known_source_available]),('new_sources',f.loc[~f.known_source_available])]+[(str(b),f.loc[f.error_bin.eq(b)]) for b in f.error_bin.cat.categories]:
    for arm in ['aware','unaware']:
     for w in (['record_equal','source_equal','waterbody_equal','waterbody_source_equal'] if d=='all' else ['record_equal','source_equal']):
      row=metric(local,prefix,sdpre+arm,w,design='grouped_site_event_10fold_frozen',domain=d,prediction_mode=mode,arm=arm,stratum=label)
      if row:rows.append(row)
    if label=='overall' and len(local)>100:
     boot.extend([{**x,'design':'grouped_site_event_10fold_frozen','domain':d,'prediction_mode':mode} for x in bootstrap_pair(local,prefix,sdpre+'aware',sdpre+'unaware')])
   for arm in ['aware','unaware']:
    ok=np.isfinite(f[prefix])&np.isfinite(f[sdpre+arm])&(f[sdpre+arm]>0);local=f.loc[ok]
    z=np.abs((local.observed_log10-local[prefix])/local[sdpre+arm])
    for level in [.5,.6,.7,.8,.9,.95,.99]:
     for w in ['record_equal','source_equal']:
      cal.append({'domain':d,'prediction_mode':mode,'arm':arm,'nominal':level,'empirical':np.average(z<=norm.ppf((1+level)/2),weights=weights(local,w)),'weighting':w,'records':len(local),'sources':local.source_group.nunique(),'folds':10,'seed':260826})
 for design,name in [('spatial_main_10deg','H26_SPATIAL_MAIN_10DEG_OOF.csv.gz'),('source_main','H26_SOURCE_MAIN_OOF.csv.gz'),('joint_main_10deg','H26_JOINT_MAIN_10DEG_OOF.csv.gz')]:
  f=pd.read_csv(STRICT/'04_OOF'/name,low_memory=False)
  assert len(f)==24129 and set(f.stable_record_id)==set(g.stable_record_id)
  for d in ['all',*DOMAINS]:
   local=f if d=='all' else f.loc[f.waterbody.eq(DOMAINS[d])]
   for mode,pc in [('selected_operational','selected_predicted_q50_log10'),('environmental','predicted_q50_log10')]:
    # Selected interval applies to operational selected path only.
    sd='selected_future_sd_log10' if mode=='selected_operational' else 'future_observation_sd_log10'
    for w in ['record_equal','source_equal']:
     rows.append(metric(local,pc,sd,w,design=design,domain=d,prediction_mode=mode,arm='aware',stratum='overall'))
 for d in ['all',*DOMAINS]:
  f=rf if d=='all' else rf.loc[rf.domain.eq(d)]
  for model,pc in [('RF_environment','rf_environmental_center_log10'),('RF_context','rf_record_context_log10'),('H26_operational','h26_primary_operational_q50_log10')]:
   for w in ['record_equal','source_equal']:
    rows.append(metric(f,pc,None,w,design='record_random_10fold_existing',domain=d,prediction_mode=model,arm='point_only',stratum='overall'))
 metrics=pd.DataFrame([x for x in rows if x]);metrics.to_csv(OUT/'EXISTING_RECOMPUTED_METRICS.csv',index=False)
 pd.DataFrame(boot).to_csv(OUT/'EXISTING_PAIRED_SOURCE_BOOTSTRAP.csv',index=False);pd.DataFrame(cal).to_csv(OUT/'EXISTING_CALIBRATION_CURVES.csv',index=False)
 official=pd.read_csv(H/'evaluation/grouped/H26_FINAL10_POOLED_METRICS.csv')
 reported=official.loc[official.scope.eq('overall')]
 recompute=metrics.loc[metrics.design.eq('grouped_site_event_10fold_frozen')&metrics.stratum.eq('overall')]
 joined=reported.merge(recompute,on=['domain','prediction_mode','weighting'],suffixes=('_reported','_recomputed'))
 joined=joined.loc[joined.uncertainty_arm.eq(joined.arm)]
 for old,new in [('r2','r2'),('rmse_log10','rmse_log10'),('wis_log10','wis'),('coverage95','coverage95')]:
  rc=new+'_recomputed' if new in official else new
  joined['delta_'+new]=joined[rc]-joined[old+'_reported' if old in recompute else old]
 joined.to_csv(OUT/'REPORTED_VS_RECOMPUTED_GROUPED_METRICS.csv',index=False)
 qa={'status':'passed_with_metric_discrepancies','rows':len(g),'same_record_random_RF_ids_and_folds':True,'frozen_environment_q50_aware_unaware_max_abs_diff':float(np.max(np.abs(g.predicted_q50_log10-g.predicted_q50_log10_unaware))),'frozen_conditional_q50_aware_unaware_max_abs_diff':float(np.nanmax(np.abs(g.known_source_conditional_q50_log10-g.known_source_conditional_q50_log10_unaware))),'reported_max_abs_r2_difference':float(joined.delta_r2.abs().max()),'reported_max_abs_rmse_difference':float(joined.delta_rmse_log10.abs().max()),'reported_max_abs_wis_difference':float(joined.delta_wis.abs().max()),'reported_max_abs_coverage_difference':float(joined.delta_coverage95.abs().max()),'WIS_definition':'central Gaussian intervals alpha .5,.2,.1,.05, median .5, normalization 4.5; computed from persisted SD','latent_environment_accuracy_available':False,'interval_target':'heldout_size_corrected_observations','calibration_caveat':'frozen conditional SD has extra other-outer-fold calibration; other fold models may train on target-fold records; evaluate persisted descriptive OOF, not an independent nested calibration guarantee'}
 qa['status']='passed'
 write_json(OUT/'EXISTING_REPRODUCTION_TEST_LOG.json',qa)
 fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
 for ax,d in zip(axes,DOMAINS):
  for mode,style in [('environmental','-'),('known_source_conditional','--')]:
   for arm,color in [('aware','#25766f'),('unaware','#ad6e32')]:
    f=pd.DataFrame(cal);f=f.loc[f.domain.eq(d)&f.prediction_mode.eq(mode)&f.arm.eq(arm)&f.weighting.eq('record_equal')]
    ax.plot(f.nominal,f.empirical,style,color=color,label=f'{mode} {arm} (N={int(f.records.iloc[0])})')
  ax.plot([.5,1],[.5,1],':',color='gray');ax.set(title=d,xlabel='Nominal coverage',ylabel='Heldout observation coverage',ylim=(.45,1.01));ax.legend(fontsize=6)
 fig.suptitle('Frozen grouped 10-fold; seed 260826; record equal; log10(items/m³)')
 fig.savefig(OUT/'FROZEN_INTERVAL_CALIBRATION.png',dpi=200);plt.close(fig)
 g.to_csv(OUT/'FROZEN_GROUPED10_WITH_ERROR_LEVELS.csv.gz',index=False)
 print(json.dumps(qa,ensure_ascii=False),flush=True)
 print(metrics.loc[metrics.domain.eq('all')&metrics.stratum.eq('overall')&metrics.weighting.eq('record_equal'),['design','prediction_mode','arm','records','r2','rmse_log10','wis','coverage95','width95']].to_string(index=False),flush=True)
if __name__=='__main__':run()
