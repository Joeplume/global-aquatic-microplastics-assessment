from common import *
from audit_existing import metric,scores,weights,bootstrap_pair
import pandas as pd,numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def point_bootstrap(f,p1,p0,B=1000):
 f=f.loc[np.isfinite(f[p1])&np.isfinite(f[p0])].copy();y=f.observed_log10.to_numpy()
 e1=f[p1].to_numpy()-y;e0=f[p0].to_numpy()-y
 tmp=pd.DataFrame({'src':f.source_group,'delta_mae':np.abs(e1)-np.abs(e0),'mse1':e1**2,'mse0':e0**2,'bias1':e1,'bias0':e0})
 agg=tmp.groupby('src').agg(n=('src','size'),mae=('delta_mae','mean'),mse1=('mse1','mean'),mse0=('mse0','mean'),bias1=('bias1','mean'),bias0=('bias0','mean'))
 rng=np.random.default_rng(261008);idx=rng.integers(0,len(agg),size=(B,len(agg)));rows=[]
 for weighting in ['record_equal','source_equal']:
  ww=agg.n.to_numpy()[idx] if weighting=='record_equal' else np.ones_like(idx,dtype=float)
  mean=lambda k:np.average(agg[k].to_numpy()[idx],weights=ww,axis=1)
  pw=agg.n.to_numpy() if weighting=='record_equal' else np.ones(len(agg));pointmean=lambda k:np.average(agg[k],weights=pw)
  for name,z,point in [('mae',mean('mae'),pointmean('mae')),('rmse',np.sqrt(mean('mse1'))-np.sqrt(mean('mse0')),np.sqrt(pointmean('mse1'))-np.sqrt(pointmean('mse0'))),('bias',mean('bias1')-mean('bias0'),pointmean('bias1')-pointmean('bias0'))]:
   rows.append({'metric':name,'weighting':weighting,'delta':point,'ci025':np.quantile(z,.025),'ci975':np.quantile(z,.975),'records':len(f),'sources':len(agg),'bootstrap_source_clusters':B,'bootstrap_seed':261008})
 return rows
def run():
 frames=[];audits=[]
 for d in DOMAINS:
  for fold in range(1,11):
   unit=OUT/'units'/d/f'fold_{fold}'
   if not (unit/'DONE.json').exists():raise RuntimeError('Incomplete tenfold: '+str(unit))
   audits.append(json.loads((unit/'DONE.json').read_text(encoding='utf-8')))
   f=pd.read_csv(unit/'COMPARISON.csv.gz');assert f.outer_validation_fold.eq(fold).all();frames.append(f)
 f=pd.concat(frames,ignore_index=True);assert len(f)==24129 and f.stable_record_id.is_unique
 g=pd.read_csv(GROUPED,low_memory=False).set_index('stable_record_id').loc[f.stable_record_id]
 assert np.array_equal(g.formal_fold,f.outer_validation_fold)
 assert np.allclose(g.observed_log10,f.observed_log10,atol=1e-6)
 assert np.array_equal(g.known_source_available,f.known_source_available)
 assert f.known_source_available.sum()==21386
 # The original environmental "unaware" runner simply omits the local known
 # error term from an aware-fitted remaining pool. A proper no-error refit
 # instead uses the independently calibrated total residual pool already
 # fit above for the nohier control. Same independently fitted BNN state and
 # calibration residuals make that the identical non-source environmental arm.
 f['full_environment_future_sd_noerror_refit']=f.nn_future_sd_noerror
 obs=pd.read_csv(OBS).set_index('stable_record_id').loc[f.stable_record_id]
 f['longitude']=obs.longitude.to_numpy();f['latitude']=obs.latitude.to_numpy()
 f['physical_site_key']=obs.longitude.round(4).astype(str).to_numpy()+'|'+obs.latitude.round(4).astype(str).to_numpy()
 f['location_seen_in_outer_training']=False
 for d in DOMAINS:
  for fold in range(1,11):
   train_sites=set(f.loc[f.domain.eq(d)&f.outer_validation_fold.ne(fold),'physical_site_key'])
   target=f.domain.eq(d)&f.outer_validation_fold.eq(fold)
   f.loc[target,'location_seen_in_outer_training']=f.loc[target,'physical_site_key'].isin(train_sites)
 f['location_status']=np.where(f.location_seen_in_outer_training,'known_coordinate_site','new_coordinate_site')
 for arm in ['error','noerror']:
  pc='known_source_conditional_q50_log10' if arm=='error' else 'hier_noerror_conditional_q50'
  sd='full_conditional_future_sd_error_uncapped' if arm=='error' else 'hier_noerror_conditional_sd_uncapped'
  f['hier_operational_q50_'+arm]=np.where(f.known_source_available,f[pc],f.predicted_q50_log10)
  envsd='full_environment_future_sd_error' if arm=='error' else 'full_environment_future_sd_noerror_refit'
  f['hier_operational_sd_'+arm]=np.where(f.known_source_available,f[sd],f[envsd])
 f['error_bin']=pd.cut(f.known_sd_log10,[-np.inf,.1,.3,1,np.inf],labels=['<=0.1','0.1-0.3','0.3-1','>1'])
 f.to_csv(OUT/'GROUPED10_COMPARISON_OOF.csv.gz',index=False)
 rows=[];boot=[];cal=[]
 models={'nonhier_noerror':('nn_retrained_q50_log10','nn_future_sd_noerror'),'nonhier_error':('nn_retrained_q50_log10','nn_future_sd_error'),'hier_noerror':('hier_operational_q50_noerror','hier_operational_sd_noerror'),'full_hier_error':('hier_operational_q50_error','hier_operational_sd_error'),'full_environment_error':('predicted_q50_log10','full_environment_future_sd_error'),'full_environment_noerror':('predicted_q50_log10','full_environment_future_sd_noerror_refit'),'full_environment_original_unaware':('predicted_q50_log10','full_environment_future_sd_noerror'),'RF_environment':('rf_environment_log10',None),'RF_context':('rf_context_log10',None)}
 for d in ['all',*DOMAINS]:
  local=f if d=='all' else f.loc[f.domain.eq(d)]
  strata=[('overall',local),('known_source',local.loc[local.known_source_available]),('new_source',local.loc[~local.known_source_available]),('known_location',local.loc[local.location_seen_in_outer_training]),('new_location',local.loc[~local.location_seen_in_outer_training])]+[(str(b),local.loc[local.error_bin.eq(b)]) for b in local.error_bin.cat.categories]
  for source_known in [True,False]:
   for place_known in [True,False]:
    strata.append((('known_source' if source_known else 'new_source')+'_'+('known_location' if place_known else 'new_location'),local.loc[local.known_source_available.eq(source_known)&local.location_seen_in_outer_training.eq(place_known)]))
  for stratum,part in strata:
   for model,(p,sd) in models.items():
    for w in (['record_equal','source_equal','waterbody_equal','waterbody_source_equal'] if d=='all' else ['record_equal','source_equal']):
     row=metric(part,p,sd,w,design='persisted_grouped10_new_fits',domain=d,prediction_mode=model,arm='uncapped_inner_calibration_no_outer_fold_adjustment',stratum=stratum)
     if row:rows.append(row)
   if len(part)>100 and part.source_group.nunique()>=2:
    for arm,p,sa,sb in [('nonhier','nn_retrained_q50_log10','nn_future_sd_error','nn_future_sd_noerror'),('hier','hier_operational_q50_error','hier_operational_sd_error','hier_operational_sd_noerror')]:
     boot.extend([{**x,'contrast':arm+'_error_minus_noerror','domain':d,'stratum':stratum,'folds':10,'seed':260826} for x in bootstrap_pair(part,p,sa,sb)])
    for label,p1,p0 in [('RF_context_minus_full_hier','rf_context_log10','hier_operational_q50_error'),('RF_environment_minus_nonhier','rf_environment_log10','nn_retrained_q50_log10'),('hier_minus_nonhier','hier_operational_q50_error','nn_retrained_q50_log10')]:
     boot.extend([{**x,'contrast':label,'domain':d,'stratum':stratum,'folds':10,'seed':260826} for x in point_bootstrap(part,p1,p0)])
  for model,(p,sd) in models.items():
   if sd is None:continue
   z=np.abs((local.observed_log10-local[p])/local[sd])
   for level in [.5,.6,.7,.8,.9,.95,.99]:
    cal.append({'domain':d,'model':model,'nominal':level,'empirical':np.mean(z<=norm.ppf((1+level)/2)),'records':len(local),'sources':local.source_group.nunique(),'folds':10,'seed':260826,'weighting':'record_equal'})
 metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'NEW_GROUPED10_METRICS.csv',index=False)
 pd.DataFrame(boot).to_csv(OUT/'NEW_PAIRED_SOURCE_BOOTSTRAP.csv',index=False);pd.DataFrame(cal).to_csv(OUT/'NEW_CALIBRATION_CURVES.csv',index=False)
 pd.DataFrame(audits).drop(columns=['full_fit','nonhier_fit','noerror_mem','memory_after']).to_csv(OUT/'TRAINING_UNIT_AUDIT_SUMMARY.csv',index=False)
 # Separate historical reproduction from new matched experiments.
 deltas=f[['stable_record_id','waterbody','outer_validation_fold','source_group']].copy()
 deltas['new_minus_frozen_environment_log10']=f.predicted_q50_log10.to_numpy()-g.predicted_q50_log10.to_numpy()
 deltas['nonhier_minus_new_full_environment_log10']=f.nn_retrained_q50_log10-f.predicted_q50_log10
 deltas.to_csv(OUT/'FROZEN_REPRODUCTION_RECORD_DIFFERENCES.csv.gz',index=False)
 qa={'status':'passed','completed_units':len(audits),'folds_per_domain':{d:10 for d in DOMAINS},'records':len(f),'known_source_records':int(f.known_source_available.sum()),'sources':f.source_group.nunique(),'site_events':f.validation_site_event_key.nunique(),'seed':260826,'epochs_max':180,'same_ids_folds_response':True,'all_nonhier_independent_fit_state_equal':all(x['nonhier_full_parameter_state_identical'] for x in audits),'environment_nonhier_mean_max_abs_difference':float(np.max(np.abs(f.nn_retrained_q50_log10-f.predicted_q50_log10))),'measurement_error_conditional_mean_max_abs_difference':float(np.nanmax(np.abs(f.hier_noerror_conditional_q50-f.known_source_conditional_q50_log10))),'new_vs_historical_environment_rmse_difference':float(np.sqrt(np.mean(deltas.new_minus_frozen_environment_log10**2))),'historical_exact_reproduction':bool(np.max(np.abs(deltas.new_minus_frozen_environment_log10))<1e-6),'frozen_results_modified':False,'new_full_interval_calibration':'uncapped inner leave-event-out/source calibration only; no other-outer-fold response calibration','RF_preprocessing':'same process feature/scaler on neural inner fit; context category and numeric medians learned only outer train','NN_scope':'BNN without source offset; same Bayesian weight prior retained. Not deterministic NN versus BNN ablation','posteriors_validated_against_latent_truth':False}
 qa['location_definition']='longitude/latitude rounded four decimals regardless source/year; seen if present in same-domain outer training; site-level overlap; spatial-block transfer is evaluated separately'
 qa['location_status_counts']=f.location_status.value_counts().to_dict()
 qa['source_bootstrap_limit']='fixed trained models, clustered across all records/folds of the same source; does not include retraining variability or multi-seed architecture selection'
 qa['noerror_environment_refit']='total residual pool with known variance zero and fresh inner calibration quantile, derived from identical independently optimized backbone; original unaware pool retained as a separate diagnostic'
 write_json(OUT/'NEW_EXPERIMENT_TEST_LOG.json',qa)
 fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
 select=metrics.loc[metrics.domain.eq('all')&metrics.stratum.eq('overall')&metrics.weighting.eq('record_equal')]
 plot=select.set_index('prediction_mode').reindex(['nonhier_noerror','nonhier_error','hier_noerror','full_hier_error','RF_environment','RF_context'])
 axes[0].barh(plot.index,plot.r2,color=['#848d9e','#848d9e','#27756d','#27756d','#b18048','#b18048']);axes[0].set(xlabel='Heldout observation R²',xlim=(-.1,1))
 interval=plot.dropna(subset=['wis']);axes[1].barh(interval.index,interval.wis,color='#27756d');axes[1].set(xlabel='WIS (lower is better)')
 fig.suptitle('New matched grouped tenfold; N=24,129; seed 260826; record equal')
 fig.savefig(OUT/'NEW_MATCHED_COMPARISON.png',dpi=220);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
 cc=pd.DataFrame(cal)
 for ax,d in zip(axes,DOMAINS):
  for model,color,style in [('nonhier_noerror','#7d8494','-'),('nonhier_error','#7d8494','--'),('hier_noerror','#27756d','-'),('full_hier_error','#27756d','--')]:
   part=cc.loc[cc.domain.eq(d)&cc.model.eq(model)]
   ax.plot(part.nominal,part.empirical,style,color=color,label=f'{model} (N={int(part.records.iloc[0])})')
  ax.plot([.5,1],[.5,1],':',color='gray');ax.set(title=d,xlabel='Nominal coverage',ylabel='Heldout observation coverage',ylim=(.4,1.01));ax.legend(fontsize=7)
 fig.suptitle('New grouped tenfold; seed 260826; record equal; inner calibration only')
 fig.savefig(OUT/'NEW_INTERVAL_CALIBRATION.png',dpi=200);plt.close(fig)
 print(json.dumps(qa,ensure_ascii=False),flush=True)
 print(select[['prediction_mode','records','r2','rmse_log10','mae','bias','wis','coverage95','width95']].to_string(index=False),flush=True)
if __name__=='__main__':run()
