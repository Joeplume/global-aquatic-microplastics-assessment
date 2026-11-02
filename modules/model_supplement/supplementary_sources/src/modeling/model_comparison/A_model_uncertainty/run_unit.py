"""Run original H26 mean/MEM code; independently retrain the no-source-layer mean.

All modifications are in-process adapters; frozen source files stay untouched.
Outer labels always come from the persisted grouped OOF, never regenerated.
"""
from common import *
import argparse,numpy as np,pandas as pd,torch,importlib,copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
torch.set_num_threads(2);torch.set_num_interop_threads(2)
p=argparse.ArgumentParser();p.add_argument('--domain',choices=DOMAINS,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--epochs',type=int,default=180);p.add_argument('--smoke',action='store_true');p.add_argument('--grid',action='store_true');a=p.parse_args()
d=a.domain;cfg=CONFIG[d];domain=DOMAINS[d]
unit=OUT/('smoke' if a.smoke else 'units')/d/f'fold_{a.fold}'
unit.mkdir(parents=True,exist_ok=True)
start=time.time()
def rngsave():return (torch.get_rng_state().clone(),np.random.get_state())
def rngrestore(s):torch.set_rng_state(s[0]);np.random.set_state(s[1])
with TrainingMutex():
 import run_h26_marine_vector_modular_bnn as strict
 import run_h26_freshwater_network_modular_bnn as fresh
 import run_h24_simple_protocol_preview as h24
 mod=importlib.import_module(cfg['runner'].stem)
 # Transition config changes only the original shared runner's aliases in this process.
 if d=='transition':mod.configure_shared_runner();runner=fresh
 else:runner=mod
 parser=mod.parser()
 args=parser.parse_args(['--candidate',cfg['candidate'],'--training',str(cfg['training']),'--prediction',str(cfg['prediction']),'--observation',str(OBS),'--caoa',str(CAOA),'--folds-csv',str(OUT/'FROZEN_GROUPED10_FOLDS.csv.gz'),'--fold',str(a.fold),'--seed','260826','--validation-scheme','random_site_event_10fold','--observation-scale-model','fixed_reported_propagation','--device','cpu','--epochs',str(a.epochs),'--posterior-draws','24','--calibration-crossfits','3','--map-cap-log10',str(cfg['cap']),'--run-root',str(unit)]+([] if a.grid else ['--oof-only']))
 frozen=pd.read_csv(GROUPED,low_memory=False)
 local_frozen=frozen.loc[frozen.waterbody.eq(domain)].copy()
 original_load=runner.load_inputs
 captured={}
 def load(*values,**kw):
  records,grid,audit=original_load(*values,**kw)
  assert set(records.stable_record_id)==set(local_frozen.stable_record_id),'loader cohort differs from frozen cohort'
  captured['records']=records;captured['grid']=grid
  return records,grid,audit
 runner.load_inputs=load
 original_split=strict.choose_outer_split
 def split(records,args):
  lookup=local_frozen.set_index('stable_record_id')
  f=records.stable_record_id.map(lookup.formal_fold).to_numpy(int)
  train=np.flatnonzero(f!=args.fold);test=np.flatnonzero(f==args.fold)
  event,_=strict.build_site_event_key(records)
  assert set(event.iloc[train]).isdisjoint(set(event.iloc[test]))
  # Check exact event definition, not only grouping counts.
  assert (event.to_numpy()==records.stable_record_id.map(lookup.validation_site_event_key).to_numpy()).all()
  captured.update(outer_train=train,test=test)
  trsrc=set(records.iloc[train].source_group);tesrc=set(records.iloc[test].source_group)
  return train,test,{'scheme':'random_site_event_10fold','kind':'persisted_frozen_grouped_labels','folds_csv_used':True,'folds_csv':str(OUT/'FROZEN_GROUPED10_FOLDS.csv.gz'),'seed':260826,'requested_fold':args.fold,'training_records':len(train),'heldout_records':len(test),'source_overlap':len(trsrc&tesrc),'train_test_site_event_overlap':0,'response_stratification_used':False}
 strict.choose_outer_split=split
 fitname='fit_model' if d=='marine' else 'fit_candidate_model'
 original_fit=getattr(runner,fitname)
 fitcalls=0
 def fit(*values,**kw):
  global fitcalls
  fitcalls+=1
  model,audit=original_fit(*values,**kw)
  if fitcalls==1:
   x,y,w,train,val,branches,scale=values[:7]
   captured.update(x=x,y=y,weights=w,fit=train,early_stop=val,branches=branches,scale=scale,fit_audit=audit)
   # Independent optimization with the same backbone, optimizer, early stop,
   # seed and maximum epochs. No fitted source offset is merely zeroed.
   post_rng=rngsave()
   nn_model,nn_audit=original_fit(*values,**kw)
   state_equal=all(torch.equal(model.state_dict()[k],nn_model.state_dict()[k]) for k in model.state_dict())
   # Test median matches official posterior draw stream by restoring post-fit RNG.
   rngrestore(post_rng)
   q,det,sd,_=strict.predict(nn_model,x[captured['test']],24)
   captured.update(nn_q=q,nn_det=det,nn_sd=sd,nn_fit_audit=nn_audit,nn_state_equal=state_equal)
   torch.save({'state_dict':nn_model.state_dict(),'fit':nn_audit,'target_scale':scale,'branches':branches,'role':'independently_retrained_BNN_without_source_offset'},unit/'NONHIER_MODEL.pt')
   torch.save({'state_dict':model.state_dict(),'fit':audit,'role':'original_modular_BNN_before_MEM'},unit/'FULL_MEAN_MODEL.pt')
   del nn_model
   rngrestore(post_rng)
  return model,audit
 setattr(runner,fitname,fit)
 # Capture outer-train calibration arrays, allowing error removal to be fit
 # independently in the variance layer without reusing the fitted error offset.
 original_scale=strict.observation_scale_fit
 def scale_fit(records,cal,test,residual,latent,known,weights,**kw):
  captured.update(cal=cal,cal_residual=residual,cal_latent=latent,known=known)
  return original_scale(records,cal,test,residual,latent,known,weights,**kw)
 strict.observation_scale_fit=scale_fit
 original_mem=strict.fit_known_source_observation_layer
 def mem(*values,**kw):
  result=original_mem(*values,**kw)
  zero_values=list(values);zero_values[6]=np.zeros_like(values[6])
  captured['noerror_mem']=original_mem(*zero_values,**kw)
  return result
 strict.fit_known_source_observation_layer=mem
 runner.train(args)
 if d=='transition':mod.rewrite_audits(args)
 aware=pd.read_csv(unit/'S0/protocol_aware'/domain/'H26_UNIT_HELDOUT.csv.gz')
 unaware=pd.read_csv(unit/'S0/detection_unaware'/domain/'H26_UNIT_HELDOUT.csv.gz')
 records=captured['records'];test=captured['test'];cal=captured['cal'];x=captured['x'];w=captured['weights']
 fullaudit=json.loads((unit/'S0/protocol_aware'/domain/'H26_UNIT_AUDIT.json').read_text(encoding='utf-8'))
 center=fullaudit['target_center_log10'];scale=fullaudit['target_scale_log10']
 nn_q=center+scale*captured['nn_q'];nn_sd=scale*captured['nn_sd']
 y=aware.observed_log10.to_numpy(float)
 # Nonhierarchical predictive scale: single residual pool, no source offsets.
 total=np.average(np.maximum(captured['cal_residual']**2-captured['cal_latent']**2,0.05**2),weights=w[cal])
 known=captured['known'];reported=np.average(known[cal]**2,weights=w[cal]);remaining=max(total-reported,0.05**2)
 nn_sds={}
 for label,meas in [('noerror',False),('error',True)]:
  cal_sd=np.sqrt(captured['cal_latent']**2+(remaining if meas else total)+(known[cal]**2 if meas else 0))
  multiplier=strict.weighted_quantile(np.abs(captured['cal_residual'])/np.maximum(cal_sd,1e-6),w[cal],.95)/1.96
  nn_sds[label]=multiplier*np.sqrt(nn_sd**2+(remaining if meas else total)+(known[test]**2 if meas else 0))
 # RF information controls use the same selected process features and train IDs.
 # Scaler from neural inner-fit set retained; categorical encodings fit outer train.
 outer=captured['outer_train'];context_columns=['detection_method','sizecorr_evidence_tier','source_group']
 cat=records.reindex(columns=context_columns).fillna('__missing__').astype(str)
 enc=OneHotEncoder(handle_unknown='ignore',sparse_output=True);enc.fit(cat.iloc[outer])
 extra=[]
 for name in ['sampling_year_effective_v1_3','qaqc_total_score','sizecorr_lower_used_um']:
  z=pd.to_numeric(records[name],errors='coerce').to_numpy(float)
  miss=~np.isfinite(z);finite=z[outer][np.isfinite(z[outer])];med=float(np.median(finite)) if len(finite) else 0
  extra.extend([np.where(miss,med,z),miss.astype(float)])
 xcontext=sparse.hstack([sparse.csr_matrix(x),sparse.csr_matrix(np.column_stack(extra)),enc.transform(cat)],format='csr')
 rfout={}
 for name,z in [('environment',x),('context',xcontext)]:
  rf=RandomForestRegressor(n_estimators=400 if not a.smoke else 20,max_features=.7,min_samples_leaf=5,n_jobs=1,random_state=260826+a.fold+1000*list(DOMAINS).index(d))
  rf.fit(z[outer],records.observed_log10.to_numpy(float)[outer],sample_weight=w[outer])
  rfout[name]=rf.predict(z[test]);del rf
 result=aware[['stable_record_id','waterbody','source_group','validation_site_event_key','outer_validation_fold','observed_log10','known_source_available','predicted_q50_log10','latent_sd_log10','known_source_conditional_q50_log10']].copy()
 result['domain']=d;result['nn_retrained_q50_log10']=nn_q
 result['nn_future_sd_noerror']=nn_sds['noerror'];result['nn_future_sd_error']=nn_sds['error']
 result['full_environment_future_sd_error']=aware.future_observation_sd_log10
 result['full_environment_future_sd_noerror']=unaware.future_observation_sd_log10
 result['full_conditional_future_sd_error']=aware.known_source_conditional_future_sd_log10
 result['full_conditional_future_sd_noerror']=unaware.known_source_conditional_future_sd_log10
 # Formal aggregation removes the development multiplier cap. Retain both
 # raw runner SD and uncapped, outer-train-only SD; no outer-fold calibration.
 ma=fullaudit['known_source_observation_layer']
 result['full_conditional_future_sd_error_uncapped']=result.full_conditional_future_sd_error*ma['aware_unclipped_interval_multiplier']/ma['aware_interval_multiplier']
 result['full_conditional_future_sd_noerror_uncapped']=result.full_conditional_future_sd_noerror*ma['unaware_unclipped_interval_multiplier']/ma['unaware_interval_multiplier']
 for name,z in rfout.items():result[f'rf_{name}_log10']=z
 # Refit no-error empirical-Bayes/MEM layer; offset math deliberately invariant
 # because known SD is absent from the mean offset estimation in official code.
 offset,available,total_sd,remaining_sd,mulaware,mulunaware,mem_audit=captured['noerror_mem']
 result['hier_noerror_conditional_q50']=np.where(available,aware.predicted_q50_log10.to_numpy()+offset,np.nan)
 result['hier_noerror_conditional_sd']=np.where(available,mulunaware*np.sqrt(result.latent_sd_log10**2+total_sd**2),np.nan)
 result['hier_noerror_conditional_sd_uncapped']=result.hier_noerror_conditional_sd*mem_audit['unaware_unclipped_interval_multiplier']/mem_audit['unaware_interval_multiplier']
 result['known_sd_log10']=known[test]
 result.to_csv(unit/'COMPARISON.csv.gz',index=False)
 frozen_compare=local_frozen.set_index('stable_record_id').loc[result.stable_record_id]
 audit={'status':'smoke_completed' if a.smoke else 'completed','domain':d,'fold':a.fold,'seed':260826,'epochs_max':a.epochs,'posterior_draws':24,'records':len(result),'source_count':result.source_group.nunique(),'site_events':result.validation_site_event_key.nunique(),'elapsed_seconds':time.time()-start,'device':'cpu','threads_max':2,'mean_fit_calls_including_crossfits':fitcalls,'nonhier_independently_retrained':True,'nonhier_full_parameter_state_identical':captured['nn_state_equal'],'nonhier_full_q50_max_abs_diff':float(np.max(np.abs(result.nn_retrained_q50_log10-result.predicted_q50_log10))),'measurement_removed_offset_max_abs_diff':float(np.nanmax(np.abs(result.hier_noerror_conditional_q50-result.known_source_conditional_q50_log10))),'original_frozen_environment_q50_max_abs_diff':float(np.max(np.abs(frozen_compare.predicted_q50_log10.to_numpy()-result.predicted_q50_log10.to_numpy()))),'original_frozen_environment_q50_rmse_diff':float(np.sqrt(np.mean((frozen_compare.predicted_q50_log10.to_numpy()-result.predicted_q50_log10.to_numpy())**2))),'full_fit':captured['fit_audit'],'nonhier_fit':captured['nn_fit_audit'],'noerror_mem':mem_audit,'training_budget':'same main mean fit and early-stop; full uncertainty additionally uses three inner source crossfits; nonhier variance reuses identical crossfits','rf_information':'environment versus context method/tier/source plus numeric year/QAQC/lower-size; outer-train fit; test categories unknown ignored','heldout_response_used_for_design':False,'checkpoint_warning':'state/scaler metadata in original H26_UNIT_AUDIT; no preexisting fold checkpoint available to establish exact historical equality','peak_memory_not_measured':True,'memory_after':memory()}
 audit['mean_fit_calls_including_crossfits']=1+len(fullaudit['observation_scale_calibration_predictions'].get('folds',[]))
 audit['independent_nonhier_main_fits']=1
 audit['checkpoint_target_center_log10']=center
 audit['checkpoint_target_scale_log10']=scale
 audit['uncapped_interval_note']='uncapped factor from inner train only; raw capped runner intervals also retained; no other outer folds used to recalibrate'
 write_json(unit/'DONE.json',audit)
 print(json.dumps({k:audit[k] for k in ['status','domain','fold','records','elapsed_seconds','nonhier_full_parameter_state_identical','original_frozen_environment_q50_rmse_diff']},ensure_ascii=False),flush=True)
