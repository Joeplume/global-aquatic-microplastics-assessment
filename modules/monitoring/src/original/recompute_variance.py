"""Isolated QA/QC downstream variance updates; original files remain read-only."""
from pathlib import Path
import argparse, importlib.util, json, hashlib, os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[2]
EXP=ROOT.parents[1]/'weighted_training'
H=PROJECT/'src/modeling/h26'
OBS=H/'data/observations/h23_observation_layer_training_records.csv.gz'
MDC=H/'code/mdc_monitoring/build_h26_final10_mdc.py'
REML=PROJECT/'figures/comparisons/.mdc_refit/refit_mdc.py'

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def components(mode):
    m=module(MDC,'source_mdc'); rows=[]; frames_all=[]; calibration={}
    oof=pd.read_csv(EXP/f'aggregate_results/OOF_{mode}.csv.gz',low_memory=False)
    for domain in ['freshwater','transition','marine']:
        frames=[];audits=[]
        for fold in range(1,11):
            frame=oof[oof.domain.eq(domain)&oof.outer_validation_fold.eq(fold)].copy()
            assert len(frame)
            auditpaths=list((EXP/f'units/{mode}/{domain}/fold_{fold}/S0/protocol_aware').glob('*/H26_UNIT_AUDIT.json'))
            assert len(auditpaths)==1
            audit=json.loads(auditpaths[0].read_text(encoding='utf-8'))
            layer=audit['known_source_observation_layer']
            known=frame.known_source_available.to_numpy(bool)
            frame['formal_fold']=fold
            # These are the exact two interval branches of the frozen runner.
            # Both branches have identical fitted concentration and source means.
            frame['known_source_conditional_future_sd_log10_aware']=frame.known_source_conditional_future_sd_log10*float(layer['aware_unclipped_interval_multiplier'])/float(layer['aware_interval_multiplier'])
            unaware=float(layer['unaware_interval_multiplier'])*np.sqrt(frame.latent_sd_log10**2+float(layer['within_source_total_sd_log10'])**2)
            frame['known_source_conditional_future_sd_log10_unaware']=np.where(known,unaware,np.nan)
            frames.append(frame);audits.append(audit)
        frames,factors=m.crossfold_conditional_calibration(frames)
        result,local=m.variance_components(domain,frames,audits)
        rows.append(result);frames_all.extend(frames);calibration[domain]=factors
        print(json.dumps({k:result[k] for k in ['domain','between_source_sd_log10','within_source_total_effective_sd_log10','reported_sd_pooled_log10']},ensure_ascii=False),flush=True)
    out=ROOT/mode;out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'H26_MDC_VARIANCE_COMPONENTS.csv',index=False)
    pd.concat(frames_all).to_csv(out/'MDC_OOF_CALIBRATED.csv.gz',index=False)
    (out/'CALIBRATION_FACTORS.json').write_text(json.dumps(calibration,indent=2),encoding='utf-8')
    return pd.DataFrame(rows)

def reml(mode,reps):
    m=module(REML,'source_reml')
    d=pd.read_csv(EXP/f'aggregate_results/OOF_{mode}.csv.gz',low_memory=False)
    o=pd.read_csv(OBS,usecols=['stable_record_id','sizecorr_known_sd_log10','unit_sd_log10'])
    d=d.merge(o,on='stable_record_id',validate='one_to_one')
    assert len(d)==24129
    d['residual_log10']=d.observed_log10-d.predicted_q50_log10
    d['known_var']=d.sizecorr_known_sd_log10**2+d.unit_sd_log10**2
    rng=np.random.default_rng(20261013);rows=[];drawrows=[]
    out=ROOT/mode;out.mkdir(parents=True,exist_ok=True)
    for domain in ['freshwater','transition','marine']:
        x=d[d.domain.eq(domain)].copy();a=m.arrays(x);base=m.fit(a)
        known=float(x.known_var.mean());base.update(domain=domain,records=len(x),sources=a[3],site_events=x.validation_site_event_key.nunique(),reported_var=known)
        for scenario,var in [('changing',base['between_var']+base['remaining_var']+known),('stable',base['remaining_var']+known),('standardized',base['remaining_var'])]:
            base['V_'+scenario]=var
            base['N30_'+scenario]=int(np.ceil(m.power_factor()*var/np.log10(.7)**2))
        rows.append(base);pd.DataFrame(rows).to_csv(out/'REFIT_POINT_ESTIMATES.csv',index=False)
        print(json.dumps(base),flush=True)
        G=a[3];g=a[2];counts=np.bincount(g,minlength=G);ks=np.bincount(g,x.known_var.to_numpy(),minlength=G)
        cache=out/f'BOOTSTRAP_{domain}.csv'
        old=pd.read_csv(cache) if cache.exists() else pd.DataFrame()
        for b in range(reps):
            mult=rng.multinomial(G,np.ones(G)/G)
            if len(old)>b:
                drawrows.append(old.iloc[b].to_dict());continue
            f=m.fit(a,mult,[base['between_var'],base['remaining_var']])
            f.update(domain=domain,replicate=b,reported_var=float(mult@ks/(mult@counts)))
            drawrows.append(f)
            if (b+1)%100==0 or b==reps-1:
                pd.DataFrame([r for r in drawrows if r['domain']==domain]).to_csv(cache,index=False)
                print(domain,b+1,flush=True)
    if drawrows:pd.DataFrame(drawrows).to_csv(out/'SOURCE_BOOTSTRAP_DRAWS.csv',index=False)
    manifest={'mode':mode,'reps':reps,'seed':20261013,'original_program':str(REML),'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [EXP/f'aggregate_results/OOF_{mode}.csv.gz',OBS,REML,MDC]},'prediction_target':'environmental OOF residuals; latent and reported variances retained as original REML method','measurement_scenarios':'unchanged definitions; standardization excludes reported variance, not natural variation'}
    (out/'VARIANCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',default='qaqc');ap.add_argument('--task',choices=['components','reml'],required=True);ap.add_argument('--reps',type=int,default=2000);a=ap.parse_args()
    components(a.mode) if a.task=='components' else reml(a.mode,a.reps)
