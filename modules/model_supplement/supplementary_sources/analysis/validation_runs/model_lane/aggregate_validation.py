"""Read isolated validation outputs and emit numerical replacement evidence."""
from pathlib import Path
import json, sys, argparse
import numpy as np
import pandas as pd
from scipy.stats import norm

P = Path(__file__).resolve().parent
Q = P.parents[1] / 'weighted_training'
sys.path.insert(0,str(Q))
import common as c
H = c.H
A = c.ROOT/'model_comparison/A_model_uncertainty'
C = c.ROOT/'model_comparison/C_caoa_validation'
DOC = P.parent
arg=argparse.ArgumentParser();arg.add_argument('--partial',action='store_true');args=arg.parse_args()
docs={x:json.loads((DOC/(x+'.json')).read_text(encoding='utf8')) for x in ['si_en','si_zh','main_en','main_zh']}
replacements=[];warnings=[];audits={}

def write(name,data):
    (P/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
def bools(s): return s.astype(str).str.lower().isin(['true','1']).to_numpy()
def table(num,row,col,new,evidence):
    for language in ['en','zh']:
        document='si_'+language;old=docs[document]['tables'][num-1]['rows'][row][col]
        if old!=str(new): replacements.append(dict(document=document,kind='table_cell',table_index=num-1,row=row,col=col,old=old,new=str(new),evidence=str(P/evidence)))
def para(pid,old,new,evidence):
    if old==new:return
    for language in ['en','zh']:
        document='main_'+language;item=next(x for x in docs[document]['paragraphs'] if x['id']==pid)
        assert item['text'].count(old)==1,(document,pid,old)
        replacements.append(dict(document=document,kind='paragraph',paragraph_id=pid,old=old,new=new,evidence=str(P/evidence)))
def score(d,q,sd=None,w=None):
    y=d.observed_log10.to_numpy(float);q=np.asarray(q,float);w=np.ones(len(d)) if w is None else np.asarray(w,float)
    ok=np.isfinite(y)&np.isfinite(q)&(w>0);y,q,w=y[ok],q[ok],w[ok]
    e=q-y;avg=lambda a:np.average(a,weights=w)
    mse=avg(e*e);variance=avg((y-avg(y))**2)
    r=dict(n=len(y),r2=1-mse/variance if variance>0 else None,rmse=float(np.sqrt(mse)),mae=float(avg(abs(e))),bias=float(avg(e)))
    if sd is not None:
        sd=np.asarray(sd,float)[ok];assert np.isfinite(sd).all() and (sd>0).all()
        wis=.5*abs(e)
        for alpha in [.5,.2,.1,.05]:
            z=norm.ppf(1-alpha/2);lo=q-z*sd;hi=q+z*sd
            wis+=alpha/2*(hi-lo+2/alpha*np.maximum(lo-y,0)+2/alpha*np.maximum(y-hi,0))
        r.update(wis=float(avg(wis/4.5)),coverage95=float(avg(abs(e)<=1.96*sd)),width95=float(avg(2*norm.ppf(.975)*sd)))
    return r

def unit(tag,domain,fold):return P/'units'/tag/domain/f'fold_{fold}'
def ready(tag,folds):
    paths=[unit(tag,d,f) for d in c.DOMAINS for f in range(1,folds+1)]
    completed=sum((p/'DONE.json').exists() for p in paths)
    audits[tag]={'complete_units':completed,'expected_units':len(paths)}
    if completed<len(paths):
        if args.partial:return False
        raise RuntimeError(f'Incomplete {tag}: {completed}/{len(paths)}')
    return True

def load(tag,folds):
    frames=[]
    for domain,waterbody in c.DOMAINS.items():
        for fold in range(1,folds+1):
            folder=unit(tag,domain,fold)/'S0/protocol_aware'/waterbody
            d=pd.read_csv(folder/'H26_UNIT_HELDOUT.csv.gz',low_memory=False)
            audit=json.loads((folder/'H26_UNIT_AUDIT.json').read_text(encoding='utf8'))
            k=bools(d.known_source_available);d['known_source_available']=k
            d['domain']=domain;d['fold']=fold
            d['q_operational']=np.where(k,d.known_source_conditional_q50_log10,d.predicted_q50_log10)
            d['sd_operational_raw']=np.where(k,d.known_source_conditional_future_sd_log10,d.future_observation_sd_log10)
            branch=audit['known_source_observation_layer']
            factor=branch['aware_unclipped_interval_multiplier']/branch['aware_interval_multiplier'] if k.any() else 1.0
            d['conditional_sd_uncapped']=d.known_source_conditional_future_sd_log10*factor
            d['sd_operational_a_uncapped']=np.where(k,d.conditional_sd_uncapped,d.future_observation_sd_log10)
            frames.append(d)
    d=pd.concat(frames,ignore_index=True)
    assert len(d)==24129 and d.stable_record_id.is_unique
    d.to_csv(P/(tag.replace('/','_')+'_OOF.csv.gz'),index=False)
    return d

# Independent descriptives are preserved. Residual variance changes with the adopted environment.
v=json.loads((P/'TABLE3_VARIANCE.json').read_text(encoding='utf8'))
print('variance keys',v.keys(),flush=True)
shares=v.get('fractions',v.get('shares',v))
if 'components' in shares:shares=shares['components']
# Exact fit output is explicitly inspected rather than guessed by a document parser.
for column,key in [(4,'geotime'),(5,'method'),(6,'source'),(7,'known_measurement'),(8,'remaining_after_known_measurement')]:
    value=shares[key]
    table(3,1,column,f'{100*value:.4f}' if column==8 else f'{100*value:.2f}','TABLE3_VARIANCE.json')
para('p0020','61.3%',f"{100*shares['source']:.1f}%",'TABLE3_VARIANCE.json')
para('p0020','4.0%',f"{100*shares['method']:.1f}%",'TABLE3_VARIANCE.json')

group=pd.read_csv(Q/'aggregate_results/OOF_qaqc.csv.gz',low_memory=False)
group['known_source_available']=bools(group.known_source_available)
gmetrics=[]
for row,qcol,known in [(1,'predicted_q50_log10',False),(2,'known_source_conditional_q50_log10',True)]:
    d=group.loc[group.known_source_available] if known else group
    r=score(d,d[qcol]);gmetrics.append(dict(target=qcol,**r))
    table(4,row,1,str(r['n']),'TABLE4_QAQC_METRICS.csv')
    table(4,row,2,f"{r['r2']:.4f}",'TABLE4_QAQC_METRICS.csv')
    for col,key in [(3,'rmse'),(4,'mae')]:table(4,row,col,f"{r[key]:.3f}" if row==1 else f"{r[key]:.4f}",'TABLE4_QAQC_METRICS.csv')
pd.DataFrame(gmetrics).to_csv(P/'TABLE4_QAQC_METRICS.csv',index=False)

if ready('random10',10):
    random=load('random10',10)
    metrics=[]
    for domain in ['overall',*c.DOMAINS]:
        d=random if domain=='overall' else random[random.domain==domain]
        for target,qcol in [('operational','q_operational'),('environment','predicted_q50_log10'),('known_source','known_source_conditional_q50_log10')]:
            e=d[d.known_source_available] if target=='known_source' else d
            metrics.append(dict(domain=domain,target=target,**score(e,e[qcol])))
    metrics=pd.DataFrame(metrics);metrics.to_csv(P/'RANDOM10_QAQC_METRICS.csv',index=False)
    overall=metrics[(metrics.domain=='overall')&(metrics.target=='operational')].iloc[0]
    known=metrics[(metrics.domain=='overall')&(metrics.target=='known_source')].iloc[0]
    table(25,2,4,f'{overall.r2:.4f}','RANDOM10_QAQC_METRICS.csv');table(25,2,5,f'{overall.rmse:.4f}','RANDOM10_QAQC_METRICS.csv')
    para('p0024','0.92',f'{known.r2:.2f}','RANDOM10_QAQC_METRICS.csv')
    warnings.append({'topic':'Main R2 population synchronization','location':'main_en/main_zh p0024','text':f"The sentence specifies record-random tenfold AND available source references. Its current matching subset has {int(known.n)} records and R2={known.r2:.6f}. Pooled operational R2={overall.r2:.6f}. The old 0.92 also occurs in grouped known-source scores; 0.92 to 0.93 is not a clean QAQC treatment-effect estimate."})
    controls=pd.read_csv(c.ROOT.parent.parent/'figures/comparisons/scripts/random_cv/METRICS.csv')
    six=controls[(controls.domain=='overall')&(~controls.model.isin(['H26','BNN_nonhier']))].copy()
    six['new_h26_rmse']=overall.rmse;six['reduction_pct']=100*(1-overall.rmse/six.RMSE)
    six.to_csv(P/'RANDOM10_FIXED_COMPARATOR_GAINS.csv',index=False)
    para('p0024','39.9–67.3%',f'{six.reduction_pct.min():.1f}–{six.reduction_pct.max():.1f}%','RANDOM10_FIXED_COMPARATOR_GAINS.csv')
    warnings.append({'topic':'Fixed conventional comparator weights','location':'Figure 2a/b and Supplementary Table 25','text':'Six traditional models and the old record-random nonhierarchical control remain frozen comparator results; only the H26 QAQC random model was retrained. Figure 2a/b therefore combines structural/model differences with QAQC weighting differences; its entire gain cannot be attributed to hierarchical structure. The independently refitted grouped neural comparison in Supplementary Table 25 uses matched QAQC weights and provides the corresponding structural-ablation evidence. No prose or figure values changed for this warning.'})
    # Static-support descriptive strata: complete frozen labels, not fold-local evidence validation.
    labels=pd.read_csv(H/'evaluation/benchmark/random/H26_RECORD_CAOA_CURRENT_DOMAIN.csv.gz',usecols=['stable_record_id','caoa_status'])
    r=random.merge(labels,on='stable_record_id',how='left',validate='one_to_one')
    assert len(r)==24129 and r.caoa_status.notna().all()
    r['tier']=r.caoa_status.map({'CAOA_in':'in','CAOA_limit':'limited','CAOA_single_source':'out','CAOA_out_unmatched':'out'})
    assert r.tier.notna().all()
    rows=[]
    for domain in ['freshwater','transition','marine','all']:
        for tier in ['in','limited','out']:
            d=r[(r.domain.eq(domain) if domain!='all' else np.ones(len(r),bool))&r.tier.eq(tier)]
            rows.append(dict(domain=domain,tier=tier,**score(d,d.q_operational,d.sd_operational_a_uncapped)))
    table19=pd.DataFrame(rows);table19.to_csv(P/'TABLE19_QAQC_STRATA.csv',index=False)
    for i,row in enumerate(rows,1):
        table(19,i,2,str(row['n']),'TABLE19_QAQC_STRATA.csv')
        for col,key,digits in [(3,'rmse',3),(4,'wis',3),(5,'coverage95',2)]:
            value=row[key]*(100 if key=='coverage95' else 1)
            table(19,i,col,f'{value:.{digits}f}','TABLE19_QAQC_STRATA.csv')
    audits['table19']={'records':len(r),'old_records':23934,'old_join_omissions_corrected':195,'missing_label_connections':0,'labels_modified':False,'tier_mapping':{'CAOA_in':'in','CAOA_limit':'limited','CAOA_single_source':'out','CAOA_out_unmatched':'out'},'interval':'conditional uncapped inner calibration; raw persisted environment for unknown sources; no outer-fold recalibration'}
    warnings.append({'topic':'Table 19 definition','location':'si_en p1585 and corresponding ZH','text':'Old table note says cross-fold interval calibration. Recomputed full-cohort table uses training-only calibration; that sentence needs synchronization. Table19 uses frozen full-data support labels and remains a descriptive stratification, not fold-local C-AOA validation. The 195-row join correction is independent of QAQC weighting.'})

# Strict spatial/source/joint results have the original operational observation target.
strictframes={};strictrows=[]
for i,design in enumerate(['spatial','source','joint'],1):
    if not ready(design,5):continue
    d=load(design,5);strictframes[design]=d
    for domain in ['all',*c.DOMAINS]:
        dd=d if domain=='all' else d[d.domain==domain]
        for target in ['operational','environment','known_source','unknown_source']:
            e=dd.loc[dd.known_source_available] if target=='known_source' else (dd.loc[~dd.known_source_available] if target=='unknown_source' else dd)
            if not len(e):continue
            q=e.predicted_q50_log10 if target in ['environment','unknown_source'] else e.q_operational
            sd=e.future_observation_sd_log10 if target in ['environment','unknown_source'] else e.sd_operational_raw
            row=dict(design=design,domain=domain,target=target,**score(e,q,sd));strictrows.append(row)
            if domain=='all' and target=='operational':
                table(5,i,1,str(row['n']),'STRICT_QAQC_METRICS.csv')
                for col,key,digits in [(2,'r2',4),(3,'rmse',3),(4,'mae',3),(5,'coverage95',2)]:
                    value=row[key]*(100 if key=='coverage95' else 1);table(5,i,col,f'{value:.{digits}f}','STRICT_QAQC_METRICS.csv')
pd.DataFrame(strictrows).to_csv(P/'STRICT_QAQC_METRICS.csv',index=False)

if 'source' in strictframes:
    d=strictframes['source'];membership=pd.read_csv(C/'SELECTION_WEIGHTS_AT_50PCT.csv.gz',low_memory=False)
    membership=membership[(membership.design=='strict_source_formal')&(membership.task=='environmental_unknown_source')]
    old=pd.read_csv(C/'MAIN_TABLE_STRICT_SOURCE_50PCT.csv')
    rows=[]
    for i,r in old.iterrows():
        m=membership[(membership.waterbody==r.waterbody)&(membership.rule==r.rule)]
        local=d[d.waterbody==r.waterbody].merge(m[['stable_record_id','selection_weight']],on='stable_record_id',validate='one_to_one')
        assert len(local)==r.total_records and local.selection_weight.notna().all()
        cluster=local.source_group.map(lambda s:'paper::'+str(s).split('paper::',1)[1].split('|',1)[0] if 'paper::' in str(s) else 'external_batch_surrogate::'+str(s))
        den=local.selection_weight.groupby(cluster).transform('sum')
        w=np.divide(local.selection_weight,den,out=np.zeros(len(local)),where=den>0)
        row=dict(waterbody=r.waterbody,rule=r.rule,total_records=len(local),**score(local,local.predicted_q50_log10,local.future_observation_sd_log10,w))
        rows.append(row)
        for col,key in [(4,'mae'),(5,'rmse'),(6,'wis'),(7,'coverage95')]:
            table(26,i+1,col,f"{row[key]*(100 if key=='coverage95' else 1):.4f}",'TABLE26_QAQC_50PCT.csv')
    pd.DataFrame(rows).to_csv(P/'TABLE26_QAQC_50PCT.csv',index=False)

# Same-backbone grouped neural comparisons; noerror is independently calibrated.
if ready('grouped_comparison',10):
    d=load('grouped_comparison',10);frames=[]
    for domain,waterbody in c.DOMAINS.items():
        for fold in range(1,11):
            u=unit('grouped_comparison',domain,fold)
            n=pd.read_csv(u/'NONHIER_COMPARISON.csv.gz')
            folder=u/'S0/detection_unaware'/waterbody
            unaware=pd.read_csv(folder/'H26_UNIT_HELDOUT.csv.gz')
            ad=json.loads((folder/'H26_UNIT_AUDIT.json').read_text(encoding='utf8'))['known_source_observation_layer']
            n['hier_sd_noerror_uncapped']=n.stable_record_id.map(unaware.set_index('stable_record_id').known_source_conditional_future_sd_log10)*(ad['unaware_unclipped_interval_multiplier']/ad['unaware_interval_multiplier'])
            frames.append(n.drop(columns=['observed_log10','source_group','known_source_available']))
    d=d.merge(pd.concat(frames),on='stable_record_id',validate='one_to_one')
    qcompare=d.set_index('stable_record_id').predicted_q50_log10-group.set_index('stable_record_id').predicted_q50_log10
    assert np.max(abs(qcompare))<1e-6,float(np.max(abs(qcompare)))
    d['hier_sd_error']=d.sd_operational_a_uncapped
    d['hier_sd_noerror']=np.where(d.known_source_available,d.hier_sd_noerror_uncapped,d.nonhier_sd_noerror)
    arms={'nonhier_noerror':('nonhier_q50_log10','nonhier_sd_noerror'),'nonhier_error':('nonhier_q50_log10','nonhier_sd_error'),'hier_noerror':('q_operational','hier_sd_noerror'),'hier_error':('q_operational','hier_sd_error')}
    rows=[]
    for population in ['All','Known source']:
        e=d if population=='All' else d[d.known_source_available]
        for arm,(q,sd) in arms.items():rows.append(dict(population=population,arm=arm,**score(e,e[q],e[sd])))
    out=pd.DataFrame(rows);out.to_csv(P/'TABLE25_GROUPED_QAQC.csv',index=False)
    for offset,population in [(9,'All'),(15,'Known source')]:
        for j,arm in enumerate(arms):
            r=out[(out.population==population)&(out.arm==arm)].iloc[0]
            for col,key in [(4,'r2'),(5,'rmse'),(6,'wis'),(7,'coverage95'),(8,'width95')]:table(25,offset+j,col,f"{r[key]*(100 if key=='coverage95' else 1):.4f}",'TABLE25_GROUPED_QAQC.csv')
    obs=pd.read_csv(c.OBS,usecols=['stable_record_id','longitude','latitude']).set_index('stable_record_id').loc[d.stable_record_id]
    d['physical_site_key']=obs.longitude.round(4).astype(str).to_numpy()+'|'+obs.latitude.round(4).astype(str).to_numpy()
    d['base_abs']=abs(d.observed_log10-d.nonhier_q50_log10);d['hier_abs']=abs(d.observed_log10-d.q_operational)
    sites=d.groupby('physical_site_key')[['base_abs','hier_abs']].mean();gain=float(100*(sites.hier_abs<sites.base_abs).mean())
    sites.to_csv(P/'FIGURE2_SITE_ERROR_COMPARISON.csv')
    write('FIGURE2_SITE_ERROR_SUMMARY.json',{'physical_sites':len(sites),'improved':int((sites.hier_abs<sites.base_abs).sum()),'improvement_pct':gain,'grouped_qaqc_environment_maxdiff_from_main':float(np.max(abs(qcompare)))})
    para('p0025','75.5%',f'{gain:.1f}%','FIGURE2_SITE_ERROR_SUMMARY.json')
    d.to_csv(P/'GROUPED_QAQC_COMPARISON_OOF.csv.gz',index=False)
    figrows=[]
    for domain in c.DOMAINS:
        e=d[d.domain==domain]
        for arm,q in [('nonhier','nonhier_q50_log10'),('hier','q_operational')]:figrows.append(dict(domain=domain,arm=arm,**score(e,e[q])))
    pd.DataFrame(figrows).to_csv(P/'FIGURE2_GROUPED_DOMAIN_METRICS.csv',index=False)
    warnings.append({'topic':'Record-random BNN fixed comparator','text':'The record-random nonhierarchical control is frozen. The grouped nonhierarchical neural model is independently refitted with QAQC weighting, and all new interval calibration remains within outer training.'})

# Field scenarios: same five records, same fixed original scenario inputs.
field=pd.read_csv(P/'TABLE27_QAQC_METRICS.csv')
for i,r in field.iterrows():
    for col,key in [(2,'rmse_log10'),(3,'mae_log10'),(4,'bias_log10'),(5,'geometric_predicted_observed_ratio'),(6,'pearson_log10_r')]:table(27,i+1,col,f'{r[key]:.4f}','TABLE27_QAQC_METRICS.csv')
warnings.append({'topic':'Field conclusion','location':'main_en/main_zh p0026','text':f"New five-site scenario geometric predicted/observed ratios range {field.geometric_predicted_observed_ratio.min():.3f}–{field.geometric_predicted_observed_ratio.max():.3f}; {(field.geometric_predicted_observed_ratio>1).sum()}/16 scenarios overestimate on average. Original broad overestimation and local-accuracy interpretation should be checked against this evidence; prose unchanged."})

# Two removal sequences x ten folds; score heldout records from removed papers only.
removal_domains=[]
for domain in c.DOMAINS:
    n=sum((unit(f'removal/r{rnd}_{pct}',domain,fold)/'DONE.json').exists() for rnd in [1,2] for pct in [30,50,80] for fold in range(1,11))
    audits['removal_'+domain]={'complete_units':n,'expected_units':60}
    if n==60:removal_domains.append(domain)
removal_ready=len(removal_domains)==3
if not args.partial and not removal_ready:raise RuntimeError('Incomplete paper-removal domains: '+str(audits))
if removal_domains:
    pairs=[]
    for pct in [30,50,80]:
        for round_ in [1,2]:
            tag=f'removal/r{round_}_{pct}'
            for domain in removal_domains:
                for fold in range(1,11):
                    u=unit(tag,domain,fold);done=json.loads((u/'DONE.json').read_text(encoding='utf8'))
                    manifest=c.ROOT/'sensitivity/paper_removal/source_manifests'/f'round_{round_:02d}'/f'fold_{fold:02d}_remove_{pct}.csv'
                    removed=set(pd.read_csv(manifest).paper_id_base)
                    e=pd.read_csv(u/'S0/protocol_aware'/c.DOMAINS[domain]/'H26_UNIT_HELDOUT.csv.gz',low_memory=False)
                    paper=e.source_group.str.extract(r'(NMP\d+)',expand=False)
                    e=e[paper.isin(removed)]
                    b=group.set_index('stable_record_id',drop=False).loc[e.stable_record_id]
                    assert np.array_equal(e.observed_log10,b.observed_log10)
                    for label,frame in [('baseline',b),('removal',e)]:
                        pairs.append(dict(removal=pct,round=round_,fold=fold,domain=domain,arm=label,**score(frame,frame.predicted_q50_log10,frame.future_observation_sd_log10)))
    pairs=pd.DataFrame(pairs);pairs.to_csv(P/'TABLE28_PAIRED_QAQC_METRICS.csv',index=False)
    wide=pairs.pivot(index=['removal','round','fold','domain'],columns='arm',values='wis').rename(columns={'baseline':'baseline_wis','removal':'removal_wis'}).reset_index()
    rows=[]
    for (pct,domain),e in wide.groupby(['removal','domain']):
        assert len(e)==20
        rows.append(dict(removal=pct,domain=domain,baseline_wis=e.baseline_wis.mean(),removal_wis=e.removal_wis.mean(),worse_pairs=int((e.removal_wis>e.baseline_wis).sum()),relative_increase_pct=100*(e.removal_wis.mean()/e.baseline_wis.mean()-1)))
    result=pd.DataFrame(rows);result.to_csv(P/'TABLE28_QAQC_SUMMARY.csv',index=False)
    recovered=[];sensitivities=[]
    for domain,f in [('transition',9),('transition',10),('marine',7)]:
        rp=unit('removal/r1_80',domain,f)/'RECOVERY_AUDIT.json'
        if rp.exists():recovered.append({'domain':domain,'fold':f,'path':str(rp)})
    for domain in ['transition','marine']:
      local_recoveries=[r for r in recovered if r['domain']==domain]
      if local_recoveries and domain in removal_domains:
        subset=wide[(wide.domain==domain)&(wide.removal==80)]
        excluded=subset[~((subset['round']==1)&subset['fold'].isin([x['fold'] for x in local_recoveries]))]
        sensitivity={'domain':domain,'removal':80,'all_pairs':len(subset),'pairs_without_guard_exceptions':len(excluded),
                     'all_relative_increase_pct':100*(subset.removal_wis.mean()/subset.baseline_wis.mean()-1),
                     'without_guard_exceptions_relative_increase_pct':100*(excluded.removal_wis.mean()/excluded.baseline_wis.mean()-1),
                     'without_guard_exceptions_worse_pairs':int((excluded.removal_wis>excluded.baseline_wis).sum()),
                     'recovery_audits':local_recoveries}
        sensitivities.append(sensitivity)
    if recovered:
        write('TABLE28_GUARD_EXCEPTION_SENSITIVITY.json',sensitivities)
        warnings.append({'topic':'Table 28 small calibration split exception','location':'80% removal; round 1 Estuary folds 9/10 and Marine fold 7','text':'Three original units stopped at the minimum-50-record inner early-stopping guard. Exact unchanged splits contained 33/40/37 records from 10/9/8 independent sources. Isolated recovery permits those partitions without changing seed, loss, weights, or the outer fold; main-model parameters and weights exactly match their pre-failure checkpoints. The all-20-pair table and sensitivities omitting these units (Estuary 18, Marine 19 pairs) are separately reported. Original failure logs are retained.','evidence':str(P/'TABLE28_GUARD_EXCEPTION_SENSITIVITY.json')})
    # Source-reference counts are independent of model weights.
    for i,oldrow in enumerate(docs['si_en']['tables'][27]['rows'][1:],1):
        if oldrow[1]=='0':continue
        domain={'Freshwater':'freshwater','Marine':'marine','Estuary':'transition'}[oldrow[0]]
        if domain not in removal_domains:continue
        r=result[(result.domain==domain)&(result.removal==int(oldrow[1]))].iloc[0]
        table(28,i,3,f'{r.baseline_wis:.4f}','TABLE28_QAQC_SUMMARY.csv');table(28,i,4,f'{r.removal_wis:.4f}','TABLE28_QAQC_SUMMARY.csv');table(28,i,5,str(int(r.worse_pairs)),'TABLE28_QAQC_SUMMARY.csv')
    if 'freshwater' in removal_domains:
        r=result[(result.domain=='freshwater')&(result.removal==80)].iloc[0]
        para('p0025','5.9%',f'{r.relative_increase_pct:.1f}%','TABLE28_QAQC_SUMMARY.csv')
        if r.relative_increase_pct<=0:warnings.append({'topic':'Paper-removal conclusion','location':'main p0025','text':'New 80% freshwater removal does not worsen mean WIS; original textual direction requires revision.'})

warnings.append({'topic':'Historical branch versus weight effect','text':'New validation uses the adopted QAQC runner. Some historical SI rows came from independent precision-refit or older calibration branches. Old-to-new changes are release changes, not all isolated QAQC treatment effects. Weight-only contrasts are available from the paired baseline/qaqc primary experiment.'})
figure={'panel_a':[], 'panel_b_current_record_random':[], 'panel_b_grouped_alternative':[], 'panel_d':[], 'panel_c':'Frozen C-AOA support proportions; not concentration-model weights. Coordinate with global lane.',
        'original_sources':{'random_comparison':str(c.ROOT.parent.parent/'figures/comparisons/scripts/random_cv/run_random.py'),
                            'grouped_comparison':str(A/'aggregate_new.py'),
                            'physical_site_aggregation':str(DOC.parent/'SI_V21_alignment_20261016_V2/finalize_table_audit.py'),
                            'legacy_ppt_builder':str(c.ROOT.parent.parent/'figures/comparisons/.build_revision2/build.mjs')},
        'legacy_builder_warning':'The legacy PPT builder contains a mixed random/grouped panel a. Use the complete record-random panel_a values here; do not rerun its old hard-coded bars.'}
if (P/'RANDOM10_QAQC_METRICS.csv').exists():
    controls=pd.read_csv(c.ROOT.parent.parent/'figures/comparisons/scripts/random_cv/METRICS.csv')
    base=controls[controls.domain.eq('overall')].copy()
    r=pd.read_csv(P/'RANDOM10_QAQC_METRICS.csv');r=r[r.domain.eq('overall')&r.target.eq('operational')].iloc[0]
    base.loc[base.model.eq('H26'),'R2']=r.r2;base.loc[base.model.eq('H26'),'RMSE']=r.rmse
    figure['panel_a']=base.to_dict('records')
    random_domain=pd.read_csv(P/'RANDOM10_QAQC_METRICS.csv')
    for domain in c.DOMAINS:
        old=controls[controls.domain.eq(domain)&controls.model.eq('BNN_nonhier')].iloc[0]
        new=random_domain[random_domain.domain.eq(domain)&random_domain.target.eq('operational')].iloc[0]
        figure['panel_b_current_record_random'].extend([dict(domain=domain,arm='nonhier_frozen',rmse=old.RMSE,r2=old.R2),dict(domain=domain,arm='hier_qaqc',rmse=new.rmse,r2=new.r2)])
if (P/'FIGURE2_GROUPED_DOMAIN_METRICS.csv').exists():figure['panel_b_grouped_alternative']=pd.read_csv(P/'FIGURE2_GROUPED_DOMAIN_METRICS.csv').to_dict('records')
if (P/'TABLE28_QAQC_SUMMARY.csv').exists():
    rr=pd.read_csv(P/'TABLE28_QAQC_SUMMARY.csv');rr=rr[rr.domain.eq('freshwater')]
    retained={0:100.,30:48.81612323,50:25.51676429,80:3.78683761}
    figure['panel_d']=[{'papers_removed_percent':0,'reference_support_lost_percent':0,'WIS_increase_percent':0}]
    for _,r in rr.iterrows():figure['panel_d'].append({'papers_removed_percent':int(r.removal),'reference_support_lost_percent':100-retained[int(r.removal)],'WIS_increase_percent':r.relative_increase_pct})
write('FIGURE2_NUMERIC_HANDOFF.json',figure)
write('replacements_model.json',{'replacements':replacements,'warnings':warnings})
write('AGGREGATE_MODEL_AUDIT.json',audits)
print(json.dumps({'replacement_cells':len(replacements),'warnings':len(warnings),'complete':removal_ready},ensure_ascii=False),flush=True)
