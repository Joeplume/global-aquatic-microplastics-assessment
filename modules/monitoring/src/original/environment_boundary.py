"""Conditional environmental-boundary bootstrap, fixed Figure 4b layout.

Source resampling weights re-estimate the environmental metric and caliper.
The complete original unique-source catalogue remains the reference evidence:
bootstrap multiplicity never supplies extra independent sources, and zero
bootstrap weight never deletes a real source. This is a conditional boundary
stability interval, not a full model/measurement-error confidence interval.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast, hashlib, json, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

O=Path(__file__).resolve().parent
R=O.parents[1]
F=R/'src/modeling/h26'
E=F/'validation'
T=E/'11_CAOA_V02_质量基础与独立验证_20260908_V1'
U=E/'15_H26_后验浓度_CAOA联合监测优先配置_20260910_V1'
DOM=['reference_surface_freshwater','transition_coastal']
MAX=400
SEED=2026101601
INPUTS=[]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def track(p):
    INPUTS.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    return p

def js(name,data):
    path=O/name
    payload=json.dumps(data,ensure_ascii=False,indent=2,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x))
    temp=path.with_name(path.name+'.writing')
    for attempt in range(10):
        try:
            temp.write_text(payload,encoding='utf8');os.replace(temp,path);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.3)

def metric(source,query,index):
    # Refit the original robust scaling and 90%-variance PCA rule.
    sampled=source[index]
    center=np.nanmedian(sampled,axis=0)
    q=np.nanquantile(sampled,[.1,.9],axis=0)
    scale=q[1]-q[0]
    if np.any(~np.isfinite(scale)|(scale<=1e-10)):
        raise ValueError('Degenerate bootstrap scale; no silent feature deletion')
    ss=np.clip((np.where(np.isfinite(source),source,center)-center)/scale,-10,10)
    qq=np.clip((np.where(np.isfinite(query),query,center)-center)/scale,-10,10)
    fit=ss[index]
    full=PCA(svd_solver='full').fit(fit)
    dim=min(12,max(2,int(np.searchsorted(np.cumsum(full.explained_variance_ratio_),.90)+1)))
    pca=PCA(n_components=dim,svd_solver='full').fit(fit)
    se=pca.transform(ss);qe=pca.transform(qq)
    tree=cKDTree(se)
    # k=2 excludes the self-identity; identical but distinct actual sources
    # legitimately have zero distance. Bootstrap copies are not in this tree.
    nearest=tree.query(se,k=2,workers=1)[0][:,1]
    cal=float(np.quantile(nearest[index],.95))
    near=tree.query(qe,k=2,workers=1)[0]
    counts=(near<=cal+1e-12).sum(axis=1).astype(np.int8)
    return qe,counts,dict(caliper=cal,dimensions=dim,center=center,scale=scale)

def calculate(parts,area,high,rep):
    first=np.full(len(area),MAX+1,dtype=np.int32)
    meta=[]
    for di,p in enumerate(parts):
        n=len(p['source'])
        # Domain-stratified source-weight bootstrap. One joint replicate is
        # used for BOTH endpoints, with fixed high mask and denominators.
        rng=np.random.default_rng(np.random.SeedSequence([SEED,max(rep,0),di]))
        index=np.arange(n) if rep<0 else rng.integers(0,n,n)
        qe,count,m=metric(p['source'],p['query'],index)
        target=p['global_index']
        first[target[count>=2]]=0
        tree=cKDTree(qe)
        for loc,budget,package in zip(p['site_index'],p['budget'],p['package']):
            hit=np.asarray(tree.query_ball_point(qe[loc],m['caliper']+1e-12),dtype=int)
            if package==1:hit=hit[count[hit]==1]
            dest=target[hit]
            first[dest]=np.minimum(first[dest],budget)
        meta.append(dict(domain=DOM[di],caliper=m['caliper'],dimensions=m['dimensions'],
            unique_reference_sources=n,bootstrap_distinct_fit_sources=len(np.unique(index)),
            baseline_in_cells=int((count>=2).sum())))
    curve=np.stack([np.cumsum(np.bincount(first,weights=area,minlength=MAX+2))[:MAX+1]/area.sum()*100,
        np.cumsum(np.bincount(first[high],weights=area[high],minlength=MAX+2))[:MAX+1]/area[high].sum()*100])
    assert np.all(np.diff(curve,axis=1)>=-1e-9) and curve.min()>=0 and curve.max()<100+1e-8
    return curve,meta,first

def prepare():
    old=track(T/'code/prepare_experiment.py')
    nodes=[n for n in ast.parse(old.read_text(encoding='utf-8-sig')).body
        if (isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='LOG_FEATURES' for v in n.targets))
        or (isinstance(n,ast.FunctionDef) and n.name=='transform')]
    env={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(old),'exec'),env)
    audit=json.loads(track(T/'audit/GLOBAL_CAOA_V02_REPRODUCTION_AUDIT.json').read_text('utf8'))['domains']
    features=sorted({f for dom in DOM for f in audit[dom]['used_features']})
    obs=pd.read_csv(track(F/'data/observations/h23_observation_layer_training_records.csv.gz'),
        usecols=['stable_record_id','waterbody','source_group'])
    raw=pd.read_csv(track(F/'data/05_process_covariates/H26_RIVER_NETWORK_PROCESS_TRAIN.csv.gz'),
        usecols=['stable_record_id',*features])
    pred=pd.read_csv(track(F/'data/05_process_covariates/H26_RIVER_NETWORK_PROCESS_PREDICTION.csv.gz'),
        usecols=['prediction_id',*features])
    train=obs.merge(raw,on='stable_record_id',validate='one_to_one')
    grid=pd.read_csv(track(U/'JOINT_STATE_MAP_GRID.csv.gz'),low_memory=False)
    g=grid[grid.waterbody.isin(DOM)&grid.analysis_assessed].copy().reset_index(drop=True)
    assert len(g)==69581 and g.prediction_id.is_unique
    seq=pd.read_csv(track(U/'CANDIDATE_POINTS.csv.gz'),low_memory=False)
    seq=seq[seq.scope.eq('terrestrial_aquatic')].sort_values('priority_order').iloc[:MAX]
    assert len(seq)==MAX and np.array_equal(seq.priority_order,np.arange(1,MAX+1))
    assert set(seq.independent_source_deployments)=={1,2}
    seq.to_csv(O/'FROZEN_FIRST400_SITES.csv',index=False)
    area=g.cell_area_km2.to_numpy();high=g.high_concentration_primary.to_numpy(bool)
    parts=[];checks=[]
    for di,dom in enumerate(DOM):
        a=audit[dom];fs=a['used_features'];local=train[train.waterbody.eq(dom)]
        sources=local.groupby('source_group',sort=False)[fs].median()
        ix=np.flatnonzero(g.waterbody.eq(dom))
        q=g.iloc[ix][['prediction_id']].merge(pred,on='prediction_id',validate='one_to_one')
        ss=np.column_stack([env['transform'](sources[f].to_numpy(float),f) for f in fs])
        qq=np.column_stack([env['transform'](q[f].to_numpy(float),f) for f in fs])
        ids=q.prediction_id.astype(str).to_numpy();lookup={pid:i for i,pid in enumerate(ids)}
        s=seq[seq.waterbody.eq(dom)]
        p=dict(source=ss,query=qq,global_index=ix,
            site_index=np.array([lookup[str(pid)] for pid in s.prediction_id]),
            budget=s.priority_order.to_numpy(int),package=s.independent_source_deployments.to_numpy(int))
        qe,counts,m=metric(ss,qq,np.arange(len(ss)))
        state=np.where(counts>=2,'in',np.where(counts==1,'limited','out'))
        mismatches=int(np.sum(state!=g.iloc[ix].caoa_analysis_state.to_numpy()))
        expected=np.array([[a['preprocessing'][f]['median_transformed'] for f in fs],
            [a['preprocessing'][f]['q10_q90_scale'] for f in fs]])
        err=float(np.max(np.abs(np.stack([m['center'],m['scale']])-expected)))
        check=dict(domain=dom,records=len(local),sources=len(ss),grid_cells=len(ix),
            caliper=m['caliper'],saved_caliper=a['nearest_source_q95_match_caliper'],
            dimensions=m['dimensions'],saved_dimensions=a['pca_dimensions'],
            preprocessing_error=err,status_mismatches=mismatches)
        assert err<1e-8 and mismatches==0 and m['dimensions']==a['pca_dimensions']
        assert abs(m['caliper']-a['nearest_source_q95_match_caliper'])<1e-8
        checks.append(check);parts.append(p)
        np.savez_compressed(O/f'PREPARED_DOMAIN_{di}.npz',**p,source_ids=sources.index.to_numpy(str),
            prediction_ids=ids,features=np.array(fs))
    orig=pd.read_csv(track(U/'JOINT_FULL_CURVE.csv.gz'))
    orig=orig[orig.scope.eq('terrestrial_aquatic')&orig.budget.le(MAX)].sort_values('budget')
    ref=orig[['overall_caoa_in_pct','high_concentration_caoa_in_pct']].to_numpy().T
    baseline,meta,first=calculate(parts,area,high,-1)
    error=float(np.max(np.abs(baseline-ref)))
    assert error<1e-8,error
    np.savez_compressed(O/'FIXED_POPULATION.npz',area=area,high=high,original_curve=ref,
        prediction_ids=g.prediction_id.to_numpy(str),baseline_first_support=first)
    js('BASELINE_AUDIT.json',dict(domains=checks,curve_max_abs_error_pp=error,
        area_denominator_km2=float(area.sum()),high_area_denominator_km2=float(area[high].sum())))
    js('INPUT_MANIFEST.json',INPUTS)
    contract=dict(seed=SEED,replicates=200,source_weight_resampling='multinomial within waterbody',
        refit=['median and q10-q90 scaling','PCA with original 90% rule and cap of 12','weighted q95 nearest-distinct-source caliper'],
        conditional_on=['all original unique reference sources retained','original first400 site identities and order',
            'original 1/2-source packages and support-union rule','original feature set','fixed q50 high-concentration mask',
            'same 69581 valid freshwater and Estuary grid cells and full-grid area weights'],
        not_included=['concentration posterior or measurement noise','environmental raster measurement error',
            'site reselection','deployment failure','future unseen source populations','marine waters'],
        interpretation='Pointwise 2.5th-97.5th percentile conditional environmental-boundary bootstrap range; not a simultaneous band or full uncertainty confidence interval',
        zero_weight_rule='Source may have zero fitting weight but remains a single real reference in classification',
        nearest_rule='Nearest distinct source computed in full fixed catalogue; distances weighted by resampled source counts for q95',
        output_version='20261016_V1')
    js('CONTRACT.json',contract)
    signature=hashlib.sha256((sha(Path(__file__))+json.dumps(contract,sort_keys=True)+json.dumps(INPUTS,sort_keys=True)).encode()).hexdigest()
    js('RUN_SIGNATURE.json',{'sha256':signature,'code_sha256':sha(Path(__file__))})
    return parts,area,high,ref,signature

def worker(rep,signature):
    out=O/'replicates'/signature[:16]/f'rep_{rep:03d}.npz'
    if out.exists():
        with np.load(out) as z:
            if str(z['signature'])==signature:return rep,z['curve'],json.loads(str(z['metadata']))
        raise RuntimeError('Existing replicate signature mismatch; use a new output version')
    parts=[]
    for di in range(2):
        with np.load(O/f'PREPARED_DOMAIN_{di}.npz') as z:
            parts.append({k:z[k] for k in ['source','query','global_index','site_index','budget','package']})
    with np.load(O/'FIXED_POPULATION.npz') as z:area=z['area'];high=z['high']
    curve,meta,_=calculate(parts,area,high,rep)
    np.savez_compressed(out,curve=curve,metadata=json.dumps(meta),signature=signature,rep=rep)
    return rep,curve,meta

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replicates',type=int,default=200);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();start=time.time();(O/'replicates').mkdir(exist_ok=True)
    js('STATUS.json',dict(status='running',stage='baseline reproduction',requested=args.replicates))
    parts,area,high,ref,signature=prepare()
    (O/'replicates'/signature[:16]).mkdir(exist_ok=True)
    print('Baseline reproduced exactly; starting conditional boundary bootstrap',flush=True)
    curves={};metas=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(worker,r,signature) for r in range(args.replicates)]
        for job in as_completed(jobs):
            r,c,m=job.result();curves[r]=c;metas.extend([dict(replicate=r,**x) for x in m])
            if len(curves)%10==0 or len(curves)==args.replicates:
                status=dict(status='running',stage='bootstrap',completed=len(curves),requested=args.replicates,elapsed_s=round(time.time()-start,1))
                js('STATUS.json',status);print(json.dumps(status),flush=True)
    samples=np.stack([curves[r] for r in range(args.replicates)])
    q=np.quantile(samples,[.025,.5,.975],axis=0)
    rows=[dict(metric=name,added_sites=n,original_curve=ref[mi,n],q025=q[0,mi,n],q50=q[1,mi,n],q975=q[2,mi,n],replicates=args.replicates)
        for mi,name in enumerate(['overall','high']) for n in range(MAX+1)]
    table=pd.DataFrame(rows)
    table.to_csv(O/'BOUNDARY_INTERVALS.csv',index=False)
    table[table.added_sites.isin([0,95,297,400])].to_csv(O/'CHECKPOINTS.csv',index=False)
    pd.DataFrame(metas).sort_values(['replicate','domain']).to_csv(O/'BOOTSTRAP_GEOMETRY.csv',index=False)
    np.savez_compressed(O/'BOOTSTRAP_CURVES.npz',curve=samples,quantiles=q)
    assert all(sha(Path(x['path']))==x['sha256'] for x in INPUTS)
    report=dict(status='complete' if args.replicates==200 else 'pilot_complete',replicates=args.replicates,
        elapsed_s=round(time.time()-start,1),all_original_inputs_unchanged=True,original_curves_unchanged=True,
        minimum_lower_percentile=float(q[0].min()),maximum_upper_percentile=float(q[2].max()),
        original_outside_pointwise_interval_counts={name:int(((ref[i]<q[0,i])|(ref[i]>q[2,i])).sum()) for i,name in enumerate(['overall','high'])},
        interval_type='conditional environmental-matching-boundary bootstrap',signature=signature)
    js('RESULT.json',report);js('STATUS.json',report)
    print(table[table.added_sites.isin([95,297])].to_string(index=False),flush=True)

if __name__=='__main__':main()
