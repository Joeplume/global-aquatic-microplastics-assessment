"""Portable downstream reproduction from frozen H26 outputs.

Does not retrain H26. The source functions are extracted unmodified from the
archived scientific programs, excluding their machine-specific I/O wrappers.
Reference files are used only after computation for equality checks.
"""
from pathlib import Path
import argparse, ast, json, os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(k,'2')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'src/original'

def definitions(filename,names,namespace,assignments=()):
    tree=ast.parse((SRC/filename).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in names)
           or (isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id in assignments for x in n.targets))]
    found={n.name for n in nodes if isinstance(n,ast.FunctionDef)}
    assert found==set(names),(filename,found,set(names))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SRC/filename),'exec'),namespace)
    return namespace

def network(out,reps):
    ns=definitions('environment_boundary.py',['metric','calculate'],
        dict(np=np,PCA=PCA,cKDTree=cKDTree,MAX=400,SEED=2026101601,
             DOM=['reference_surface_freshwater','transition_coastal']))
    parts=[]
    for i in range(2):
        z=np.load(ROOT/f'inputs/network/PREPARED_DOMAIN_{i}.npz')
        parts.append({k:z[k] for k in ['source','query','global_index','site_index','budget','package']})
    pop=np.load(ROOT/'inputs/network/FIXED_POPULATION.npz')
    point,meta,first=ns['calculate'](parts,pop['area'],pop['high'],-1)
    assert len(first)==69581
    reference=pd.read_csv(ROOT/'reference/network/FOUR_STRATEGY_CURVES.csv.gz')
    ref=reference[reference.strategy.eq('joint_concentration_then_caoa')].sort_values('budget').iloc[:401]
    reference_curve=ref[['overall_caoa_in_pct','high_concentration_caoa_in_pct']].to_numpy().T
    err=float(np.max(np.abs(point-reference_curve)));assert err<1e-8,err
    pd.DataFrame({'added_sites':np.arange(401),'overall_support_pct':point[0],'high_support_pct':point[1]}).to_csv(out/'COVERAGE_RECOMPUTED.csv',index=False)
    old_draws=np.load(ROOT/'reference/network/BOOTSTRAP_CURVES.npz')['curve']
    draws=[];errors=[]
    for rep in range(reps):
        c,_,_=ns['calculate'](parts,pop['area'],pop['high'],rep)
        draws.append(c);errrep=float(np.max(np.abs(c-old_draws[rep])));errors.append(errrep)
        assert errrep<1e-8,(rep,errrep)
    if draws:np.savez_compressed(out/'BOUNDARY_RECOMPUTED_DRAWS.npz',curve=np.stack(draws))
    if reps==200:
        q=np.quantile(np.stack(draws),[.025,.5,.975],axis=0)
        pd.DataFrame([dict(metric=name,added_sites=k,q025=q[0,i,k],q50=q[1,i,k],q975=q[2,i,k])
                      for i,name in enumerate(['overall','high']) for k in range(401)]).to_csv(out/'BOUNDARY_INTERVALS_RECOMPUTED.csv',index=False)
    return dict(target95_first_site=int(np.flatnonzero(point[0]>=95)[0]),
        checkpoints={str(k):dict(overall=float(point[0,k]),high=float(point[1,k])) for k in [95,297,305]},
        maximum_reference_difference=err,boundary_replicates_actually_recomputed=reps,
        maximum_boundary_draw_difference=max(errors) if errors else None,metric_details=[{k:v for k,v in m.items() if k not in ['center','scale']} for m in meta],
        scope='fixed current site order and high-concentration mask; refit source-weight environmental boundary, not concentration-posterior or field-performance uncertainty')

def reml(out,reps=0):
    ns=definitions('reml.py',['arrays','fit','power_factor'],dict(np=np,pd=pd,minimize=minimize,norm=norm))
    d=pd.read_csv(ROOT/'inputs/mdc/OOF_REML_INPUT.csv.gz')
    d['residual_log10']=d.observed_log10-d.predicted_q50_log10
    d['known_var']=d.sizecorr_known_sd_log10**2+d.unit_sd_log10**2
    rows=[];drawrows=[];rng=np.random.default_rng(20261013)
    for dom in ['freshwater','transition','marine']:
        x=d[d.domain.eq(dom)].copy();a=ns['arrays'](x);r=ns['fit'](a)
        known=float(x.known_var.mean())
        r.update(domain=dom,records=len(x),sources=a[3],site_events=x.validation_site_event_key.nunique(),reported_var=known)
        for name,var in [('changing',r['between_var']+r['remaining_var']+known),('stable',r['remaining_var']+known),('standardized',r['remaining_var'])]:
            r['V_'+name]=var;r['N30_'+name]=int(np.ceil(ns['power_factor']()*var/np.log10(.7)**2))
        rows.append(r)
        if reps:
            G=a[3];g=a[2];counts=np.bincount(g,minlength=G);ks=np.bincount(g,x.known_var.to_numpy(),minlength=G)
            # Advance the original RNG by all 2000 draws per domain even in a
            # short verification run, so later domains retain original seeds.
            for bi in range(2000):
                mult=rng.multinomial(G,np.ones(G)/G)
                if bi>=reps:continue
                fit=ns['fit'](a,mult,[r['between_var'],r['remaining_var']])
                fit.update(domain=dom,replicate=bi,reported_var=float(mult@ks/(mult@counts)));drawrows.append(fit)
    table=pd.DataFrame(rows);table.to_csv(out/'REML_REFITTED.csv',index=False)
    old=pd.read_csv(ROOT/'reference/REFIT_POINT_ESTIMATES.csv').set_index('domain')
    current=table.set_index('domain');cols=['between_var','remaining_var','reported_var','N30_changing','N30_stable','N30_standardized']
    assert np.allclose(current[cols],old.loc[current.index,cols],rtol=1e-6,atol=1e-6)
    if drawrows:
        draws=pd.DataFrame(drawrows);draws.to_csv(out/'REML_BOOTSTRAP_RECOMPUTED.csv',index=False)
        olddraw=pd.read_csv(ROOT/'reference/SOURCE_BOOTSTRAP_DRAWS.csv').set_index(['domain','replicate'])
        actual=draws.set_index(['domain','replicate']);cols=['between_var','remaining_var','reported_var']
        assert np.allclose(actual[cols],olddraw.loc[actual.index,cols],rtol=1e-5,atol=1e-5)
        if reps==2000:
            summary=[]
            for dom,z in draws.groupby('domain',sort=False):
                for scenario,v in [('changing',z.between_var+z.remaining_var+z.reported_var),('stable',z.remaining_var+z.reported_var),('standardized',z.remaining_var)]:
                    n=np.ceil(ns['power_factor']()*v/np.log10(.7)**2)
                    summary.append(dict(domain=dom,scenario=scenario,replicates=2000,q025=float(np.quantile(n,.025)),q50=float(np.quantile(n,.5)),q975=float(np.quantile(n,.975))))
            pd.DataFrame(summary).to_csv(out/'N30_BOOTSTRAP_INTERVALS_RECOMPUTED.csv',index=False)
    return dict(records=len(d),actual_reml_refitted=True,bootstrap_replicates_per_domain_actually_recomputed=reps,counts=table[['domain','N30_changing','N30_stable','N30_standardized']].to_dict('records'),
        standardization='complete removal of reported size/unit variance only; hypothetical scenario, not empirical performance',
        scope='six annual occasions, AR1 rho .3, 30% decline, two-sided alpha .05, power .8; independent stations')

def costs(out):
    ns=dict(np=np,pd=pd)
    ns['OLD']=json.loads((ROOT/'inputs/costs/V1_CONFIG.json').read_text('utf-8-sig'))
    ns['V2CFG']=json.loads((ROOT/'inputs/costs/V2_CONFIG.json').read_text('utf-8-sig'))
    definitions('cost_common.py',['haversine'],ns,assignments=['CFG','DOM','CATS'])
    names=[n.name for n in ast.parse((SRC/'cost_engine.py').read_text('utf-8-sig')).body if isinstance(n,ast.FunctionDef)]
    definitions('cost_engine.py',names,ns)
    rows=[]
    for k in [95,305]:
        q=pd.read_csv(ROOT/f'inputs/costs/SELECTED_{k}.csv',low_memory=False)
        for capacity in ns['CFG']['capacity_branches']:
            for access in ns['CFG']['access_branches']:
                for level in ns['CFG']['levels']:
                    z=ns['network'](q,level=level,capacity=capacity,access=access,bridge=True)
                    vals={c:float(z[c].sum()) for c in z if c.endswith('_eur')}
                    assert np.isclose(vals['six_resource_eur'],vals['baseline_resource_eur']+5*vals['annual_resource_eur'])
                    rows.append(dict(checkpoint=k,capacity_branch=capacity,access_branch=access,cost_level=level,**vals))
    table=pd.DataFrame(rows);table.to_csv(out/'BASELINE_COSTS_RECOMPUTED.csv',index=False)
    old=pd.read_csv(ROOT/'reference/BASELINE_COSTS_95_305.csv')
    keys=['checkpoint','capacity_branch','access_branch','cost_level'];cols=[c for c in table if c.endswith('_eur')]
    a=table.set_index(keys).sort_index();b=old.set_index(keys).sort_index()
    diff=float(np.max(np.abs(a[cols].to_numpy()-b[cols].to_numpy())))
    assert np.allclose(a[cols],b[cols],rtol=1e-10,atol=1e-7)
    return dict(scenarios=len(table),maximum_reference_difference_eur=diff,
        currency='2026 EUR scenario prices ex VAT',scope='baseline + five annual follow-ups; capacity and access are scenarios, not established local facts',
        baseline_ranges=table.groupby('checkpoint').baseline_resource_eur.agg(['min','max']).reset_index().to_dict('records'))

def institutions(out):
    d=pd.read_csv(ROOT/'inputs/institutions/matches_305.csv')
    distances=d.nearest_documented_env_km_within_search.to_numpy(float)
    rows=[]
    for radius in [50,100,200]:
        matched=np.isfinite(distances)&(distances<=radius)
        assert np.array_equal(matched,d[f'matched_{radius}km'].to_numpy(bool))
        rows.append(dict(radius_km=radius,sites=len(d),unmatched_sites=int((~matched).sum()),unmatched_pct=float((~matched).mean()*100)))
    pd.DataFrame(rows).to_csv(out/'ENVIRONMENTAL_INSTITUTION_RECOMPUTED.csv',index=False)
    return dict(distance_derived=rows,scope='frozen documented environmental-institution distances; no new institutional capability verification')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out','--output',type=Path,default=ROOT/'outputs')
    ap.add_argument('--stage',choices=['all','network','reml','costs'],default='all')
    ap.add_argument('--boundary-reps',type=int,default=3,help='0-200; 3 is numerical replicate verification, 200 rebuilds published interval')
    ap.add_argument('--reml-reps',type=int,default=0,help='0-2000; 2000 regenerates source-bootstrap intervals per domain')
    args=ap.parse_args();assert 0<=args.boundary_reps<=200 and 0<=args.reml_reps<=2000;args.out.mkdir(parents=True,exist_ok=True)
    result={'scope':'actual downstream recalculation from frozen upstream model outputs; not end-to-end H26 training'}
    if args.stage in ['all','network']:result['network']=network(args.out,args.boundary_reps);print('network complete',flush=True)
    if args.stage in ['all','reml']:result['reml']=reml(args.out,args.reml_reps);print('REML complete',flush=True)
    if args.stage in ['all','costs']:result['costs']=costs(args.out);result['institutions']=institutions(args.out);print('costs complete',flush=True)
    (args.out/'REPRODUCTION_REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
