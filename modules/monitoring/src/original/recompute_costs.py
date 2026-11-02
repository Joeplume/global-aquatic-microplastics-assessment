"""Reuse frozen cost units and routes, recomputing basin N from QA/QC OOF residuals."""
from recompute_variance import ROOT,PROJECT,EXP,H
import json,sys,os,hashlib,time
from pathlib import Path
import numpy as np,pandas as pd
V2=PROJECT/'analysis/监测实施与成本_20261010_V2'
V3=PROJECT/'src/modeling/08_H26监测实施成本审计重算_20261010_V3'
OUT=ROOT/'qaqc/costs'
for folder in ['inputs','results','audit']:(OUT/folder).mkdir(parents=True,exist_ok=True)

def save(q,name):q.to_csv(OUT/name,index=False)
def js(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda a:a.item() if isinstance(a,np.generic) else str(a)),encoding='utf-8')
def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)

def cost_namespace():
    p=V3/'src/common.py';s=p.read_text(encoding='utf-8-sig')
    s=s.replace("ROOT=Path(__file__).resolve().parents[1]",f"ROOT=Path({str(OUT)!r})")
    s=s.replace('PROJECT=ROOT.parents[2]',f"PROJECT=Path({str(PROJECT)!r})")
    ns={'__file__':str(p),'__name__':'isolated_cost_common'};exec(compile(s,str(p),'exec'),ns)
    ep=V3/'src/engine.py';exec(compile(ep.read_text(encoding='utf-8-sig').replace('from common import *',''),str(ep),'exec'),ns)
    return ns

def update_components():
    ev=pd.read_csv(V2/'inputs/LOCAL_EVENT_RESIDUALS.csv.gz',low_memory=False)
    q=pd.read_csv(EXP/'aggregate_results/OOF_qaqc.csv.gz',low_memory=False).set_index('stable_record_id')
    assert ev.stable_record_id.is_unique and ev.stable_record_id.isin(q.index).all()
    assert np.allclose(ev.observed_log10,q.loc[ev.stable_record_id,'observed_log10'])
    ev['residual_log10']=q.loc[ev.stable_record_id,'residual_log10'].to_numpy()
    ev['known_source_conditional_q50_log10']=q.loc[ev.stable_record_id,'known_source_conditional_q50_log10'].to_numpy()
    save(ev,'inputs/LOCAL_EVENT_RESIDUALS_QAQC.csv.gz')
    p=V2/'src/basins.py';s=p.read_text(encoding='utf-8-sig').split('\ndef run():')[0]
    s=s.replace('from common import *','').replace('from local_costs import network','')
    s=s.replace("ROOT/'inputs/LOCALIZED_GRID.csv.gz'",f"Path({str(V2/'inputs/LOCALIZED_GRID.csv.gz')!r})")
    s=s.replace("ROOT/'inputs/LOCAL_EVENT_RESIDUALS.csv.gz'","ROOT/'inputs/LOCAL_EVENT_RESIDUALS_QAQC.csv.gz'")
    s=s.replace("F/'04_MDC与监测政策/01_source_mdc/H26_MDC_VARIANCE_COMPONENTS.csv'",f"Path({str(ROOT/'qaqc/H26_MDC_VARIANCE_COMPONENTS.csv')!r})")
    ns=dict(Path=Path,np=np,pd=pd,ROOT=OUT,CFG=json.loads((V2/'CONFIG.json').read_text(encoding='utf-8')),DOM=['reference_surface_freshwater','transition_coastal'],save=save,js=js,__name__='isolated_components')
    exec(compile(s,str(p),'exec'),ns);g,b,th=ns['components']()
    return g,b,th,ns['allocate']

def run():
    g,b,th,allocate=update_components()
    g=pd.read_csv(V3/'inputs/GRID_V3.csv.gz',low_memory=False);g=g[g.basin_key.ne('unassigned')].reset_index(drop=True)
    ns=cost_namespace();network=ns['network'];CFG=ns['CFG'];rows=[]
    checkpoint=OUT/'results/BASIN_GLOBAL_COSTS_QAQC.csv'
    if checkpoint.exists():rows=pd.read_csv(checkpoint).to_dict('records')
    keys={(r['scope'],r['protocol'],float(r['measurement_reduction']),r['capacity_branch'],r['access_branch'],r['cost_level']) for r in rows}
    for scope in ['all_assessed_basins','existing_baseline_basins']:
        bx=b if scope=='all_assessed_basins' else b[b.existing_baseline_eligible]
        gx=g[g.basin_key.isin(bx.basin_key)].reset_index(drop=True)
        for red in [0.,.25,.5,.75]:
            N=np.maximum(1,np.ceil(bx.event_variance_log10_sq*(1-red*bx.measurement_fraction_assumed)/th)).astype(int)
            counts=allocate(gx,bx,N)
            assert np.array_equal(pd.Series(counts).groupby(gx.basin_key).sum().reindex(bx.basin_key),N)
            for protocol in (['qualified_diverse','regional_common','single_workflow'] if red==0 else ['regional_common','single_workflow']):
                for cap in CFG['capacity_branches']:
                    for access in CFG['access_branches']:
                        for level in CFG['levels']:
                            key=(scope,protocol,red,cap,access,level)
                            if key in keys:continue
                            z=network(gx,counts,level=level,capacity=cap,access=access,coarse=True,protocol=protocol,scenario_id='QAQC_'+'_'.join(map(str,key)))
                            cols=[c for c in z if c.endswith('_eur')]
                            assert np.allclose(z.six_resource_eur,z.baseline_resource_eur+5*z.annual_resource_eur)
                            rows.append(dict(scope=scope,protocol=protocol,capacity_branch=cap,access_branch=access,cost_level=level,measurement_reduction=red,basin_units=len(bx),station_equivalents=int(N.sum()),**{c:float(z[c].sum()) for c in cols}))
                            save(pd.DataFrame(rows),'results/BASIN_GLOBAL_COSTS_QAQC.csv')
                log(scope,red,protocol,'completed',len(rows))
    tt=pd.DataFrame(rows)
    ref=tt[(tt.protocol=='qualified_diverse')&(tt.measurement_reduction==0)][['scope','capacity_branch','access_branch','cost_level','station_equivalents','six_resource_eur','six_cash_eur','annual_resource_eur','training_organization_baseline_resource_eur']].rename(columns={'station_equivalents':'reference_sites','six_resource_eur':'reference_resource_eur','six_cash_eur':'reference_cash_eur','annual_resource_eur':'reference_annual_eur','training_organization_baseline_resource_eur':'reference_initial_support_eur'})
    be=tt.merge(ref,on=['scope','capacity_branch','access_branch','cost_level'],validate='many_to_one')
    be['resource_saving_pct']=100*(1-be.six_resource_eur/be.reference_resource_eur)
    be['net_resource_saving_eur']=be.reference_resource_eur-be.six_resource_eur
    be['saved_station_equivalents']=be.reference_sites-be.station_equivalents
    annual=be.reference_annual_eur-be.annual_resource_eur;extra=be.training_organization_baseline_resource_eur-be.reference_initial_support_eur
    be['break_even_followup_years']=np.where(annual>0,np.maximum(extra,0)/annual,np.nan)
    save(be,'results/BASIN_METHOD_BENEFITS_QAQC.csv')
    js('audit/COST_UPDATE_MANIFEST.json',dict(status='complete',scenarios=len(tt),original_components_program=str(V2/'src/basins.py'),cost_engine=str(V3/'src/engine.py'),price_rules_unchanged=True,grid_and_routes_unchanged=True,updates='QAQC OOF residuals and domain variance components propagated to independent station counts, allocations and exact event cost engine',inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [EXP/'aggregate_results/OOF_qaqc.csv.gz',ROOT/'qaqc/H26_MDC_VARIANCE_COMPONENTS.csv',V3/'src/engine.py',V3/'src/common.py']}))

def spatial():
    ns=cost_namespace();network=ns['network'];CFG=ns['CFG']
    q=pd.read_csv(V3/'inputs/CANDIDATES_V3.csv.gz',low_memory=False).set_index('prediction_id',drop=False)
    rows=[];pointrows=[];audit=[]
    # First check the frozen original 297-site accounting with its original order.
    old=pd.read_csv(PROJECT/'analysis/监测实施与成本_20261009_V1/inputs/ORIGINAL_JOINT_ORDER.csv.gz')
    qs=q.set_index('candidate_index',drop=False).loc[old.candidate_index.iloc[:297]].reset_index(drop=True)
    prior=pd.read_csv(V3/'results/BASELINE_GLOBAL_COSTS.csv')
    for cap in CFG['capacity_branches']:
        for access in CFG['access_branches']:
            z=network(qs,level='mid',capacity=cap,access=access,bridge=True)
            ref=prior[(prior.checkpoint==297)&(prior.capacity_branch==cap)&(prior.access_branch==access)&(prior.cost_level=='mid')].iloc[0]
            actual=float(z.baseline_resource_eur.sum());assert np.isclose(actual,ref.baseline_resource_eur,rtol=1e-10)
            audit.append(dict(capacity=cap,access=access,original_baseline_resources=actual,reference=float(ref.baseline_resource_eur),reproduced=True))
    for n in [95,297]:
        sites=pd.read_csv(ROOT.parent/f'global_lane/network/SELECTED_{n}.csv').sort_values('priority_order')
        assert sites.prediction_id.isin(q.index).all()
        s=q.loc[sites.prediction_id].reset_index(drop=True)
        assert np.array_equal(s.prediction_id,sites.prediction_id)
        for cap in CFG['capacity_branches']:
            for access in CFG['access_branches']:
                for level in CFG['levels']:
                    z=network(s,level=level,capacity=cap,access=access,bridge=True,scenario_id=f'QAQC_{n}_{cap}_{access}_{level}')
                    cols=[c for c in z if c.endswith('_eur')]
                    rows.append(dict(checkpoint=n,capacity_branch=cap,access_branch=access,cost_level=level,**{c:float(z[c].sum()) for c in cols}))
                    pointrows.append(z.assign(checkpoint=n,capacity_branch=cap,access_branch=access,cost_level=level))
        save(s,f'inputs/SELECTED_{n}_COST_INPUTS.csv')
    save(pd.DataFrame(rows),'results/BASELINE_GLOBAL_COSTS_QAQC.csv')
    save(pd.concat(pointrows,ignore_index=True),'results/BASELINE_POINT_COSTS_QAQC.csv.gz')
    support=[]
    for n in [95,297]:
        s=pd.read_csv(OUT/f'inputs/SELECTED_{n}_COST_INPUTS.csv')
        support.append(dict(sites=n,no_ror_50km=int(s.ror_count_50km.eq(0).sum()),no_ror_50km_pct=float(s.ror_count_50km.eq(0).mean()*100)))
    save(pd.DataFrame(support),'results/ROR_SUPPORT_COUNTS_QAQC.csv')
    js('audit/SPATIAL_COST_AUDIT.json',dict(status='complete',old_297_point_accounting_checks=audit,new_layout_source=str(ROOT.parent/'global_lane/network'),prices_capabilities_access_rules_unchanged=True,ror_counts=support))
    print(pd.DataFrame(rows).query("checkpoint==297 and cost_level=='mid'")[['capacity_branch','access_branch','baseline_resource_eur']].to_string(index=False),flush=True)

if __name__=='__main__':spatial() if len(sys.argv)>1 and sys.argv[1]=='spatial' else run()
