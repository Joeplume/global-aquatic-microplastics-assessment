"""Basin evaluation, fixed-repeat design, local variance and scenario costs.

The route/capability columns are frozen implementation assumptions. This runner
recomputes consequences of those assumptions; it does not verify access on site.
"""
from pathlib import Path
import ast, json, math, argparse
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import norm,chi2
from reproduce_monitoring import ROOT,SRC,definitions

def management(out):
    ns=dict(np=np,pd=pd,math=math,Any=Any)
    tree=ast.parse((SRC/'management.py').read_text('utf-8-sig'))
    assigns=[x.id for n in tree.body if isinstance(n,ast.Assign) for x in n.targets if isinstance(x,ast.Name)]
    definitions('management.py',['require','slope_factor','mdc_log10','required_sites','decline_pct','variance_table','pareto_curve','choose_monitoring_arms','build_network_packages'],ns,assigns)
    inp=ROOT/'inputs/management'
    comp=pd.read_csv(inp/'H26_MDC_VARIANCE_COMPONENTS.csv')
    grid=pd.read_csv(inp/'grid_support_v02.csv.gz',dtype={'basin_id':str})
    grid['caoa_status_current']=np.where(grid.native_v02_code.eq(0),'CAOA_in',grid.native_v02_status)
    sites=pd.read_csv(inp/'H26_CURRENT_NETWORK_SITES.csv.gz',dtype={'basin_id':str})
    weights=pd.read_csv(inp/'H26_EMISSION_BASIN_WEIGHTS.csv.gz',dtype={'basin_id':str})
    metrics=pd.read_csv(inp/'bridge_basin_requirements_v02.csv.gz',dtype={'basin_id':str})
    sigma=comp.set_index('domain').within_source_total_effective_sd_log10.to_dict()
    ablation=ns['variance_table'](comp)
    _,selected=ns['choose_monitoring_arms'](grid,sites,weights,sigma)
    packs=ns['build_network_packages'](selected,metrics,ablation)
    curve=ns['pareto_curve'](packs,'joint')
    packs.to_csv(out/'FIXED_REPEAT_PACKAGES.csv.gz',index=False);curve.to_csv(out/'FIXED_REPEAT_CURVE.csv',index=False)
    old=pd.read_csv(ROOT/'reference/FIXED_REPEAT_CURVE.csv')
    assert len(curve)==len(old) and np.allclose(curve.emission_weighted_satisfied_basin_share,old.emission_weighted_satisfied_basin_share)
    act=packs[~packs.joint_baseline & packs.joint_feasible & packs.joint_cost.gt(0)].copy()
    act['benefit_per_point']=act.emission_weight/act.joint_cost
    first20=act.sort_values(['benefit_per_point','emission_weight','basin_id'],ascending=[False,False,True],kind='mergesort').head(20)
    first20.to_csv(out/'FIRST20_PACKAGES.csv',index=False)
    old20=pd.read_csv(ROOT/'reference/FIRST20_PACKAGES.csv',dtype={'basin_id':str})
    assert first20.basin_id.tolist()==old20.basin_id.tolist()
    gn=grid[grid.latitude.ge(-60)&grid.latitude.lt(60)].copy();sn=sites[sites.latitude.ge(-60)&sites.latitude.lt(60)].copy()
    _,selected=ns['choose_monitoring_arms'](gn,sn,weights,sigma)
    selected['caoa_comparable']=selected.caoa_in_area_share.ge(.5-1e-15)
    selected['mdc_detectable']=selected.detectable_current_reference.fillna(False).astype(bool)
    selected['four_state']=np.select([selected.caoa_comparable&selected.mdc_detectable,selected.caoa_comparable&~selected.mdc_detectable,~selected.caoa_comparable&selected.mdc_detectable],['comparable_and_detectable','comparable_not_detectable','detectable_comparability_insufficient'],default='both_insufficient')
    spot=pd.read_csv(inp/'H26_SPOT_UNMANAGED_EMISSION_GRID_FREEZE.csv.gz',dtype={'basin_id':str})
    spot=spot[spot.latitude.ge(-60)&spot.latitude.lt(60)&spot.basin_id.notna()&spot.basin_id.ne('UNROUTED_SPOT')]
    selected['nonpolar_emission_t_yr']=selected.basin_id.map(spot.groupby('basin_id').spot_debris_emission_t_yr.sum()).fillna(0)
    disp=selected[selected.selected_waterbody.notna()&np.isfinite(selected.caoa_in_area_share)].copy()
    rows=[dict(state=state,basins=len(z),basin_pct=len(z)/len(disp)*100,emission_pct=z.nonpolar_emission_t_yr.sum()/disp.nonpolar_emission_t_yr.sum()*100) for state,z in disp.groupby('four_state')]
    summary=pd.DataFrame(rows).sort_values('state');summary.to_csv(out/'NONPOLAR_STATES_RECOMPUTED.csv',index=False)
    old=pd.read_csv(ROOT/'reference/NONPOLAR_STATES_SUMMARY.csv').sort_values('state')
    assert summary.state.tolist()==old.state.tolist() and np.allclose(summary[['basins','basin_pct','emission_pct']],old[['basins','basin_pct','emission_pct']])
    fw=gn[gn.waterbody.eq('reference_surface_freshwater')&gn.natural_water_exists]
    passing=set(disp.loc[disp.four_state.eq('comparable_and_detectable'),'basin_id'])
    area=100*fw.loc[fw.basin_id.isin(passing),'area_weight'].sum()/fw.area_weight.sum()
    return dict(assessed_basins=len(disp),freshwater_pass_area_pct=float(area),states=rows,first20_basin_ids=first20.basin_id.tolist(),
        variance_input='crossfold-calibrated H26_MDC_VARIANCE_COMPONENTS, not REML Table18')

def local_variance(out):
    inp=ROOT/'inputs/basins'
    s=(SRC/'local_basins.py').read_text('utf-8-sig')
    s=s.replace("ROOT/'inputs/LOCALIZED_GRID.csv.gz'", "INPUT/'LOCALIZED_GRID.csv.gz'")
    s=s.replace("ROOT/'inputs/LOCAL_EVENT_RESIDUALS.csv.gz'", "INPUT/'LOCAL_EVENT_RESIDUALS_QAQC.csv.gz'")
    s=s.replace("F/'04_MDC与监测政策/01_source_mdc/H26_MDC_VARIANCE_COMPONENTS.csv'", "COMPONENT_FILE")
    def save(d,path):
        dest=out/path;dest.parent.mkdir(parents=True,exist_ok=True);d.to_csv(dest,index=False)
    def js(path,data):
        dest=out/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(data,indent=2),encoding='utf8')
    ns=dict(np=np,pd=pd,norm=norm,chi2=chi2,INPUT=inp,COMPONENT_FILE=ROOT/'inputs/management/H26_MDC_VARIANCE_COMPONENTS.csv',
            CFG=json.loads((ROOT/'inputs/costs/V2_CONFIG.json').read_text('utf-8-sig')),DOM=['reference_surface_freshwater','transition_coastal'],save=save,js=js)
    nodes=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in ['components','allocate']]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SRC/'local_basins.py'),'exec'),ns)
    g,b,threshold=ns['components']()
    old=pd.read_csv(ROOT/'reference/BASIN_LOCAL_VARIANCE_AND_REQUIRED_SITES.csv').set_index('basin_key')
    a=b.set_index('basin_key');assert set(a.index)==set(old.index)
    assert np.allclose(a.event_variance_log10_sq,old.loc[a.index,'event_variance_log10_sq'],rtol=1e-9)
    assert np.array_equal(a.required_independent_sites,old.loc[a.index,'required_independent_sites'])
    return g,b,threshold,ns['allocate']

def basin_costs(out,full=False):
    _,b,threshold,allocate=local_variance(out)
    g=pd.read_csv(ROOT/'inputs/basins/GRID_V3.csv.gz',low_memory=False);g=g[g.basin_key.ne('unassigned')].reset_index(drop=True)
    ns=dict(np=np,pd=pd,OLD=json.loads((ROOT/'inputs/costs/V1_CONFIG.json').read_text('utf-8-sig')))
    definitions('cost_common.py',['haversine'],ns,['CFG','DOM','CATS'])
    definitions('cost_engine.py',['route_parameters','routes','network'],ns)
    rows=[]
    for scope in ['all_assessed_basins','existing_baseline_basins']:
        bx=b if scope=='all_assessed_basins' else b[b.existing_baseline_eligible]
        gx=g[g.basin_key.isin(bx.basin_key)].reset_index(drop=True)
        for red in ([0.,.25,.5,.75] if full else [0.,.5]):
            N=np.maximum(1,np.ceil(bx.event_variance_log10_sq*(1-red*bx.measurement_fraction_assumed)/threshold)).astype(int)
            counts=allocate(gx,bx,N)
            assert np.array_equal(pd.Series(counts).groupby(gx.basin_key).sum().reindex(bx.basin_key),N)
            protocols=(['qualified_diverse','regional_common','single_workflow'] if red==0 else ['regional_common','single_workflow']) if full else (['qualified_diverse'] if red==0 else ['regional_common'])
            for protocol in protocols:
                for cap in ns['CFG']['capacity_branches']:
                    for access in ns['CFG']['access_branches']:
                        for level in (ns['CFG']['levels'] if full else ['mid']):
                            z=ns['network'](gx,counts,level=level,capacity=cap,access=access,coarse=True,protocol=protocol)
                            rows.append(dict(scope=scope,protocol=protocol,measurement_reduction=red,capacity_branch=cap,access_branch=access,cost_level=level,basin_units=len(bx),station_equivalents=int(N.sum()),**{c:float(z[c].sum()) for c in z if c.endswith('_eur')}))
                print(scope,red,protocol,'finished',flush=True)
    result=pd.DataFrame(rows);result.to_csv(out/'BASIN_COSTS_RECOMPUTED.csv',index=False)
    old=pd.read_csv(ROOT/'reference/BASIN_GLOBAL_COSTS_QAQC.csv')
    keys=['scope','protocol','measurement_reduction','capacity_branch','access_branch','cost_level']
    a=result.set_index(keys);old=old.set_index(keys).loc[a.index]
    cols=['station_equivalents','baseline_resource_eur','annual_resource_eur','six_resource_eur','six_cash_eur']
    assert np.allclose(a[cols],old[cols],rtol=1e-9,atol=1e-5)
    return dict(basins=len(b),existing_baseline_basins=int(b.existing_baseline_eligible.sum()),station_equivalents=int(b.required_independent_sites.sum()),cost_scenarios_actually_recomputed=len(result),all_cost_scenarios=bool(full),
        variance_basis_counts=b.variance_basis.value_counts().to_dict())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['management','basins','all'],default='management')
    ap.add_argument('--full-cost-grid',action='store_true',help='all 216 low/mid/high scenarios; otherwise selected paired mid-price checks')
    ap.add_argument('--out','--output',type=Path,default=ROOT/'basin_outputs');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    report={}
    if a.stage in ['management','all']:report['management']=management(a.out)
    if a.stage in ['basins','all']:report['basins']=basin_costs(a.out,a.full_cost_grid)
    (a.out/'BASIN_REPRODUCTION_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
