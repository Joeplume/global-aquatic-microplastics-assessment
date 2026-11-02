from recompute_variance import ROOT,PROJECT,H,module
import json,numpy as np,pandas as pd
OLD=PROJECT/'src/modeling/99_临时与待确认/H26_独立检测分支重拟合_20260905'
MAN=OLD/'results/H26_独立检测分支管理可评估性_20260905'
FIG=PROJECT/'figures/Figure4/20260908_V02_knee_population'
NONPOLAR=PROJECT/'07_SUBMISSION_REVISION/Aquatic_microplastics_NW_CURRENT_20260911/04_治理变化的可检出性/03结果/19_非极区MDC'

def save(d,out,name):d.to_csv(out/name,index=False)
def run(mode='qaqc'):
    out=ROOT/mode;out.mkdir(exist_ok=True)
    mg=module(OLD/'code/run_unlocked_management_evaluability.py','management')
    comp=pd.read_csv(ROOT/'qaqc/H26_MDC_VARIANCE_COMPONENTS.csv' if mode=='qaqc' else MAN/'frozen_inputs/H26_MDC_VARIANCE_COMPONENTS.csv')
    grid=pd.read_csv(FIG/'grid_support_v02.csv.gz',dtype={'basin_id':str})
    grid['caoa_status_current']=np.where(grid.native_v02_code.eq(0),'CAOA_in',grid.native_v02_status)
    sites=pd.read_csv(MAN/'frozen_inputs/H26_CURRENT_NETWORK_SITES.csv.gz',dtype={'basin_id':str})
    weights=pd.read_csv(MAN/'frozen_inputs/H26_EMISSION_BASIN_WEIGHTS.csv.gz',dtype={'basin_id':str})
    metrics=pd.read_csv(FIG/'bridge_basin_requirements_v02.csv.gz',dtype={'basin_id':str})
    sigma=comp.set_index('domain').within_source_total_effective_sd_log10.to_dict()
    ablation=mg.variance_table(comp);save(ablation,out,'VARIANCE_SCENARIOS.csv')
    arms,selected=mg.choose_monitoring_arms(grid,sites,weights,sigma)
    packs=mg.build_network_packages(selected,metrics,ablation);curve=mg.pareto_curve(packs,'joint')
    save(packs,out,'FIXED_REPEAT_PACKAGES.csv.gz');save(curve,out,'FIXED_REPEAT_CURVE.csv')
    y=curve.emission_weighted_satisfied_basin_share.to_numpy()*100
    first=int(np.flatnonzero(y>y[0]+1e-10)[0]);gain90=int(np.flatnonzero(y>=y[0]+.9*(y[-1]-y[0])-1e-10)[0])
    fixed=dict(baseline_pct=float(y[0]),first_gain_budget=first,gain90_budget=gain90,gain90_pct=float(y[gain90]),final_budget=len(y)-1,final_pct=float(y[-1]))
    act=packs[~packs.joint_baseline & packs.joint_feasible & packs.joint_cost.gt(0)].copy()
    act['benefit_per_point']=act.emission_weight/act.joint_cost
    act=act.sort_values(['benefit_per_point','emission_weight','basin_id'],ascending=[False,False,True],kind='mergesort')
    save(act.head(20),out,'FIRST20_PACKAGES.csv')
    # The publication's basin-state panel uses a different, explicitly nonpolar denominator.
    gn=grid[grid.latitude.ge(-60)&grid.latitude.lt(60)].copy();sn=sites[sites.latitude.ge(-60)&sites.latitude.lt(60)].copy()
    arms,selected=mg.choose_monitoring_arms(gn,sn,weights,sigma)
    selected['caoa_comparable']=selected.caoa_in_area_share.ge(.5-1e-15)
    selected['mdc_detectable']=selected.detectable_current_reference.fillna(False).astype(bool)
    assessed=selected.selected_waterbody.notna()&np.isfinite(selected.caoa_in_area_share)
    selected['four_state']=np.select([selected.caoa_comparable&selected.mdc_detectable,selected.caoa_comparable&~selected.mdc_detectable,~selected.caoa_comparable&selected.mdc_detectable],['comparable_and_detectable','comparable_not_detectable','detectable_comparability_insufficient'],default='both_insufficient')
    spot=pd.read_csv(MAN/'frozen_inputs/H26_SPOT_UNMANAGED_EMISSION_GRID_FREEZE.csv.gz',dtype={'basin_id':str})
    spot=spot[spot.latitude.ge(-60)&spot.latitude.lt(60)&spot.basin_id.notna()&spot.basin_id.ne('UNROUTED_SPOT')]
    em=spot.groupby('basin_id').spot_debris_emission_t_yr.sum()
    selected['nonpolar_emission_t_yr']=selected.basin_id.map(em).fillna(0)
    disp=selected[assessed].copy();save(disp,out,'NONPOLAR_BASIN_STATES.csv.gz')
    p545=packs[packs.basin_id.isin(disp.basin_id)].copy()
    p545['emission_weight']/=p545.emission_weight.sum()
    c545=mg.pareto_curve(p545,'joint')
    save(p545,out,'FIXED_REPEAT_PACKAGES_545.csv.gz');save(c545,out,'FIXED_REPEAT_CURVE_545.csv')
    summaries=[]
    for state,z in disp.groupby('four_state'):
        summaries.append(dict(state=state,basins=len(z),basin_pct=len(z)/len(disp)*100,emission_pct=z.nonpolar_emission_t_yr.sum()/disp.nonpolar_emission_t_yr.sum()*100))
    save(pd.DataFrame(summaries),out,'NONPOLAR_STATES_SUMMARY.csv')
    polygon=H/'data/06_geospatial_assets/natural_earth_10m_admin1_marine/marine/ne_10m_geography_marine_polys.shp'
    # Reuse the original assigned marine cell footprints, not old MDC values.
    gm=gn.copy();gm.loc[gm.waterbody.eq('marine'),'natural_water_exists']=True
    marine,mc,aud=mg.build_marine_si(gm,sn,polygon,ablation)
    save(marine,out,'MARINE_BASIN_MDC.csv')
    save(mc,out,'MARINE_CELL_BASIN.csv.gz')
    mm=marine[marine.design.eq('fixed_repeat_stable_source_historical_method')&marine.caoa_threshold.eq(.5)]
    mpass=mm.detects_30pct_decline&mm.caoa_comparable
    fw=gn[gn.waterbody.eq('reference_surface_freshwater')&gn.natural_water_exists]
    passing=set(disp.loc[disp.four_state.eq('comparable_and_detectable'),'basin_id'])
    cohort_dir=NONPOLAR.parent/'第四章管理可评估性补算'
    co=pd.read_csv(cohort_dir/'results/PROJECTED_SITE_COHORTS.csv.gz',dtype={'unit_id':str})
    co=co[co.latitude.ge(-60)&co.latitude.lt(60)]
    rc=module(cohort_dir/'recalculate.py','cohort_design')
    add=co.groupby(['waterbody','unit_id']).slope_information.sum()
    continued=disp.copy()
    continued['added_information']=[add.get((r.selected_waterbody,r.basin_id),0.) for r in continued.itertuples()]
    continued['trend_information']=continued.current_sites*rc.BASE_INFO+continued.added_information
    continued['mdc_2030_log10']=rc.mdc(continued.sigma_reference_log10.to_numpy(),continued.trend_information.to_numpy())
    continued['joint_2030']=continued.caoa_comparable&continued.mdc_2030_log10.le(rc.TARGET+1e-12)
    save(continued,out,'CONTINUATION_2030_STATES.csv')
    ocean_continued=mm.copy()
    ocean_continued['added_information']=[add.get(('marine',r.marine_basin),0.) for r in ocean_continued.itertuples()]
    ocean_continued['trend_information']=ocean_continued.current_fixed_sites*rc.BASE_INFO+ocean_continued.added_information
    ocean_continued['mdc_2030_log10']=rc.mdc(ocean_continued.sigma_log10.to_numpy(),ocean_continued.trend_information.to_numpy())
    ocean_continued['joint_2030']=ocean_continued.caoa_comparable&ocean_continued.mdc_2030_log10.le(rc.TARGET+1e-12)
    save(ocean_continued,out,'CONTINUATION_2030_MARINE.csv')
    overview=dict(fixed_repeat=fixed,assessed_basins=len(disp),freshwater_joint_pass_area_pct=float(fw.loc[fw.basin_id.isin(passing),'area_weight'].sum()/fw.area_weight.sum()*100),marine_joint_pass_area_pct=float(mm.loc[mpass,'area_weight_sum'].sum()/mm.area_weight_sum.sum()*100),marine_passing=int(mpass.sum()),nonpolar_summary=summaries,continuation_land_passing=int(continued.joint_2030.sum()),continuation_added_land_passing=int((continued.joint_2030&~continued.mdc_detectable).sum()),continuation_marine_passing=int(ocean_continued.joint_2030.sum()),continuation_added_marine_passing=int((ocean_continued.joint_2030&~ocean_continued.detects_30pct_decline).sum()),cohort_total_nonpolar=len(co))
    (out/'MANAGEMENT_SUMMARY.json').write_text(json.dumps(overview,indent=2),encoding='utf-8');print(json.dumps(overview),flush=True)

if __name__=='__main__':
    import sys
    run(sys.argv[1] if len(sys.argv)>1 else 'qaqc')
