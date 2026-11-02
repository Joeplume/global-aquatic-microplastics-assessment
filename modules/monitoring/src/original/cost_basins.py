from common import *
from engine import network
import gc

def allocate(g,b,ns):
    counts=np.zeros(len(g),int);targets=dict(zip(b.basin_key,ns))
    for key,idx in g.groupby('basin_key',sort=False).indices.items():
        n=int(targets[key]);a=g.cell_area_km2.iloc[idx].to_numpy();x=n*a/a.sum();k=np.floor(x).astype(int)
        k[np.argsort(-(x-k),kind='stable')[:n-int(k.sum())]]+=1;counts[idx]=k
    return counts

def run():
    status('basin_scenarios_running')
    g=pd.read_csv(ROOT/'inputs/GRID_V3.csv.gz',low_memory=False);g=g[g.basin_key!='unassigned'].reset_index(drop=True)
    b=pd.read_csv(V2/'results/BASIN_LOCAL_VARIANCE_AND_REQUIRED_SITES.csv');ps=pd.read_csv(V2/'results/BASIN_POWER_SENSITIVITY.csv.gz');f=pd.read_csv(V2/'results/BASIN_STATION_EQUIVALENT_ALLOCATION.csv.gz',usecols=['prediction_id','station_equivalents','scope'])
    outputs=[];countries=[];totals=[];checks=[];point_paths=[]
    for scope in ['all_assessed_basins','existing_baseline_basins']:
        bx=b if scope=='all_assessed_basins' else b[b.existing_baseline_eligible]
        gx=g[g.basin_key.isin(bx.basin_key)].reset_index(drop=True)
        original=f[f.scope==scope].set_index('prediction_id').station_equivalents.reindex(gx.prediction_id).fillna(0).to_numpy(int)
        for red in [0.,.25,.5,.75]:
            if red==0:
                counts=original;N=bx.required_independent_sites.to_numpy(int)
                checks.append(dict(check=scope+'_frozen_counts',passed=bool(np.array_equal(counts,allocate(gx,bx,N)))))
            else:
                z=ps[(ps.rho==.3)&(ps.common_natural_fraction==0)&(ps.measurement_reduction==red)].set_index('basin_key')
                N=z.loc[bx.basin_key,'required_independent_sites'].to_numpy(int);counts=allocate(gx,bx,N)
            got=pd.Series(counts).groupby(gx.basin_key).sum().reindex(bx.basin_key)
            checks.append(dict(check=f'{scope}_{red}_basin_N',passed=bool(np.array_equal(got,N))))
            if red==0:
                save(pd.DataFrame(dict(prediction_id=gx.prediction_id,basin_key=gx.basin_key,station_equivalents=counts,scope=scope,physical_locations_verified=False)),'inputs/FROZEN_STATION_EQUIVALENTS_'+scope+'.csv.gz')
            for protocol in (['qualified_diverse','regional_common','single_workflow'] if red==0 else ['regional_common','single_workflow']):
                for cap in CFG['capacity_branches']:
                    for access in CFG['access_branches']:
                        for level in CFG['levels']:
                            name=f'{scope}_red{red}_{protocol}_{cap}_{access}_{level}'
                            z=network(gx,counts,level=level,capacity=cap,access=access,coarse=True,protocol=protocol,scenario_id=name)
                            cols=[c for c in z if c.endswith('_eur')]
                            agg=z.groupby(['basin_key','waterbody'],sort=False)[['station_equivalents','field_events_six_occasions','trips_per_occasion']+cols].sum().reset_index()
                            agg['occupied_cells']=z.groupby('basin_key',sort=False).size().reindex(agg.basin_key).to_numpy()
                            agg=agg.merge(bx[['basin_key','variance_basis','local_df','existing_baseline_eligible','required_independent_sites','measurement_fraction_assumed','longitude','latitude','area_km2']],on='basin_key',validate='one_to_one')
                            for k,v in dict(scope=scope,protocol=protocol,capacity_branch=cap,access_branch=access,cost_level=level,measurement_reduction=red,size_window_um='10-5000',sampling_occasions=6,route_feasibility_verified=False,target_contract_verified=False).items():agg[k]=v
                            for c in CATS:agg[c+'_share']=agg[c+'_six_resource_eur']/agg.six_resource_eur
                            outputs.append(agg)
                            co=z.groupby(['country_iso3','waterbody'],dropna=False)[['station_equivalents']+cols].sum().reset_index()
                            for k,v in dict(scope=scope,protocol=protocol,capacity_branch=cap,access_branch=access,cost_level=level,measurement_reduction=red).items():co[k]=v
                            countries.append(co)
                            sums=dict(scope=scope,protocol=protocol,capacity_branch=cap,access_branch=access,cost_level=level,measurement_reduction=red,basin_units=len(bx),station_equivalents=int(N.sum()),**{c:float(agg[c].sum()) for c in cols})
                            totals.append(sums)
                            checks.append(dict(check=name+'_cash_resource_basin_country_closure',passed=bool(np.isclose(agg.six_resource_eur.sum(),co.six_resource_eur.sum()) and np.isclose(agg.six_cash_eur.sum(),co.six_cash_eur.sum()) and np.allclose(agg.six_resource_eur,agg.baseline_resource_eur+5*agg.annual_resource_eur))))
                            if red==0 and protocol=='qualified_diverse':
                                path='results/POINT_COSTS_'+name+'.csv.gz';save(z,path);point_paths.append(path)
                                # Full event/allocation books at central prices, every unknown/route branch.
                                if level=='mid':
                                    dl,e,a,t=network(gx,counts,level=level,capacity=cap,access=access,coarse=True,protocol=protocol,scenario_id=name,ledger=True)
                                    save(e,'results/EVENTS_'+name+'.csv.gz');save(a,'results/ALLOCATIONS_'+name+'.csv.gz');save(t,'results/TRIPS_'+name+'.csv.gz')
                                    checks.append(dict(check=name+'_ledger_allocation_closure',passed=bool(np.isclose(e.resource_eur.sum(),a.resource_eur.sum()) and np.isclose(e.cash_eur.sum(),a.cash_eur.sum()))))
                                    del dl,e,a,t;gc.collect()
                            del z
                log('Basin scenarios',scope,'measurement reduction',red,'protocol',protocol)
                # Recoverable aggregate checkpoints; no finished work lost on interruption.
                save(pd.concat(outputs,ignore_index=True),'results/BASIN_COSTS_V3.csv.gz');save(pd.DataFrame(totals),'results/BASIN_GLOBAL_COSTS_V3.csv');status('basin_scenarios_running',completed_scenario_rows=len(totals),scope=scope,measurement_reduction=red,protocol=protocol)
    out=pd.concat(outputs,ignore_index=True);tt=pd.DataFrame(totals);co=pd.concat(countries,ignore_index=True)
    save(out,'results/BASIN_COSTS_V3.csv.gz');save(co,'results/BASIN_COUNTRY_COSTS_V3.csv.gz');save(tt,'results/BASIN_GLOBAL_COSTS_V3.csv')
    ref=tt[(tt.protocol=='qualified_diverse')&(tt.measurement_reduction==0)][['scope','capacity_branch','access_branch','cost_level','station_equivalents','six_resource_eur','six_cash_eur','annual_resource_eur','training_organization_baseline_resource_eur']].rename(columns={'station_equivalents':'reference_sites','six_resource_eur':'reference_resource_eur','six_cash_eur':'reference_cash_eur','annual_resource_eur':'reference_annual_eur','training_organization_baseline_resource_eur':'reference_initial_support_eur'})
    benefits=tt.merge(ref,on=['scope','capacity_branch','access_branch','cost_level'],validate='many_to_one')
    benefits['saved_station_equivalents']=benefits.reference_sites-benefits.station_equivalents
    benefits['net_resource_saving_eur']=benefits.reference_resource_eur-benefits.six_resource_eur
    benefits['net_cash_saving_eur']=benefits.reference_cash_eur-benefits.six_cash_eur
    benefits['resource_saving_pct']=100*benefits.net_resource_saving_eur/benefits.reference_resource_eur
    annual_saving=benefits.reference_annual_eur-benefits.annual_resource_eur
    extra_start=benefits.training_organization_baseline_resource_eur-benefits.reference_initial_support_eur
    benefits['break_even_followup_years']=np.where(annual_saving>0,np.maximum(extra_start,0)/annual_saving,np.nan)
    benefits['conditional_not_empirical']=True;save(benefits,'results/BASIN_METHOD_BENEFITS.csv')
    sensitivity=ps.groupby(['rho','measurement_reduction','common_natural_fraction']).agg(units=('basin_key','size'),attainable_units=('attainable','sum'),required_sites=('required_independent_sites','sum')).reset_index();sensitivity['unattainable_units']=sensitivity.units-sensitivity.attainable_units
    save(sensitivity,'results/POWER_VARIANCE_SENSITIVITY_SEPARATE.csv')
    js('audit/BASIN_CHECKS.json',checks);assert all(t['passed'] for t in checks)
    js('audit/BASIN_DELIVERY.json',dict(point_cost_files=point_paths,independent_scopes=True,basin_units=13279,existing_baseline_subset=547,local_variance_units=int((b.local_df>0).sum()),unassigned_cells=4077,field_occasions=6,event_ledger_prices='mid; all two capacity x two route branches and both scopes',other_prices='same physical events and unit prices scaled by frozen level; reproduce with scripts',limitations='independent station equivalents; AR1 planning model; grid-footprint areas; travel, preservation and capacity scenarios'))
    log('Basin scenarios complete',len(tt),'tests',len(checks))
if __name__=='__main__':run()
