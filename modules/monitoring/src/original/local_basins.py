from common import *
from local_costs import network
from scipy.stats import norm,chi2

def components():
 g=pd.read_csv(ROOT/'inputs/LOCALIZED_GRID.csv.gz',low_memory=False)
 g=g[g.basin_key.ne('unassigned')].copy()
 ev=pd.read_csv(ROOT/'inputs/LOCAL_EVENT_RESIDUALS.csv.gz',low_memory=False)
 comp=pd.read_csv(F/'04_MDC与监测政策/01_source_mdc/H26_MDC_VARIANCE_COMPONENTS.csv').set_index('domain')
 mapping={DOM[0]:'freshwater',DOM[1]:'transition'}
 # Each source x site-event contributes once. Source means removed; source is not assumed to be a laboratory.
 e=ev.groupby(['basin_key','source_group','site_event_key'],as_index=False).agg(residual=('residual_log10','mean'))
 e['source_n']=e.groupby(['basin_key','source_group']).residual.transform('size')
 e['source_mean']=e.groupby(['basin_key','source_group']).residual.transform('mean')
 e['squared_within']=(e.residual-e.source_mean)**2
 loc=e[e.source_n>=2].groupby('basin_key').agg(sse=('squared_within','sum'),events=('site_event_key','size'),sources=('source_group','nunique'))
 loc['df']=loc.events-loc.sources
 # Same source, exact coordinates, method and year: diagnostic only unless enough separate time series.
 annual=ev.groupby(['basin_key','source_group','exact_location_key','detection_method','sampling_year_effective_v1_3'],dropna=False).observed_log10.mean().reset_index()
 ts=[]
 for key,z in annual.groupby(['basin_key','source_group','exact_location_key','detection_method'],dropna=False):
  if len(z)<3:continue
  t=z.sampling_year_effective_v1_3.to_numpy(float);y=z.observed_log10.to_numpy(float)
  if not np.isfinite(t).all():continue
  X=np.c_[np.ones(len(t)),t-t.mean()];r=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
  ts.append(dict(basin_key=key[0],source=key[1],site=key[2],method=key[3],years=len(z),sse=float(r@r),df=len(z)-2))
 tdf=pd.DataFrame(ts);save(tdf,'results/REPEATED_SITE_YEAR_DIAGNOSTICS.csv')
 tlocal=tdf.groupby('basin_key').agg(sse=('sse','sum'),df=('df','sum'),series=('site','size')) if len(tdf) else pd.DataFrame()
 records=[]
 for key,z in g.groupby('basin_key',sort=True):
  water=z.waterbody.iloc[0];d=comp.loc[mapping[water]];v0=d.within_source_total_effective_sd_log10**2
  local=loc.loc[key] if key in loc.index else None
  df=int(local['df']) if local is not None else 0;sse=float(local.sse) if local is not None else 0
  basis='local_source_centered_event_proxy' if df else 'waterbody_borrowed_variance'
  nts=int(tlocal.loc[key,'series']) if key in tlocal.index else 0;td=int(tlocal.loc[key,'df']) if key in tlocal.index else 0
  # Rare eligible repeated-year case. Still not a verified complete six-occasion future panel.
  if nts>=3 and td>=6:
   df=td;sse=float(tlocal.loc[key,'sse']);basis='local_repeated_year_detrended_proxy'
  v=(sse+CFG['shrinkage_prior_df']*v0)/(df+CFG['shrinkage_prior_df'])
  reported_fraction=min(d.reported_sd_pooled_log10**2/v0,1.)*.5
  supportarea=z.loc[z.caoa_analysis_state.eq('in'),'cell_area_km2'].sum()/z.cell_area_km2.sum()
  obs=int(z.observed_records.gt(0).sum())
  records.append(dict(basin_key=key,waterbody=water,basin_id=z.basin_id.iloc[0],cells=len(z),area_km2=z.cell_area_km2.sum(),supported_fraction=supportarea,observed_cells=obs,existing_baseline_eligible=bool(obs>0 and supportarea>=.5),variance_basis=basis,local_df=df,repeat_series=nts,repeat_df=td,event_variance_log10_sq=v,measurement_fraction_assumed=reported_fraction,borrowed_domain_variance=v0,variance_prior_df=20,longitude=np.average(z.longitude,weights=z.cell_area_km2),latitude=np.average(z.latitude,weights=z.cell_area_km2)))
 b=pd.DataFrame(records)
 t=np.arange(6);X=np.c_[np.ones(6),t];R=CFG['rho']**np.abs(t[:,None]-t[None,:]);factor=np.linalg.inv(X.T@np.linalg.solve(R,X))[1,1]*25
 # Exact two-sided Gaussian power critical noncentrality, common for all units.
 cutoff=norm.ppf(.975);lo,hi=0.,10.
 for _ in range(60):
  mid=(lo+hi)/2
  if norm.sf(cutoff-mid)+norm.cdf(-cutoff-mid)>=.8:hi=mid
  else:lo=mid
 threshold=(abs(np.log10(.7))/hi)**2/factor
 b['required_independent_sites']=np.maximum(1,np.ceil(b.event_variance_log10_sq/threshold)).astype(int)
 # Borrowed event-variance uncertainty is a model-based sensitivity, not an empirical local CI.
 rng=np.random.default_rng(CFG['seed']);dof=b.local_df.to_numpy()+20
 draws=b.event_variance_log10_sq.to_numpy()[:,None]*dof[:,None]/rng.chisquare(dof[:,None],size=(len(b),500))
 ndraw=np.ceil(draws/threshold)
 b['n_variance_sensitivity_p025']=np.quantile(ndraw,.025,axis=1);b['n_variance_sensitivity_p975']=np.quantile(ndraw,.975,axis=1)
 save(b,'results/BASIN_LOCAL_VARIANCE_AND_REQUIRED_SITES.csv')
 powerrows=[]
 for rho in [0,.3,.6]:
  R=rho**np.abs(t[:,None]-t[None,:]);fac=np.linalg.inv(X.T@np.linalg.solve(R,X))[1,1]*25
  th=(abs(np.log10(.7))/hi)**2/fac
  for red in CFG['measurement_reductions']:
   vv=b.event_variance_log10_sq.to_numpy()*(1-red*b.measurement_fraction_assumed.to_numpy())
   natural=b.event_variance_log10_sq.to_numpy()*(1-b.measurement_fraction_assumed.to_numpy())
   for common in [0,.01,.05]:
    floor=common*natural;ok=floor<th;N=np.full(len(b),np.nan);N[ok]=np.maximum(1,np.ceil((vv[ok]-floor[ok])/(th-floor[ok])))
    powerrows.append(pd.DataFrame(dict(basin_key=b.basin_key,rho=rho,measurement_reduction=red,common_natural_fraction=common,required_independent_sites=N,attainable=ok)))
 save(pd.concat(powerrows,ignore_index=True),'results/BASIN_POWER_SENSITIVITY.csv.gz')
 z=rng.standard_normal(500);null=float((abs(z)>cutoff).mean());v=b.event_variance_log10_sq.to_numpy();N=b.required_independent_sites.to_numpy();theory=norm.sf(cutoff-abs(np.log10(.7))/np.sqrt(factor*v/N))+norm.cdf(-cutoff-abs(np.log10(.7))/np.sqrt(factor*v/N))
 alternative=np.mean(np.abs(z[None,:]+abs(np.log10(.7))/np.sqrt(factor*v[:,None]/N[:,None]))>cutoff,axis=1)
 save(pd.DataFrame(dict(basin_key=b.basin_key,theoretical_power=theory,monte_carlo_power_500=alternative,null_false_positive_500=null)),'results/BASIN_POWER_CHECK.csv')
 js('audit/VARIANCE_AUDIT.json',dict(basins=len(b),basis_counts=b.variance_basis.value_counts().to_dict(),existing_baseline_eligible=int(b.existing_baseline_eligible.sum()),repeat_time_series=len(tdf),null_mc_500=null,minimum_theoretical_power=float(theory.min()),unit='log10 concentration site-event variance, not latent map uncertainty',boundary='OOF within-source event residuals are planning proxies, not measured annual variability. Power assumes independent stations and fixed AR1 rho.'))
 return g,b,threshold

def allocate(g,b,nvals):
 # Cost allocation only: spread station equivalents proportionally over each basin's assessed footprint.
 # Multiple equivalents in a 0.5-degree cell require future distinct subgrid locations and independence validation.
 counts=np.zeros(len(g),int);targets=dict(zip(b.basin_key,nvals))
 for key,idx in g.groupby('basin_key',sort=False).indices.items():
  n=int(targets[key]);area=g.iloc[idx].cell_area_km2.to_numpy();exact=n*area/area.sum();k=np.floor(exact).astype(int)
  k[np.argsort(-(exact-k),kind='stable')[:n-int(k.sum())]]+=1;counts[idx]=k
 return counts

def run():
 status('basin_costs_running');g,b,threshold=components();g=g.reset_index(drop=True)
 allrows=[];countryrows=[];allocations=[];maintenance=[]
 for scope in ['all_assessed_basins','existing_baseline_basins']:
  bx=b if scope=='all_assessed_basins' else b[b.existing_baseline_eligible]
  gx=g[g.basin_key.isin(bx.basin_key)].reset_index(drop=True)
  # Persistent shared laboratory fees pooled globally once, allocated to basin and country costs.
  for red in CFG['measurement_reductions']:
   N=np.maximum(1,np.ceil(bx.event_variance_log10_sq*(1-red*bx.measurement_fraction_assumed)/threshold)).astype(int)
   counts=allocate(gx,bx,N)
   for route in (['qualified_diverse','regional_common','single_workflow'] if red==0 else ['regional_common','single_workflow']):
    for level in ['low','mid','high']:
     cost=network(gx,counts,level,route=route,bridge=False)
     agg=cost.groupby(['basin_key','waterbody'],sort=False).agg(station_equivalents=('station_equivalents','sum'),occupied_cells=('prediction_id','size'),initial_setup_eur=('initial_setup_eur','sum'),baseline_eur=('baseline_eur','sum'),annual_eur=('annual_eur','sum'),five_year_eur=('five_year_eur','sum'),conversion_eur=('method_conversion_eur','sum'),validation_eur=('method_validation_eur','sum'),annual_support_travel_eur=('annual_mission_travel_eur','sum')).reset_index()
     agg=agg.merge(bx[['basin_key','variance_basis','existing_baseline_eligible','local_df']],on='basin_key',validate='one_to_one');agg['scope']=scope;agg['route']=route;agg['measurement_reduction']=red;agg['cost_level']=level
     agg['subgrid_siting_required']=agg.station_equivalents>agg.occupied_cells
     allrows.append(agg)
     cc=cost.groupby(['country_iso3','country','continent','waterbody'],dropna=False)[['station_equivalents','initial_setup_eur','baseline_eur','annual_eur','five_year_eur']].sum().reset_index();cc['scope']=scope;cc['route']=route;cc['measurement_reduction']=red;cc['cost_level']=level;countryrows.append(cc)
     if red==0 and route=='qualified_diverse' and level=='mid':
      cost['scope']=scope;cost['physical_locations_verified']=False;cost['cell_requires_subgrid_stations']=cost.station_equivalents>1
      allocations.append(cost)
   log('Basin cost scenarios',scope,'variance reduction',red)
  old=gx[gx.observed_records>0]
  if len(old):
   for level in ['low','mid','high']:
    co=network(old,level=level);m=co.groupby(['basin_key','waterbody'],sort=False).agg(observed_grid_proxies=('prediction_id','size'),maintenance_annual_eur=('annual_eur','sum')).reset_index();m['scope']=scope;m['cost_level']=level;maintenance.append(m)
 out=pd.concat(allrows,ignore_index=True);countries=pd.concat(countryrows,ignore_index=True)
 save(out,'results/BASIN_NETWORK_COSTS.csv.gz');save(countries,'results/BASIN_NETWORK_COUNTRY_COSTS.csv');save(pd.concat(allocations,ignore_index=True),'results/BASIN_STATION_EQUIVALENT_ALLOCATION.csv.gz');save(pd.concat(maintenance,ignore_index=True),'results/EXISTING_GRID_MAINTENANCE_COSTS.csv')
 sums=out.groupby(['scope','route','measurement_reduction','cost_level']).agg(basin_units=('basin_key','size'),station_equivalents=('station_equivalents','sum'),occupied_cells=('occupied_cells','sum'),baseline_eur=('baseline_eur','sum'),annual_eur=('annual_eur','sum'),five_year_eur=('five_year_eur','sum'),conversion_eur=('conversion_eur','sum'),validation_eur=('validation_eur','sum')).reset_index()
 ref=sums[sums.route.eq('qualified_diverse')][['scope','cost_level','five_year_eur','annual_eur']].rename(columns={'five_year_eur':'reference_five_year_eur','annual_eur':'reference_annual_eur'})
 sums=sums.merge(ref,on=['scope','cost_level'],validate='many_to_one');sums['saving_pct']=100*(1-sums.five_year_eur/sums.reference_five_year_eur)
 sums['payback_years']=np.where(sums.reference_annual_eur>sums.annual_eur,(sums.conversion_eur+sums.validation_eur)/(sums.reference_annual_eur-sums.annual_eur),np.nan)
 save(sums,'results/BASIN_METHOD_COST_SUMMARY.csv')
 tests=[]
 for key,z in out.groupby(['scope','route','measurement_reduction','cost_level']):
  a=countries[(countries.scope==key[0])&(countries.route==key[1])&(countries.measurement_reduction==key[2])&(countries.cost_level==key[3])]
  tests.append(dict(check='country_reconciliation_'+str(key),passed=bool(np.isclose(z.five_year_eur.sum(),a.five_year_eur.sum()))))
  tests.append(dict(check='six_occasions_'+str(key),passed=bool(np.allclose(z.five_year_eur,z.baseline_eur+5*z.annual_eur))))
 tests.append(dict(check='basin_scope_not_spatial_95_297',passed=len(b)>297))
 js('audit/BASIN_COST_TESTS.json',tests);assert all(x['passed'] for x in tests)
 log('Basin costs complete',len(b),'units; existing baseline eligible',int(b.existing_baseline_eligible.sum()))
if __name__=='__main__':run()
