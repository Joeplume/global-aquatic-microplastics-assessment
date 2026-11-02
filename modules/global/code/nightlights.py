"""Adopted all-country night-light algorithm, unchanged except paths and output location. Current concentration grid affects only separate concentration diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
F='night_lights_radiance_2020'
AR='footprint_area_wgs84_km2'
def read(p,**kw):return pd.read_csv(p,**kw)
def qval(d,col,q):
    z=d.sort_values(col)
    return float(z[col].iloc[np.searchsorted(z[AR].cumsum(),z[AR].sum()*q)])
def run(INPUT,OUT,grid):
    def save(df,name):df.to_csv(OUT/name,index=False)
    country_table=read(INPUT/'country_frame.csv')
    UN=set(country_table.iso3)
    assert len(UN)==193
    names=country_table.set_index('iso3').country.to_dict()
    cov=read(INPUT/'nightlights.csv.gz',usecols=['prediction_id',F,F+'__status'])
    assert cov.prediction_id.is_unique and grid.prediction_id.is_unique
    grid=grid[grid.waterbody.eq('reference_surface_freshwater')].merge(cov,on='prediction_id',validate='one_to_one')
    grid['valid_light']=grid[F].notna()&grid[F].ge(0)&~grid[F+'__status'].str.contains('unresolved|imputed|not_applicable|missing',case=False,na=True)
    train=read(INPUT/'record_locations.csv.gz',usecols=['stable_record_id','m5_grid_id','prediction_domain_for_record','sampling_year_effective_v1_3','country_iso3_v3'])
    assert len(train)==34853 and train.stable_record_id.is_unique
    records=train[train.prediction_domain_for_record.eq('reference_surface_freshwater')].copy()
    records['prediction_id']=records.m5_grid_id+'__reference_surface_freshwater'
    cohort=read(INPUT/'model_cohort.csv.gz',usecols=['stable_record_id'])
    assert cohort.stable_record_id.is_unique
    records['in_model_cohort']=records.stable_record_id.isin(set(cohort.stable_record_id))
    allids=set(records.prediction_id)
    modelids=set(records.loc[records.in_model_cohort,'prediction_id'])
    record_n=records.groupby('prediction_id').size()
    grid['sampled_all']=grid.prediction_id.isin(allids)
    grid['sampled_model']=grid.prediction_id.isin(modelids)
    grid['records_all']=grid.prediction_id.map(record_n).fillna(0).astype(int)
    
    # Record-level eligibility audit keeps unknown/no-grid/nonmember/light-missing distinct.
    r=records.merge(grid[['prediction_id','country_iso3','valid_light']],on='prediction_id',how='left',validate='many_to_one')
    r['eligibility']='eligible_location'
    r.loc[r.valid_light.eq(False),'eligibility']='missing_or_invalid_night_light'
    r.loc[r.country_iso3.notna()&~r.country_iso3.isin(UN),'eligibility']='outside_UN193_country_frame'
    r.loc[r.country_iso3.isna(),'eligibility']='unassigned_country_or_missing_grid'
    save(r[['stable_record_id','prediction_id','sampling_year_effective_v1_3','country_iso3_v3','country_iso3','in_model_cohort','eligibility']], 'RECORD_ELIGIBILITY.csv')
    target=grid[grid.valid_light&grid.country_iso3.isin(UN)].copy()
    assert target.prediction_id.is_unique
    country_rows=[]
    eligibility=[]
    variants=[('all_countries_all_years',target,'sampled_all'),
              ('model_cohort_locations',target,'sampled_model'),
              ('finite_concentration_target',target[target.q50_log10.notna()],'sampled_all'),
              ('latitude_below60',target[target.latitude.abs().lt(60)],'sampled_all')]
    for variant,t,sample_col in variants:
        for iso in sorted(UN):
            d=t[t.country_iso3.eq(iso)].copy()
            observed=d[d[sample_col]].copy()
            reason='eligible'
            if d.empty:reason='no_valid_target_grid'
            elif observed.empty:reason='no_sampled_cell_with_valid_light'
            elif d[F].nunique()<2:reason='no_national_night_light_contrast'
            elif d.development_group.nunique()!=1 or d.development_group.iloc[0] not in ['developed','developing']:reason='unassigned_or_inconsistent_development_group'
            eligibility.append({'variant':variant,'country_iso3':iso,'country':names[iso],'target_cells':len(d),
                                'sampled_cells':len(observed),'status':reason})
            if reason!='eligible':continue
            grouped=d.groupby(F)[AR].sum().sort_index()
            cdf=(grouped.cumsum()-.5*grouped)/grouped.sum()
            assert abs(np.average(d[F].map(cdf),weights=d[AR])-.5)<1e-12
            v=float(np.average(observed[F].map(cdf),weights=observed[AR]))
            med_n=qval(d,F,.5);med_o=qval(observed,F,.5);q75=qval(d,F,.75)
            row={'variant':variant,'country_iso3':iso,'country':names[iso],'development_group':d.development_group.iloc[0],
                 'target_cells':len(d),'sampled_cells':len(observed),'target_area_km2':d[AR].sum(),
                 'sampled_area_pct':100*observed[AR].sum()/d[AR].sum(),
                 'mean_sampled_national_percentile':v,'positive_percentile_shift':v>.5+1e-12,
                 'national_median_light':med_n,'sampled_median_light':med_o,'positive_median_shift':med_o>med_n,
                 'national_bright_quartile_threshold':q75,'bright_target_area_pct':100*d.loc[d[F].ge(q75),AR].sum()/d[AR].sum(),
                 'sampled_cells_in_bright_quartile_pct':100*observed[F].ge(q75).mean()}
            # Concentration is not used in selection or in the night-light indicator.
            # Keep this model-conditional diagnostic separate, for the Brazil sentence only.
            df=d[d.q50_log10.notna()];of=observed[observed.q50_log10.notna()]
            row['finite_target_cells']=len(df);row['finite_sampled_cells']=len(of)
            row['sampled_vs_national_predicted_geomean_pct']=100*(10**(np.average(of.q50_log10,weights=of[AR])-np.average(df.q50_log10,weights=df[AR]))-1) if len(df) and len(of) else np.nan
            country_rows.append(row)
    countries=pd.DataFrame(country_rows)
    save(countries,'COUNTRY_RESULTS.csv')
    elig=pd.DataFrame(eligibility);save(elig,'COUNTRY_ELIGIBILITY.csv')
    groups=[]
    for variant,d in countries.groupby('variant'):
        for minimum in [1,5]:
            z=d[d.sampled_cells.ge(minimum)]
            for group in ['developed','developing','all']:
                a=z if group=='all' else z[z.development_group.eq(group)]
                groups.append({'variant':variant,'minimum_sampled_cells':minimum,'group':group,'eligible_countries':len(a),
                    'positive_percentile_countries':int(a.positive_percentile_shift.sum()),
                    'positive_percentile_pct':100*a.positive_percentile_shift.mean(),
                    'positive_median_countries':int(a.positive_median_shift.sum()),
                    'positive_median_pct':100*a.positive_median_shift.mean(),
                    'equal_country_mean_percentile':a.mean_sampled_national_percentile.mean(),
                    'area_weighted_country_mean_percentile':np.average(a.mean_sampled_national_percentile,weights=a.target_area_km2) if len(a) else np.nan})
    groups=pd.DataFrame(groups);save(groups,'GROUP_RESULTS.csv')
    
    # Independent numerical check: rank via pairwise ordering of aggregated light values,
    # instead of the grouped cumulative implementation above. Test every primary country.
    validation=[]
    primary=countries[countries.variant.eq('all_countries_all_years')]
    for row in primary.itertuples():
        d=target[target.country_iso3.eq(row.country_iso3)].sort_values(F)
        x=d[F].to_numpy();w=d[AR].to_numpy();cw=np.r_[0,np.cumsum(w)]
        o=d[d.sampled_all];ox=o[F].to_numpy()
        left=np.searchsorted(x,ox,'left');right=np.searchsorted(x,ox,'right')
        ranks=(cw[left]+.5*(cw[right]-cw[left]))/cw[-1]
        independently=float(np.average(ranks,weights=o[AR]))
        assert abs(independently-row.mean_sampled_national_percentile)<1e-10
        validation.append({'country_iso3':row.country_iso3,'difference':independently-row.mean_sampled_national_percentile})
    save(pd.DataFrame(validation),'INDEPENDENT_CHECK.csv')
    
    primary_group=groups[(groups.variant=='all_countries_all_years')&groups.minimum_sampled_cells.eq(1)]
    assert primary_group.loc[primary_group.group.ne('all'),'eligible_countries'].sum()==len(primary)
    assert not primary.country_iso3.isin(['TWN','HKG','MAC']).any()
    return primary_group.to_dict("records")
