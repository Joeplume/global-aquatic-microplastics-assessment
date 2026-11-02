"""Arithmetic extracted unchanged from the adopted recompute_global.py. Only input paths are portable."""
from pathlib import Path
import numpy as np
import pandas as pd
AR='footprint_area_wgs84_km2'
SP='footprint_area_frozen_sphere_km2'
WBS=['reference_surface_freshwater','transition_coastal','marine']
INPUT=Path(__file__).resolve().parents[1]/'inputs'
def read(path,**kwargs):return pd.read_csv(path,low_memory=False,**kwargs)

def mean(d,area):
    return float(np.average(d.q50_log10,weights=d[area]))

def threshold(d):
    s=d.sort_values('q50_log10',kind='stable')
    return float(np.interp(.75,(s[SP].cumsum()-.5*s[SP])/s[SP].sum(),s.q50_log10))

def summaries(g):
    f=g[g.q50_log10.notna()].copy()
    out=[]
    for wb in WBS:
        d=f[f.waterbody.eq(wb)]
        for scope,z in [('All finite',d),('Environmental AOA in',d[d.aoa_status.eq('AOA_in')]),('Full cross-study support',d[d.baseline_matched_sources.ge(2)])]:
            out.append({'waterbody':wb,'population':scope,'cells':len(z),'area_km2':z[AR].sum(),'mean_log10':mean(z,AR),'geomean_items_m3':10**mean(z,AR)})
    hi=f[f.q50_log10.gt(3)]
    shares=[{'waterbody':wb,'high_cells':int(hi.waterbody.eq(wb).sum()),'high_area_km2':hi.loc[hi.waterbody.eq(wb),AR].sum(),'share_pct':100*hi.loc[hi.waterbody.eq(wb),AR].sum()/hi[AR].sum()} for wb in WBS]
    return pd.DataFrame(out),pd.DataFrame(shares)

def ranks(g,roster):
    fw=g[g.waterbody.eq(WBS[0])&g.q50_log10.notna()]
    rows=[]
    for iso in sorted(roster):
        d=fw[fw.country_iso3.eq(iso)];s=d[d.baseline_matched_sources.ge(2)]
        rows.append({'iso3':iso,'country':d.country.iloc[0],'all_recalc_cells':len(d),'all_recalc_area_km2':d[SP].sum(),
            'all_recalc_mean_q50_log10':mean(d,SP),'caoa_cells':len(s),'caoa_area_km2':s[SP].sum(),
            'caoa_mean_q50_log10':mean(s,SP),'support_pct':100*s[SP].sum()/d[SP].sum()})
    r=pd.DataFrame(rows)
    for col,label in [('all_recalc_mean_q50_log10','rank_all'),('caoa_mean_q50_log10','rank_caoa')]:
        r[label]=r[col].rank(method='min',ascending=False).astype(int)
    rng=np.random.default_rng(261014)
    for scope in ['all','caoa']:
        boot=np.empty((800,len(r)))
        for j,iso in enumerate(r.iso3):
            d=fw[fw.country_iso3.eq(iso)].copy()
            if scope=='caoa':d=d[d.baseline_matched_sources.ge(2)].copy()
            d['wq']=d.q50_log10*d[SP]
            b=d.groupby('block5_id')[[SP,'wq']].sum();n=len(b)
            mult=rng.multinomial(n,np.ones(n)/n,size=800)
            boot[:,j]=(mult@b.wq)/(mult@b[SP])
        br=pd.DataFrame(boot).rank(axis=1,ascending=False,method='min').to_numpy()
        r[scope+'_rank_q025']=np.quantile(br,.025,axis=0)
        r[scope+'_rank_q975']=np.quantile(br,.975,axis=0)
    r['rank_change']=r.rank_all-r.rank_caoa
    return r

def high_gaps(g,roster):
    f=g[g.q50_log10.notna()].copy();cut=[]
    for wb,d in f.groupby('waterbody'):
        q75=threshold(d)
        cut.append({'waterbody':wb,'weighted_q75_log10':q75,'weighted_q75_items_m3':10**q75,'area_basis':SP})
        g.loc[g.waterbody.eq(wb),'high_threshold_updated_q75']=q75
    g['high_updated']=g.q50_log10.ge(g.high_threshold_updated_q75)
    fw=g[g.waterbody.eq(WBS[0])&g.q50_log10.notna()]
    gaps=[]
    for iso in sorted(roster):
        d=fw[fw.country_iso3.eq(iso)&fw.latitude.abs().lt(66.5)]
        hi=d.high_updated;gap=d.baseline_matched_sources.lt(2)
        ha=d.loc[hi,AR].sum();ga=d.loc[hi&gap,AR].sum()
        gaps.append({'country':d.country.iloc[0],'country_iso3':iso,'high_prediction_footprint_area_km2':ha,'high_caoa_gap_area_km2':ga,
            'high_gap_fraction_of_country_high_area':ga/ha if ha else np.nan,'total_area_km2':d[AR].sum(),'support_area_km2':d.loc[~gap,AR].sum()})
    d=fw[fw.high_updated&fw.baseline_matched_sources.lt(2)]
    regional={'numerator_area_km2':float(d.loc[d.development_group.eq('developing'),SP].sum()),'denominator_area_km2':float(d[SP].sum()),
        'share_pct':float(100*d.loc[d.development_group.eq('developing'),SP].sum()/d[SP].sum())}
    return pd.DataFrame(cut),pd.DataFrame(gaps),regional

def ratios(g):
    rec=read(INPUT/'eligible_records.csv.gz')
    countries=set(rec.loc[rec.sampling_year_effective_v1_3.eq(2020)&rec.core_complete,'country_iso3'])
    observed=set(read(INPUT/'observed_2020.csv').prediction_id)
    assert len(countries)==23
    d=g[g.waterbody.eq(WBS[0])&g.q50_log10.notna()&g.country_iso3.isin(countries)&g.latitude.abs().lt(60)].copy()
    d['observed']=d.prediction_id.isin(observed)
    supported=d[d.baseline_matched_sources.ge(2)].copy()
    common=set(supported.loc[supported.observed,'country_iso3']);assert len(common)==21
    rows=[]
    for scope,z in [('current23_all',d),('paired21_all',d[d.country_iso3.isin(common)]),('paired21_supported',supported[supported.country_iso3.isin(common)])]:
        ca=z.groupby('country_iso3')[AR].sum();oa=z[z.observed].groupby('country_iso3')[AR].sum()
        assert ca.index.difference(oa.index).empty
        _,ix=np.unique(z.block5_id.astype(str),return_inverse=True);n=int(ix.max()+1)
        mult=np.random.default_rng(26101910).multinomial(n,np.full(n,1/n),size=500)
        for footprint in ['observed','all']:
            w=z[AR].to_numpy().copy()
            if footprint=='observed':w*=z.observed.to_numpy()*z.country_iso3.map(ca/oa).to_numpy()
            means=[];den=[];num=[]
            for group in ['developed','developing']:
                wg=w*z.development_group.eq(group).to_numpy()
                den.append(np.bincount(ix,weights=wg,minlength=n));num.append(np.bincount(ix,weights=wg*z.q50_log10,minlength=n))
                means.append(float(np.sum(wg*z.q50_log10)/wg.sum()))
            bm=(mult@np.array(num).T)/(mult@np.array(den).T);br=10**(bm[:,1]-bm[:,0])
            rows.append({'scope':scope,'footprint':footprint,'countries':len(ca),'cells':int((w>0).sum()),'ratio':10**(means[1]-means[0]),'q025':np.nanquantile(br,.025),'q975':np.nanquantile(br,.975)})
    return pd.DataFrame(rows)

def brazil(g):
    feat='night_lights_radiance_2020'
    cov=read(INPUT/'nightlights.csv.gz',usecols=['prediction_id',feat,feat+'__status'])
    d=g[g.waterbody.eq(WBS[0])&g.country_iso3.eq('BRA')].merge(cov,on='prediction_id',validate='one_to_one')
    d=d[d[feat].notna()&d[feat].ge(0)&~d[feat+'__status'].str.contains('unresolved|imputed|not_applicable|missing',case=False,na=True)]
    train=read(INPUT/'record_locations.csv.gz',usecols=['m5_grid_id','prediction_domain_for_record'])
    observed=set(train.loc[train.prediction_domain_for_record.eq(WBS[0]),'m5_grid_id']+'__'+WBS[0])
    d=d[d.q50_log10.notna()];o=d[d.prediction_id.isin(observed)]
    assert len(o)==21
    return {'country':'Brazil','finite_target_cells':len(d),'finite_sampled_cells':len(o),
        'sampled_geomean_items_m3':10**mean(o,AR),'national_geomean_items_m3':10**mean(d,AR),
        'sampled_vs_national_predicted_geomean_pct':100*(10**(mean(o,AR)-mean(d,AR))-1),
        'night_light_metrics_changed':False}
