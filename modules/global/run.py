"""Reaggregate the adopted frozen numerical inputs. This entry does not refit a concentration model."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'2')
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
import global_statistics as s
import nightlights

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'output');ap.add_argument('--skip-mass',action='store_true'); a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    g=pd.read_csv(ROOT/'inputs/prediction_grid.csv.gz',low_memory=False)
    assert g.prediction_id.is_unique
    roster=set(pd.read_csv(ROOT/'inputs/country_roster.csv').iso3)
    summary,shares=s.summaries(g)
    ranks=s.ranks(g,roster)
    cut,gaps,reg=s.high_gaps(g.copy(),roster)
    ratio=s.ratios(g)
    brazil=pd.DataFrame([s.brazil(g)])
    outputs={'SI06.csv':summary,'SI07.csv':shares,'SI08.csv':ranks,'SI11_high_gap.csv':pd.DataFrame([reg]),'SI16.csv':ratio,'SI17.csv':gaps,'BRAZIL.csv':brazil,'HIGH_THRESHOLDS.csv':cut}
    nout=out/'nightlights';nout.mkdir(exist_ok=True)
    nsummary=nightlights.run(ROOT/'inputs',nout,g.copy())
    if not a.skip_mass:
        z=np.load(ROOT/'inputs/mass_parameters.npz');indices=np.random.default_rng(2601003).choice(500,200,replace=False)
        rows=[]
        for short,wb in zip(['freshwater','transition','marine'],s.WBS):
            raw=np.load(ROOT/'inputs'/f'latent_draws_{short}.npz')
            q=g.set_index('prediction_id').loc[raw['prediction_ids']]
            draw=raw['draws'];assert draw.shape==(200,len(q))
            w=q[s.SP].to_numpy();w=w/w.sum()
            num=10**np.sum(q.q50_log10.to_numpy()*w)
            nd=10**np.sum(draw*w[None,:],axis=1,dtype=np.float64)
            factor=z[wb+'__total']*z[wb+'__size_count_multiplier']/1000;fp=np.median(factor)
            m=np.quantile(nd*fp,[.025,.975]);j=np.quantile(nd*factor[indices],[.025,.975])
            rows.append({'waterbody':wb,'finite_cells':len(q),'number_geomean_items_m3':num,'mass_geomean_ug_L':num*fp,'mass_model_q025':m[0],'mass_model_q975':m[1],'mass_joint_q025':j[0],'mass_joint_q975':j[1],'mass_factor_ug_per_item':fp*1000})
        outputs['SI20.csv']=pd.DataFrame(rows)
    # Country statistics use the adopted UN193 attribution; denominator is all freshwater records.
    rec=pd.read_csv(ROOT/'inputs/record_locations.csv.gz')
    aliases=pd.read_csv(ROOT/'inputs/country_aliases.csv').set_index('raw_iso3').country_iso3.to_dict()
    frame=pd.read_csv(ROOT/'inputs/country_frame.csv');un=set(frame.iso3)
    rec['country']=rec.country_iso3_v3.replace(aliases)
    counts=rec[rec.country.isin(un)].groupby('country').size().reindex(sorted(un),fill_value=0)
    fw=rec[rec.prediction_domain_for_record.eq(s.WBS[0])]
    fc=fw[fw.country.isin(un)].groupby('country').size().reindex(sorted(un),fill_value=0)
    top=fc.reset_index(name='n').sort_values(['n','country'],ascending=[False,True]).head(15)
    land=pd.read_csv(ROOT/'inputs/land_areas.csv');land=land[land.iso3.isin(un)].groupby('iso3').area.sum()
    country_summary={'records':len(rec),'freshwater_records':len(fw),'no_record_countries':int(counts.eq(0).sum()),'top15_freshwater_share_pct':100*top.n.sum()/len(fw),'top15_land_share_pct':100*land.reindex(top.country).sum()/land.sum(),'area_input':'Frozen Natural Earth WGS84 polygon area calculation; country attribution reapplied, geometry not regenerated.'}
    (out/'COUNTRY_SUMMARY.json').write_text(json.dumps(country_summary,indent=2),encoding='utf8')
    dj=pd.read_csv(ROOT/'inputs/danjiangkou_predictions.csv');rows=[]
    for v,z in dj.groupby('variant',sort=False):
        y=np.log10(z.observed_items_L.to_numpy());p=z.predicted_log10_items_L.to_numpy();err=p-y
        rows.append({'variant':v,'n':len(z),'rmse_log10':np.sqrt(np.mean(err**2)),'mae_log10':np.mean(abs(err)),'bias_log10':err.mean(),'geometric_predicted_observed_ratio':10**err.mean(),'pearson_log10_r':np.corrcoef(y,p)[0,1],'spearman_rho':pd.Series(y).rank().corr(pd.Series(p).rank())})
    outputs['SI23.csv']=pd.DataFrame(rows)
    lib=pd.read_csv(ROOT/'inputs/library_events.csv.gz',low_memory=False)
    cohort=set(pd.read_csv(ROOT/'inputs/model_cohort.csv.gz').stable_record_id)
    counts=[]
    for wb,z in lib.groupby('waterbody'):
        f=z[z.stable_record_id.isin(cohort)]
        counts.append({'waterbody':wb,'library_records':len(z),'library_sources':z.source_group.nunique(),'model_records':len(f),'excluded_records':len(z)-len(f),'model_sources':f.source_group.nunique()})
    pd.DataFrame(counts).to_csv(out/'SI01_SI29_counts.csv',index=False)
    t=pd.read_csv(ROOT/'inputs/time_association.csv')
    t=t[t.scheme.eq('paper_heldout_10fold')&t.stratum.eq('all')&t.arm.isin(['U','B'])].copy()
    u=t.loc[t.arm.eq('U'),'conditional_mean_log10_per_year'].iloc[0]
    t['attenuation_pct']=100*(1-t.conditional_mean_log10_per_year/u)
    t.to_csv(out/'SI02.csv',index=False)
    checks=[]
    for name,d in outputs.items():
        d.to_csv(out/name,index=False)
        old=ROOT/'expected'/name
        if old.exists():
            expected=pd.read_csv(old)
            assert list(d.columns)==list(expected.columns),(name,'columns')
            assert d.shape==expected.shape,(name,'shape')
            diffs={}
            for c in d:
                if pd.api.types.is_numeric_dtype(d[c]) and pd.api.types.is_numeric_dtype(expected[c]):
                    assert np.allclose(d[c],expected[c],equal_nan=True,rtol=1e-10,atol=1e-9),(name,c)
                    diffs[c]=float(np.nanmax(np.abs(d[c].to_numpy(dtype=float)-expected[c].to_numpy(dtype=float))))
                else:assert d[c].fillna('').astype(str).tolist()==expected[c].fillna('').astype(str).tolist(),(name,c)
            checks.append({'table':name,'rows':len(d),'matched_adopted_result':True,'max_absolute_difference':max(diffs.values(),default=0)})
    old=pd.read_csv(ROOT/'expected/NIGHTLIGHT_GROUPS.csv');new=pd.read_csv(nout/'GROUP_RESULTS.csv')
    pd.testing.assert_frame_equal(old,new,check_exact=False,rtol=1e-10,atol=1e-10)
    checks.append({'table':'NIGHTLIGHT_GROUPS.csv','matched_adopted_result':True,'rows':len(new)})
    result={'status':'passed','entry_point':'frozen QAQC-weighted prediction surface, fixed support classifications, record locations and calibrated latent draws; no model refit','prediction_cells':len(g),'finite_prediction_cells':int(g.q50_log10.notna().sum()),'checks':checks,'nightlight_primary':nsummary,'libraries':{'numpy':np.__version__,'pandas':pd.__version__},'scope':'Reproduces SI6/7/8/11(high-gap component)/16/17/19(night lights)/20 and Brazil comparison; no new C-AOA estimation or training.'}
    (out/'RUN_REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'status':'passed','tables_checked':len(checks),'output':str(out)},ensure_ascii=False))
if __name__=='__main__':main()
