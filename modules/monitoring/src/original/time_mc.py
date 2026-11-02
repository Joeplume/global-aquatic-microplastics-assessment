"""Known-covariance forward temporal MC, using frozen MEM variance anchors.
No historical trend is inferred; source identities are composite source proxies.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='2'
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]; W=ROOT.parents[1]
SEED=26100840; REPS=5000; ALPHA=.05; POWER=.8
SITES=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,20000]
DECLINES=[0,.1,.2,.3,.5]
METHODS=[('source_changing',0),('stable_source',0)]+[('stable_standardized',r) for r in [.25,.5,.75,1.]]
SCENARIOS=[('primary_annual',.3,1,0.,0.),('independent_annual',0.,1,0.,0.),
 ('high_ar_annual',.7,1,0.,0.),('balanced_seasonal',.3,4,0.,.25),
 ('drifting_seasonal',.3,1,0.,.25),('spatial_common05',.3,1,.05,0.),
 ('high_ar_spatial_common05',.7,4,.05,.25),('marine_empirical_ar',.8519,1,0.,0.)]

def schedule(visits,scenario):
    year=np.repeat(np.arange(6),visits)
    phase=np.tile((np.arange(visits)+.5)/visits,6) if visits>1 else np.repeat(.5,6)
    if scenario=='drifting_seasonal':phase=.1+.13*year
    t=year+phase
    return t,year,phase

def covariance(row,n,method,reduction,rho,visits,spatial,season,scenario):
    t,year,phase=schedule(visits,scenario); m=len(t)
    ve=float(row.within_source_total_effective_sd_log10)**2
    vk=float(row.reported_sd_pooled_log10)**2
    vs=float(row.between_source_sd_log10)**2
    # Half of reported component remains natural event variance. This split is a sensitivity assumption.
    natural=max(ve-vk,0)+.5*vk; qa=.5*vk*(1-reduction)
    group_sizes=np.minimum(10,np.maximum(n-np.arange((n+9)//10)*10,0))
    group_weight_sq=float(np.sum((group_sizes/n)**2))
    natural_scale=spatial+(1-spatial)/n
    corr=np.eye(m) if rho==0 else rho**np.abs(t[:,None]-t[None,:])
    c=natural*natural_scale*corr + qa/n*np.eye(m)
    source_corr=(year[:,None]==year[None,:]).astype(float) if method=='source_changing' else np.ones((m,m))
    c+=vs*group_weight_sq*source_corr
    if season:
        s=np.column_stack([np.sin(2*np.pi*phase),np.cos(2*np.pi*phase)])
        c+=season**2*natural_scale*(s@s.T)
    # Each sample mean uses known planned covariance, including source sharing.
    c+=np.eye(m)*1e-12
    x=np.column_stack([np.ones(m),t-t[0]])
    ci=cho_solve(cho_factor(c,lower=True),x)
    h=np.linalg.solve(x.T@ci,ci.T)[1]*5
    var=float(h@c@h)
    # Deliberately misspecified independent event analysis of exactly the same draws.
    h0=np.linalg.solve(x.T@x,x.T)[1]*5
    naive=(ve/n)*float(h0@h0)
    return c,h,var,h0,naive,group_weight_sq

def analytic_power(v,d):
    effect=abs(np.log10(1-d))/np.sqrt(v); z=norm.ppf(1-ALPHA/2)
    return float(norm.cdf(-z-effect)+norm.sf(z-effect))

def run():
    config=dict(seed=SEED,mc_reps=REPS,alpha=ALPHA,target_power=POWER,site_counts=SITES,
      declines=DECLINES,methods=METHODS,scenarios=SCENARIOS,source_sites_per_system=10,
      natural_reported_fraction=.5,standardization_reduces_only_remaining_QA_variance=True,
      event_calendar='2025 baseline + 2026-2030 follow-ups; 5-year concentration decline',
      significance_test='two-sided known-covariance GLS z; variance-estimation uncertainty not included',
      weighting='equal sites and events; shared systems retained',folds=10,new_model_fit=False,
      mc_max_se=float(np.sqrt(.25/REPS)),power80_se=float(np.sqrt(.8*.2/REPS)))
    (ROOT/'TIME_CONFIG.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    comp=pd.read_csv(W/'h26/04_MDC与监测政策/01_source_mdc/H26_MDC_VARIANCE_COMPONENTS.csv')
    rows=[]; targets=[]; anchor=[]; tests=[]
    with threadpool_limits(limits=2):
      for di,row in enumerate(comp.itertuples()):
       for si,(scenario,rho,visits,spatial,season) in enumerate(SCENARIOS):
        if scenario=='marine_empirical_ar' and row.domain!='marine':
         for method,reduction in METHODS:
          targets.append(dict(waterbody=row.domain,scenario=scenario,method=method,qa_variance_reduction=reduction,
            decline_fraction=.3,required_sites=None,status='not_assessed_no_domain_empirical_autocorrelation',mc_reps=0))
         continue
        t,_,_=schedule(visits,scenario)
        # Common random numbers for paired designs and N. No test responses are used in design.
        rng=np.random.default_rng(SEED+di*100+si); eps=rng.standard_normal((REPS,len(t)))
        for method,reduction in METHODS:
         for n in SITES:
          c,h,var,h0,naive,gws=covariance(row,n,method,reduction,rho,visits,spatial,season,scenario)
          noise=eps@np.linalg.cholesky(c).T
          z0=noise@h/np.sqrt(var); zn=noise@h0/np.sqrt(naive)
          for d in DECLINES:
           shift=np.log10(1-d)
           reject=np.abs(z0+shift/np.sqrt(var))>norm.ppf(.975)
           pn=float(reject.mean()); se=float(np.sqrt(pn*(1-pn)/REPS))
           bad=np.abs(zn+shift/np.sqrt(naive))>norm.ppf(.975)
           rows.append(dict(waterbody=row.domain,scenario=scenario,method=method,qa_variance_reduction=reduction,
             sites=n,events_per_site=6*visits,total_field_events=n*6*visits,decline_fraction=d,
             power_or_fpr=pn,mc_se=se,mc_low=max(0,pn-1.96*se),mc_high=min(1,pn+1.96*se),
             analytic_power=analytic_power(var,d),naive_independence_power_or_fpr=float(bad.mean()),
             contrast_variance_log10=var,source_group_weight_squared=gws,seed=SEED+di*100+si,
             mc_reps=REPS,folds=10,evidence='conditional_simulation_not_field_validation'))
         for d in [.1,.2,.3,.5]:
          if analytic_power(covariance(row,20000,method,reduction,rho,visits,spatial,season,scenario)[2],d)<POWER:
            target=None;status='not_reached_in_1_to_20000_sites'
          else:
            lo,hi=1,20000
            while lo<hi:
              mid=(lo+hi)//2
              if analytic_power(covariance(row,mid,method,reduction,rho,visits,spatial,season,scenario)[2],d)>=POWER:hi=mid
              else:lo=mid+1
            target=lo;status='conditional_exact_integer_under_declared_covariance'
          p=None;se=None
          if target:
            c,h,var,*_=covariance(row,target,method,reduction,rho,visits,spatial,season,scenario)
            z=(eps@np.linalg.cholesky(c).T)@h/np.sqrt(var)
            p=float((np.abs(z+np.log10(1-d)/np.sqrt(var))>norm.ppf(.975)).mean())
            se=float(np.sqrt(p*(1-p)/REPS))
          targets.append(dict(waterbody=row.domain,scenario=scenario,method=method,qa_variance_reduction=reduction,
             decline_fraction=d,required_sites=target,total_field_events=None if target is None else target*6*visits,
             events_per_site=6*visits,power_mc_at_target=p,mc_se=se,status=status,seed=SEED+di*100+si,mc_reps=REPS,
             calibration_cost='not_assessed_no_empirical_standardization_quote'))
       ve=float(row.within_source_total_effective_sd_log10)**2;vk=float(row.reported_sd_pooled_log10)**2
       anchor.append(dict(waterbody=row.domain,source_var=float(row.between_source_sd_log10)**2,
         effective_within_var=ve,reported_var=vk,residual_var=max(ve-vk,0),retained_natural_var=max(ve-vk,0)+.5*vk,
         eligible_qa_var=.5*vk,reported_qa_fraction_assumption=.5,source_semantics='composite source proxy; not verified lab'))
       c1,h,v1,*_=covariance(row,100,'stable_source',0,.3,1,0,0,'primary_annual')
       tests.append(dict(test=f'{row.domain}_stable_bias_cancels',passed=abs(float(h.sum()))<1e-10,value=float(h.sum())))
       # Legacy direct deletion endpoint reconstructed separately; never used as primary.
       t=np.arange(6.);x=np.column_stack([np.ones(6),t]);cor=.3**np.abs(t[:,None]-t[None,:])
       vf=float((np.linalg.inv(x.T@np.linalg.solve(cor,x))[1,1])*25)
       q=(norm.ppf(.975)+norm.ppf(.8))**2*vf/np.log10(.7)**2
       tests.append(dict(test=f'{row.domain}_legacy_full_variance_deletion',passed=True,
         stable_sites_approx=int(np.ceil(ve*q)),deletion_sites_approx=int(np.ceil(max(ve-vk,0)*q)),
         evidence='only_formula_reproduction; deletion_not_empirically_achieved'))
       print(time.strftime('%H:%M:%S'),row.domain,'temporal complete',flush=True)
    df=pd.DataFrame(rows); tg=pd.DataFrame(targets)
    df.to_csv(ROOT/'results/TIME_MC_CURVES.csv.gz',index=False,compression='gzip')
    tg.to_csv(ROOT/'results/TIME_POWER80_TARGETS.csv',index=False)
    pd.DataFrame(anchor).to_csv(ROOT/'audit/TIME_VARIANCE_ASSUMPTIONS.csv',index=False)
    null=df[df.decline_fraction.eq(0)].copy()
    null.to_csv(ROOT/'results/TIME_NULL_FPR.csv',index=False)
    tests.extend([dict(test='correct_model_null_mean',passed=abs(null.power_or_fpr.mean()-.05)<.01,value=float(null.power_or_fpr.mean())),
      dict(test='MC_power_matches_analytic_mean_absolute_error',passed=(df.power_or_fpr-df.analytic_power).abs().mean()<.01,
        value=float((df.power_or_fpr-df.analytic_power).abs().mean()))])
    (ROOT/'audit/TIME_TESTS.json').write_text(json.dumps(tests,indent=2,default=lambda x: x.item()),encoding='utf-8')
    assert all(x['passed'] for x in tests)
    primary=tg[tg.scenario.eq('primary_annual')&tg.decline_fraction.eq(.3)]
    primary.to_csv(ROOT/'results/TIME_PRIMARY_30PCT.csv',index=False)
    print(primary.to_string(index=False),flush=True)
    print('null range',null.power_or_fpr.min(),null.power_or_fpr.max(),'naive max',null.naive_independence_power_or_fpr.max(),flush=True)

if __name__=='__main__':run()
