"""Conditional variance-layer refit; never retrains or modifies frozen H26 predictions."""
from pathlib import Path
import json, hashlib, argparse, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
M = ROOT.parents[2] / 'src/modeling'
OLD = M/'99_临时与待确认/H26_独立检测分支重拟合_20260905'
OOF = OLD/'results/mdc/H26_FINAL10_OOF.csv.gz'
OBS = M/'h26/data/observations/h23_observation_layer_training_records.csv.gz'

def load():
    d = pd.read_csv(OOF, low_memory=False)
    o = pd.read_csv(OBS, usecols=['stable_record_id','sizecorr_known_sd_log10','unit_sd_log10'])
    assert d.stable_record_id.is_unique and o.stable_record_id.is_unique
    d = d.merge(o, on='stable_record_id', how='left', validate='one_to_one', indicator=True)
    assert d['_merge'].eq('both').all()
    assert np.allclose(d.residual_log10, d.observed_log10-d.predicted_q50_log10_aware,atol=3e-7)
    d['residual_log10'] = d.observed_log10-d.predicted_q50_log10_aware
    assert d[['source_group','validation_site_event_key']].notna().all().all()
    assert d[['sizecorr_known_sd_log10','unit_sd_log10','latent_sd_log10']].notna().all().all()
    d['known_var'] = d.sizecorr_known_sd_log10**2+d.unit_sd_log10**2
    return d

def arrays(d):
    g, groups = pd.factorize(d.source_group, sort=True)
    y = d.residual_log10.to_numpy(float)
    v = (d.known_var+d.latent_sd_log10**2).to_numpy(float)
    return y,v,g,len(groups)

def fit(a, multiplicity=None, start=None):
    y,v,g,G = a
    m = np.ones(G) if multiplicity is None else multiplicity
    n = np.bincount(g, minlength=G)
    N = m@n
    # Independent source blocks: V_g = diag(known variance + sigma2) + tau2 11'.
    # Sherman-Morrison reduces exact REML objective to O(records + sources).
    def objective(theta, details=False):
        tau,sigma = theta
        diag = v+sigma
        inv = 1/diag
        s = np.bincount(g, inv, minlength=G)
        q = np.bincount(g, inv*y, minlength=G)
        r = np.bincount(g, inv*y*y, minlength=G)
        ld = np.bincount(g, np.log(diag), minlength=G)+np.log1p(tau*s)
        A = m@(s/(1+tau*s))
        B = m@(q/(1+tau*s))
        C = m@(r-tau*q*q/(1+tau*s))
        val = .5*(m@ld+np.log(A)+C-B*B/A+(N-1)*np.log(2*np.pi))
        return (val,B/A) if details else val
    starts = [start] if start is not None else [[.5,.2],[2.,.05],[.1,1.]]
    fits = [minimize(objective, s, method='L-BFGS-B', bounds=[(0,100),(1e-8,100)],
                     options={'ftol':1e-11,'gtol':1e-5,'maxiter':200}) for s in starts]
    r = min(fits, key=lambda x:x.fun)
    if not r.success:
        rr = minimize(objective, r.x, method='Powell', bounds=[(0,100),(1e-8,100)],
                      options={'ftol':1e-10,'xtol':1e-8,'maxiter':300})
        if rr.success and rr.fun <= r.fun+1e-5: r=rr
    assert r.success, r.message
    return {'between_var':float(r.x[0]),'remaining_var':float(r.x[1]),
            'intercept':float(objective(r.x,True)[1]),'objective':float(r.fun)}

def power_factor():
    t=np.arange(6); X=np.column_stack([np.ones(6),t]); R=.3**np.abs(t[:,None]-t)
    h=np.linalg.solve(X.T@np.linalg.solve(R,X),np.linalg.solve(R,X).T)[1]*5
    return float((norm.ppf(.975)+norm.ppf(.8))**2*(h@R@h))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--replicates',type=int,default=0); args=ap.parse_args()
    d=load(); rng=np.random.default_rng(20261013); rows=[]; drawrows=[]
    for domain in ['freshwater','transition','marine']:
        x=d[d.domain.eq(domain)].copy(); a=arrays(x); base=fit(a)
        known=float(x.known_var.mean()); base.update(domain=domain,records=len(x),sources=a[3],
            site_events=x.validation_site_event_key.nunique(),reported_var=known)
        for scenario,var in [('changing',base['between_var']+base['remaining_var']+known),
                             ('stable',base['remaining_var']+known),('standardized',base['remaining_var'])]:
            base['N30_'+scenario]=int(np.ceil(power_factor()*var/np.log10(.7)**2))
        rows.append(base); print(json.dumps(base),flush=True)
        G=a[3]; g=a[2]; counts=np.bincount(g,minlength=G)
        knownsum=np.bincount(g,x.known_var.to_numpy(),minlength=G)
        for b in range(args.replicates):
            m=rng.multinomial(G,np.ones(G)/G)
            f=fit(a,m,[base['between_var'],base['remaining_var']])
            f.update(domain=domain,replicate=b,reported_var=float(m@knownsum/(m@counts)))
            drawrows.append(f)
            if (b+1)%100==0: print(domain,b+1,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'REFIT_POINT_ESTIMATES.csv',index=False)
    if drawrows: pd.DataFrame(drawrows).to_csv(ROOT/'SOURCE_BOOTSTRAP_DRAWS.csv',index=False)
    manifest={'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OOF,OBS]},
        'replicates':args.replicates,'seed':20261013,'method':'source-cluster percentile bootstrap of heteroscedastic random-intercept REML',
        'conditional_on':['frozen OOF predictions','reported size/unit variances','latent marginal variances treated as independent known diagonal noise','six annual occasions','AR1 rho=0.3','independent sites'],
        'not_included':['BNN retraining uncertainty','estimated temporal covariance','spatial covariance','longitudinal validation'],
        'standardized_scenario':'hypothetical complete removal of reported size/unit variance; not pure natural variance'}
    (ROOT/'REFIT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__': main()
