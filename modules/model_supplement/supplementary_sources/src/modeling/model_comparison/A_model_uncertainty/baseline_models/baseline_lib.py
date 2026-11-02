"""Frozen-cohort baselines. All writable artifacts belong to this extension."""
import os,sys,time,json
from pathlib import Path
S=Path(__file__).resolve().parent
A=S.parent
sys.path.insert(0,str(A))
import common as old
os.environ['MPLCONFIGDIR']=str(S/'mpl_cache')
sys.dont_write_bytecode=True
import numpy as np,pandas as pd,torch
from scipy import sparse
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
import run_h26_marine_vector_modular_bnn as strict
import run_h26_freshwater_network_modular_bnn as fresh
import run_h26_transition_outlet_plume_bnn as transition
import run_h24_simple_protocol_preview as h24
torch.set_num_threads(2)
CONFIG=json.loads((S/'CONFIG.json').read_text(encoding='utf8'))

def load_domain(d,fold):
    o=pd.read_csv(A/'GROUPED10_COMPARISON_OOF.csv.gz',low_memory=False)
    o=o.loc[o.domain.eq(d)].reset_index(drop=True)
    dom=old.DOMAINS[d]
    u=A/f'units/{d}/fold_{fold}'
    package = next(p for p in Path(__file__).resolve().parents if p.name == 'model_supplement')
    parameters = json.loads((package/'data/analysis_parameters.json').read_text(encoding='utf8'))['baseline_folds'][f'{d}/fold_{fold}']
    audit = parameters
    done = parameters['training']
    names=audit['process_scaler']['features']
    raw=pd.read_csv(old.CONFIG[d]['training'],low_memory=False).drop_duplicates('stable_record_id').set_index('stable_record_id')
    assert set(o.stable_record_id)<=set(raw.index)
    r=raw.loc[o.stable_record_id].reset_index()
    obs=pd.read_csv(old.OBS,low_memory=False).drop_duplicates('stable_record_id').set_index('stable_record_id')
    for c in ['detection_method','sizecorr_evidence_tier','sampling_year_effective_v1_3','qaqc_total_score','sizecorr_lower_used_um']:
        if c not in r:r[c]=obs.loc[o.stable_record_id,c].to_numpy()
    if 'positive_water_depth_m' in names and 'positive_water_depth_m' not in r:
        r['positive_water_depth_m']=np.maximum(-pd.to_numeric(r['water_depth_m'],errors='coerce'),0)
    if 'h26_cmems_positive_convergence_sinv' in names and 'h26_cmems_positive_convergence_sinv' not in r:
        r['h26_cmems_positive_convergence_sinv']=np.maximum(pd.to_numeric(r['h26_cmems_convergence_sinv'],errors='coerce'),0)
    assert all(c in r for c in names),[c for c in names if c not in r]
    # Source, response, event and fold authority is the retained A OOF, not a reconstructed join.
    for c in ['source_group','observed_log10','validation_site_event_key']:r[c]=o[c].to_numpy()
    tr=np.flatnonzero(o.outer_validation_fold.to_numpy()!=fold);te=np.flatnonzero(o.outer_validation_fold.to_numpy()==fold)
    assert set(r.iloc[tr].validation_site_event_key).isdisjoint(set(r.iloc[te].validation_site_event_key))
    ps=audit['process_scaler']
    scaler=h24.ProcessScaler(tuple(names),np.array(ps['centers']),np.array(ps['q10_q90_scales']))
    x=scaler.transform(r)
    # Original A training weights were inverse full-domain source counts. Preserve this for matched RF arm.
    cnt=r.source_group.value_counts();w=r.source_group.map(1/cnt).to_numpy(float);w/=np.mean(w[tr])
    return o,r,x,tr,te,w,audit,done

def encodings(r,x,fit):
    se=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32)
    se.fit(r[['source_group']].iloc[fit].astype(str));zs=se.transform(r[['source_group']].astype(str))
    cat=r[['detection_method','sizecorr_evidence_tier']].fillna('__missing__').astype(str)
    pe=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32);pe.fit(cat.iloc[fit])
    nums=[];meta=[]
    for c in ['sampling_year_effective_v1_3','qaqc_total_score','sizecorr_lower_used_um']:
        z=pd.to_numeric(r[c],errors='coerce').to_numpy(float);missing=~np.isfinite(z)
        finite=z[fit][np.isfinite(z[fit])];med=float(np.median(finite)) if len(finite) else 0
        q=np.quantile(finite,[.1,.9]) if len(finite) else [0,1];scale=max(float(q[1]-q[0]),1e-8)
        nums.extend([np.clip((np.where(missing,med,z)-med)/scale,-4,4),missing.astype(float)])
        meta.append({'name':c,'median':med,'q10_q90_scale':scale})
    zp=sparse.hstack([sparse.csr_matrix(np.column_stack(nums)),pe.transform(cat)],format='csr')
    xe=sparse.csr_matrix(x)
    return {'environment':xe,'source_only':sparse.hstack([xe,zs],format='csr'),'context':sparse.hstack([xe,zs,zp],format='csr')},zp,{'source_categories':se.categories_[0].tolist(),'protocol_categories':[v.tolist() for v in pe.categories_],'protocol_numeric':meta,'imputation_scaling_fit_records':len(fit),'protocol_transform':'median/q10-q90, missing flag, clip +/-4; train-only categorical encoding'}

def inner_x(r,names,fit):
    centers=[];scales=[]
    for name in names:
        z=h24.numeric(r[name]);z=np.sign(z)*np.log1p(np.abs(z)) if name in h24.caoa.LOG_FEATURES or name=='plastic_waste_generation_t' else z
        finite=z[fit][np.isfinite(z[fit])];centers.append(float(np.median(finite)) if len(finite) else 0)
        scales.append(max(float(np.diff(np.quantile(finite,[.1,.9]))[0]),1e-8) if len(finite) else 1)
    scaler=h24.ProcessScaler(tuple(names),np.array(centers),np.array(scales))
    return scaler.transform(r),{'features':names,'centers':centers,'q10_q90_scales':scales,'fit_records':len(fit)}

def lme_fit(X,y,source):
    """Exact profiled random-intercept REML, with redundant fixed columns removed by SVD."""
    X=np.asarray(X,dtype=np.float64);y=np.asarray(y,float)
    U,s,Vt=np.linalg.svd(X,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-10));V=Vt[:rank].T;B=X@V
    groups,gi=np.unique(np.asarray(source).astype(str),return_inverse=True);n=np.bincount(gi).astype(float)
    sx=np.zeros((len(groups),rank));np.add.at(sx,gi,B);sy=np.bincount(gi,weights=y)
    xx=B.T@B;xy=B.T@y;yy=y@y;df=len(y)-rank
    def solve(lam,ret=False):
        c=lam/(1+lam*n);Sx=xx-(sx.T*c)@sx;t=xy-sx.T@(c*sy)
        sign,logdet=np.linalg.slogdet(Sx)
        if sign<=0:return np.inf
        b=np.linalg.solve(Sx,t);rss=max(float(yy-np.dot(c,sy*sy)-b@t),1e-12)
        sig=rss/df;obj=df*np.log(sig)+np.log1p(lam*n).sum()+logdet
        if ret:return b,sig,c*(sy-sx@b),float(obj)
        return float(obj)
    opt=minimize_scalar(lambda z:solve(np.exp(z)),bounds=(-12,12),method='bounded',options={'maxiter':100,'xatol':1e-6})
    lam=float(np.exp(opt.x));null=solve(0)
    if null<opt.fun:lam=0
    b,sig,ri,obj=solve(lam,True);beta=V@b
    return beta,dict(zip(groups,ri)),{'implementation':'exact Woodbury profiled REML','converged':bool(opt.success),'optimizer_message':str(opt.message),'optimizer_evaluations':int(opt.nfev),'variance_ratio':lam,'residual_variance':sig,'source_intercept_variance':lam*sig,'fixed_rank':rank,'fixed_columns':X.shape[1],'training_records':len(y),'training_sources':len(groups),'restricted_objective':obj,'null_boundary_selected':lam==0,'weighted_likelihood':False}

def backbone(d,audit,done,scale):
    names=audit['process_scaler']['features'];branches={k:[names.index(v) for v in vals if v in names] for k,vals in audit['branches'].items()}
    if d=='freshwater':b=fresh.modular.ModularBNN(branches,scale,old.CONFIG[d]['cap'])
    elif d=='transition':b=fresh.DirectionalFreshwaterBNN(branches,done['full_fit']['direction_constraints'],scale,old.CONFIG[d]['cap'])
    else:b=strict.SharedDirectionalMarineBNN(branches,done['full_fit']['physics_direction_constraints'],scale,old.CONFIG[d]['cap'])
    assert sum(p.numel() for p in b.parameters())==done['full_fit']['parameters']
    return b

class FixedSource(torch.nn.Module):
    def __init__(self,bone,nsource,nprotocol):
        super().__init__();self.bone=bone;self.source=torch.nn.Parameter(torch.zeros(nsource));self.protocol=torch.nn.Parameter(torch.zeros(nprotocol))
    def forward(self,x,si,p,sample=False):
        env=self.bone(x,sample=sample);source=torch.where(si>=0,self.source[si.clamp(min=0)]-self.source.mean(),torch.zeros_like(env))
        return env+source+(p@self.protocol if self.protocol.numel() else 0)

def nn_fit(r,x,fit,validation,d,audit,done,protocol,seed,max_epochs,select):
    cfg=CONFIG['models']['NN_fixed_source'];torch.manual_seed(seed);np.random.seed(seed)
    src=r.source_group.astype(str).to_numpy();cats=np.unique(src[fit]);mp=dict(zip(cats,range(len(cats))));si=np.array([mp.get(v,-1) for v in src])
    mats,p,encoding=encodings(r,x,fit);p=p.toarray().astype(np.float32) if protocol else np.zeros((len(r),0),np.float32)
    # Standardization uses inner-fit source medians, then full outer-training source medians on refit.
    y=r.observed_log10.to_numpy(float);sy=r.iloc[fit].groupby('source_group').observed_log10.median().to_numpy()
    center=float(np.median(sy));scale=max(float(np.std(sy)),.25)
    model=FixedSource(backbone(d,audit,done,scale),len(cats),p.shape[1])
    params=[{'params':model.bone.parameters(),'lr':cfg['backbone_lr'],'weight_decay':cfg['backbone_weight_decay']},{'params':[model.source,model.protocol],'lr':cfg['source_head_lr'],'weight_decay':0}]
    opt=torch.optim.Adam(params)
    xt=torch.as_tensor(x,dtype=torch.float32);yt=torch.as_tensor((y-center)/scale,dtype=torch.float32);st=torch.as_tensor(si);pt=torch.as_tensor(p)
    counts=pd.Series(src[fit]).value_counts();weight=np.array([1/counts.get(s,1) for s in src]);wt=torch.as_tensor(weight,dtype=torch.float32)
    initial={k:v.detach().clone() for k,v in model.bone.state_dict().items()};best=np.inf;bestepoch=1;stale=0;trace=[]
    for ep in range(1,max_epochs+1):
        model.train();perm=np.random.permutation(fit)
        for start in range(0,len(perm),4096):
            ix=perm[start:start+4096];opt.zero_grad();pred=model(xt[ix],st[ix],pt[ix],sample=True)
            point=torch.nn.functional.smooth_l1_loss(pred,yt[ix],reduction='none',beta=cfg['Huber_beta'])
            loss=(point*wt[ix]).sum()/wt[ix].sum()+cfg['KL_coefficient']*model.bone.kl()/len(fit)
            parts=model.bone.components(xt[ix],sample=True)
            loss=loss+cfg['branch_shrinkage']*sum(v.square().mean() for v in parts.values())
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if select:
            model.eval()
            with torch.no_grad():e=(model(xt[validation],st[validation],pt[validation],sample=False)-yt[validation]).numpy()
            vw=np.array([1/c for c in pd.Series(src[validation]).map(pd.Series(src[validation]).value_counts())])
            rmse=float(np.sqrt(np.average(e*e,weights=vw)));trace.append({'epoch':ep,'inner_RMSE_standardized':rmse})
            if rmse<best-1e-6:best=rmse;bestepoch=ep;stale=0
            else:stale+=1
            if stale>=cfg['patience']:break
        else:bestepoch=ep
    change=max(float((v-initial[k]).abs().max()) for k,v in model.bone.state_dict().items() if v.dtype.is_floating_point)
    info={'source_coefficients_jointly_optimized':True,'source_coefficients_initialization':'zero','source_fixed_effect_prior':None,'source_coefficient_sd_standardized':float(model.source.detach().std(unbiased=False)),'source_max_abs_standardized':float(model.source.detach().abs().max()),'backbone_max_parameter_change':change,'backbone_class':type(model.bone).__name__,'backbone_parameter_count':done['full_fit']['parameters'],'source_parameters':len(cats),'protocol_parameters':p.shape[1],'epochs_executed':ep,'selected_epoch':bestepoch,'inner_best_RMSE_standardized':best if select else None,'training_records':len(fit),'target_center_log10':center,'target_scale_log10':scale,'source_categories':cats.tolist(),'encoding':encoding,'trace':trace,'fit_seed':seed,'gradient_norm_clip':5,'source_identifiability':'subtract equal-source mean intercept in forward; unseen sources zero'}
    return model,(xt,st,pt),center,scale,info

def nn_predict(model,tensors,rows,center,scale):
    x,si,p=tensors;model.eval()
    with torch.no_grad():
        draws=torch.stack([model.bone(x[rows],sample=True) for _ in range(24)])
        env=torch.quantile(draws,.5,dim=0)
        ri=torch.where(si[rows]>=0,model.source[si[rows].clamp(min=0)]-model.source.mean(),torch.zeros_like(env))
        proto=p[rows]@model.protocol if model.protocol.numel() else 0
        return center+scale*env.numpy(),center+scale*(env+ri+proto).numpy()
