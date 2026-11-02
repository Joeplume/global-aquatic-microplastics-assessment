"""Three actual environment-only fits on the frozen H26 site-event tenfold."""
import os, sys, json, time, subprocess
from pathlib import Path
OUT = Path(__file__).resolve().parent
S = OUT.parent
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(S))
from baseline_lib import load_domain, inner_x, old, np, pd, torch, GroupShuffleSplit, threadpool_limits
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
import joblib

CONFIG = {
    'ordinary_controls': ['RF', 'HistGradientBoosting', 'SVR', 'MLP', 'RBF_NN', 'LinearRegression'],
    'primary_models': ['H26', 'BNN_nonhier'],
    'new_models': ['MLP', 'RBF_NN', 'LinearRegression'],
    'cohort_N': 24129, 'outer_folds': 'reuse original frozen site-event grouped tenfold',
    'inputs': 'environment only for all six ordinary controls and nonhierarchical BNN',
    'H26_boundary': 'source-conditioned H26 when training source available; environment fallback for unseen source',
    'target': 'frozen observed_log10; log10(items/m3); 10-5000 um',
    'weights': 'same frozen inverse full-domain source counts as existing environment RF/HGB/SVR; normalized on fit subset; no response-dependent weights',
    'MLP': {'hidden': [64, 32], 'activation': 'ReLU', 'loss': 'weighted MSE', 'optimizer': 'Adam', 'lr': .001,
            'weight_decay': .0001, 'batch': 512, 'max_epochs': 200, 'patience': 25,
            'selection': 'one inner 20% site-event grouped validation; weighted RMSE selects epoch; refit all outer training',
            'target_scaling': 'fit-subset weighted mean and standard deviation, floor 0.25'},
    'RBF_NN': {'hidden_units': 96, 'activation': 'Gaussian exp(-squared_distance/(2*width^2))',
               'centers': 'train-only weighted MiniBatchKMeans, n_init=3, max_iter=100',
               'widths': 'median distance to 5 nearest other fitted centers; minimum 0.1',
               'output': 'train-only weighted ridge with intercept, alpha=1; target standardized on fit subset',
               'training': 'standard two-stage RBF neural network; hidden centers unsupervised, output supervised'},
    'LinearRegression': {'implementation': 'sklearn weighted ordinary least squares with intercept; no source terms'},
    'resources': 'one active training process; shared existing Windows mutex; CPU two threads; no installation',
    'selection_boundary': 'all architecture and hyperparameters fixed before fits; no outer-test selection',
    'reused_input_sha256': {p.name: old.sha(p) for p in [S/'SIX_BASELINES_GROUPED10_OOF.csv.gz', S/'SIX_MODEL_POINT_METRICS.csv', S.parent/'GROUPED10_COMPARISON_OOF.csv.gz']},
}

def target_scale(y, w, fit):
    c = float(np.average(y[fit], weights=w[fit]))
    s = max(float(np.sqrt(np.average((y[fit]-c)**2, weights=w[fit]))), .25)
    return c, s

def mlp_fit(x, y, w, fit, validation, seed, epochs, select):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1],64), torch.nn.ReLU(),
                                torch.nn.Linear(64,32), torch.nn.ReLU(), torch.nn.Linear(32,1))
    c, sc = target_scale(y,w,fit)
    xx = torch.tensor(x,dtype=torch.float32)
    yy = torch.tensor((y-c)/sc,dtype=torch.float32)
    ww = torch.tensor(w/w[fit].mean(),dtype=torch.float32)
    optim = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    best, bestep, stale, trace = float('inf'),1,0,[]
    for ep in range(1,epochs+1):
        model.train()
        perm = rng.permutation(fit)
        for start in range(0,len(perm),512):
            ix = perm[start:start+512]
            optim.zero_grad()
            err = model(xx[ix]).flatten()-yy[ix]
            loss = (err.square()*ww[ix]).sum()/ww[ix].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            optim.step()
        if select:
            model.eval()
            with torch.no_grad():
                e = (model(xx[validation]).flatten()-yy[validation]).numpy()*sc
            score = float(np.sqrt(np.average(e*e,weights=w[validation])))
            trace.append({'epoch':ep,'weighted_RMSE_log10':score})
            if score < best-1e-6: best,bestep,stale = score,ep,0
            else: stale += 1
            if stale >= 25: break
        else: bestep = ep
    return model,c,sc,{'epochs_executed':ep,'selected_epoch':bestep,'trace':trace,
                       'target_center':c,'target_scale':sc,'fit_records':len(fit)}

def run_unit(domain,fold,model_name):
    unit = OUT/f'units/{model_name}/{domain}/fold_{fold}'
    unit.mkdir(parents=True,exist_ok=True)
    if (unit/'DONE.json').exists(): return
    started=time.time()
    with old.TrainingMutex(), threadpool_limits(limits=2):
        o,r,x,tr,te,w,audit,done = load_domain(domain,fold)
        y=r.observed_log10.to_numpy(float)
        seed=261009+fold+1000*list(old.DOMAINS).index(domain)
        result=o.iloc[te][['stable_record_id','domain','outer_validation_fold','observed_log10']].copy()
        if model_name=='LinearRegression':
            est=LinearRegression(n_jobs=1).fit(x[tr],y[tr],sample_weight=w[tr])
            pred=est.predict(x[te])
            joblib.dump({'estimator':est,'scaler':audit['process_scaler']},unit/'MODEL.joblib')
            detail={'rank':int(est.rank_),'features':x.shape[1],'intercept':float(est.intercept_)}
        elif model_name=='RBF_NN':
            km=MiniBatchKMeans(n_clusters=min(96,len(tr)),random_state=seed,n_init=3,
                              max_iter=100,batch_size=1024).fit(x[tr],sample_weight=w[tr])
            centers=km.cluster_centers_
            dd=cdist(centers,centers)
            np.fill_diagonal(dd,np.inf)
            widths=np.maximum(np.median(np.sort(dd,axis=1)[:,:min(5,len(centers)-1)],axis=1),.1)
            phi=np.exp(-cdist(x,centers,'sqeuclidean')/(2*widths[None,:]**2))
            c,sc=target_scale(y,w,tr)
            est=Ridge(alpha=1,solver='cholesky').fit(phi[tr],(y[tr]-c)/sc,sample_weight=w[tr])
            pred=c+sc*est.predict(phi[te])
            joblib.dump({'output':est,'centers':centers,'widths':widths,'center':c,'scale':sc,
                         'scaler':audit['process_scaler']},unit/'MODEL.joblib')
            detail={'hidden_units':len(centers),'width_min':float(widths.min()),'width_max':float(widths.max()),
                    'target_center':c,'target_scale':sc,'kmeans_n_iter':int(km.n_iter_)}
        else:
            fr,vr=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(tr,groups=r.validation_site_event_key.iloc[tr]))
            fit,val=tr[fr],tr[vr]
            assert set(r.validation_site_event_key.iloc[fit]).isdisjoint(set(r.validation_site_event_key.iloc[val]))
            xi,inner_scaler=inner_x(r,audit['process_scaler']['features'],fit)
            sel,c,sc,selection=mlp_fit(xi,y,w,fit,val,seed,200,True)
            del sel
            net,c,sc,refit=mlp_fit(x,y,w,tr,None,seed,selection['selected_epoch'],False)
            net.eval()
            with torch.no_grad(): pred=c+sc*net(torch.tensor(x[te],dtype=torch.float32)).flatten().numpy()
            torch.save({'state_dict':net.state_dict(),'architecture':[x.shape[1],64,32,1],
                        'target_center':c,'target_scale':sc,'process_scaler':audit['process_scaler']},unit/'MODEL.pt')
            detail={'inner_selection':selection,'inner_scaler':inner_scaler,'full_outer_refit':refit,'inner_event_overlap':0}
        assert len(pred)==len(te) and np.isfinite(pred).all()
        result['environment']=pred
        result.to_csv(unit/'OOF.csv.gz',index=False)
        old.write_json(unit/'DONE.json',{'status':'completed','model':model_name,'domain':domain,'fold':fold,
                       'test_records':len(te),'train_records':len(tr),'outer_event_overlap':0,
                       'outer_test_response_used':False,'inputs':'environment only','seed':seed,
                       'elapsed_seconds':time.time()-started,'process_scaler':audit['process_scaler'],'details':detail})

def run_all():
    if (OUT/'CONFIG.json').exists(): assert json.loads((OUT/'CONFIG.json').read_text(encoding='utf8'))==CONFIG
    else: old.write_json(OUT/'CONFIG.json',CONFIG)
    start=time.time();complete=0
    for domain in old.DOMAINS:
        for fold in range(1,11):
            for name in ['LinearRegression','RBF_NN','MLP']:
                old.write_json(OUT/'STATUS.json',{'status':'running','active':{'model':name,'domain':domain,'fold':fold},'completed':complete,'total':90,'elapsed_seconds':time.time()-start})
                unit=OUT/f'units/{name}/{domain}/fold_{fold}';unit.mkdir(parents=True,exist_ok=True)
                with (unit/'run.log').open('a',encoding='utf8') as log:
                    rc=subprocess.call([sys.executable,'-B',__file__,'--unit',domain,str(fold),name],stdout=log,stderr=subprocess.STDOUT,cwd=str(OUT))
                if rc:
                    old.write_json(OUT/'STATUS.json',{'status':'failed','unit':str(unit),'returncode':rc,'completed':complete})
                    raise RuntimeError('unit failed; see '+str(unit/'run.log'))
                complete+=1
                print(json.dumps({'model':name,'domain':domain,'fold':fold,'completed':complete,'total':90}),flush=True)
    old.write_json(OUT/'STATUS.json',{'status':'training_completed','completed':complete,'total':90,'elapsed_seconds':time.time()-start})

if __name__=='__main__':
    if len(sys.argv)>1: run_unit(sys.argv[2],int(sys.argv[3]),sys.argv[4])
    else: run_all()
