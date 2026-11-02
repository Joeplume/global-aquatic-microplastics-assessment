"""Refit seven baselines against persisted H26 record-random outer folds.
Frozen model sources are imported read-only. All outputs stay next to this file.
"""
import os, sys, json, time, warnings, hashlib
from pathlib import Path
OUT = Path(__file__).resolve().parent
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(OUT/'mpl_cache')
sys.dont_write_bytecode = True
M = (next(p for p in Path(__file__).resolve().parents if p.name == 'modules')/'model/project/src/modeling')
A = M/'model_comparison/A_model_uncertainty'
S = A/'baseline_models'
sys.path.insert(0,str(S))
import baseline_lib as lib
sys.path.insert(0,str(S/'baseline_runs'))
import run_new_controls as simple
os.environ['MPLCONFIGDIR'] = str(OUT/'mpl_cache')
np,pd,torch=lib.np,lib.pd,lib.torch
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import r2_score,mean_squared_error
from scipy.spatial.distance import cdist
MODELS=['LinearRegression','RBF_NN','HistGradientBoosting','SVR','RF','MLP','BNN_nonhier']

def write(name,obj):
    if name=='STATUS.json':
        name=f'status/STATUS_{time.time_ns()}.json'
    for attempt in range(20):
        try:
            lib.old.write_json(OUT/name,obj)
            return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.2)
def sha(p): return lib.old.sha(p)

def run():
    start=time.time()
    oof=pd.read_csv(lib.old.RANDOM,low_memory=False)
    assert len(oof)==24129 and oof.stable_record_id.is_unique
    print('OOF columns: '+','.join(oof.columns),flush=True)
    assert 'outer_validation_fold' in oof
    contract={'outer_folds':'exact H26 persisted record-random tenfold labels; no redraw',
        'N':len(oof),'target':'original observed_log10, log10(items/m3)',
        'models':MODELS,'H26':'reused frozen primary operational posterior median',
        'baseline_inputs':'same domain-specific environmental/process features as earlier grouped controls; no source or protocol predictors',
        'boundary':'H26 uses source-conditioned prediction where available; matched random folds do not equalize information or establish causal benefit of hierarchy',
        'preprocessing':'re-estimated using fit rows only, original signed-log1p and ProcessScaler',
        'weights':'inverse OUTER-TRAIN source counts, normalized in training; no heldout target use',
        'inner_split':'fixed seed random 80/20 split of outer training only',
        'BNN':'original selected modular backbone without source-offset layer; original Adam/Huber/KL/shrinkage and 180 epochs/patience24; inner checkpoint selection; 24 posterior draws',
        'MLP':simple.CONFIG['MLP'],'RBF_NN':simple.CONFIG['RBF_NN'],
        'RF':{'trees':400,'max_features':.7,'min_samples_leaf':5},
        'HGB':{'iterations':180,'max_leaf_nodes':31,'min_samples_leaf':20,'learning_rate':.05,'l2':1,'early_stopping':False},
        'SVR':{'kernel':'rbf','C':10,'epsilon':.1,'gamma':'scale','tol':.001,'max_iter':200000},
        'inputs':{'H26_OOF':{'path':str(lib.old.RANDOM),'sha256':sha(lib.old.RANDOM)},
                  'baseline_lib':sha(S/'baseline_lib.py'),'simple_code':sha(S/'baseline_runs/run_new_controls.py')},
        'resources':'one training process; shared Windows mutex; CPU2 threads; resumable units'}
    write('CONTRACT.json',contract)
    completed=0
    with lib.old.TrainingMutex(), lib.threadpool_limits(2):
        for di,d in enumerate(lib.old.DOMAINS):
            old_o,r,_,_,_,_,audit,done=lib.load_domain(d,1)
            lookup=oof.set_index('stable_record_id').loc[r.stable_record_id]
            assert np.allclose(r.observed_log10,lookup.observed_log10,rtol=0,atol=1e-7)
            r['observed_log10']=lookup.observed_log10.to_numpy()
            folds=lookup.outer_validation_fold.to_numpy(int)
            assert set(folds)==set(range(1,11))
            y=r.observed_log10.to_numpy(float)
            names=audit['process_scaler']['features']
            branches={k:[names.index(v) for v in vs] for k,vs in audit['branches'].items()}
            for fold in range(1,11):
                tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
                assert not set(tr)&set(te)
                seed=261013+di*1000+fold
                counts=r.source_group.iloc[tr].value_counts()
                w=r.source_group.map(lambda s:1/counts.get(s,1)).to_numpy(float);w/=w[tr].mean()
                x,scaler=lib.inner_x(r,names,tr)
                fr,vr=next(ShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(tr))
                fit,val=tr[fr],tr[vr]
                xi,iscale=lib.inner_x(r,names,fit)
                for modelname in MODELS:
                    u=OUT/f'units/{d}/{fold}/{modelname}';u.mkdir(parents=True,exist_ok=True)
                    if (u/'DONE.json').exists():completed+=1;continue
                    write('STATUS.json',{'status':'running','domain':d,'fold':fold,'model':modelname,'completed':completed,'total':210,'seconds':time.time()-start})
                    ts=time.time();detail={}
                    if modelname=='LinearRegression':
                        est=LinearRegression(n_jobs=1).fit(x[tr],y[tr],sample_weight=w[tr]);pred=est.predict(x[te])
                    elif modelname=='RBF_NN':
                        km=MiniBatchKMeans(n_clusters=96,random_state=seed,n_init=3,max_iter=100,batch_size=1024).fit(x[tr],sample_weight=w[tr])
                        centers=km.cluster_centers_;dd=cdist(centers,centers);np.fill_diagonal(dd,np.inf)
                        widths=np.maximum(np.median(np.sort(dd,axis=1)[:,:5],axis=1),.1)
                        phi=np.exp(-cdist(x,centers,'sqeuclidean')/(2*widths[None,:]**2))
                        c,sc=simple.target_scale(y,w,tr)
                        est=Ridge(alpha=1,solver='cholesky').fit(phi[tr],(y[tr]-c)/sc,sample_weight=w[tr]);pred=c+sc*est.predict(phi[te])
                    elif modelname in ['RF','HistGradientBoosting','SVR']:
                        if modelname=='RF':est=RandomForestRegressor(n_estimators=400,max_features=.7,min_samples_leaf=5,n_jobs=1,random_state=seed)
                        elif modelname=='HistGradientBoosting':est=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=31,min_samples_leaf=20,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=seed)
                        else:est=SVR(kernel='rbf',C=10,epsilon=.1,gamma='scale',tol=.001,max_iter=200000,cache_size=256)
                        with warnings.catch_warnings(record=True) as ww:
                            warnings.simplefilter('always');est.fit(x[tr],y[tr],sample_weight=w[tr])
                        detail={'warnings':[str(v.message) for v in ww],'fit_status':int(getattr(est,'fit_status_',0))}
                        pred=est.predict(x[te])
                    elif modelname=='MLP':
                        sel,c,sc,selection=simple.mlp_fit(xi,y,w,fit,val,seed,200,True);del sel
                        net,c,sc,refit=simple.mlp_fit(x,y,w,tr,None,seed,selection['selected_epoch'],False)
                        net.eval()
                        with torch.no_grad():pred=c+sc*net(torch.tensor(x[te],dtype=torch.float32)).flatten().numpy()
                        torch.save({'state_dict':net.state_dict(),'scaler':scaler,'target_center':c,'target_scale':sc},u/'MODEL.pt')
                        detail={'selection':selection,'refit':refit};del net
                    else:
                        sy=r.iloc[fit].groupby('source_group').observed_log10.median().to_numpy()
                        c=float(np.median(sy));sc=max(float(np.std(sy)),.25);yy=(y-c)/sc
                        if d=='marine':
                            net,detail=lib.strict.fit_model(xi,yy,w,fit,val,branches,sc,candidate=lib.old.CONFIG[d]['candidate'],physics_signs=[],branch_signs=done['full_fit']['physics_direction_constraints'],seed=seed,epochs=180,total_cap_log10=3.,device=torch.device('cpu'))
                        else:
                            directions=done['full_fit'].get('direction_constraints',{})
                            net,detail=lib.fresh.fit_candidate_model(xi,yy,w,fit,val,branches,sc,candidate=lib.old.CONFIG[d]['candidate'],direction_orientations=directions,seed=seed,epochs=180,total_cap_log10=4.,device=torch.device('cpu'))
                        q,*_=lib.strict.predict(net,xi[te],24);pred=c+sc*q
                        torch.save({'state_dict':net.state_dict(),'scaler':iscale,'target_center':c,'target_scale':sc,'branches':audit['branches'],'details':detail},u/'MODEL.pt');del net
                    assert len(pred)==len(te) and np.isfinite(pred).all()
                    result=pd.DataFrame({'stable_record_id':r.stable_record_id.iloc[te].to_numpy(),'domain':d,'outer_validation_fold':fold,'observed_log10':y[te],'predicted_log10':pred})
                    result.to_csv(u/'OOF.csv.gz',index=False)
                    lib.old.write_json(u/'DONE.json',{'model':modelname,'domain':d,'fold':fold,'train_N':len(tr),'test_N':len(te),'seed':seed,'train_test_record_overlap':0,'outer_test_response_used':False,'elapsed_seconds':time.time()-ts,'scaler':scaler,'inner_scaler':iscale,'detail':detail})
                    completed+=1;print(json.dumps({'completed':completed,'total':210,'domain':d,'fold':fold,'model':modelname,'seconds':round(time.time()-ts,1)}),flush=True)
    aggregate(oof)
    write('STATUS.json',{'status':'completed','completed':completed,'total':210,'seconds':time.time()-start})

def aggregate(oof):
    rows=[];allpred=[]
    for m in MODELS:
        frames=[pd.read_csv(OUT/f'units/{d}/{f}/{m}/OOF.csv.gz') for d in lib.old.DOMAINS for f in range(1,11)]
        z=pd.concat(frames,ignore_index=True);assert len(z)==24129 and z.stable_record_id.is_unique
        z=z.set_index('stable_record_id').loc[oof.stable_record_id].reset_index()
        assert np.array_equal(z.outer_validation_fold,oof.outer_validation_fold)
        assert np.allclose(z.observed_log10,oof.observed_log10,atol=1e-10,rtol=0)
        z['model']=m;allpred.append(z)
    h=oof[['stable_record_id','outer_validation_fold','observed_log10']].copy()
    h['domain']=oof['domain'] if 'domain' in oof else oof.waterbody.map({v:k for k,v in lib.old.DOMAINS.items()})
    h['predicted_log10']=oof.primary_predicted_q50_log10;h['model']='H26';allpred.append(h)
    full=pd.concat(allpred,ignore_index=True);full.to_csv(OUT/'EIGHT_MODELS_RANDOM_OOF.csv.gz',index=False)
    for (m,d),z in list(full.groupby(['model','domain']))+[((m,'overall'),z) for m,z in full.groupby('model')]:
        rows.append({'model':m,'domain':d,'N':len(z),'R2':r2_score(z.observed_log10,z.predicted_log10),'RMSE':float(np.sqrt(mean_squared_error(z.observed_log10,z.predicted_log10)))})
    pd.DataFrame(rows).to_csv(OUT/'METRICS.csv',index=False)
    write('figure_data.json',{'models':[x for x in rows if x['domain']=='overall'],
          'domain_rmse':[x for x in rows if x['model'] in ['H26','BNN_nonhier'] and x['domain']!='overall']})
    write('VALIDATION.json',{'all_8_models_N':24129,'fold_labels_exact':True,'targets_exact':True,'unique_OOF_once_per_record':True,'H26_R2':next(x['R2'] for x in rows if x['model']=='H26' and x['domain']=='overall'),'source_unchanged':sha(lib.old.RANDOM)})

if __name__=='__main__':run()
