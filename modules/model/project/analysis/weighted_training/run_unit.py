from common import *
import argparse, ast, inspect, importlib, time
import pandas as pd, numpy as np, torch

p=argparse.ArgumentParser()
p.add_argument('--domain',choices=DOMAINS,required=True)
p.add_argument('--fold',type=int,required=True)
p.add_argument('--mode',choices=['baseline','uniform','qaqc','missing_low','missing_high'],required=True)
p.add_argument('--epochs',type=int,default=180)
p.add_argument('--smoke',action='store_true')
p.add_argument('--oof-only',action='store_true')
a=p.parse_args()
torch.set_num_threads(2);torch.set_num_interop_threads(2)
unit=Path(os.environ['NMP_MODEL_OUTPUT'])/('smoke' if a.smoke else 'units')/a.mode/a.domain/f'fold_{a.fold}'
unit.mkdir(parents=True,exist_ok=True)
run_signature={'contract_sha256':sha(OUT/'CONTRACT.json'),'common_sha256':sha(OUT/'common.py'),
 'runner_sha256':sha(__file__),'epochs':a.epochs,'grid_exported':not a.oof_only}
if (unit/'DONE.json').exists():
    prior=json.loads((unit/'DONE.json').read_text(encoding='utf-8'))
    assert prior.get('run_signature')==run_signature,'Existing result has a different run signature; use a new output directory'
    print('ALREADY_COMPLETE '+str(unit),flush=True);sys.exit(0)
start=time.time(); cfg=CONFIG[a.domain];domain=DOMAINS[a.domain]
if memory()['available_bytes']<2*2**30: raise RuntimeError('Less than 2 GiB available RAM')
import run_h26_marine_vector_modular_bnn as strict
import run_h26_freshwater_network_modular_bnn as fresh
captured={};fit_ledgers=[]

def data_weights(base,fit_rows):
    ledger=captured['ledger']
    assert len(base)==len(ledger)
    mult=quality_multiplier(ledger,base,fit_rows,a.mode)
    use=base if a.mode in ('baseline','uniform') else np.asarray(base)*mult
    assert np.array_equal(base,captured['base_weights'])
    fit_rows=np.asarray(fit_rows,int)
    fit_ledgers.append({'fit_call':len(fit_ledgers)+1,'train_n':len(fit_rows),
        'quality_factor_min':float(mult[fit_rows].min()),'quality_factor_max':float(mult[fit_rows].max()),
        'unscored_train_n':int(ledger.iloc[fit_rows].raw_quality_factor.isna().sum()),
        'base_weight_sum':float(np.sum(base[fit_rows])), 'adjusted_weight_sum':float(np.sum(use[fit_rows]))})
    if len(fit_ledgers)==1:
        out=ledger.copy();out['base_source_weight']=base;out['quality_multiplier']=mult;out['fit_weight']=use
        out['outer_train']=np.isin(np.arange(len(ledger)),captured['outer_train'])
        out['main_fit']=np.isin(np.arange(len(ledger)),fit_rows)
        out.to_csv(unit/'RECORD_WEIGHTS.csv.gz',index=False)
    return (use/np.mean(use)).astype(np.float32)

def patch_fit(module,name,target):
    original=getattr(module,name)
    tree=ast.parse(inspect.getsource(original))
    changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in node.targets):
            node.value=ast.parse('__qaqc_data_weights(weights, train_rows)',mode='eval').body;changed+=1
    assert changed==1,(name,changed)
    ast.fix_missing_locations(tree)
    namespace=original.__globals__.copy();namespace['__qaqc_data_weights']=data_weights
    exec(compile(tree,str(unit/(name+'_adapter.py')),'exec'),namespace)
    function=namespace[name]
    (unit/(name+'_adapter.py')).write_text(ast.unparse(tree)+'\n',encoding='utf-8')
    def wrapped(*values,**kwargs):
        result=function(*values,**kwargs)
        # Source-state and scaler are also saved by the original runner in map mode.
        if len(fit_ledgers)==1:
            torch.save(result[0].state_dict(),unit/'MEAN_STATE.pt')
        return result
    setattr(module,name,wrapped)

patch_fit(strict,'fit_model','fit_weight')
patch_fit(fresh,'fit_candidate_model','fitted_weight')
mod=importlib.import_module(cfg['module'])
if a.domain=='transition': mod.configure_shared_runner();runner=fresh
else: runner=mod
frozen=pd.read_csv(OUT/'FROZEN_GROUPED10_FOLDS.csv.gz',low_memory=False)
local=frozen.loc[frozen.waterbody.eq(domain)].set_index('stable_record_id')
original_load=runner.load_inputs
def load(*args,**kwargs):
    records,grid,audit=original_load(*args,**kwargs)
    assert records.stable_record_id.is_unique
    assert set(records.stable_record_id)==set(local.index),'Changed training cohort'
    captured['ledger']=record_scores(records)
    captured['records']=records
    captured['base_weights']=strict.h24.source_equal_weights(records.source_group)
    captured['grid_count']=len(grid)
    captured['record_order_sha256']=hashlib.sha256('\n'.join(records.stable_record_id.astype(str)).encode()).hexdigest()
    return records,grid,audit
runner.load_inputs=load
def split(records,args):
    fold=records.stable_record_id.map(local.formal_fold).to_numpy(int)
    train=np.flatnonzero(fold!=a.fold);test=np.flatnonzero(fold==a.fold)
    event,_=strict.build_site_event_key(records)
    assert set(event.iloc[train]).isdisjoint(set(event.iloc[test]))
    assert (event.to_numpy()==records.stable_record_id.map(local.validation_site_event_key).to_numpy()).all()
    captured['outer_train']=train;captured['test']=test
    return train,test,{'scheme':'random_site_event_10fold','kind':'persisted_frozen_grouped_labels','folds_csv_used':True,'seed':260826,'requested_fold':a.fold,'training_records':len(train),'heldout_records':len(test),'source_overlap':len(set(records.iloc[train].source_group)&set(records.iloc[test].source_group)),'train_test_site_event_overlap':0,'response_stratification_used':False}
strict.choose_outer_split=split
args=mod.parser().parse_args([
 '--candidate',cfg['candidate'],'--training',str(cfg['training']),'--prediction',str(cfg['prediction']),
 '--observation',str(OBS),'--caoa',str(CAOA),'--folds-csv',str(OUT/'FROZEN_GROUPED10_FOLDS.csv.gz'),
 '--fold',str(a.fold),'--seed','260826','--validation-scheme','random_site_event_10fold',
 '--observation-scale-model','fixed_reported_propagation','--device','cpu','--epochs',str(a.epochs),
 '--posterior-draws','24','--calibration-crossfits','3','--map-cap-log10',str(cfg['cap']),
 '--run-root',str(unit)]+(['--oof-only'] if a.oof_only else []))
runner.train(args)
if a.domain=='transition': mod.rewrite_audits(args)
output=unit/'S0/protocol_aware'/domain
heldout=pd.read_csv(output/'H26_UNIT_HELDOUT.csv.gz',low_memory=False)
assert set(heldout.stable_record_id)==set(captured['records'].iloc[captured['test']].stable_record_id)
assert heldout.predicted_q50_log10.notna().all()
assert len(fit_ledgers)==4,fit_ledgers
if not a.oof_only:
    grid=pd.read_csv(output/'H26_UNIT_GRID.csv.gz',low_memory=False)
    assert len(grid)==captured['grid_count']
audit={'status':'completed','mode':a.mode,'domain':a.domain,'fold':a.fold,'records':len(heldout),
 'run_signature':run_signature,
 'epochs_max':a.epochs,'posterior_draws':24,'grid_exported':not a.oof_only,
 'record_order_sha256':captured['record_order_sha256'],'fit_calls':fit_ledgers,'elapsed_seconds':time.time()-start,
 'base_weights_unchanged':True,'evaluation_weights_unchanged':True,'early_stopping_weights_unchanged':True,
 'source_offset_rule_unchanged':True,'known_measurement_sd_unchanged':True,'memory':memory()}
write_json(unit/'DONE.json',audit)
print(json.dumps({k:audit[k] for k in ['status','mode','domain','fold','records','elapsed_seconds']}),flush=True)
