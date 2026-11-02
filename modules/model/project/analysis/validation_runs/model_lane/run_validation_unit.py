"""QAQC main-model validation extensions. Original runners stay read-only."""
from pathlib import Path
import os, sys, json, ast, inspect, importlib, hashlib, argparse, time

OUT = Path(__file__).resolve().parent
QAQC = OUT.parents[1] / 'weighted_training'
sys.path.insert(0, str(QAQC))
import common as c
os.environ['MPLCONFIGDIR'] = str(Path(os.environ['NMP_MODEL_OUTPUT']) / 'mpl_cache')
import numpy as np
import pandas as pd
import torch

p = argparse.ArgumentParser()
p.add_argument('--design', choices=['random10', 'spatial', 'source', 'joint', 'removal', 'grouped_comparison'], required=True)
p.add_argument('--domain', choices=c.DOMAINS, required=True)
p.add_argument('--fold', type=int, required=True)
p.add_argument('--round', type=int, default=1)
p.add_argument('--removal', type=int, default=80)
p.add_argument('--epochs', type=int, default=180)
a = p.parse_args()
torch.set_num_threads(2); torch.set_num_interop_threads(2)
tag = a.design if a.design != 'removal' else f'removal/r{a.round}_{a.removal}'
unit = Path(os.environ['NMP_MODEL_OUTPUT']) / 'units' / tag / a.domain / f'fold_{a.fold}'
unit.mkdir(parents=True, exist_ok=True)
signature = {'script': c.sha(__file__), 'qaqc_common': c.sha(QAQC/'common.py'), 'args': vars(a)}
if (unit/'DONE.json').exists():
    old = json.loads((unit/'DONE.json').read_text(encoding='utf-8'))
    assert old['signature'] == signature
    print('ALREADY_COMPLETE', unit, flush=True); sys.exit(0)
start = time.time()
cfg = c.CONFIG[a.domain]; waterbody = c.DOMAINS[a.domain]
import run_h26_marine_vector_modular_bnn as strict
import run_h26_freshwater_network_modular_bnn as fresh
captured = {}; calls = []

def adjusted(base, rows):
    mult = c.quality_multiplier(captured['ledger'], base, rows, 'qaqc')
    value = np.asarray(base) * mult
    calls.append({'train_n': len(rows), 'base_sum': float(np.sum(base[rows])), 'adjusted_sum': float(np.sum(value[rows]))})
    if len(calls) == 1:
        ledger = captured['ledger'].copy()
        ledger['base_weight'] = base; ledger['quality_multiplier'] = mult; ledger['fit_weight'] = value
        ledger['main_fit'] = np.isin(np.arange(len(base)), rows)
        ledger.to_csv(unit/'RECORD_WEIGHTS.csv.gz', index=False)
    return (value/value.mean()).astype(np.float32)

def patch(module, name, weightname):
    old = getattr(module, name)
    tree = ast.parse(inspect.getsource(old)); count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == weightname for t in node.targets):
            node.value = ast.parse('__quality_weights(weights, train_rows)', mode='eval').body; count += 1
    assert count == 1
    ns = old.__globals__.copy(); ns['__quality_weights'] = adjusted
    ast.fix_missing_locations(tree); exec(compile(tree, str(unit/(name+'_adapter.py')), 'exec'), ns)
    (unit/(name+'_adapter.py')).write_text(ast.unparse(tree)+'\n', encoding='utf-8')
    setattr(module, name, ns[name])

patch(strict, 'fit_model', 'fit_weight'); patch(fresh, 'fit_candidate_model', 'fitted_weight')
mod = importlib.import_module(cfg['module'])
if a.domain == 'transition': mod.configure_shared_runner(); runner = fresh
else: runner = mod
frozen = pd.read_csv(c.GROUPED, low_memory=False)
local = frozen.loc[frozen.waterbody.eq(waterbody)].set_index('stable_record_id')
load_original = runner.load_inputs
def load(*args, **kw):
    records, grid, audit = load_original(*args, **kw)
    assert set(records.stable_record_id) == set(local.index)
    captured['records'] = records; captured['ledger'] = c.record_scores(records)
    return records, grid, audit
runner.load_inputs = load
split_original = strict.choose_outer_split
strict_files = {'spatial': ('strict_spatial_block', 'H26_FOLDS_SPATIAL_MAIN_10DEG.csv.gz', 3),
                'source': ('strict_source_group', 'H26_FOLDS_SOURCE_MAIN.csv.gz', 0),
                'joint': ('strict_spatial_source_purged', 'H26_FOLDS_JOINT_MAIN_10DEG.csv.gz', 0)}
if a.design == 'random10':
    fold_file = c.H/'evaluation/benchmark/random/H26_RANDOM_RECORD10_PRIMARY_OOF.csv.gz'
    fold_frame = pd.read_csv(fold_file, usecols=['stable_record_id', 'outer_validation_fold']).set_index('stable_record_id')
    scheme = 'random_record_10fold'; crossfits = 3
elif a.design in strict_files:
    scheme, filename, crossfits = strict_files[a.design]
    fold_file = c.STRICT/'folds'/filename
else:
    fold_file = QAQC/'FROZEN_GROUPED10_FOLDS.csv.gz'
    fold_frame = pd.read_csv(fold_file).set_index('stable_record_id').rename(columns={'formal_fold':'outer_validation_fold'})
    scheme = 'random_site_event_10fold'; crossfits = 3

def split(records, args):
    if a.design in strict_files:
        train, test, audit = split_original(records, args)
    else:
        fold = records.stable_record_id.map(fold_frame.outer_validation_fold).to_numpy(int)
        train, test = np.flatnonzero(fold != a.fold), np.flatnonzero(fold == a.fold)
        audit = {'scheme': scheme, 'kind': 'persisted_frozen_labels', 'folds_csv_used': True,
                 'seed': 260826, 'requested_fold': a.fold, 'training_records': len(train), 'heldout_records': len(test)}
        if a.design == 'removal':
            manifest = c.ROOT/'sensitivity/paper_removal/source_manifests'/f'round_{a.round:02d}'/f'fold_{a.fold:02d}_remove_{a.removal}.csv'
            removed = set(pd.read_csv(manifest).paper_id_base)
            paper = records.source_group.str.extract(r'(NMP\d+)', expand=False)
            train = train[~paper.iloc[train].isin(removed).to_numpy()]
            captured['removed_papers'] = removed
            audit.update(removal_manifest=str(manifest), removal_manifest_sha256=c.sha(manifest), training_records=len(train))
        event, _ = strict.build_site_event_key(records)
        overlap = len(set(event.iloc[train]) & set(event.iloc[test]))
        if a.design != 'random10': assert overlap == 0
        audit.update(source_overlap=len(set(records.iloc[train].source_group)&set(records.iloc[test].source_group)), train_test_site_event_overlap=overlap)
    captured['outer_train'] = train; captured['test'] = test
    return train, test, audit
strict.choose_outer_split = split

# Capture same-weight independently fitted no-source model when requested.
fitname = 'fit_model' if a.domain == 'marine' else 'fit_candidate_model'
original_fit = getattr(runner, fitname); fitcount = [0]
def fit(*values, **kw):
    fitcount[0] += 1
    result = original_fit(*values, **kw)
    if fitcount[0] == 1:
        x, y, w, fr, vr, branches, scale = values[:7]
        captured.update(x=x, weights=w, model=result[0], fit=fr, val=vr, branches=branches, target_scale=scale)
        torch.save(result[0].state_dict(), unit/'MEAN_STATE.pt')
        if a.design == 'grouped_comparison':
            after = (torch.get_rng_state().clone(), np.random.get_state())
            net, detail = original_fit(*values, **kw)
            torch.set_rng_state(after[0]); np.random.set_state(after[1])
            q, _, sd, _ = strict.predict(net, x[captured['test']], 24)
            captured.update(nonhier_q=q, nonhier_sd=sd, nonhier_fit=detail)
            torch.save(net.state_dict(), unit/'NONHIER_STATE.pt')
            del net
            torch.set_rng_state(after[0]); np.random.set_state(after[1])
    return result
setattr(runner, fitname, fit)
original_scale = strict.observation_scale_fit
def scale_fit(records, cal, test, residual, latent, known, weights, **kw):
    captured.update(cal=cal, cal_residual=residual, cal_latent=latent, known=known)
    return original_scale(records, cal, test, residual, latent, known, weights, **kw)
strict.observation_scale_fit = scale_fit
args = mod.parser().parse_args(['--candidate',cfg['candidate'],'--training',str(cfg['training']),'--prediction',str(cfg['prediction']),
    '--observation',str(c.OBS),'--caoa',str(c.CAOA),'--folds-csv',str(fold_file),'--fold',str(a.fold),'--seed','260826',
    '--validation-scheme',scheme,'--observation-scale-model','fixed_reported_propagation','--device','cpu','--epochs',str(a.epochs),
    '--posterior-draws','24','--calibration-crossfits',str(crossfits),'--map-cap-log10',str(cfg['cap']), '--run-root',str(unit),'--oof-only'])
runner.train(args)
if a.domain == 'transition': mod.rewrite_audits(args)
folder = unit/'S0/protocol_aware'/waterbody
held = pd.read_csv(folder/'H26_UNIT_HELDOUT.csv.gz', low_memory=False)
audit = json.loads((folder/'H26_UNIT_AUDIT.json').read_text(encoding='utf-8'))
assert set(held.stable_record_id) == set(captured['records'].iloc[captured['test']].stable_record_id)
assert np.isfinite(held.predicted_q50_log10).all()
if a.design == 'grouped_comparison':
    center = audit['target_center_log10']; scale = audit['target_scale_log10']; test = captured['test']; cal = captured['cal']
    result = held[['stable_record_id','observed_log10','source_group','known_source_available']].copy()
    result['nonhier_q50_log10'] = center + scale*captured['nonhier_q']
    w = captured['weights']; residual = captured['cal_residual']; latent = captured['cal_latent']; known = captured['known']
    total = np.average(np.maximum(residual**2-latent**2,.05**2),weights=w[cal]); reported = np.average(known[cal]**2,weights=w[cal])
    remaining = max(total-reported,.05**2)
    for label, error in [('noerror',False),('error',True)]:
        sdcal = np.sqrt(latent**2+(remaining if error else total)+(known[cal]**2 if error else 0))
        multiplier = strict.weighted_quantile(abs(residual)/np.maximum(sdcal,1e-6),w[cal],.95)/1.96
        result['nonhier_sd_'+label] = multiplier*np.sqrt((scale*captured['nonhier_sd'])**2+(remaining if error else total)+(known[test]**2 if error else 0))
    result.to_csv(unit/'NONHIER_COMPARISON.csv.gz',index=False)
done = {'status':'completed','signature':signature,'design':a.design,'domain':a.domain,'fold':a.fold,'records':len(held),
        'epochs':a.epochs,'calibration_crossfits':crossfits,'fit_calls':calls,'elapsed_seconds':time.time()-start,
        'fold_file':str(fold_file),'fold_sha256':c.sha(fold_file),'prediction_target':'environment and source-conditional kept separate'}
c.write_json(unit/'DONE.json',done)
print(json.dumps({k:done[k] for k in ['status','design','domain','fold','records','elapsed_seconds']}),flush=True)
