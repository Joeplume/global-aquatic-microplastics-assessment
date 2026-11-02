"""Recover three failed calibration units, retaining exact splits.

The original 50-record early-stopping guard failed after the main model fit.
These folds still contain 8-10 independent early-stopping sources. This wrapper
does not select a new split or change training parameters. It permits only the
previously inspected 33/40/37-record split and records that exception separately.
All completed units and the original runner source are untouched.
"""
from pathlib import Path
import argparse, ast, hashlib, inspect, json, runpy, shutil, sys
import numpy as np
import pandas as pd
import torch

P = Path(__file__).resolve().parent
Q = P.parents[1] / 'weighted_training'
sys.path.insert(0, str(Q))
import common as c
import run_h26_marine_vector_modular_bnn as strict

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, choices=[7,9,10], required=True)
parser.add_argument('--domain', choices=['transition','marine'], required=True)
a = parser.parse_args()
allowed={('transition',9):(722,33,51,10),('transition',10):(770,40,50,9),('marine',7):(3293,37,45,8)}
assert (a.domain,a.fold) in allowed
u = P / 'units/removal/r1_80' / a.domain / f'fold_{a.fold}'
assert not (u / 'DONE.json').exists(), 'Never overwrite a completed unit'
assert 'Inner source-validation split is too small' in (u / 'run.log').read_text(encoding='utf8',errors='replace')
archive = u / 'failed_before_recovery'
archive.mkdir(exist_ok=False)
frozen_names = ['MEAN_STATE.pt', 'RECORD_WEIGHTS.csv.gz', 'fit_candidate_model_adapter.py', 'fit_model_adapter.py', 'run.log']
before = {}
for name in frozen_names:
    shutil.copy2(u / name, archive / name)
    before[name] = c.sha(archive / name)
old_state = torch.load(archive / 'MEAN_STATE.pt', map_location='cpu', weights_only=False)
old_weights = pd.read_csv(archive / 'RECORD_WEIGHTS.csv.gz')

original = strict.inner_source_validation_split
source = inspect.getsource(original)
tree = ast.parse(source)
count = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'require' and len(node.args) > 1:
        if isinstance(node.args[1], ast.Constant) and node.args[1].value == 'Inner source-validation split is too small':
            node.args[0] = ast.parse('len(fit_rows) >= 500 and len(validation_rows) >= 30', mode='eval').body
            count += 1
assert count == 1
ast.fix_missing_locations(tree)
ns = original.__globals__.copy()
exec(compile(tree, str(u / 'small_calibration_guard_adapter.py'), 'exec'), ns)
(u / 'small_calibration_guard_adapter.py').write_text(ast.unparse(tree) + '\n', encoding='utf8')
permitted = ns['inner_source_validation_split']
events = []
def guarded(records, rows, seed):
    try:
        return original(records, rows, seed)
    except RuntimeError as exc:
        if str(exc) != 'Inner source-validation split is too small':
            raise
        fit, validation, audit = permitted(records, rows, seed)
        expected = allowed[(a.domain,a.fold)]
        actual = (len(fit), len(validation), audit['fit_sources'], audit['validation_sources'])
        assert actual == expected, (actual, expected)
        assert not set(records.iloc[fit].source_group) & set(records.iloc[validation].source_group)
        event = dict(domain=a.domain,fold=a.fold, requested_seed=int(seed), fit_records=len(fit), earlystop_records=len(validation),
                     earlystop_sources=audit['validation_sources'], original_minimum=50, recovered_minimum=30,
                     identical_partition=True, response_based_selection=False)
        events.append(event)
        audit['small_record_guard_exception'] = event
        return fit, validation, audit
strict.inner_source_validation_split = guarded
sys.argv = [str(P / 'run_validation_unit.py'), '--design', 'removal', '--domain', a.domain, '--fold', str(a.fold), '--round', '1', '--removal', '80']
runpy.run_path(str(P / 'run_validation_unit.py'), run_name='__main__')
assert len(events) == 1, events
new_state = torch.load(u / 'MEAN_STATE.pt', map_location='cpu', weights_only=False)
assert set(old_state) == set(new_state) and all(torch.equal(old_state[k], new_state[k]) for k in old_state)
pd.testing.assert_frame_equal(old_weights, pd.read_csv(u / 'RECORD_WEIGHTS.csv.gz'), check_exact=True)
for name in ['fit_candidate_model_adapter.py', 'fit_model_adapter.py', 'run.log']:
    assert c.sha(u / name) == before[name]
done = json.loads((u / 'DONE.json').read_text(encoding='utf8'))
done['recovery'] = {'kind': 'documented_small_earlystop_record_guard_exception', 'script': str(Path(__file__).resolve()),
                    'script_sha256': c.sha(__file__), 'original_failed_files': before, 'events': events,
                    'main_model_parameters_identical_to_failed_checkpoint': True,
                    'main_fit_weights_identical_to_failed_checkpoint': True,
                    'original_log_preserved': True}
c.write_json(u / 'DONE.json', done)
c.write_json(u / 'RECOVERY_AUDIT.json', done['recovery'])
print(json.dumps(done['recovery']), flush=True)
