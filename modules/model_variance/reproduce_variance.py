"""Refit SI3 crossed variance components from the actual matched cohort and QAQC OOF."""
from pathlib import Path
import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='2'
import ast,argparse,json,warnings
import numpy as np,pandas as pd,statsmodels.api as sm
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=HERE/'results');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
core=pd.read_csv(HERE/'data/MATCHED_RECORD_LEDGER.csv')
oof=pd.read_csv(HERE/'data/OOF_qaqc.csv.gz',low_memory=False).set_index('stable_record_id').loc[core.stable_record_id]
assert len(core)==975 and oof.index.is_unique
core['residual_log10']=oof.observed_log10.to_numpy()-oof.predicted_q50_log10.to_numpy();core['all_group']='all'
reference=pd.read_csv(HERE/'data/REFERENCE_RESIDUALS.csv')
assert core.stable_record_id.equals(reference.stable_record_id)
assert np.allclose(core.residual_log10,reference.residual_log10,atol=1e-12,rtol=0)
core.to_csv(a.output/'MATCHED_QAQC_RESIDUALS.csv',index=False)
tree=ast.parse((HERE/'original_decomposition.py').read_text(encoding='utf8'));fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='fit_variance_components')
ns={'np':np,'pd':pd,'sm':sm,'warnings':warnings};exec(compile(ast.Module(body=[fn],type_ignores=[]),'original_decomposition.py','exec'),ns)
result=ns['fit_variance_components'](core)
(a.output/'VARIANCE_COMPONENTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
old=json.loads((HERE/'data/REFERENCE_VARIANCE.json').read_text(encoding='utf8'));delta={k:result['shares'][k]-old['shares'][k] for k in old['shares']}
table=json.loads((HERE/'data/si_v22_7_tables.json').read_text(encoding='utf8'))[2]
audit={'converged':result['converged'],'n':result['n_rows'],'sources':result['n_sources'],'matched_strata':result['n_geotime_strata'],'maximum_residual_difference':float(abs(core.residual_log10-reference.residual_log10).max()),'share_differences':delta,'matches_reference_within_1e-7':all(abs(v)<1e-7 for v in delta.values()),'SI3_reference_table':table,'known_measurement_is_capped':result['known_measurement_capped_by_residual_component'],'warnings':result['warnings']}
(a.output/'VALIDATION.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'converged':result['converged'],'source_share':result['shares']['source'],'method_share':result['shares']['method'],'reference_match':audit['matches_reference_within_1e-7']}))
