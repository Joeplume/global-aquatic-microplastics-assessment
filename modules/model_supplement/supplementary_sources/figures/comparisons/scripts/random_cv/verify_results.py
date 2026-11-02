"""Independent per-record, fold, metric and convergence checks for the figure."""
from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
from sklearn.metrics import r2_score,mean_squared_error
O=Path(__file__).parent
C=json.loads((O/'CONTRACT.json').read_text(encoding='utf8'))
origin=Path(C['inputs']['H26_OOF']['path'])
assert hashlib.sha256(origin.read_bytes()).hexdigest()==C['inputs']['H26_OOF']['sha256']
ref=pd.read_csv(origin).set_index('stable_record_id')
pred=pd.read_csv(O/'EIGHT_MODELS_RANDOM_OOF.csv.gz')
metrics=pd.read_csv(O/'METRICS.csv')
assert len(pred)==24129*8 and pred.model.nunique()==8
for name,z in pred.groupby('model'):
    assert z.stable_record_id.is_unique and set(z.stable_record_id)==set(ref.index)
    z=z.set_index('stable_record_id').loc[ref.index]
    assert np.array_equal(z.outer_validation_fold,ref.outer_validation_fold)
    assert np.allclose(z.observed_log10,ref.observed_log10,rtol=0,atol=1e-12)
    assert np.isfinite(z.predicted_log10).all()
    for domain,q in [('overall',z)]+list(z.groupby('domain')):
        row=metrics[(metrics.model==name)&(metrics.domain==domain)].iloc[0]
        assert row.N==len(q)
        assert abs(row.R2-r2_score(q.observed_log10,q.predicted_log10))<1e-12
        assert abs(row.RMSE-np.sqrt(mean_squared_error(q.observed_log10,q.predicted_log10)))<1e-12
    if name=='H26':assert np.allclose(z.predicted_log10,ref.primary_predicted_q50_log10,rtol=0,atol=1e-12)
unit_reports=[];warnings=[]
for domain in ['freshwater','transition','marine']:
    for fold in range(1,11):
        for model in C['models']:
            p=O/f'units/{domain}/{fold}/{model}/DONE.json'
            d=json.loads(p.read_text(encoding='utf8'))
            assert d['model']==model and d['domain']==domain and d['fold']==fold
            assert not d['outer_test_response_used'] and d['train_test_record_overlap']==0
            if d['detail'].get('fit_status',0)!=0 or d['detail'].get('warnings'):warnings.append({'path':str(p),'details':d['detail']})
            unit_reports.append({'model':model,'domain':domain,'fold':fold,'test_N':d['test_N'],'elapsed_seconds':d['elapsed_seconds']})
assert len(unit_reports)==210
pd.DataFrame(unit_reports).to_csv(O/'UNIT_MANIFEST.csv',index=False)
report={'status':'PASS' if not warnings else 'WARN','models':8,'records_per_model':24129,
        'completed_new_training_units':210,'same_fold_ids':True,'same_observed_targets':True,
        'H26_unchanged':True,'all_metrics_independently_recomputed':True,'convergence_warnings':warnings,
        'prediction_sha256':hashlib.sha256((O/'EIGHT_MODELS_RANDOM_OOF.csv.gz').read_bytes()).hexdigest(),
        'runner_sha256':hashlib.sha256((O/'run_random.py').read_bytes()).hexdigest()}
(O/'INDEPENDENT_QA.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
print(metrics[metrics.domain.eq('overall')].to_string(index=False))
