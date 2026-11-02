"""Isolated paired QA/QC experiment. No frozen input or model is overwritten."""
import os, sys, json, hashlib, math, ctypes
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = (next(p for p in Path(__file__).resolve().parents if p.name == 'modules')/'model/project/src/modeling')
H = ROOT / 'h26'
CODE = H / 'code'
STRICT = H / 'validation/transfer_validation'
OBS = H / 'data/observations/h23_observation_layer_training_records.csv.gz'
GROUPED = H / 'evaluation/grouped/H26_FINAL10_OOF.csv.gz'
CAOA = H / 'data/04_folds_aoa_caoa/H26_CAOA_V0_3_SOURCE_BRIDGE_GRID.csv.gz'
SCORES = ROOT.parent / 'paper_selection_prisma_audit/data/formal_library_qaqc_paper_summary_20260716.csv'
DOMAINS = {'freshwater':'reference_surface_freshwater','transition':'transition_coastal','marine':'marine'}
CONFIG = {
 'freshwater': {'candidate':'network_normalized_fixed_slope_missing', 'training':H/'data/05_process_covariates/H26_RIVER_NETWORK_PROCESS_TRAIN.csv.gz', 'prediction':H/'data/05_process_covariates/H26_RIVER_NETWORK_PROCESS_PREDICTION.csv.gz', 'module':'run_h26_freshwater_network_modular_bnn', 'cap':4},
 'transition': {'candidate':'outlet_plume_signed_delivery_process', 'training':STRICT/'features/H26_TRANSITION_VECTOR_PLUME_PRESSURE_DELIVERY_TRAIN.csv.gz', 'prediction':STRICT/'features/H26_TRANSITION_VECTOR_PLUME_PRESSURE_DELIVERY_PREDICTION.csv.gz', 'module':'run_h26_transition_outlet_plume_bnn', 'cap':4},
 'marine': {'candidate':'transport_path_potential_shared_directional', 'training':STRICT/'features/marine_path_v4/H26_OCEAN_TRANSPORT_GRAPH_TRAIN.csv.gz', 'prediction':STRICT/'features/marine_path_v4/H26_OCEAN_TRANSPORT_GRAPH_PREDICTION.csv.gz', 'module':'run_h26_marine_vector_modular_bnn', 'cap':3},
}
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['MPLCONFIGDIR'] = str(OUT/'mpl_cache')
sys.dont_write_bytecode = True
for folder in ['mem_uncertainty','validation','model/freshwater','model/marine','model/transition_coastal']:
    sys.path.insert(0, str(CODE/folder))

def write_json(path, obj):
    def clean(x):
        if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [clean(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x): return None
        return x
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, default=str, allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''): h.update(data)
    return h.hexdigest()

def memory():
    class M(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ['total','available','totalpage','availablepage','totalvirtual','availablevirtual','extended']]
    m=M();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {'available_bytes':m.available,'total_bytes':m.total,'load_percent':m.load}

def record_scores(records):
    import pandas as pd
    import numpy as np
    table=pd.read_csv(SCORES,low_memory=False).set_index('paper_id')
    assert table.index.is_unique
    paper=records.source_group.astype(str).str.extract(r'(NMP\d+)',expand=False)
    ans=records[['stable_record_id','source_group']].copy()
    ans['paper_id']=paper
    for col in ['total_numeric_score','numeric_dims','na_dims_count','qaqc_eligible','criteria_final_scores']:
        ans[col]=paper.map(table[col])
    score=pd.to_numeric(ans.total_numeric_score,errors='coerce')
    complete=ans.numeric_dims.eq(7)&ans.na_dims_count.eq(0)&score.between(0,14)
    ans['quality_status']=np.where(complete,'complete_paper_score',np.where(score.notna(),'incomplete_paper_score','unscored_external_or_unmatched'))
    ans['raw_quality_factor']=np.where(complete,(score+1)/15,np.nan)
    if 'qaqc_total_score' in records:
        old=pd.to_numeric(records.qaqc_total_score,errors='coerce')
        matched=score.notna()&old.notna()
        assert np.allclose(score[matched],old[matched]),'Frozen scores differ from published QAQC summary'
    return ans

def quality_multiplier(ledger,base,fit_rows,mode):
    """Normalize with training rows only; missing=1 in the primary analysis."""
    import numpy as np
    base=np.asarray(base,float)
    q=ledger.raw_quality_factor.to_numpy(float)
    complete=np.isfinite(q)
    fit_rows=np.asarray(fit_rows,int)
    scored_fit=fit_rows[complete[fit_rows]]
    assert len(scored_fit)>0
    mult=np.ones(len(q),float)
    if mode in ('baseline','uniform'): return mult
    if mode=='qaqc':
        mu=np.average(q[scored_fit],weights=base[scored_fit])
        mult[complete]=q[complete]/mu
    elif mode in ('missing_low','missing_high'):
        # Paper/source-level quantile, not record-count-weighted imputation.
        source_values=ledger.iloc[scored_fit].drop_duplicates('source_group').raw_quality_factor.to_numpy(float)
        fill=float(np.quantile(source_values,0.25 if mode=='missing_low' else 0.75))
        q=np.where(complete,q,fill)
        mult=q/np.average(q[fit_rows],weights=base[fit_rows])
    else: raise ValueError(mode)
    assert np.isfinite(mult).all() and (mult>0).all()
    assert np.isclose(np.sum(base[fit_rows]*mult[fit_rows]),np.sum(base[fit_rows]),rtol=1e-10)
    if mode=='qaqc': assert np.all(mult[~complete]==1)
    return mult
