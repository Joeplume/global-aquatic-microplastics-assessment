import os,sys,json,hashlib,ctypes,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
H=ROOT/'h26'
SUP=H/'validation'
STRICT=SUP/'transfer_validation'
CODE=H/'code'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
 os.environ[k]='2'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']=str(OUT/'mpl_cache')
sys.dont_write_bytecode=True
for p in [CODE/'mem_uncertainty',CODE/'validation',CODE/'model/freshwater',CODE/'model/marine',CODE/'model/transition_coastal']:
 sys.path.insert(0,str(p))
DOMAINS={'freshwater':'reference_surface_freshwater','transition':'transition_coastal','marine':'marine'}
OBS=H/'data/observations/h23_observation_layer_training_records.csv.gz'
GROUPED=H/'evaluation/grouped/H26_FINAL10_OOF.csv.gz'
RANDOM=H/'evaluation/benchmark/random/H26_RANDOM_RECORD10_PRIMARY_OOF.csv.gz'
CAOA=H/'data/04_folds_aoa_caoa/H26_CAOA_V0_3_SOURCE_BRIDGE_GRID.csv.gz'
CONFIG={
 'freshwater':{'candidate':'network_normalized_fixed_slope_missing','training':H/'data/05_process_covariates/H26_RIVER_NETWORK_PROCESS_TRAIN.csv.gz','prediction':H/'data/05_process_covariates/H26_RIVER_NETWORK_PROCESS_PREDICTION.csv.gz','runner':CODE/'model/freshwater/run_h26_freshwater_network_modular_bnn.py','cap':4.0},
 'transition':{'candidate':'outlet_plume_signed_delivery_process','training':STRICT/'features/H26_TRANSITION_VECTOR_PLUME_PRESSURE_DELIVERY_TRAIN.csv.gz','prediction':STRICT/'features/H26_TRANSITION_VECTOR_PLUME_PRESSURE_DELIVERY_PREDICTION.csv.gz','runner':CODE/'model/transition_coastal/run_h26_transition_outlet_plume_bnn.py','cap':4.0},
 'marine':{'candidate':'transport_path_potential_shared_directional','training':STRICT/'features/marine_path_v4/H26_OCEAN_TRANSPORT_GRAPH_TRAIN.csv.gz','prediction':STRICT/'features/marine_path_v4/H26_OCEAN_TRANSPORT_GRAPH_PREDICTION.csv.gz','runner':CODE/'model/marine/run_h26_marine_vector_modular_bnn.py','cap':3.0}}
def write_json(p,obj):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 def clean(v):
  if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [clean(x) for x in v]
  if isinstance(v,float) and not math.isfinite(v):return None
  return v
 Path(p).write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def memory():
 class M(ctypes.Structure):
  _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ['total','available','totalpage','availablepage','totalvirtual','availablevirtual','extended']]
 m=M();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
 return {'physical_total_bytes':m.total,'physical_available_bytes':m.available,'load_percent':m.load}
class TrainingMutex:
 def __enter__(self):
  k=ctypes.WinDLL('kernel32',use_last_error=True)
  k.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_bool,ctypes.c_wchar_p];k.CreateMutexW.restype=ctypes.c_void_p
  k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];k.WaitForSingleObject.restype=ctypes.c_ulong
  k.ReleaseMutex.argtypes=[ctypes.c_void_p];k.CloseHandle.argtypes=[ctypes.c_void_p]
  self.k=k;self.handle=k.CreateMutexW(None,False,'Local\\H26Validation20261008Training')
  if not self.handle:raise ctypes.WinError(ctypes.get_last_error())
  while True:
   result=k.WaitForSingleObject(self.handle,1000)
   if result in (0,0x80):break
   if result!=0x102:raise RuntimeError('mutex wait failed '+str(result))
  if memory()['physical_available_bytes']<2*2**30:raise RuntimeError('available RAM <2GiB')
  return self
 def __exit__(self,*args):
  self.k.ReleaseMutex(self.handle);self.k.CloseHandle(self.handle)
