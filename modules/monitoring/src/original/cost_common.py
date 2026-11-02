import os, sys, json, hashlib, time
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='2'
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'audit/mpl_cache'))
PROJECT=ROOT.parents[2]
PLAN=PROJECT/'analysis'
V1=PLAN/'监测实施与成本_20261009_V1'
V2=PLAN/'监测实施与成本_20261010_V2'
RULES=PLAN/'监测实施与成本_20261010_V3/执行规则.txt'
OLD=json.loads((V1/'CONFIG.json').read_text(encoding='utf-8-sig'))
V2CFG=json.loads((V2/'CONFIG.json').read_text(encoding='utf-8-sig'))
DOM=['reference_surface_freshwater','transition_coastal']
CATS=['equipment_materials','sampling_labor','analysis_QA','training_organization','transport']
CFG=dict(seed=26101071,paired_layout_seed=OLD['seed'],replicates=500,
    occasions=[0,1,2,3,4,5],size_window_um='10-5000',currency='EUR_ex_VAT',
    capacity_branches=['available_unrecorded','support_needed'],
    access_branches=['shore_if_feasible','boat_if_required'],
    levels=['low','mid','high'],field_hours=4.,field_people=2,workday_hours=8.,
    team_tile_degrees=10,team_station_capacity=50,course_team_capacity=20,
    lab_capacity_analyses_per_year=250,blank_fraction=.15,duplicate_fraction=.15,
    field_days_land=2,field_days_boat=5,vehicle_station_cap=4,boat_station_cap=8,
    rental_billing='transit plus onsite occupation; driver/skipper and fuel included',
    within_cell_interstation_km=5.,land_speed_kmh=40.,boat_speed_kmh=20.,
    remote_port_distance_km=200.,shore_verified=False,
    default_provider_distance_km=500.,provider_size_gap_module_eur_mid=2500.,
    field_team_course_eur_mid=4000.,organization_eur_mid=8000.,
    annual_coordination_per_capacity_slot_eur_mid=500.,
    cash_labor_available_fraction=0.,cash_labor_support_fraction=1.,
    price_label='2026 EUR planning scenarios; 2024 price indices; no target-window quote',
    independent_station_locations_verified=False)
def save(q,name):
    q.to_csv(ROOT/name,index=False,encoding='utf-8-sig',compression='infer')
def js(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def status(phase,**kw):
    path=ROOT/'STATUS.json'
    try:previous=json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    except json.JSONDecodeError:previous={}
    previous.update(status='running',phase=phase,updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
    js('STATUS.json',previous)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)
def haversine(lon1,lat1,lon2,lat2):
    a,b,c,d=np.deg2rad([lon1,lat1,lon2,lat2]);v=np.sin((d-b)/2)**2+np.cos(b)*np.cos(d)*np.sin((c-a)/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(v,0,1)))
def legacy():
    # Execute definitions only: neither legacy common.py nor run() is executed.
    s=(V2/'src/local_costs.py').read_text(encoding='utf-8-sig').replace('from common import *','')
    ns=dict(np=np,pd=pd,CFG=V2CFG,OLD_CFG=OLD,DOM=DOM,__name__='legacy_read_only')
    exec(compile(s,str(V2/'src/local_costs.py'),'exec'),ns)
    return ns
