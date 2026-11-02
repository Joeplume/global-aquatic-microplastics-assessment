"""Refresh concentration-dependent spatial orders, same candidate/geometry contract.

The 200 fixed-layout environmental-boundary replicates are calculated separately
from the 500 historical source-package simulations. Both use the adopted
spatial-support definition.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import ast,json,hashlib,heapq,gc,sys,time,argparse
from pathlib import Path
from typing import Any,Iterable
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent
PLAN=O.parents[1];ROOT=PLAN.parent
H=ROOT/'src/modeling/h26'
E=H/'validation'
U=E/'15_H26_后验浓度_CAOA联合监测优先配置_20260910_V1'
V=E/'16_联合配置转折点与效率_20260910_V1'
B=PLAN/'Figure4b_environment_boundary_20261016_V1'
N=O/'network';N.mkdir(exist_ok=True)
DOMAINS=['reference_surface_freshwater','transition_coastal'];DOM=DOMAINS
SCOPES={'terrestrial_aquatic':DOMAINS};BUDGET_NODES=[0,95,251,297,400,1000]
MAX=400;SEED=2026101601

def js(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def functions(path,names):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name in names]
    assert {x.name for x in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),globals())

functions(U/'code/run_experiment.py',['require','area_percentiles','nearest_observation_distance','build_candidate_sets','action_cover','greedy_sequence','concentration_order','target_arrays','sequence_curve'])
functions(B/'run.py',['metric','calculate'])

def candidate_inputs():
    g=pd.read_csv(U/'JOINT_STATE_MAP_GRID.csv.gz',low_memory=False)
    cohort=pd.read_csv(E/'transfer_validation/00_合同/H26_FROZEN_24129_COHORT.csv.gz',usecols=['longitude','latitude','waterbody'])
    g['nearest_observation_km']=nearest_observation_distance(g,cohort)
    geometry={};g['embedding_position']=-1
    for i,d in enumerate(DOMAINS):
        z=np.load(B/f'PREPARED_DOMAIN_{i}.npz',allow_pickle=True);ids=z['prediction_ids']
        emb,counts,m=metric(z['source'],z['query'],np.arange(len(z['source'])))
        ix=g.index[g.prediction_id.isin(ids)].to_numpy()
        assert list(g.loc[ix,'prediction_id'])==list(ids)
        assert np.array_equal(np.where(counts>=2,'in',np.where(counts==1,'limited','out')),g.loc[ix,'caoa_analysis_state'])
        g.loc[ix,'embedding_position']=np.arange(len(ids))
        geometry[d]={'X':emb,'caliper':m['caliper']}
    c,allsets,limitedsets,audit=build_candidate_sets(g,geometry)
    c['scope']='terrestrial_aquatic'
    prior=pd.read_csv(V/'CANDIDATE_UNIVERSE.csv.gz')
    assert list(c.prediction_id)==list(prior.prediction_id)
    return g,c,allsets,limitedsets,audit

def curve(g,c,aa,ll,order,strategy):
    out,_=sequence_curve(g,c,aa,ll,'terrestrial_aquatic',strategy,order,1000)
    return out

def historical(g,c,aa,ll,newhigh):
    """Identical source RNG order to 500x1000 historical-package repetitions.

    Method draws never gate the original support status. Only the independent
    source-package random stream is required to reconstruct coverage exactly.
    """
    hist=json.loads((U/'HISTORICAL_GEOGRAPHIC_MONTE_CARLO_DETAIL.json').read_text(encoding='utf-8'))
    order=pd.read_csv(V/'ORDER_geographic_coverage_explicit_bridge.csv.gz')
    idx=order.candidate_index.to_numpy(int);assert len(idx)==1000
    weights=c.cell_area_km2.to_numpy();hiold=c.high_concentration_primary.to_numpy(bool)
    p=g[g.waterbody.isin(DOMAINS)&g.analysis_assessed]
    oldhigh=p.high_concentration_primary.to_numpy(bool);area=p.cell_area_km2.to_numpy()
    support=p.caoa_analysis_state.eq('in').to_numpy()
    newh=newhigh[g.index.get_indexer(p.index)]
    newhc=newhigh[c['index'].to_numpy(int)]
    oldden=area[oldhigh].sum();newden=area[newh].sum()
    basetotal=area[support].sum();baseold=area[support&oldhigh].sum();basenew=area[support&newh].sum()
    rsource=np.random.default_rng(260910)
    probability=np.array([hist[d]['p_two_or_more_independent_sources_per_observed_cell'] for d in c.loc[idx,'waterbody']])
    arrays=np.empty((500,3,1001),np.float32)
    for rep in range(500):
        first=np.full(len(c),1001,np.int16)
        two=rsource.random(1000)<probability
        for k,i in enumerate(idx,1):
            target=aa[i] if two[k-1] else ll[i]
            first[target]=np.minimum(first[target],k)
        arrays[rep,0]=(basetotal+np.cumsum(np.bincount(first,weights=weights,minlength=1002))[:1001])/area.sum()*100
        arrays[rep,1]=(baseold+np.cumsum(np.bincount(first[hiold],weights=weights[hiold],minlength=1002))[:1001])/oldden*100
        arrays[rep,2]=(basenew+np.cumsum(np.bincount(first[newhc],weights=weights[newhc],minlength=1002))[:1001])/newden*100
    orig=pd.read_csv(U/'FOUR_STRATEGY_CURVES.csv.gz')
    orig=orig[orig.scope.eq('terrestrial_aquatic')&orig.strategy.eq('geographic_coverage_historical_mc')].sort_values('budget')
    checks={}
    for channel,label in [(0,'overall_caoa_in_pct'),(1,'high_concentration_caoa_in_pct')]:
        a=arrays[:,channel].astype(float)
        for suffix,est in [('',a.mean(0)),('_low',np.quantile(a,.025,axis=0)),('_high',np.quantile(a,.975,axis=0))]:
            error=float(np.max(np.abs(est-orig[label+suffix].to_numpy())))
            assert error<2e-5,(label,suffix,error)
            checks[label+suffix]=error
    rows=[]
    for k in range(1001):
        row={'strategy':'geographic_coverage_historical_mc','scope':'terrestrial_aquatic','budget':k}
        for channel,label in [(0,'overall_caoa_in_pct'),(2,'high_concentration_caoa_in_pct')]:
            a=arrays[:,channel,k].astype(float)
            row[label]=a.mean();row[label+'_low']=np.quantile(a,.025);row[label+'_high']=np.quantile(a,.975)
        rows.append(row)
    np.savez_compressed(N/'HISTORICAL_SOURCE_PACKAGE_CURVES.npz',overall=arrays[:,0],high_old=arrays[:,1],high_new=arrays[:,2])
    return pd.DataFrame(rows),checks

def main_orders():
    g,c,aa,ll,audit=candidate_inputs()
    print('candidate neighborhoods ready',len(c),flush=True)
    weights=c.cell_area_km2.to_numpy();oldhigh=c.high_concentration_primary.to_numpy(bool)
    oldjoint,_=greedy_sequence(c,aa,ll,'terrestrial_aquatic',oldhigh,weights,'joint')
    expected=pd.read_csv(V/'ORDER_joint_concentration_then_caoa.csv.gz')
    oldjoint.to_csv(N/'RECONSTRUCTED_OLD_ORDER.csv',index=False)
    assert list(oldjoint.prediction_id.iloc[:400])==list(expected.prediction_id.iloc[:400]),'Original first400 candidate/order reconstruction mismatch'
    assert set(oldjoint.prediction_id)==set(expected.prediction_id)
    tie_mismatches=np.flatnonzero(oldjoint.prediction_id.to_numpy()!=expected.prediction_id.to_numpy()).tolist()
    oldcurve=curve(g,c,aa,ll,oldjoint,'joint_concentration_then_caoa')
    prior=pd.read_csv(U/'FOUR_STRATEGY_CURVES.csv.gz')
    prior=prior[prior.scope.eq('terrestrial_aquatic')&prior.strategy.eq('joint_concentration_then_caoa')].sort_values('budget')
    assert np.max(np.abs(oldcurve.overall_caoa_in_pct-prior.overall_caoa_in_pct.to_numpy()))<1e-8
    assert np.max(np.abs(oldcurve.high_concentration_caoa_in_pct-prior.high_concentration_caoa_in_pct.to_numpy()))<1e-8
    new=pd.read_csv(O/'GRID_UPDATED.csv.gz',usecols=['prediction_id','q50_log10']).set_index('prediction_id')
    cuts=pd.read_csv(O/'HIGH_THRESHOLDS.csv').set_index('waterbody').weighted_q75_log10
    newq=new.loc[g.prediction_id,'q50_log10'].to_numpy()
    newhigh=np.isfinite(newq)&(newq>=g.waterbody.map(cuts).to_numpy())
    history,mc_checks=historical(g,c,aa,ll,newhigh)
    print('historical repetitions exactly reproduced and updated',flush=True)
    g['q50_log10']=newq;g['posterior_median_items_m3']=10**newq;g['high_concentration_primary']=newhigh
    for d in DOMAINS:
        mask=g.waterbody.eq(d)&g.analysis_assessed
        g.loc[mask,'domain_concentration_percentile']=area_percentiles(g.loc[mask,'q50_log10'].to_numpy(),g.loc[mask,'cell_area_km2'].to_numpy())
    for col in ['q50_log10','posterior_median_items_m3','high_concentration_primary','domain_concentration_percentile']:
        c[col]=g.loc[c['index'],col].to_numpy()
    newhi=c.high_concentration_primary.to_numpy(bool)
    joint,_=greedy_sequence(c,aa,ll,'terrestrial_aquatic',newhi,weights,'joint')
    caoa,_=greedy_sequence(c,aa,ll,'terrestrial_aquatic',newhi,weights,'caoa')
    con=concentration_order(c,'terrestrial_aquatic',1000)
    curves=[history]
    for name,order in [('joint_concentration_then_caoa',joint),('caoa_single_target',caoa),('concentration_priority',con)]:
        order.to_csv(N/f'ORDER_{name}.csv.gz',index=False)
        curves.append(curve(g,c,aa,ll,order,name))
    allc=pd.concat(curves,ignore_index=True);allc.to_csv(N/'FOUR_STRATEGY_CURVES.csv.gz',index=False)
    allc[allc.budget.isin([95,297])].to_csv(N/'TABLE10_SPATIAL_CHECKPOINTS.csv',index=False)
    joint[joint.priority_order.le(95)].to_csv(N/'SELECTED_95.csv',index=False)
    joint[joint.priority_order.le(297)].to_csv(N/'SELECTED_297.csv',index=False)
    first400=joint.iloc[:400];assert len(first400)==400
    first400.to_csv(N/'FIXED_FIRST400_SITES.csv',index=False)
    p=g[g.waterbody.isin(DOMAINS)&g.analysis_assessed].copy().reset_index(drop=True)
    parts=[]
    for i,d in enumerate(DOMAINS):
        z=np.load(B/f'PREPARED_DOMAIN_{i}.npz',allow_pickle=True);seq=first400[first400.waterbody.eq(d)]
        pos={str(pid):j for j,pid in enumerate(z['prediction_ids'])}
        part={k:z[k] for k in ['source','query','global_index']}
        part.update(site_index=np.array([pos[str(pid)] for pid in seq.prediction_id]),budget=seq.priority_order.to_numpy(int),package=seq.independent_source_deployments.to_numpy(int))
        np.savez_compressed(N/f'PREPARED_DOMAIN_{i}.npz',**part,prediction_ids=z['prediction_ids'].astype(str))
        parts.append(part)
    ref=allc[allc.strategy.eq('joint_concentration_then_caoa')].sort_values('budget').iloc[:401]
    original=ref[['overall_caoa_in_pct','high_concentration_caoa_in_pct']].to_numpy().T
    point,meta,_=calculate(parts,p.cell_area_km2.to_numpy(),p.high_concentration_primary.to_numpy(bool),-1)
    e=float(np.max(np.abs(original-point)));assert e<1e-8,e
    np.savez_compressed(N/'FIXED_POPULATION.npz',area=p.cell_area_km2.to_numpy(),high=p.high_concentration_primary.to_numpy(bool),prediction_ids=p.prediction_id.to_numpy(str),original_curve=original)
    jcurve=allc[allc.strategy.eq('joint_concentration_then_caoa')]
    target=lambda col,x:int(jcurve.loc[jcurve[col].ge(x-1e-8),'budget'].min())
    result={'status':'orders_complete_boundary_pending','original_first400_sequence_exactly_reproduced':True,'original_full_order_equal_gain_tie_swap_zero_based':tie_mismatches,'original_full_curve_reproduced_atol':1e-8,'original_history_mc_reproduced_maxerror':mc_checks,'candidates':len(c),'sequence_length':len(joint),'same_old_target_high_98_2008_sites':target('high_concentration_caoa_in_pct',98.2008),'same_old_target_overall_95_7846_sites':target('overall_caoa_in_pct',95.7846),'all_high_supported_first_site':target('high_concentration_caoa_in_pct',100.),'boundary_point_vs_greedy_maxerror':e,'candidate_audit':audit,'files':{'old_program':str(U/'code/run_experiment.py'),'old_boundary':str(B/'run.py'),'new_q50':str(O/'GRID_UPDATED.csv.gz'),'high_thresholds':str(O/'HIGH_THRESHOLDS.csv')}}
    js(N/'NETWORK_AUDIT.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

def boundary_worker(rep):
    p=N/'boundary_replicates'/f'rep_{rep:03d}.npz'
    if p.exists():
        z=np.load(p);return rep,z['curve'],json.loads(str(z['metadata']))
    parts=[]
    for i in range(2):
        z=np.load(N/f'PREPARED_DOMAIN_{i}.npz');parts.append({k:z[k] for k in ['source','query','global_index','site_index','budget','package']})
    z=np.load(N/'FIXED_POPULATION.npz')
    curve,meta,_=calculate(parts,z['area'],z['high'],rep)
    np.savez_compressed(p,curve=curve,metadata=json.dumps(meta))
    return rep,curve,meta

def boundary():
    (N/'boundary_replicates').mkdir(exist_ok=True)
    curves={};meta=[]
    with ProcessPoolExecutor(max_workers=1) as pool:
        for done in as_completed([pool.submit(boundary_worker,i) for i in range(200)]):
            i,c,m=done.result();curves[i]=c;meta.extend([{'replicate':i,**x} for x in m])
            if len(curves)%20==0:print('boundary',len(curves),'/200',flush=True)
    samples=np.stack([curves[i] for i in range(200)]);q=np.quantile(samples,[.025,.5,.975],axis=0)
    z=np.load(N/'FIXED_POPULATION.npz');ref=z['original_curve']
    rows=[{'metric':name,'added_sites':k,'original_curve':ref[i,k],'q025':q[0,i,k],'q50':q[1,i,k],'q975':q[2,i,k],'replicates':200} for i,name in enumerate(['overall','high']) for k in range(401)]
    table=pd.DataFrame(rows);table.to_csv(N/'BOUNDARY_INTERVALS.csv',index=False)
    table[table.added_sites.isin([0,95,297,400])].to_csv(N/'TABLE30_CHECKPOINTS.csv',index=False)
    pd.DataFrame(meta).to_csv(N/'BOOTSTRAP_GEOMETRY.csv',index=False)
    np.savez_compressed(N/'BOOTSTRAP_CURVES.npz',curve=samples,quantiles=q)
    audit=json.loads((N/'NETWORK_AUDIT.json').read_text(encoding='utf-8'));audit['status']='complete';audit['boundary_replicates']=200
    audit['contract']='New QAQC-derived high mask and joint sequence; fixed during boundary resampling. Original support features, catalogue, scaling/PCA/q95-caliper bootstrap and union rules unchanged.'
    js(N/'NETWORK_AUDIT.json',audit)
    print(table[table.added_sites.isin([95,297])].to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['orders','boundary','all'],default='all');a=parser.parse_args()
    if a.stage in ['orders','all']:main_orders()
    if a.stage in ['boundary','all']:boundary()
