"""95-site priority option and >=95% overall-support target; frozen model and rules."""
from pathlib import Path
import os,sys,json,hashlib,ast,importlib.util,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='2'
sys.dont_write_bytecode=True
import numpy as np,pandas as pd
from sklearn.neighbors import NearestNeighbors
O=Path(__file__).resolve().parent;P=O.parent.parent;OLD=O.parent/'validation_runs';N=OLD/'global_lane/network'
for d in ['results','audit','costs/inputs','costs/results','deliverables']:(O/d).mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(O/'audit/mpl_cache')
manifest={}
def read(p,**kw):
 p=Path(p);manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p,**kw)
def save(df,name):df.to_csv(O/'results'/name,index=False)
def js(name,data):(O/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def log(*a):print(time.strftime('%H:%M:%S'),*a,flush=True)
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
net=loadmod('frozen_network',OLD/'global_lane/recompute_network.py')
DOM=net.DOMAINS
def allocation(k):
 x=np.array([675.,317.])*k/992.;a=np.floor(x).astype(int)
 for j in np.argsort(-(x-a),kind='stable')[:k-int(a.sum())]:a[j]+=1
 return a

def coverage():
 curves=read(N/'FOUR_STRATEGY_CURVES.csv.gz');joint=curves[curves.strategy.eq('joint_concentration_then_caoa')].sort_values('budget')
 target=joint[joint.overall_caoa_in_pct.ge(95)].iloc[0];budget=int(target.budget)
 assert budget==305 and float(joint[joint.budget.eq(304)].overall_caoa_in_pct.iloc[0])<95
 order=read(N/'ORDER_joint_concentration_then_caoa.csv.gz')
 for k in [95,297,budget]:
  s=order.sort_values('priority_order').head(k).copy();assert len(s)==k and s.prediction_id.is_unique
  if k in [95,297]:assert s.prediction_id.tolist()==read(N/f'SELECTED_{k}.csv').prediction_id.tolist()
  save(s,f'SELECTED_{k}.csv')
 bd=read(N/'BOUNDARY_INTERVALS.csv');save(bd[bd.added_sites.isin([0,95,305,400])],'BOUNDARY_CHECKPOINTS.csv')
 geo=read(N/'GEOGRAPHIC_EXPLICIT_BRIDGE_CURVE.csv.gz')
 out=[]
 for sid,t in [('joint_concentration_then_caoa',joint),('concentration_priority',curves[curves.strategy.eq('concentration_priority')]),('geographic_coverage_explicit_bridge',geo)]:
  for k in [95,297,305]:
   r=t[t.budget.eq(k)].iloc[0]
   out.append(dict(strategy_id=sid,budget=k,overall_support_pct=r.overall_caoa_in_pct,high_support_pct=r.high_concentration_caoa_in_pct,overall_low_pct=r.overall_caoa_in_pct,overall_high_pct=r.overall_caoa_in_pct,high_low_pct=r.high_concentration_caoa_in_pct,high_high_pct=r.high_concentration_caoa_in_pct))
 save(pd.DataFrame(out),'COMPARATORS_DETERMINISTIC.csv')
 js('audit/TARGET.json',dict(target_overall_percent=95,minimum_joint_sites=budget,prior_site_coverage=float(joint[joint.budget.eq(304)].overall_caoa_in_pct.iloc[0]),overall=float(target.overall_caoa_in_pct),high=float(target.high_concentration_caoa_in_pct),all_high_first=270,layout_unchanged=True))
 log('95% target',budget,float(target.overall_caoa_in_pct))

def natural():
 g=read(net.U/'JOINT_STATE_MAP_GRID.csv.gz',low_memory=False)
 new=read(OLD/'global_lane/GRID_UPDATED.csv.gz',usecols=['prediction_id','q50_log10']).set_index('prediction_id')
 cut=read(OLD/'global_lane/HIGH_THRESHOLDS.csv').set_index('waterbody').weighted_q75_log10
 g['newhigh']=new.loc[g.prediction_id,'q50_log10'].to_numpy()>=g.waterbody.map(cut).to_numpy()
 history=json.loads((net.U/'HISTORICAL_GEOGRAPHIC_MONTE_CARLO_DETAIL.json').read_text('utf-8'))
 sites=read(net.E/'17_2030自然增长完整点位支持评估_20260910_V1/FROZEN_6030_PROJECTED_SITES_ANNOTATED.csv.gz',low_memory=False)
 pop=g[g.waterbody.isin(DOM)&g.analysis_assessed];support=pop.caoa_analysis_state.eq('in')
 area=pop.cell_area_km2.to_numpy();vec=np.column_stack([area,area*pop.newhigh.to_numpy()]);base=vec[support].sum(0);den=vec.sum(0)
 T=net.E/'11_CAOA_V02_质量基础与独立验证_20260908_V1/code/prepare_experiment.py'
 tree=ast.parse(T.read_text('utf-8-sig'));nodes=[x for x in tree.body if (isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='LOG_FEATURES' for t in x.targets)) or (isinstance(x,ast.FunctionDef) and x.name=='transform')]
 env={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(T),'exec'),env)
 checkpoints=[95,297,305];sim={k:np.tile(base,(500,1)) for k in checkpoints};exact={k:base.copy() for k in checkpoints}
 domaincurves=[];domaincounts=[]
 for di,d in enumerate(DOM):
  z=np.load(net.B/f'PREPARED_DOMAIN_{di}.npz',allow_pickle=True)
  s=sites[sites.waterbody.eq(d)].sort_values(['projection_year','original_row']).reset_index(drop=True)
  domaincounts.append(len(s));fs=list(z['features'])
  field=net.H/'data/05_process_covariates/H26_RIVER_NETWORK_PROCESS_PREDICTION.csv.gz'
  raw=read(field,usecols=['prediction_id',*fs]).set_index('prediction_id').loc[s.prediction_id]
  sq=np.column_stack([env['transform'](raw[f].to_numpy(float),f) for f in fs])
  emb,_,meta=net.metric(z['source'],np.concatenate([z['query'],sq]),np.arange(len(z['source'])))
  target=g.set_index('prediction_id').loc[z['prediction_ids']];gapmask=target.caoa_analysis_state.ne('in').to_numpy();gap=target[gapmask]
  nn=NearestNeighbors(radius=meta['caliper'],algorithm='ball_tree').fit(emb[np.flatnonzero(gapmask)])
  neighbors=nn.radius_neighbors(emb[len(z['query']):],return_distance=False)
  w=gap.cell_area_km2.to_numpy();v=np.column_stack([w,w*gap.newhigh.to_numpy()]);limited=gap.caoa_analysis_state.eq('limited').to_numpy()
  cover=np.zeros((500,len(gap)),bool);seen=np.zeros(len(gap),bool);surv=np.ones(len(gap));run=np.zeros((500,2));ex=np.zeros(2);hist=[ex.copy()]
  rng=np.random.default_rng(26091017+di);prob=history[d]['p_two_or_more_independent_sources_per_observed_cell'];mcmax=allocation(305)[di]
  for i,t in enumerate(neighbors,1):
   lt=t[limited[t]];lt=lt[~seen[lt]];seen[lt]=True;delta=v[lt].sum(0);ex+=delta
   ot=t[~limited[t]];ex+=(v[ot]*surv[ot,None]*prob).sum(0);surv[ot]*=1-prob
   if i<=mcmax:
    run+=delta;reps=np.flatnonzero(rng.random(500)<prob)
    if len(ot) and len(reps):
     ix=np.ix_(reps,ot);fresh=~cover[ix];run[reps]+=fresh@v[ot];cover[ix]=True
   for k in checkpoints:
    if i==allocation(k)[di]:sim[k]+=run;exact[k]+=ex
   hist.append(ex.copy())
  domaincurves.append(np.array(hist));log('Historical domain complete',d,len(s))
  save(s.head(mcmax).assign(comparison_budget=305),f'HISTORICAL_305_DOMAIN_{di}.csv')
 maxn=sum(domaincounts);histrows=[]
 for k in range(maxn+1):
  a=allocation(k)
  if any(a[i]>domaincounts[i] for i in range(2)):break
  cov=100*(base+domaincurves[0][a[0]]+domaincurves[1][a[1]])/den
  histrows.append(dict(budget=k,freshwater_sites=a[0],estuary_sites=a[1],overall_support_pct=cov[0],high_support_pct=cov[1]))
 hist=pd.DataFrame(histrows);save(hist,'HISTORICAL_FULL_EXPECTED_CURVE.csv')
 rows=[]
 for k in checkpoints:
  a=100*sim[k]/den;b=100*exact[k]/den
  rows.append(dict(strategy_id='natural_continuation_proportional_domains',budget=k,overall_support_pct=b[0],high_support_pct=b[1],overall_low_pct=np.quantile(a[:,0],.025),overall_high_pct=np.quantile(a[:,0],.975),high_low_pct=np.quantile(a[:,1],.025),high_high_pct=np.quantile(a[:,1],.975)))
  save(pd.DataFrame(a,columns=['overall','high']).assign(replicate=np.arange(1,501)),f'HISTORICAL_DRAWS_{k}.csv')
 prior=read(N/'FIGURE4C_FOUR_STRATEGIES_95_297.csv');nr=pd.DataFrame(rows)
 for k in [95,297]:
  a=nr[nr.budget.eq(k)].iloc[0];b=prior[prior.strategy_id.eq(a.strategy_id)&prior.budget.eq(k)].iloc[0]
  for c in ['overall_support_pct','high_support_pct','overall_low_pct','overall_high_pct','high_low_pct','high_high_pct']:assert np.isclose(a[c],b[c],atol=1e-7,rtol=0),(k,c,a[c],b[c])
 t=pd.concat([read(O/'results/COMPARATORS_DETERMINISTIC.csv'),nr],ignore_index=True)
 rank={'joint_concentration_then_caoa':0,'concentration_priority':1,'geographic_coverage_explicit_bridge':2,'natural_continuation_proportional_domains':3}
 t['rank']=t.strategy_id.map(rank);t=t.sort_values(['budget','rank']).drop(columns='rank');save(t[t.budget.isin([95,305])],'FOUR_STRATEGIES_95_305.csv')
 hit=hist[hist.overall_support_pct.ge(95)]
 time_result=dict(annual_additions=213,joint_sites=305,joint_years=305/213,historical_max_tested_sites=int(hist.budget.max()),historical_max_coverage=float(hist.overall_support_pct.iloc[-1]),historical_sites_95=int(hit.budget.iloc[0]) if len(hit) else None,paired_years_saved=(int(hit.budget.iloc[0])-305)/213 if len(hit) else None,legacy_planning_reference_sites=2900,legacy_reference_years_difference=(2900-305)/213,legacy_reference_not_verified_as_95pct_threshold=True,old_95_297_reproduced=True)
 js('results/TIME_TARGET95.json',time_result);log('TIME',time_result)

def costs():
 V3=P/'src/modeling/08_H26监测实施成本审计重算_20261010_V3';out=O/'costs'
 p=V3/'src/common.py';s=p.read_text('utf-8-sig').replace('ROOT=Path(__file__).resolve().parents[1]',f'ROOT=Path({str(out)!r})').replace('PROJECT=ROOT.parents[2]',f'PROJECT=Path({str(P)!r})')
 ns={'__file__':str(p),'__name__':'isolated_cost'};exec(compile(s,str(p),'exec'),ns)
 ep=V3/'src/engine.py';exec(compile(ep.read_text('utf-8-sig').replace('from common import *',''),str(ep),'exec'),ns)
 q=read(V3/'inputs/CANDIDATES_V3.csv.gz',low_memory=False).set_index('prediction_id',drop=False);rows=[];country=[];support=[];point=[]
 for k in [95,297,305]:
  ids=read(O/f'results/SELECTED_{k}.csv').prediction_id;s=q.loc[ids].reset_index(drop=True)
  support.append(dict(sites=k,no_ror_50km=int(s.ror_count_50km.eq(0).sum()),no_ror_50km_pct=float(s.ror_count_50km.eq(0).mean()*100)))
  s.to_csv(out/f'inputs/SELECTED_{k}.csv',index=False)
  for cap in ns['CFG']['capacity_branches']:
   for access in ns['CFG']['access_branches']:
    for level in ns['CFG']['levels']:
     z=ns['network'](s,level=level,capacity=cap,access=access,bridge=True,scenario_id=f'target95_{k}_{cap}_{access}_{level}')
     cols=[c for c in z if c.endswith('_eur')]
     row=dict(checkpoint=k,capacity_branch=cap,access_branch=access,cost_level=level,**{c:float(z[c].sum()) for c in cols});rows.append(row)
     if k in [95,305]:point.append(z.assign(checkpoint=k,capacity_branch=cap,access_branch=access,cost_level=level))
  log('Costs completed',k)
 prior=read(OLD/'monitor_lane/qaqc/costs/results/BASELINE_GLOBAL_COSTS_QAQC.csv')
 t=pd.DataFrame(rows)
 for _,a in t[t.checkpoint.isin([95,297])].iterrows():
  b=prior[(prior.checkpoint==a.checkpoint)&(prior.capacity_branch==a.capacity_branch)&(prior.access_branch==a.access_branch)&(prior.cost_level==a.cost_level)].iloc[0]
  for c in [c for c in t if c.endswith('_eur')]:assert np.isclose(a[c],b[c],rtol=1e-10),c
 save(t[t.checkpoint.isin([95,305])],'BASELINE_COSTS_95_305.csv');save(pd.DataFrame(support),'INSTITUTION_SUPPORT_COUNTS.csv')
 pd.concat(point,ignore_index=True).to_csv(O/'results/POINT_COSTS_95_305.csv.gz',index=False)
 log(t.query("checkpoint==305 and cost_level=='mid'")[['capacity_branch','access_branch','baseline_resource_eur']].to_dict('records'))
 js('audit/COST_RULES.json',dict(prices_and_cost_engine_unchanged=True,old_95_297_reproduced=True,no_new_capacity_claims=True,source_engine=str(ep)))

if __name__=='__main__':
 which=sys.argv[1] if len(sys.argv)>1 else 'all'
 if which in ['coverage','all']:coverage()
 if which in ['natural','all']:natural()
 if which in ['costs','all']:costs()
 js('audit/INPUT_HASHES_'+which+'.json',manifest)
