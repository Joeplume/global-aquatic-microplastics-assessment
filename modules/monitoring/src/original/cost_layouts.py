from common import *
from engine import network, route_parameters
from scipy import sparse
import math, heapq

def distance(x,y,a,b):
    x,y,a,b=map(math.radians,(x,y,a,b));v=math.sin((b-y)/2)**2+math.cos(y)*math.cos(b)*math.sin((a-x)/2)**2
    return 12742.0176*math.asin(min(1,math.sqrt(max(0,v))))

class PrefixCost:
    """Same baseline event formula as engine.network; all three prices paired.

    Includes discrete QC counts, finite capacity slots and course/team limits.
    A route prefix is a feasible, priority-ordered planning itinerary.
    """
    def __init__(self,q,access):
        self.q=q;self.mode,self.arrival,self.speed,self.maxwork,self.cap=route_parameters(q,access)
        self.r=q.price_factor.to_numpy(float);self.lon=q.deployment_lon.to_numpy();self.lat=q.deployment_lat.to_numpy();self.dist=q.provider_distance_km.to_numpy()
        self.teams=q.team_region.to_numpy();self.countries=q.country_iso3.fillna('UNK').to_numpy();self.providers=q.provider_id.to_numpy()
        ps=[OLD['cost_levels'][l] for l in CFG['levels']]
        self.scale=np.array([p['support_multiplier'] for p in ps]);self.setup=np.array([p['setup'] for p in ps]);self.field=np.array([p['field_event'] for p in ps]);self.detect=np.array([p['detection'] for p in ps]);self.ship=np.array([p['shipping'] for p in ps]);self.land=np.array([p['land_hour'] for p in ps]);self.boat=np.array([p['boat_hour'] for p in ps])
    def run(self,order,support=True):
        res=np.zeros((len(order)+1,3));cash=res.copy();total=np.zeros(3);ctotal=total.copy();regions={};countryteams={};labs={};trips={}
        for k,i in enumerate(order,1):
            r=self.r[i];wage=self.field*.8/8*r;team=self.teams[i];country=self.countries[i];lab=(country,self.providers[i]);cf=1 if support else 0
            field=8*wage+self.field*.2*r;total+=field;ctotal+=8*wage*cf+self.field*.2*r
            oldn=regions.get(team,0);newn=oldn+1;regions[team]=newn;nt0=math.ceil(oldn/50);nt1=math.ceil(newn/50)
            if nt1>nt0:
                equipment=self.setup*(1+.5*support)*(.9+.1*r)
                total+=equipment if support else .05*equipment;ctotal+=equipment if support else 0
                if support:
                    org=8000*self.scale*r;total+=org;ctotal+=org
                    cc=countryteams.get(country,0);countryteams[country]=cc+1
                    if math.ceil((cc+1)/20)>math.ceil(cc/20):
                        training=4000*self.scale*(.5+.5*r);total+=training;ctotal+=training
            prior=labs.get(lab,(0,0.));ln,dsum=prior;nn=ln+1;ndsum=dsum+self.dist[i]
            def labprice(n,ds):
                if not n:return np.zeros(3)
                samples=2*(n+2*math.ceil(.15*n));slots=math.ceil(samples/247)
                analyzed=samples+slots*3;d=ds/n
                return analyzed*(self.detect*(.6+.4*r)+self.ship*(.5+.5*r)+.02*self.scale*d)+slots*500*self.scale*r+2500*self.scale+(2500*self.scale if support else 0)
            delta=labprice(nn,ndsum)-labprice(ln,dsum);total+=delta;ctotal+=delta;labs[lab]=(nn,ndsum)
            key=(team,self.mode[i]);t=trips.get(key);move=0. if t is None else distance(t['lon'],t['lat'],self.lon[i],self.lat[i])*1.5/self.speed[i]
            if t is None or t['work']+4+move>self.maxwork[i]+1e-9 or t['n']>=self.cap[i]:t=dict(work=0.,moving=0.,arrival=0.,lon=self.lon[i],lat=self.lat[i],n=0);trips[key]=t;move=0.
            def tripprice(t):
                if not t['n']:return np.zeros(3),np.zeros(3)
                h=t['arrival']+t['moving'];days=math.ceil((h+t['n']*4)/8)
                shore=self.mode[i] in ['land_reach_sample','shore_bridge_assumed'];rate=self.land*(.4+.6*r) if shore else self.boat*(.5+.5*r)
                other=(h+t['n']*4)*rate+(days-1)*2*60*self.scale*r+days*2*30*self.scale*r;labor=h*2*wage
                return other+labor,other+labor*cf
            before,beforec=tripprice(t)
            t['work']+=4+move;t['moving']+=move;t['arrival']=max(t['arrival'],self.arrival[i]);t['n']+=1;t['lon']=self.lon[i];t['lat']=self.lat[i]
            after,afterc=tripprice(t);total+=after-before;ctotal+=afterc-beforec
            res[k]=total;cash[k]=ctotal
        return res,cash

def baseline():
    q=pd.read_csv(ROOT/'inputs/CANDIDATES_V3.csv.gz',low_memory=False);o=pd.read_csv(V1/'inputs/ORIGINAL_JOINT_ORDER.csv.gz').candidate_index.to_numpy(int)
    rows=[];sites=[];events=[];alloc=[];trip=[];diag=[]
    for n in [95,297]:
        s=q.iloc[o[:n]].reset_index(drop=True)
        for cap in CFG['capacity_branches']:
            for access in CFG['access_branches']:
                for level in CFG['levels']:
                    name=f'baseline_{n}_{cap}_{access}_{level}'
                    z,e,a,t=network(s,level=level,capacity=cap,access=access,bridge=True,ledger=True,scenario_id=name)
                    sites.append(z.assign(checkpoint=n));events.append(e);alloc.append(a);trip.append(t.assign(scenario_id=name))
                    vals={c:float(z[c].sum()) for c in z if c.endswith('_eur')};rows.append(dict(checkpoint=n,capacity_branch=cap,access_branch=access,cost_level=level,**vals))
                    for stage in ['capability','services','routes']:
                        w=z if stage=='routes' else network(s,level=level,capacity=cap,access=access,bridge=True,stage=stage)
                        diag.append(dict(checkpoint=n,capacity_branch=cap,access_branch=access,cost_level=level,stage=stage,baseline_resource_eur=w.baseline_resource_eur.sum(),six_resource_eur=w.six_resource_eur.sum(),baseline_cash_eur=w.baseline_cash_eur.sum(),sequential_diagnostic_not_causal=True))
    save(pd.concat(sites),'results/BASELINE_POINT_COSTS.csv.gz');save(pd.DataFrame(rows),'results/BASELINE_GLOBAL_COSTS.csv');save(pd.concat(events),'results/BASELINE_COST_EVENTS.csv.gz');save(pd.concat(alloc),'results/BASELINE_EVENT_ALLOCATIONS.csv.gz');save(pd.concat(trip),'results/BASELINE_TRIPS.csv.gz')
    z=pd.concat(sites);cols=[c for c in z if c.endswith('_eur')]
    save(z.groupby(['checkpoint','capacity_branch','access_branch','cost_level','country_iso3'],dropna=False)[cols].sum().reset_index(),'results/BASELINE_COUNTRY_COSTS.csv')
    d=pd.DataFrame(diag);v2=pd.read_csv(V2/'results/BASELINE_GLOBAL_COSTS_V2.csv');v2=v2[v2.access=='usual']
    d=d.merge(v2[['checkpoint','cost_level','baseline_eur']],on=['checkpoint','cost_level'],validate='many_to_one').rename(columns={'baseline_eur':'V2_baseline_eur'});d['change_from_V2_eur']=d.baseline_resource_eur-d.V2_baseline_eur
    save(d,'results/SEQUENTIAL_ACCOUNTING_DIAGNOSTICS.csv')
    tests=[]
    for cap in CFG['capacity_branches']:
        for access in CFG['access_branches']:
            pc=PrefixCost(q,access);rr,cc=pc.run(o[:297],support=cap=='support_needed')
            for n in [95,297]:
                for j,level in enumerate(CFG['levels']):
                    ref=pd.DataFrame(rows).query('checkpoint == @n and capacity_branch == @cap and access_branch == @access and cost_level == @level').iloc[0]
                    tests.append(dict(check=f'prefix_event_formula_{n}_{cap}_{access}_{level}',passed=bool(np.isclose(rr[n,j],ref.baseline_resource_eur,rtol=1e-10) and np.isclose(cc[n,j],ref.baseline_cash_eur,rtol=1e-10)),difference_eur=float(rr[n,j]-ref.baseline_resource_eur)))
    js('audit/PREFIX_FORMULA_TESTS.json',tests);assert all(x['passed'] for x in tests),[x for x in tests if not x['passed']]
    log('Baseline ledgers and sequential diagnostics saved')

def make_heuristic(q,mat,base):
    # Lazy gain/standalone marginal proxy; actual shared event fees assessed later.
    w=q.cell_area_km2.to_numpy();wh=w*q.high_concentration_primary.to_numpy();covered=np.zeros(len(q),bool);selected=np.zeros(len(q),bool);out=[]
    proxy=350/50+120+2*180*1.3+q.travel_hours.to_numpy()*np.where(q.waterbody.eq(DOM[0]),60,400)+8000/50+4000/1000
    for weights,target,prior,den in [(wh,OLD['high_target_pct'],base['high_area'],base['high_denominator']),(w,OLD['overall_target_pct'],base['overall_area'],base['denominator'])]:
        heap=[(-v/proxy[i],i,-1) for i,v in enumerate(mat.dot(weights*(~covered))) if not selected[i]];heapq.heapify(heap)
        while heap:
            if (prior+weights[covered].sum())/den*100>=target-1e-8 and len(out)>=95:break
            _,i,stamp=heapq.heappop(heap);ix=mat.indices[mat.indptr[i]:mat.indptr[i+1]];gain=weights[ix[~covered[ix]]].sum()/proxy[i]
            if stamp!=len(out):heapq.heappush(heap,(-gain,i,len(out)));continue
            out.append(i);selected[i]=True;covered[ix]=True
    return np.r_[out,np.flatnonzero(~selected)].astype(int)

def run_paired():
    status('paired_500_running',completed_replicates=0)
    q=pd.read_csv(ROOT/'inputs/CANDIDATES_V3.csv.gz',low_memory=False);assert np.array_equal(q.candidate_index,np.arange(len(q)))
    orig=pd.read_csv(V1/'inputs/ORIGINAL_JOINT_ORDER.csv.gz').candidate_index.to_numpy(int);n=len(q)
    base=json.loads((V1/'audit/PREFLIGHT.json').read_text(encoding='utf-8'))['baselines'];mat=sparse.load_npz(V1/'inputs/REPAIR_ACTIONS.npz').tocsr()
    w=q.cell_area_km2.to_numpy();wh=w*q.high_concentration_primary.to_numpy()
    def coverage(order,ks):
        covered=np.zeros(n,bool);sa=base['overall_area'];sh=base['high_area'];result={0:(sa/base['denominator']*100,sh/base['high_denominator']*100)};ks=set(map(int,ks))
        for k,i in enumerate(order[:max(ks,default=0)],1):
            ix=mat.indices[mat.indptr[i]:mat.indptr[i+1]];new=ix[~covered[ix]];sa+=w[new].sum();sh+=wh[new].sum();covered[ix]=True
            if k in ks:result[k]=(sa/base['denominator']*100,sh/base['high_denominator']*100)
        return result
    spatial=pd.read_csv(V1/'results/SPATIAL_COMPARISONS_500.csv');hist=pd.read_csv(V1/'results/HISTORICAL_REUSE_COMPARISONS_500.csv')
    frozen=pd.concat([spatial[spatial.strategy.isin(['original_joint','stratified_random'])],hist]).query("cost_level == 'mid'")
    density=pd.read_csv(V1/'inputs/HISTORICAL_20NN_DENSITY.csv').sort_values('candidate_index').historical_20nn_density.to_numpy()
    domain=q.waterbody.to_numpy();slots=list(domain[orig]);rng=np.random.default_rng(OLD['seed'])
    for d in DOM:slots.extend([d]*(int(np.sum(domain==d))-slots.count(d)))
    tail=np.asarray(slots[len(orig):],object);rng.shuffle(tail);slots=np.r_[np.asarray(slots[:len(orig)]),tail]
    bydom={d:np.flatnonzero(domain==d) for d in DOM};slotidx={d:np.flatnonzero(slots==d) for d in DOM}
    pc={a:PrefixCost(q,a) for a in CFG['access_branches']};budget={};rows=[];fingerprints=[];validations=[]
    for a in CFG['access_branches']:
        for c in CFG['capacity_branches']:
            rr,cc=pc[a].run(orig[:297],c=='support_needed')
            for j,l in enumerate(CFG['levels']):
                for k in [95,297]:budget[a,c,l,k]=(rr[k,j],cc[k,j])
    def evaluate(order,strategy,rep):
        f=frozen[(frozen.strategy==strategy)&(frozen.replicate==rep)]
        targets={r.criterion:(int(r.sites),float(r.overall_pct),float(r.high_pct)) for r in f[f.comparison=='same_target'].itertuples()}
        if not targets:
            allcov=coverage(order,range(1,6001));targets={}
            for label in ['high','overall','both']:
                hit=[k for k,(a,h) in allcov.items() if (label=='overall' or h>=OLD['high_target_pct']-1e-8) and (label=='high' or a>=OLD['overall_target_pct']-1e-8)]
                if hit:targets[label]=(min(hit),*allcov[min(hit)])
        lim=max(600,297,max((v[0] for v in targets.values()),default=0));lim=min(lim,len(order))
        cache={}
        while True:
            truncated=False
            for access in CFG['access_branches']:
                for cap in CFG['capacity_branches']:
                    rr,cc=pc[access].run(order[:lim],cap=='support_needed');cache[access,cap]=(rr,cc)
                    for j,level in enumerate(CFG['levels']):
                        rb,cb=budget[access,cap,level,297]
                        if (rr[-1,j]<=rb or cc[-1,j]<=cb) and lim<len(order):truncated=True
            if not truncated:break
            lim=min(len(order),lim*2)
        entries=[];needed={95,297}
        for access in CFG['access_branches']:
            for cap in CFG['capacity_branches']:
                rr,cc=cache[access,cap]
                # Cost need not be strictly monotone when shared shipping QA is regrouped.
                # Same budget uses the best feasible prefix, never searchsorted blindly.
                for j,level in enumerate(CFG['levels']):
                    for k in [95,297]:entries.append((access,cap,level,'same_N',str(k),k,rr[k,j],cc[k,j],None,None))
                    for label,(k,a,h) in targets.items():entries.append((access,cap,level,'same_target',label,k,rr[k,j],cc[k,j],a,h))
                    for k in [95,297]:
                        rb,cb=budget[access,cap,level,k]
                        possible=np.flatnonzero(rr[:,j]<=rb+1e-7);ix=int(possible[-1]);needed.add(ix)
                        entries.append((access,cap,level,'same_budget_resource',str(k),ix,rr[ix,j],cc[ix,j],None,None))
                        possible=np.flatnonzero(cc[:,j]<=cb+1e-7);ix=int(possible[-1]);needed.add(ix)
                        entries.append((access,cap,level,'same_budget_cash',str(k),ix,rr[ix,j],cc[ix,j],None,None))
                        if ix==lim and lim<len(order):raise RuntimeError('Budget curve truncation: extend evaluated prefix')
        cov=coverage(order,needed)
        for access,cap,level,comp,crit,k,re,ce,a,h in entries:
            a,h=cov[k] if a is None else (a,h)
            rows.append(dict(strategy=strategy,replicate=rep,capacity_branch=cap,access_branch=access,cost_level=level,comparison=comp,criterion=crit,sites=k,baseline_resource_eur=re,baseline_cash_eur=ce,overall_pct=a,high_pct=h,attained=True,conditional_feasibility_unverified=True))
        fingerprint=hashlib.sha256(np.asarray(order,dtype='<i4').tobytes()).hexdigest();fingerprints.append(dict(strategy=strategy,replicate=rep,full_order_sha256=fingerprint,evaluated_max_prefix=lim,paired_seed=OLD['seed']))
        if rep in [0,249,499] or rep==-1:
            cs=coverage(order,[v[0] for v in targets.values()])
            for label,(k,a,h) in targets.items():validations.append(dict(strategy=strategy,replicate=rep,target=label,sites=k,overall_delta=cs[k][0]-a,high_delta=cs[k][1]-h,passed=bool(np.allclose(cs[k],[a,h],atol=1e-8))))
    histpriority=np.empty(n,int)
    for d,ix in bydom.items():histpriority[slotidx[d]]=ix[np.argsort(-density[ix],kind='stable')]
    heuristic=make_heuristic(q,mat,base);save(q.iloc[heuristic[:1000]].assign(priority_order=np.arange(1,1001)),'results/V3_COST_HEURISTIC_ORDER.csv.gz')
    evaluate(np.r_[orig,np.setdiff1d(np.arange(n),orig)],'original_joint',-1);evaluate(histpriority,'historical_20nn_priority',-1);evaluate(heuristic,'V3_cost_heuristic',-1)
    for rep in range(500):
        u=rng.random(n).clip(1e-12,1-1e-12)
        for strategy,key in [('stratified_random',u),('historical_20nn_weighted',-np.log1p(-u)/density)]:
            order=np.empty(n,int)
            for d,ix in bydom.items():order[slotidx[d]]=ix[np.argsort(key[ix],kind='stable')]
            evaluate(order,strategy,rep)
        if (rep+1)%25==0:
            save(pd.DataFrame(rows),'results/PAIRED_LAYOUT_COMPARISONS_500.csv.gz');save(pd.DataFrame(fingerprints),'audit/LAYOUT_FINGERPRINTS.csv');status('paired_500_running',completed_replicates=rep+1);log('Paired layouts',rep+1)
    out=pd.DataFrame(rows);save(out,'results/PAIRED_LAYOUT_COMPARISONS_500.csv.gz');save(pd.DataFrame(fingerprints),'audit/LAYOUT_FINGERPRINTS.csv');js('audit/FROZEN_TARGET_REPRODUCTION.json',validations);assert all(v['passed'] for v in validations)
    group=['strategy','capacity_branch','access_branch','cost_level','comparison','criterion']
    summary=out.groupby(group)[['sites','baseline_resource_eur','baseline_cash_eur','overall_pct','high_pct']].quantile([.025,.5,.975]).reset_index().rename(columns={'level_6':'layout_quantile'})
    save(summary,'results/LAYOUT_COMPARISON_QUANTILES.csv');log('500 paired layout comparisons saved',len(out))
if __name__=='__main__':baseline();run_paired()
