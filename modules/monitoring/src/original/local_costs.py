from common import *

def point_parts(q,level='mid',access=1.,visiting=False):
 """Unit costs at each point, excluding shared programme fees. Price index is not a wage observation."""
 p=OLD_CFG['cost_levels'][level];scale=p['support_multiplier'];r=q.price_factor.to_numpy(float)
 fresh=q.waterbody.eq(DOM[0]).to_numpy();b=q.support_type.eq('B_organization_and_technical').to_numpy()
 h=q.travel_hours.to_numpy(float)*access
 # Preserve 120 EUR reference field event: 2 people x 4 h x 12 EUR/h + 24 EUR materials.
 wage=p['field_event']*.8/(CFG['field_people']*CFG['field_hours'])*r
 equipment=p['setup']*(1+.5*b)*(.9+.1*r)
 field_labor=CFG['field_people']*CFG['field_hours']*wage
 materials=p['field_event']*.2*r
 transport=h*np.where(fresh,p['land_hour']*(.4+.6*r),p['boat_hour']*(.5+.5*r))
 travel_labor=h*CFG['field_people']*wage
 days=np.maximum(1,np.ceil((h+CFG['field_hours'])/8))
 lodge=(days-1)*CFG['field_people']*CFG['lodging_eur_mid']*scale*r
 perdiem=days*CFG['field_people']*CFG['perdiem_eur_mid']*scale*r
 # Imported instruments/reagents share not multiplied by general local price index.
 analysis=p['detection']*(.6+.4*r)
 qa=analysis*CFG['routine_QA_fraction']
 shipping=p['shipping']*(.5+.5*r)+.02*scale*q.nearest_ror_km.to_numpy(float)
 maintenance=.05*equipment
 return pd.DataFrame(dict(equipment_eur=equipment,field_labor_eur=field_labor,field_materials_eur=materials,local_transport_eur=transport,travel_labor_eur=travel_labor,lodging_eur=lodge,perdiem_eur=perdiem,analysis_eur=analysis,qa_eur=qa,shipping_eur=shipping,maintenance_eur=maintenance,local_wage_assumption_eur_hour=wage,travel_hours=h))

def mission(distance,destination_price,scale=1.,workdays=2):
 """Two visiting staff from a nearest-institution proxy. No verified road/flight itinerary."""
 distance=np.asarray(distance);air=distance>CFG['fly_threshold_km']
 hours=np.where(air,12+2*distance/600,distance*1.5/40*2)
 fare=np.where(air,2*(CFG['airfare_fixed_eur_mid']+CFG['airfare_per_km_eur_mid']*2*distance),hours*60)
 labor=2*CFG['expert_hour_eur_mid']*(workdays*8+hours)
 nights=np.maximum(1,np.ceil(hours/8)+workdays-1)
 accommodation=2*nights*CFG['lodging_eur_mid']*destination_price
 per_diem=2*(nights+1)*CFG['perdiem_eur_mid']*destination_price
 return np.array([fare,labor,accommodation,per_diem])*scale

def network(q,counts=None,level='mid',route='qualified_diverse',access=1.,bridge=False,visiting=False,already_standardized=False):
 """Allocate each shared fee once per institution-country-domain service group.
 Counts are station equivalents allocated to cells, not co-located physical sites.
 """
 q=q.reset_index(drop=True);counts=np.ones(len(q),int) if counts is None else np.asarray(counts,int)
 keep=counts>0;q=q.loc[keep].reset_index(drop=True);counts=counts[keep]
 unit=point_parts(q,level,access);v=unit.filter(regex='_eur$').multiply(counts,axis=0)
 p=OLD_CFG['cost_levels'][level];s=p['support_multiplier']
 shared=['training','organization','initial_comparison','annual_comparison','annual_coordination','method_conversion','method_validation','initial_mission_travel','initial_mission_labor','initial_mission_lodging','initial_mission_perdiem','annual_mission_travel','annual_mission_labor','annual_mission_lodging','annual_mission_perdiem']
 for k in shared:v[k+'_eur']=0.
 group,_=pd.factorize(q.service_group);ng=group.max()+1;ns=np.bincount(group,weights=counts)
 alloc=counts/ns[group];number=np.ceil(ns*(1.3+(1.3 if bridge else 0))/(250-3))
 price=np.bincount(group,weights=counts*q.price_factor.to_numpy())/ns;baseprice=.5+.5*price
 B=np.zeros(ng,bool);A=B.copy();d=np.zeros(ng)
 np.maximum.at(B,group,q.support_type.eq('B_organization_and_technical').to_numpy())
 np.maximum.at(A,group,q.support_type.ne('C_add_sampling').to_numpy())
 np.maximum.at(d,group,q.nearest_ror_km.to_numpy())
 amounts=dict(training=number*8000*s*baseprice*A,organization=number*8000*s*price*B,initial_comparison=number*2500*s,annual_comparison=number*3*p['detection']*(.6+.4*price),annual_coordination=number*500*s*price)
 if route!='qualified_diverse':
  amounts['method_conversion']=number*s*(2500 if route=='regional_common' else 27000)*(0 if already_standardized else 1)
  amounts['method_validation']=number*5000*s
 mi=mission(d,price,s,2)*A;ma=mission(d,price,s,1)*A*(B|visiting)
 for k,x,y in zip(['travel','labor','lodging','perdiem'],mi,ma):amounts['initial_mission_'+k]=number*x;amounts['annual_mission_'+k]=number*y
 for k,x in amounts.items():v[k+'_eur']=x[group]*alloc
 v['bridge_eur']=unit.analysis_eur.to_numpy()*1.3*counts*bool(bridge)
 initial=['equipment','training','organization','initial_comparison','method_conversion','method_validation','initial_mission_travel','initial_mission_labor','initial_mission_lodging','initial_mission_perdiem']
 annual=['field_labor','field_materials','local_transport','travel_labor','lodging','perdiem','analysis','qa','shipping','maintenance','annual_comparison','annual_coordination','annual_mission_travel','annual_mission_labor','annual_mission_lodging','annual_mission_perdiem']
 v['initial_setup_eur']=v[[x+'_eur' for x in initial]].sum(axis=1)
 v['annual_eur']=v[[x+'_eur' for x in annual]].sum(axis=1)
 annual_mission=v[[f'annual_mission_{k}_eur' for k in ['travel','labor','lodging','perdiem']]].sum(axis=1)
 # Initial mission covers baseline commissioning; annual visiting support starts in follow-up year 1.
 v['baseline_eur']=v.initial_setup_eur+v.annual_eur-v.maintenance_eur-annual_mission+v.bridge_eur
 v['five_year_eur']=v.baseline_eur+5*v.annual_eur
 v['station_equivalents']=counts
 meta=q[['prediction_id','basin_key','waterbody','country_iso3','country','continent','support_type','service_group','price_factor','price_evidence','deployment_lon','deployment_lat','nearest_ror_km','published_route_context']].copy()
 return pd.concat([meta,v],axis=1)

def run():
 status('point_costs_running')
 g=pd.read_csv(ROOT/'inputs/LOCALIZED_GRID.csv.gz',low_memory=False);q=pd.read_csv(ROOT/'inputs/LOCALIZED_CANDIDATES.csv.gz',low_memory=False)
 orig=pd.read_csv(track(V1/'inputs/ORIGINAL_JOINT_ORDER.csv.gz')).prediction_id.to_numpy()
 site=[];globalrows=[];country=[];unitrows=[]
 for level in ['low','mid','high']:
  u=point_parts(g,level)
  A=g.support_type.ne('C_add_sampling').to_numpy();B=g.support_type.eq('B_organization_and_technical').to_numpy();scale=OLD_CFG['cost_levels'][level]['support_multiplier']
  u['standalone_initial_support_mission_eur']=mission(g.nearest_ror_km.to_numpy(),g.price_factor.to_numpy(),scale,2).sum(axis=0)*A
  u['standalone_annual_support_mission_eur']=mission(g.nearest_ror_km.to_numpy(),g.price_factor.to_numpy(),scale,1).sum(axis=0)*B
  u['standalone_mission_cost_is_not_additive']=True
  unitrows.append(pd.concat([g[['prediction_id','basin_key','waterbody','country_iso3','support_type','price_factor','price_evidence','deployment_lon','deployment_lat']],u],axis=1).assign(cost_level=level))
  for n in [95,297]:
   sel=q.set_index('prediction_id').loc[orig[:n]].reset_index()
   for access,af in [('usual',1),('restricted',1.5),('difficult',2.5)]:
    z=network(sel,level=level,access=af,bridge=True);z['checkpoint']=n;z['cost_level']=level;z['access']=access
    site.append(z);vals=z.filter(regex='_eur$').sum().to_dict();globalrows.append(dict(checkpoint=n,cost_level=level,access=access,**vals))
    c=z.groupby(['country_iso3','country','continent'],dropna=False)[list(vals)].sum().reset_index();c['checkpoint']=n;c['cost_level']=level;c['access']=access;country.append(c)
 save(pd.concat(unitrows,ignore_index=True),'results/ALL_GRID_UNIT_COSTS.csv.gz')
 save(pd.concat(site,ignore_index=True),'results/BASELINE_SITE_LEDGER.csv.gz');save(pd.DataFrame(globalrows),'results/BASELINE_GLOBAL_COSTS_V2.csv');save(pd.concat(country,ignore_index=True),'results/BASELINE_COUNTRY_COSTS_V2.csv')
 # Eligibility/quality retained: US/China/Europe not assigned zero support merely because a standard exists.
 tests=[]
 a=q.head(10);a.loc[a.index[:3],'support_type']=['A_technical_support','B_organization_and_technical','C_add_sampling']
 f=network(a);r=network(a.iloc[::-1]);tests.append(dict(check='order_invariant_total',passed=bool(np.isclose(f.five_year_eur.sum(),r.five_year_eur.sum()))))
 tests.append(dict(check='six_occasions',passed=bool(np.allclose(f.five_year_eur,f.baseline_eur+5*f.annual_eur))))
 tests.append(dict(check='point_costs_finite',passed=bool(np.isfinite(pd.concat(unitrows).filter(regex='_eur$')).all().all())))
 js('audit/POINT_COST_TESTS.json',tests);assert all(t['passed'] for t in tests)
 log('Point costs complete; spatial revisit budgets explicitly separate from basin power designs')
if __name__=='__main__':run()
