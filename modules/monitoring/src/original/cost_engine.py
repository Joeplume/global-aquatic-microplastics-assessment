"""Resource events, cash/resource distinction, and bounded trip proxies.

No fine station coordinates are generated. Existing capability is a scenario,
not an observation. Analytical prices include instruments and operator time.
"""
from common import *

def route_parameters(q,access):
    fresh=q.waterbody.eq(DOM[0]).to_numpy()
    shore=fresh | ((access=='shore_if_feasible') & q.waterbody.eq(DOM[1]).to_numpy())
    mode=np.where(shore,np.where(fresh,'land_reach_sample','shore_bridge_assumed'),np.where(q.ports_distance_km.to_numpy()>CFG['remote_port_distance_km'],'remote_vessel_team','local_workboat_assumed'))
    h=np.where(shore,2*1.5*q.cities_distance_km.to_numpy()/CFG['land_speed_kmh'],q.travel_hours.to_numpy())
    # Arrival includes both outbound and return. Boat transit remains the frozen port proxy.
    speed=np.where(shore,CFG['land_speed_kmh'],CFG['boat_speed_kmh'])
    maxwork=np.where(shore,CFG['field_days_land'],CFG['field_days_boat'])*CFG['workday_hours']
    cap=np.where(shore,CFG['vehicle_station_cap'],CFG['boat_station_cap'])
    return mode,h,speed,maxwork,cap

def routes(q,counts,access,coarse=False,trace=False):
    mode,arrival,speed,maxwork,cap=route_parameters(q,access)
    n=len(q); travel=np.zeros(n); nights=np.zeros(n); days=np.zeros(n); trip_count=np.zeros(n,int); traces=[]
    if coarse:
        # Distinct independent-station equivalents; only intra-cell distance scenarios.
        intra=CFG['within_cell_interstation_km']*1.5/speed
        k=np.minimum(cap,np.maximum(1,np.floor((maxwork+intra)/(CFG['field_hours']+intra)).astype(int)))
        trips=np.ceil(counts/k).astype(int);last=counts-(trips-1)*k
        moving=np.maximum(k-1,0)*intra;lastmove=np.maximum(last-1,0)*intra
        travel=trips*arrival+np.maximum(counts-trips,0)*intra
        dt=np.maximum(1,np.ceil((arrival+k*CFG['field_hours']+moving)/8));dl=np.maximum(1,np.ceil((arrival+last*CFG['field_hours']+lastmove)/8))
        days=(trips-1)*dt+dl;nights=days-trips;trip_count=trips
        for i in range(n) if trace else []:
            traces.append(dict(route_event_key='coarse_'+str(i),mode=mode[i],stations=int(counts[i]),trips=int(trips[i]),station_capacity=int(k[i]),shared_return_hours=float(arrival[i]),interstation_hours=float((counts[i]-trips[i])*intra[i]),total_person_team_hours=float(travel[i]+counts[i]*4),workdays=int(days[i]),planning_only=True))
    else:
        open_trips={};finished=[]
        for i,row in enumerate(q.itertuples(index=False)):
            key=(row.team_region,mode[i])
            t=open_trips.get(key)
            move=0. if t is None else float(haversine(t['lon'],t['lat'],row.deployment_lon,row.deployment_lat))*1.5/speed[i]
            if t is None or t['work']+4+move>maxwork[i]+1e-9 or len(t['idx'])>=cap[i]:
                if t is not None:finished.append(t)
                t=dict(idx=[],work=0.,moving=0.,arrival=0.,lon=row.deployment_lon,lat=row.deployment_lat,mode=mode[i]);open_trips[key]=t;move=0.
            t['idx'].append(i);t['work']+=4+move;t['moving']+=move;t['arrival']=max(t['arrival'],arrival[i]);t['lon']=row.deployment_lon;t['lat']=row.deployment_lat
        finished+=list(open_trips.values())
        for j,t in enumerate(finished):
            ix=np.asarray(t['idx']);h=t['arrival']+t['moving'];d=max(1,int(np.ceil((h+len(ix)*4)/8)))
            # Sample-day limits cover on-site and interstation time; transit days are added.
            travel[ix]=h/len(ix);days[ix]=d/len(ix);nights[ix]=(d-1)/len(ix);trip_count[ix[0]]=1
            traces.append(dict(route_event_key='trip_'+str(j),mode=t['mode'],stations=len(ix),trips=1,shared_return_hours=t['arrival'],interstation_hours=t['moving'],total_person_team_hours=h+len(ix)*4,workdays=d,planning_only=True))
    return dict(mode=mode,travel=travel,nights=nights,days=days,trips=trip_count,traces=traces)

def network(q,counts=None,level='mid',capacity='support_needed',access='boat_if_required',bridge=False,coarse=False,protocol='qualified_diverse',scenario_id='network',ledger=False,stage='routes'):
    q=q.reset_index(drop=True);counts=np.ones(len(q),int) if counts is None else np.asarray(counts,int)
    keep=counts>0;q=q.loc[keep].reset_index(drop=True);counts=counts[keep];n=len(q)
    p=OLD['cost_levels'][level];scale=p['support_multiplier'];r=q.price_factor.to_numpy(float)
    wage=p['field_event']*.8/8*r;support=(capacity=='support_needed') & ~q.get('target_contract_verified',pd.Series(False,index=q.index)).to_numpy(bool)
    need_org=(capacity=='support_needed') & ~q.get('sampling_organization_verified',pd.Series(False,index=q.index)).to_numpy(bool)
    support=support|need_org
    laborcash=np.where(need_org,CFG['cash_labor_support_fraction'],CFG['cash_labor_available_fraction'])
    resource=np.zeros((n,2,5));cash=resource.copy();events=[];alloc=[]
    def emit(name,cat,phase,ix,weights,quantity,unit_price,provider,includes,excludes,cash_fraction=1.,source='V1/V2 reference unit prices; V3 scenario',unit_name='resource_units',allocation_rule='station-equivalent proportional within event coverage'):
        if quantity==0:return
        ix=np.asarray(ix,int);weights=np.asarray(weights,float);weights=weights/weights.sum()
        amount=float(quantity*unit_price);ci=CATS.index(cat);ph=0 if phase=='baseline' else 1
        resource[ix,ph,ci]+=weights*amount;cf=np.broadcast_to(cash_fraction,len(ix));cash[ix,ph,ci]+=weights*amount*cf
        if ledger:
            eid=f'{scenario_id}:{len(events):08d}'
            events.append(dict(cost_event_id=eid,event=name,category=cat,phase=phase,occasion=0 if ph==0 else '1-5 annual identical purchases',provider=provider,payee=provider,quantity=quantity,quantity_unit=unit_name,unit_price_eur=unit_price,resource_eur=amount,cash_eur=float((weights*amount*cf).sum()),coverage_station_equivalents=int(counts[ix].sum()),includes=includes,excludes=excludes,source_assumption=source,allocation_rule=allocation_rule,currency='EUR_ex_VAT',size_window_um='10-5000',scenario_id=scenario_id))
            for j,w,c in zip(ix,weights,cf):alloc.append(dict(cost_event_id=eid,prediction_id=q.prediction_id.iloc[j],basin_key=q.basin_key.iloc[j],country_iso3=q.country_iso3.iloc[j],weight=w,resource_eur=w*amount,cash_eur=w*amount*c,scenario_id=scenario_id))
    cellgroups=q.groupby('team_region',sort=False).indices
    def cell(name,cat,phase,values,cf=1.,includes='',excludes='',quantities=None,unit_name='planning_component_eur'):
        if not ledger:
            ci=CATS.index(cat);ph=0 if phase=='baseline' else 1
            resource[:,ph,ci]+=values;cash[:,ph,ci]+=values*cf
            return
        for key,ix in cellgroups.items():
            ix=np.asarray(ix);ix=ix[np.asarray(values)[ix]>0]
            if not len(ix):continue
            qty=float(np.asarray(quantities)[ix].sum()) if quantities is not None else float(np.asarray(values)[ix].sum())
            emit(name,cat,phase,ix,np.asarray(values)[ix],qty,float(np.asarray(values)[ix].sum())/qty,str(key),includes,excludes,np.asarray(cf)[ix] if np.ndim(cf) else cf,unit_name=unit_name,allocation_rule='actual component usage share')
    for phase,mult in [('baseline',1),('followup',5)]:
        cell('field_sampling_labor','sampling_labor',phase,counts*8*wage*mult,laborcash,'2 sampling staff x 4 hours per station/event','analytical staff; driver and skipper',counts*8*mult,'person_hours')
        cell('field_consumables','equipment_materials',phase,counts*p['field_event']*.2*r*mult,includes='sample collection materials',excludes='instrument/analytical operator',quantities=counts*mult,unit_name='field_events')
    # Sampling equipment/team ownership shared; no analytical instrument purchase.
    shared_key=q.team_region if stage!='capability' else q.service_group
    groups=shared_key.groupby(shared_key,sort=False).groups
    for team,idx in groups.items():
        ix=np.asarray(list(idx));ns=counts[ix].sum();price=np.average(r[ix],weights=counts[ix]);need=bool(need_org[ix].any())
        teams=int(np.ceil(ns/CFG['team_station_capacity'])) if stage!='capability' else int(np.ceil(ns*1.3/247))
        equip=p['setup']*(1+.5*need)*(.9+.1*price)
        emit('sampling_equipment' if need else 'existing_equipment_use','equipment_materials','baseline',ix,counts[ix],teams,equip if need else .05*equip,team,'shared sampling kit; existing asset annual use if available','analytical equipment',1 if need else 0)
        emit('sampling_equipment_maintenance','equipment_materials','followup',ix,counts[ix],5*teams,.05*equip,team,'five annual kit maintenance/use charges','new kit purchase',1 if need else 0)
        if need:emit('organization_commissioning','training_organization','baseline',ix,counts[ix],teams,CFG['organization_eur_mid']*scale*price,team,'local organization/data commissioning','course; routine annual administration')
    # Courses shared across teams and matrices; no nearest-ROR or water-domain course key.
    for country,idx in q.groupby('country_iso3',dropna=False,sort=False).indices.items():
        ix=np.asarray(idx);active=ix[need_org[ix]]
        if len(active):
            if stage=='capability':courses=len(pd.unique(q.service_group.iloc[active]))
            else:
                teams=sum(int(np.ceil(counts[j].sum()/CFG['team_station_capacity'])) for _,j in q.iloc[active].groupby('team_region',sort=False).groups.items())
                courses=int(np.ceil(teams/CFG['course_team_capacity']))
            price=np.average(r[active],weights=counts[active])
            emit('field_team_course_all_in','training_organization','baseline',active,counts[active],courses,CFG['field_team_course_eur_mid']*scale*(.5+.5*price),'course_provider_'+str(country),'instructor; travel; course materials; field/matrix modules','ongoing sampling labor; annual laboratory throughput')
    for key,idx in q.groupby(['country_iso3','provider_id'],dropna=False,sort=False).indices.items():
        ix=np.asarray(idx);ns=int(counts[ix].sum());price=np.average(r[ix],weights=counts[ix]);unit=p['detection']*(.6+.4*price)
        nb=int(np.ceil(ns*.15));nd=int(np.ceil(ns*.15));primary=ns+nb+nd;bridged=(ns+nb+nd) if bridge else 0
        slots=int(np.ceil(max(primary+bridged,primary)/247))
        for phase,mult in [('baseline',1),('followup',5)]:
            for name,num in [('primary_count_analysis',ns),('field_procedure_blank_analysis',nb),('duplicate_analysis',nd),('independent_source_bridge_analysis',bridged if phase=='baseline' else 0),('intercomparison_analysis',slots*3)]:
                emit(name,'analysis_QA',phase,ix,counts[ix],num*mult,unit,key[1],'pretreatment; instrument; analytical operator; count endpoint required','shipping; field labor; no additional analysis multiplier')
            distance=np.average(q.provider_distance_km.iloc[ix],weights=counts[ix]);shipping=p['shipping']*(.5+.5*price)+.02*scale*distance
            emit('sample_shipping','transport',phase,ix,counts[ix],(primary+(bridged if phase=='baseline' else 0)+slots*3)*mult,shipping,key[1],'shipping distance proxy and handling','field arrival; target preservation not verified')
            emit('capacity_slot_coordination','training_organization',phase,ix,counts[ix],slots*mult,CFG['annual_coordination_per_capacity_slot_eur_mid']*scale*price,key[1],'booking/admin for finite 250 analyses/year slots','analytical operator; repeated training')
        emit('initial_cross_source_validation','analysis_QA','baseline',ix,counts[ix],1,2500*scale,key[1],'startup method comparison management','analysis already charged explicitly')
        # All concrete providers have existing equipment, QA, and some water method.
        # A size/endpoint gap requires an adaptation module, never a new full laboratory.
        need=bool(support[ix].any());gap=not bool(q.target_contract_verified.iloc[ix].all())
        if need and gap:emit('target_window_adaptation','training_organization','baseline',ix,counts[ix],1,CFG['provider_size_gap_module_eur_mid']*scale,key[1],'10-5000 um/count/matrix contract adaptation, expert work all in','new analytical instruments; general laboratory training')
        if protocol!='qualified_diverse' and need:
            extra=2500 if protocol=='regional_common' else 12000
            emit('common_workflow_alignment','training_organization','baseline',ix,counts[ix],1,extra*scale,key[1],'additional shared workflow alignment','already-qualified method full retraining')
            emit('common_workflow_validation','analysis_QA','baseline',ix,counts[ix],1,5000*scale,key[1],'between-provider validation administration','separately enumerated analyses')
    rt=routes(q,counts,access,coarse,trace=ledger)
    if stage!='routes':
        rt['travel']=q.travel_hours.to_numpy()*counts;rt['days']=np.maximum(1,np.ceil((q.travel_hours.to_numpy()+4)/8))*counts;rt['nights']=rt['days']-counts;rt['trips']=counts.copy()
    shore=np.isin(rt['mode'],['land_reach_sample','shore_bridge_assumed']);rate=np.where(shore,p['land_hour']*(.4+.6*r),p['boat_hour']*(.5+.5*r))
    for phase,mult in [('baseline',1),('followup',5)]:
        rental_hours=rt['travel']+counts*CFG['field_hours']
        cell('field_vehicle_or_boat','transport',phase,rental_hours*rate*mult,includes='vehicle/boat transit AND onsite occupation; fuel; driver/skipper included',excludes='two sampling staff wages',quantities=rental_hours*mult,unit_name='vehicle_boat_hours')
        cell('sampling_staff_travel_labor','sampling_labor',phase,rt['travel']*2*wage*mult,laborcash,'sampling staff travel time','driver/skipper already in rental',rt['travel']*2*mult,'person_hours')
        cell('accommodation','transport',phase,rt['nights']*2*60*scale*r*mult,includes='overnight room only',excludes='meals/incidentals',quantities=rt['nights']*2*mult,unit_name='person_nights')
        cell('meals_incidentals','transport',phase,rt['days']*2*30*scale*r*mult,includes='meals and incidental day allowance',excludes='room cost',quantities=rt['days']*2*mult,unit_name='person_days')
    meta=q[['prediction_id','basin_key','waterbody','country_iso3','country','continent','deployment_lon','deployment_lat','provider_id','provider_name','capability_evidence_v3','ability_unverified','target_contract_verified','team_region','price_factor','price_evidence']].copy()
    meta['station_equivalents']=counts;meta['sampling_occasions']=6;meta['field_events_six_occasions']=counts*6;meta['size_window_um']='10-5000';meta['capacity_branch']=capacity;meta['access_branch']=access;meta['cost_level']=level;meta['protocol']=protocol;meta['scenario_id']=scenario_id
    meta['independent_source_deployments']=q.independent_source_deployments.to_numpy() if 'independent_source_deployments' in q else np.nan
    meta['source_independence_evidence']='frozen repair action contract; organizations/labs do not establish independence' if bridge else 'basin trend design; repair action contract not applicable'
    meta['support_need_scenario']=np.where(need_org,'B_organization_support_assumed',np.where(support,'A_technical_support_assumed','C_existing_capacity_assumed'))
    meta['ability_evidence_status']='unverified at point; service evidence retained separately';meta['field_mode']=rt['mode'];meta['trips_per_occasion']=rt['trips'];meta['travel_hours_per_occasion']=rt['travel'];meta['physical_locations_verified']=False;meta['subgrid_siting_required']=counts>1
    for j,c in enumerate(CATS):
        meta[c+'_baseline_resource_eur']=resource[:,0,j];meta[c+'_baseline_cash_eur']=cash[:,0,j]
        meta[c+'_six_resource_eur']=resource[:,:,j].sum(axis=1);meta[c+'_six_cash_eur']=cash[:,:,j].sum(axis=1)
    meta['baseline_resource_eur']=resource[:,0].sum(axis=1);meta['baseline_cash_eur']=cash[:,0].sum(axis=1)
    meta['annual_resource_eur']=resource[:,1].sum(axis=1)/5;meta['annual_cash_eur']=cash[:,1].sum(axis=1)/5
    meta['six_resource_eur']=resource.sum(axis=(1,2));meta['six_cash_eur']=cash.sum(axis=(1,2))
    for c in CATS:meta[c+'_share']=meta[c+'_six_resource_eur']/meta.six_resource_eur
    if ledger:
        ev=pd.DataFrame(events);al=pd.DataFrame(alloc)
        assert ev.cost_event_id.is_unique
        assert np.isclose(ev.resource_eur.sum(),meta.six_resource_eur.sum())
        assert np.isclose(ev.cash_eur.sum(),meta.six_cash_eur.sum())
        chk=al.groupby('cost_event_id')[['resource_eur','cash_eur']].sum().reindex(ev.cost_event_id)
        assert np.allclose(chk.resource_eur,ev.resource_eur)
        assert np.allclose(chk.cash_eur,ev.cash_eur)
        return meta,ev,al,pd.DataFrame(rt['traces'])
    return meta
