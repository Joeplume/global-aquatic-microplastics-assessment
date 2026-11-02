"""Aggregate all ten real map folds using the unchanged original aggregation functions."""
from pathlib import Path
import argparse,ast,warnings
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--training-output',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
tree=ast.parse((HERE/'provenance/original_aggregate.py').read_text(encoding='utf8'))
names={'numeric','cell_area','modal_strings','aggregate_grid'}
body=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
assert len(body)==len(names)
ns={'np':np,'pd':pd,'warnings':warnings,'GRID_NUMERIC':['latent_q50_log10','latent_sd_log10','future_observation_sd_log10'],'IDENTITY':['prediction_id','longitude','latitude','waterbody'],'DOMAINS':{'freshwater':'reference_surface_freshwater','transition':'transition_coastal','marine':'marine'}}
exec(compile(ast.Module(body=body,type_ignores=[]),'original_aggregate.py','exec'),ns)
frames=[]
for domain in ns['DOMAINS']:
 units=[(fold,a.training_output/'units/qaqc'/domain/f'fold_{fold}') for fold in range(1,11)]
 frames.append(ns['aggregate_grid'](units,domain,10))
a.output.parent.mkdir(parents=True,exist_ok=True)
pd.concat(frames,ignore_index=True).to_csv(a.output,index=False)
print('WROTE',a.output)
