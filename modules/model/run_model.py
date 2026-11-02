"""Run the actual three-domain H26 model from frozen prepared input tables."""
from pathlib import Path
import argparse, os, subprocess, sys, json

HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--design',choices=['grouped10','random10','spatial','source','joint','grouped_comparison'],default='grouped10')
p.add_argument('--domain',choices=['freshwater','transition','marine','all'],default='freshwater')
p.add_argument('--fold',default='1',help='Integer fold or all')
p.add_argument('--output',type=Path,default=HERE/'results')
p.add_argument('--epochs',type=int,default=180)
p.add_argument('--with-maps',action='store_true',help='Export full prediction grids (grouped10 only)')
p.add_argument('--check-only',action='store_true')
a=p.parse_args()
a.output=a.output.resolve()
if a.with_maps and a.design!='grouped10': p.error('--with-maps requires grouped10')
files=json.loads((HERE/'MODEL_FILES.json').read_text(encoding='utf8'))
missing=[x['package_path'] for x in files if not (HERE/x['package_path']).is_file()]
if missing: raise FileNotFoundError('Missing input files: '+repr(missing))
a.output.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env['NMP_MODEL_OUTPUT']=str(a.output);env['PYTHONUTF8']='1';env['PYTHONDONTWRITEBYTECODE']='1'
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: env[k]='2'
domains=['freshwater','transition','marine'] if a.domain=='all' else [a.domain]
nf=5 if a.design in ['spatial','source','joint'] else 10
folds=list(range(1,nf+1)) if a.fold=='all' else [int(a.fold)]
if any(f<1 or f>nf for f in folds): p.error(f'fold must be 1..{nf}')
commands=[]
for d in domains:
 for f in folds:
    if a.design=='grouped10':
        script=HERE/'project/analysis/weighted_training/run_unit.py'
        args=['--domain',d,'--fold',str(f),'--mode','qaqc','--epochs',str(a.epochs)]
        if not a.with_maps: args+=['--oof-only']
    else:
        script=HERE/'project/analysis/validation_runs/model_lane/run_validation_unit.py'
        args=['--design',a.design,'--domain',d,'--fold',str(f),'--epochs',str(a.epochs)]
    commands.append([sys.executable,'-X','utf8',str(script),*args])
(a.output/'run_plan.json').write_text(json.dumps({'design':a.design,'units':len(commands),'epochs':a.epochs,'formal_epochs':180,'full_dataset':True,'commands':commands},ensure_ascii=False,indent=2),encoding='utf8')
if a.check_only:
 print(json.dumps({'input_files':len(files),'scheduled_units':len(commands),'status':'inputs_present'}));sys.exit(0)
for cmd in commands: subprocess.run(cmd,cwd=HERE,env=env,check=True)
print('COMPLETED',len(commands),'training units.')
