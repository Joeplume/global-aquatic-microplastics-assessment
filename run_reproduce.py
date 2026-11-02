"""Run the model and downstream analyses."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['verify', 'results', 'results-full', 'model-fold', 'model-all'], default='results')
    p.add_argument('--output', default='outputs')
    p.add_argument('--domain', choices=['freshwater', 'transition', 'marine'], default='freshwater')
    p.add_argument('--fold', type=int, choices=range(1, 11), default=1)
    p.add_argument('--design', default='grouped10')
    args = p.parse_args()
    config = json.loads((root/'workflow.json').read_text(encoding='utf-8'))
    if args.stage not in config['stages']:
        raise SystemExit('This stage is not available in this package. See README.md.')
    requested_output = Path(args.output)
    out = (requested_output if requested_output.is_absolute() else root/requested_output).resolve()
    if any(out == root/x or (root/x) in out.parents for x in ['modules', 'reference', 'inputs']):
        raise SystemExit('Choose an output directory outside modules, reference and inputs.')
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Choose a new, empty output directory.')
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'input_manifest.json').read_text(encoding='utf-8'))
    absent, changed = [], []
    for item in manifest:
        f = root/item['path']
        if not f.is_file():
            absent.append(item['path'])
        elif args.stage == 'verify':
            h = hashlib.sha256()
            with f.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
            if h.hexdigest() != item['sha256']: changed.append(item['path'])
    if absent or changed:
        msg={'missing':absent, 'changed':changed}
        (out/'INPUT_ERRORS.json').write_text(json.dumps(msg,indent=2),encoding='utf-8')
        raise SystemExit('Input check failed. See INPUT_ERRORS.json.')
    result={'stage':args.stage, 'starting_point':config['stages'][args.stage]['starting_point'],
            'status':'running', 'jobs':[], 'scientific_checks':[]}
    summary=out/'RUN_SUMMARY.json'
    snapshot_number=0
    def save():
        nonlocal snapshot_number
        snapshot_number+=1
        target=summary if result['status']!='running' else out/f'PROGRESS_{snapshot_number:03d}.json'
        # Each closed progress file is immutable. This also avoids sync-client
        # locks while a reviewer inspects progress during a longer computation.
        with target.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2)
    save()
    env=os.environ.copy()
    env['PYTHONUTF8']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    env['MPLCONFIGDIR']=str(out/'plot_cache')
    substitutions={'root':str(root),'out':str(out),'python':sys.executable,
                   'domain':args.domain,'fold':str(args.fold),'design':args.design}
    for job in config['stages'][args.stage]['jobs']:
        argv=[v.format(**substitutions) for v in job['argv']]
        log=out/(job['name']+'.log');start=time.monotonic()
        print('Running '+job['name'],flush=True)
        with log.open('w',encoding='utf-8') as stream:
            completed=subprocess.run(argv,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        entry={'name':job['name'],'exit_code':completed.returncode,
               'elapsed_seconds':round(time.monotonic()-start,2),'log':log.name}
        result['jobs'].append(entry)
        if completed.returncode:
            result['status']='failed';save()
            raise SystemExit(f'{job["name"]} failed. Read {log}; no success is inferred from partial files.')
        for name in job.get('reports',[]):
            report = Path(name.format(**substitutions))
            if not report.is_file():
                result['status']='failed';save()
                raise SystemExit('Expected execution report is missing: '+str(report))
            result['scientific_checks'].append({'job':job['name'], 'report':str(report.relative_to(out))})
        save()
    result['status']='completed'
    result['interpretation']=config['stages'][args.stage]['interpretation']
    save()
    print('Completed. '+result['interpretation'],flush=True)


if __name__=='__main__':
    main()
