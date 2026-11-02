from recompute_variance import ROOT,PROJECT
from pathlib import Path
import hashlib,json
p=PROJECT/'src/modeling/model_comparison/D_monitoring_simulation/src/time_mc.py'
out=ROOT/'qaqc/time_mc'
for folder in ['results','audit']:(out/folder).mkdir(parents=True,exist_ok=True)
s=p.read_text(encoding='utf-8-sig')
s=s.replace("ROOT=Path(__file__).resolve().parents[1]; W=ROOT.parents[1]",f"ROOT=Path({str(out)!r}); W=ROOT.parents[1]")
s=s.replace("W/'h26/04_MDC与监测政策/01_source_mdc/H26_MDC_VARIANCE_COMPONENTS.csv'",f"Path({str(ROOT/'qaqc/H26_MDC_VARIANCE_COMPONENTS.csv')!r})")
ns={'__file__':str(p),'__name__':'isolated_time_mc'}
exec(compile(s,str(p),'exec'),ns)
ns['run']()
(out/'RECALCULATION_MANIFEST.json').write_text(json.dumps({'original_source':str(p),'original_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'changed_input':'QAQC domain variance components only','unchanged':['5000 paired Monte Carlo draws','seeds','source clustering','six occasions','variance-reduction definitions','site limits','evaluation formulas']},indent=2),encoding='utf-8')
