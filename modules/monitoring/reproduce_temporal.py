"""Run the actual 5000-draw temporal simulation with portable input/output paths."""
from pathlib import Path
import argparse,json
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--out','--output',type=Path,default=ROOT/'temporal_outputs');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    for sub in ['results','audit']:(a.out/sub).mkdir(exist_ok=True)
    src=ROOT/'src/original/time_mc.py';s=src.read_text('utf-8-sig')
    s=s.replace("ROOT=Path(__file__).resolve().parents[1]; W=ROOT.parents[1]",'ROOT=OUTPUT; W=ROOT')
    s=s.replace("W/'h26/04_MDC与监测政策/01_source_mdc/H26_MDC_VARIANCE_COMPONENTS.csv'",'COMPONENT_FILE')
    ns=dict(__file__=str(src),__name__='portable_temporal',OUTPUT=a.out,COMPONENT_FILE=ROOT/'inputs/management/H26_MDC_VARIANCE_COMPONENTS.csv')
    exec(compile(s,str(src),'exec'),ns);ns['run']()
    old=pd.read_csv(ROOT/'reference/TIME_PRIMARY_30PCT.csv')
    actual=pd.read_csv(a.out/'results/TIME_PRIMARY_30PCT.csv')
    keys=['waterbody','scenario','method','qa_variance_reduction']
    actual=actual.set_index(keys).sort_index();old=old.set_index(keys).sort_index()
    assert actual.required_sites.fillna(-1).equals(old.required_sites.fillna(-1))
    (a.out/'REPRODUCTION_REPORT.json').write_text(json.dumps(dict(actual_temporal_mc_rerun=True,reps=5000,targets_equal=True,scope='conditional covariance simulation; not measured longitudinal validation'),indent=2),encoding='utf8')
if __name__=='__main__':main()
