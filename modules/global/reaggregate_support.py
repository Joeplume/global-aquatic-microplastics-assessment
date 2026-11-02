"""Reaggregate SI5 from the adopted grid fields, without redefining C-AOA."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
g=pd.read_csv(ROOT/'inputs/prediction_grid.csv.gz',low_memory=False)
g=g[g.q50_log10.notna()]
rows=[]
domains=['reference_surface_freshwater','transition_coastal','marine']
for scope in ['all_finite','environmental_AOA_in']:
    d=g if scope=='all_finite' else g[g.aoa_status.eq('AOA_in')]
    for wb in domains+['all']:
        z=d if wb=='all' else d[d.waterbody.eq(wb)]
        supported=z.baseline_matched_sources.ge(2)
        area=z.footprint_area_wgs84_km2
        rows.append(dict(scope=scope,waterbody=wb,cells=len(z),full_support=int(supported.sum()),
          less_than_full=int((~supported).sum()),less_than_full_cells_pct=100*(~supported).mean(),
          footprint_km2=area.sum(),less_than_full_area_pct=100*area[~supported].sum()/area.sum()))
out=pd.DataFrame(rows)
out.to_csv(a.output/'SI05_RECOMPUTED.csv',index=False)
expected=json.loads((ROOT/'expected/SI05_display.json').read_text('utf8'))['rows'][1:]
for row,reference in zip(rows,expected):
    assert [row[k] for k in ['cells','full_support','less_than_full']]==[int(v) for v in reference[2:5]]
    assert abs(row['less_than_full_cells_pct']-float(reference[5]))<=.0051
    assert abs(row['footprint_km2']-float(reference[6]))<=.051
    assert abs(row['less_than_full_area_pct']-float(reference[7]))<=.0051
(a.output/'CHECK.json').write_text(json.dumps({'status':'passed','rows':len(rows),
    'scope':'Frozen adopted C-AOA/area fields reaggregated; no new boundary estimation.',
    'comparison':'Exact counts; manuscript display precision for area and percentages.'},indent=2),encoding='utf8')
print('SI5: all eight rows match the manuscript display.')
