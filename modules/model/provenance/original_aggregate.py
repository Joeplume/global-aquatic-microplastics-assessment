"""Paired QA/QC evaluation; frozen inputs and manuscript products remain unchanged."""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from common import OUT, H, GROUPED, DOMAINS, sha, write_json

REGION_MAP = H / 'validation/02_CAOA相对AOA增量验证_20260901/data/CELL_REGION_ASSIGNMENT.csv.gz'
COUNTRY_SET = H / 'validation/05_既有结论存活检验_20260902/data/COUNTRY_RANK_BOOTSTRAP_PRIMARY_70.csv'
POINT_COLUMNS = {
    'environment': 'predicted_q50_log10',
    'conditional_available': 'known_source_conditional_q50_log10',
    'operational_known_else_environment': 'operational_q50_log10',
}
SD_COLUMNS = {
    'environment': 'future_observation_sd_log10',
    'conditional_available': 'known_source_conditional_future_sd_log10',
    'operational_known_else_environment': 'operational_future_sd_log10',
}
GRID_NUMERIC = ['latent_q50_log10', 'latent_sd_log10', 'future_observation_sd_log10']
IDENTITY = ['prediction_id', 'longitude', 'latitude', 'waterbody']


def boolean(series):
    values = series.astype(str).str.lower()
    if not values.isin(['true', 'false', '1', '0']).all():
        raise ValueError('Invalid boolean values: ' + repr(values.unique().tolist()))
    return values.isin(['true', '1']).to_numpy()


def numeric(series):
    return pd.to_numeric(series, errors='coerce').to_numpy(float)


def wmean(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else math.nan


def quantile(values, p):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, p)) if len(values) else math.nan


def cell_area(latitude):
    latitude = np.deg2rad(np.asarray(latitude, float))
    return 6371.0088 ** 2 * np.deg2rad(.5) * (np.sin(latitude + np.deg2rad(.25)) - np.sin(latitude - np.deg2rad(.25)))


def file_record(path, hash_file=False):
    result = {'path': str(path), 'exists': path.exists()}
    if path.exists():
        result.update(bytes=path.stat().st_size, modified_ns=path.stat().st_mtime_ns)
        if hash_file:
            result['sha256'] = sha(path)
    return result


def save_csv(frame, path):
    pd.DataFrame(frame).to_csv(path, index=False, encoding='utf-8-sig')


def uncapped_factor(audit, conditional):
    if conditional:
        branch = audit['known_source_observation_layer']
        raw = branch.get('aware_unclipped_interval_multiplier')
        used = branch.get('aware_interval_multiplier')
        if branch.get('outer_heldout_response_used') is not False:
            raise ValueError('Conditional interval calibration is not confirmed training-only')
    else:
        branch = audit['observation_scale_model']
        raw = branch.get('unclipped_interval_multiplier')
        used = branch.get('interval_multiplier_from_inner_source_holdout')
        if audit.get('observation_scale_calibration_predictions', {}).get('outer_heldout_response_used') is not False:
            raise ValueError('Environmental observation interval calibration is not confirmed training-only')
    if raw is None or used is None or not np.isfinite(float(raw)) or not np.isfinite(float(used)) or float(raw) <= 0 or float(used) <= 0:
        raise ValueError('Missing or invalid interval multipliers; uncapped intervals cannot be inferred')
    return float(raw) / float(used)


def load_oof(unit, domain, fold, mode, expected):
    folder = unit / 'S0/protocol_aware' / DOMAINS[domain]
    frame = pd.read_csv(folder / 'H26_UNIT_HELDOUT.csv.gz', low_memory=False)
    ledger = pd.read_csv(unit / 'RECORD_WEIGHTS.csv.gz', low_memory=False)
    audit = json.loads((folder / 'H26_UNIT_AUDIT.json').read_text(encoding='utf-8'))
    if not frame.stable_record_id.is_unique or not ledger.stable_record_id.is_unique:
        raise ValueError('Nonunique record IDs')
    if set(frame.stable_record_id) != set(expected.stable_record_id):
        raise ValueError('Heldout record population differs from persisted folds')
    if not frame.outer_validation_fold.eq(fold).all():
        raise ValueError('Fold labels differ')
    reference = expected.set_index('stable_record_id').loc[frame.stable_record_id]
    if 'validation_site_event_key' in reference and not np.array_equal(reference.validation_site_event_key.to_numpy(), frame.validation_site_event_key.to_numpy()):
        raise ValueError('Site-event keys differ from persisted folds')
    if not frame.waterbody.eq(DOMAINS[domain]).all():
        raise ValueError('Waterbody labels differ')
    for name in ['base_source_weight', 'quality_multiplier', 'quality_status', 'raw_quality_factor']:
        frame[name] = frame.stable_record_id.map(ledger.set_index('stable_record_id')[name])
    if not (np.isfinite(frame.base_source_weight) & frame.base_source_weight.gt(0)).all():
        raise ValueError('Missing or nonpositive original source-equal weights')
    known = boolean(frame.known_source_available)
    frame['known_source_available'] = known
    frame['mode'] = mode
    frame['domain'] = domain
    for target in ['environment', 'conditional_available']:
        sd = SD_COLUMNS[target]
        frame[sd + '_uncapped_training_only'] = numeric(frame[sd]) * uncapped_factor(audit, target == 'conditional_available')
    frame['operational_q50_log10'] = np.where(known, frame.known_source_conditional_q50_log10, frame.predicted_q50_log10)
    frame['operational_future_sd_log10'] = np.where(known, frame.known_source_conditional_future_sd_log10, frame.future_observation_sd_log10)
    frame['operational_future_sd_log10_uncapped_training_only'] = np.where(
        known, frame.known_source_conditional_future_sd_log10_uncapped_training_only,
        frame.future_observation_sd_log10_uncapped_training_only)
    return frame


def strata(frame):
    yield 'all_records', frame
    yield 'known_source', frame.loc[frame.known_source_available]
    yield 'unknown_source', frame.loc[~frame.known_source_available]


def point_metrics(frame, prediction, weights):
    y, p = numeric(frame.observed_log10), numeric(frame[prediction])
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(weights) & (weights > 0)
    y, p, weights = y[valid], p[valid], weights[valid]
    result = {'records_eligible': len(frame), 'records_evaluated': int(valid.sum()), 'records_na': int((~valid).sum())}
    result.update(r2=math.nan, rmse_log10=math.nan, mae_log10=math.nan, bias_log10=math.nan)
    if not len(y):
        return {**result, 'status': 'no_evaluable_records'}
    e = p - y
    mse = wmean(e ** 2, weights)
    variance = wmean((y - wmean(y, weights)) ** 2, weights)
    result.update(r2=1 - mse / variance if variance > 0 else math.nan,
                  rmse_log10=math.sqrt(mse), mae_log10=wmean(np.abs(e), weights), bias_log10=wmean(e, weights))
    return {**result, 'status': 'ok' if variance > 0 else 'constant_response_r2_undefined'}


def interval_metrics(frame, prediction, sd_column, weights):
    y, p, sd = numeric(frame.observed_log10), numeric(frame[prediction]), numeric(frame[sd_column])
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(sd) & (sd > 0) & np.isfinite(weights) & (weights > 0)
    y, p, sd, weights = y[valid], p[valid], sd[valid], weights[valid]
    result = {'records_eligible': len(frame), 'records_evaluated': int(valid.sum()), 'records_na': int((~valid).sum()),
              'coverage95': math.nan, 'mean_width95_log10': math.nan, 'wis_log10': math.nan}
    if not len(y):
        return {**result, 'status': 'no_evaluable_records'}
    wis = .5 * np.abs(y - p)
    for alpha in [.5, .2, .1, .05]:
        z = norm.ppf(1 - alpha / 2)
        lo, hi = p - z * sd, p + z * sd
        wis += alpha / 2 * (hi - lo + 2 / alpha * np.maximum(lo - y, 0) + 2 / alpha * np.maximum(y - hi, 0))
    result.update(coverage95=wmean(np.abs(y - p) <= norm.ppf(.975) * sd, weights),
                  mean_width95_log10=wmean(2 * norm.ppf(.975) * sd, weights), wis_log10=wmean(wis / 4.5, weights))
    return {**result, 'status': 'ok'}


def evaluate_oof(frame):
    points, intervals = [], []
    for domain in ['all', *DOMAINS]:
        local = frame if domain == 'all' else frame.loc[frame.domain.eq(domain)]
        for stratum, part in strata(local):
            for target, prediction in POINT_COLUMNS.items():
                eligible = part.loc[part.known_source_available] if target == 'conditional_available' else part
                for weighting in ['record_equal', 'base_source_equal']:
                    weights = np.ones(len(eligible)) if weighting == 'record_equal' else numeric(eligible.base_source_weight)
                    label = 'base_source_equal_domain_normalized' if domain == 'all' and weighting == 'base_source_equal' else weighting
                    keys = {'mode': frame['mode'].iloc[0], 'domain': domain, 'stratum': stratum, 'prediction_target': target,
                            'weighting': label, 'sources': eligible.source_group.nunique(), 'folds': eligible.outer_validation_fold.nunique(),
                            'truth': 'heldout_harmonized_observation', 'unit': 'log10(items/m3)'}
                    points.append({**keys, **point_metrics(eligible, prediction, weights)})
                    for label, suffix in [('raw_persisted', ''), ('uncapped_training_only', '_uncapped_training_only')]:
                        intervals.append({**keys, 'interval_version': label, 'interval_target': 'future_observation',
                                          **interval_metrics(eligible, prediction, SD_COLUMNS[target] + suffix, weights)})
    return points, intervals


def paired_oof(base, other, mode):
    base = base.sort_values('stable_record_id').reset_index(drop=True)
    other = other.sort_values('stable_record_id').reset_index(drop=True)
    if not np.array_equal(base.stable_record_id, other.stable_record_id):
        raise ValueError('Paired OOF ID mismatch')
    for field in ['waterbody', 'source_group', 'outer_validation_fold', 'known_source_available', 'validation_site_event_key']:
        if not np.array_equal(base[field], other[field]):
            raise ValueError('Paired OOF mismatch: ' + field)
    for field in ['observed_log10', 'base_source_weight']:
        if not np.allclose(numeric(base[field]), numeric(other[field]), rtol=0, atol=1e-10, equal_nan=True):
            raise ValueError('Paired OOF mismatch: ' + field)
    output = base[['stable_record_id', 'source_group', 'domain', 'waterbody', 'outer_validation_fold', 'known_source_available', 'observed_log10', 'base_source_weight', 'quality_status']].copy()
    output['comparison_mode'] = mode
    rows = []
    for target, prediction in POINT_COLUMNS.items():
        delta = numeric(other[prediction]) - numeric(base[prediction])
        if target == 'conditional_available':
            delta[~base.known_source_available.to_numpy()] = np.nan
        output[target + '_delta_log10'] = delta
        for domain in ['all', *DOMAINS]:
            local = output if domain == 'all' else output.loc[output.domain.eq(domain)]
            for stratum, part in strata(local):
                values = numeric(part[target + '_delta_log10'])
                rows.append({'comparison_mode': mode, 'domain': domain, 'stratum': stratum, 'prediction_target': target,
                             'records_eligible': len(part), 'records_finite': int(np.isfinite(values).sum()),
                             'median_delta_log10': quantile(values, .5), 'median_absolute_delta_log10': quantile(np.abs(values), .5),
                             'p95_absolute_delta_log10': quantile(np.abs(values), .95), 'max_absolute_delta_log10': float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else math.nan})
    return output, rows


def modal_strings(arrays):
    stack = np.stack(arrays)
    labels = sorted(np.unique(stack).tolist())
    counts = np.stack([(stack == label).sum(axis=0) for label in labels])
    return np.asarray(labels, dtype=object)[np.argmax(counts, axis=0)]


def aggregate_grid(units, domain, expected_folds):
    reference = None
    fields = {column: [] for column in GRID_NUMERIC}
    status = {'aoa_status': [], 'caoa_status': []}
    for fold, unit in units:
        path = unit / 'S0/protocol_aware' / DOMAINS[domain] / 'H26_UNIT_GRID.csv.gz'
        frame = pd.read_csv(path, usecols=[*IDENTITY, *GRID_NUMERIC, *status], low_memory=False).sort_values('prediction_id').reset_index(drop=True)
        if not frame.prediction_id.is_unique or not frame.waterbody.eq(DOMAINS[domain]).all():
            raise ValueError('Invalid grid identity or domain')
        if reference is None:
            reference = frame[IDENTITY].copy()
        elif not reference.equals(frame[IDENTITY]):
            raise ValueError('Grid identity, coordinates or waterbody differ across folds')
        for field in fields:
            fields[field].append(numeric(frame[field]))
        for field in status:
            status[field].append(frame[field].fillna('NA').astype(str).to_numpy())
    if len(units) != expected_folds:
        raise ValueError('Incorrect number of grid folds')
    q, latent, future = [np.stack(fields[name]) for name in GRID_NUMERIC]
    result = reference
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result['q50_log10'] = np.nanmedian(q, axis=0)
        result['finite_prediction_folds'] = np.isfinite(q).sum(axis=0)
        fold_sd = np.nanstd(q, axis=0, ddof=1)
        fold_sd = np.where(np.isfinite(fold_sd), fold_sd, 0)
        result['q50_fold_sd_log10'] = fold_sd
        result['latent_state_sd_log10'] = np.sqrt(np.nanmean(latent ** 2, axis=0) + fold_sd ** 2)
        result['future_observation_sd_log10'] = np.nanmedian(future, axis=0)
    for field, arrays in status.items():
        result[field] = modal_strings(arrays)
    result['domain'] = domain
    result['folds'] = expected_folds
    result['cell_area_km2'] = cell_area(result.latitude)
    return result


def grid_pair_summary(part, keys):
    valid = part.baseline_finite & part.comparison_finite
    local = part.loc[valid]
    row = {**keys, 'grid_cells_total': len(part), 'paired_finite_cells': len(local),
           'baseline_only_finite_cells': int((part.baseline_finite & ~part.comparison_finite).sum()),
           'comparison_only_finite_cells': int((~part.baseline_finite & part.comparison_finite).sum()),
           'paired_grid_area_km2': float(local.cell_area_km2.sum()), 'area_definition': 'full_0.5_degree_grid_cell_area_not_water_surface_area'}
    if local.empty:
        return {**row, 'status': 'no_paired_finite_cells'}
    baseline, comparison, weights = numeric(local.baseline_q50_log10), numeric(local.comparison_q50_log10), numeric(local.cell_area_km2)
    diff = comparison - baseline
    bmean, cmean = wmean(baseline, weights), wmean(comparison, weights)
    btop, ctop = baseline >= np.quantile(baseline, .75), comparison >= np.quantile(comparison, .75)
    intersection, union = btop & ctop, btop | ctop
    rank = float(spearmanr(baseline, comparison).statistic) if len(local) > 1 and np.ptp(baseline) > 0 and np.ptp(comparison) > 0 else math.nan
    row.update(status='ok', baseline_geometric_mean_items_m3=10 ** bmean, comparison_geometric_mean_items_m3=10 ** cmean,
               geometric_mean_relative_change_pct=100 * (10 ** (cmean - bmean) - 1), median_absolute_delta_log10=quantile(np.abs(diff), .5),
               p95_absolute_delta_log10=quantile(np.abs(diff), .95), area_weighted_delta_log10=wmean(diff, weights),
               spearman_grid_rank=rank, top_quartile_definition='within_comparison_population_unweighted_cell_quantile_ties_retained',
               baseline_top_quartile_cells=int(btop.sum()), comparison_top_quartile_cells=int(ctop.sum()),
               top_quartile_jaccard_cells=float(intersection.sum() / union.sum()),
               top_quartile_jaccard_area=float(weights[intersection].sum() / weights[union].sum()),
               baseline_top_quartile_retained_area_fraction=float(weights[intersection].sum() / weights[btop].sum()))
    for field in ['latent_state_sd_log10', 'future_observation_sd_log10']:
        row['baseline_area_weighted_' + field] = wmean(numeric(local['baseline_' + field]), weights)
        row['comparison_area_weighted_' + field] = wmean(numeric(local['comparison_' + field]), weights)
        row['area_weighted_delta_' + field] = wmean(numeric(local['comparison_' + field]) - numeric(local['baseline_' + field]), weights)
    return row


def compare_grids(base, other, mode, mapping, countries, issues):
    base, other = [x.sort_values('prediction_id').reset_index(drop=True) for x in [base, other]]
    if not base[IDENTITY].equals(other[IDENTITY]):
        raise ValueError('Paired grid IDs or coordinates differ: ' + mode)
    out = base[[*IDENTITY, 'domain', 'cell_area_km2', 'folds']].copy()
    out['comparison_mode'] = mode
    for field in ['q50_log10', 'latent_state_sd_log10', 'future_observation_sd_log10', 'finite_prediction_folds']:
        out['baseline_' + field] = base[field].to_numpy()
        out['comparison_' + field] = other[field].to_numpy()
        out['delta_' + field] = numeric(other[field]) - numeric(base[field])
    out['baseline_finite'] = np.isfinite(base.q50_log10)
    out['comparison_finite'] = np.isfinite(other.q50_log10)
    out['baseline_cross_study_supported'] = base.caoa_status.eq('CAOA_in').to_numpy()
    out['baseline_environment_supported'] = base.aoa_status.eq('AOA_in').to_numpy()
    mismatch = int((out.baseline_finite != out.comparison_finite).sum())
    if mismatch:
        issues.append({'kind': 'finite_grid_mask_mismatch', 'mode': mode, 'cells': mismatch})
    for status in ['caoa_status', 'aoa_status']:
        changed = int(base[status].ne(other[status]).sum())
        if changed:
            issues.append({'kind': 'support_annotation_changed', 'annotation': status, 'mode': mode, 'cells': changed, 'comparison_mask': 'baseline_frozen'})
    if mapping is not None:
        out = out.merge(mapping[['prediction_id', 'country', 'continent']], on='prediction_id', how='left', validate='one_to_one')
    summaries, regions, country_rows = [], [], []
    for domain in ['all', *DOMAINS]:
        frame = out if domain == 'all' else out.loc[out.domain.eq(domain)]
        for support, part in [('all_finite', frame), ('baseline_CAOA_in', frame.loc[frame.baseline_cross_study_supported])]:
            summaries.append(grid_pair_summary(part, {'comparison_mode': mode, 'domain': domain, 'support_scope': support}))
            if mapping is not None:
                for name, regional in part.groupby('continent', dropna=False):
                    regions.append(grid_pair_summary(regional, {'comparison_mode': mode, 'domain': domain, 'support_scope': support,
                                                                'region_type': 'frozen_continent_label', 'region': name}))
            if domain == 'freshwater' and mapping is not None and countries is not None:
                for country in countries:
                    country_rows.append(grid_pair_summary(part.loc[part.country.eq(country)],
                        {'comparison_mode': mode, 'domain': domain, 'support_scope': support, 'country': country,
                         'country_set': 'existing_primary_70_no_new_eligibility_filter'}))
    country_frame = pd.DataFrame(country_rows)
    if len(country_frame):
        for prefix in ['baseline', 'comparison']:
            country_frame[prefix + '_rank_desc'] = country_frame.groupby(['comparison_mode', 'support_scope'])[prefix + '_geometric_mean_items_m3'].rank(ascending=False, method='min')
        country_frame['rank_change'] = country_frame.comparison_rank_desc - country_frame.baseline_rank_desc
    return out, summaries, regions, country_frame


def metric_differences(rows, measures, extra_keys):
    frame = pd.DataFrame(rows)
    keys = ['domain', 'stratum', 'prediction_target', 'weighting', *extra_keys]
    baseline = frame.loc[frame['mode'].eq('baseline'), [*keys, 'records_evaluated', *measures]]
    comparisons = []
    for mode in frame['mode'].unique():
        if mode == 'baseline':
            continue
        other = frame.loc[frame['mode'].eq(mode), [*keys, 'records_evaluated', *measures]]
        pair = baseline.merge(other, on=keys, how='outer', validate='one_to_one', suffixes=('_baseline', '_comparison'))
        pair['comparison_mode'] = mode
        pair['same_evaluated_count'] = pair.records_evaluated_baseline.eq(pair.records_evaluated_comparison)
        for name in measures:
            pair['delta_' + name] = pair[name + '_comparison'] - pair[name + '_baseline']
        comparisons.append(pair)
    return pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()


def historical_refit_comparison(baseline):
    historical = pd.read_csv(GROUPED, low_memory=False).set_index('stable_record_id')
    if not historical.index.is_unique or not baseline.stable_record_id.isin(historical.index).all():
        raise ValueError('Historical FINAL10 does not uniquely cover the new baseline record IDs')
    historical = historical.loc[baseline.stable_record_id].reset_index()
    baseline = baseline.reset_index(drop=True)
    known_old = boolean(historical.known_source_available)
    source_labels_match = np.array_equal(baseline.source_group, historical.source_group)
    fold_column = 'formal_fold' if 'formal_fold' in historical else 'outer_validation_fold'
    folds_match = np.array_equal(baseline.outer_validation_fold, historical[fold_column])
    targets_match = np.allclose(numeric(baseline.observed_log10), numeric(historical.observed_log10), equal_nan=True, atol=1e-6, rtol=0)
    rows = []
    for target, column in [('environment', 'predicted_q50_log10'), ('conditional_available', 'known_source_conditional_q50_log10')]:
        delta = numeric(baseline[column]) - numeric(historical[column])
        if target == 'conditional_available':
            delta[~(known_old & baseline.known_source_available.to_numpy())] = np.nan
        for domain in ['all', *DOMAINS]:
            in_domain = np.ones(len(baseline), bool) if domain == 'all' else baseline.domain.eq(domain).to_numpy()
            for stratum, source_mask in [('all_records', np.ones(len(baseline), bool)),
                                         ('known_source', baseline.known_source_available.to_numpy()),
                                         ('unknown_source', ~baseline.known_source_available.to_numpy())]:
                mask = in_domain & source_mask
                values = delta[mask]
                valid = np.isfinite(values)
                rows.append({'domain': domain, 'stratum': stratum, 'prediction_target': target,
                             'comparison': 'new_baseline_without_QAQC_minus_historical_FINAL10_not_QAQC_effect',
                             'records_in_scope': int(mask.sum()), 'records_paired_finite': int(valid.sum()),
                             'sources_match': source_labels_match, 'folds_match': folds_match, 'observed_response_match': targets_match,
                             'known_source_flag_changes': int((known_old[mask] != baseline.known_source_available.to_numpy()[mask]).sum()),
                             'median_delta_log10': quantile(values, .5), 'median_absolute_delta_log10': quantile(np.abs(values), .5),
                             'p95_absolute_delta_log10': quantile(np.abs(values), .95),
                             'rmse_between_predictions_log10': math.sqrt(float(np.mean(values[valid] ** 2))) if valid.any() else math.nan,
                             'max_absolute_delta_log10': float(np.max(np.abs(values[valid]))) if valid.any() else math.nan,
                             'exact_within_1e6': bool(np.all(np.abs(values[valid]) <= 1e-6)) if valid.any() else None})
    return rows


def run(args):
    modes = list(dict.fromkeys(['baseline', *args.modes]))
    folds = [1] if args.smoke else list(range(1, 11))
    destination = OUT / ('aggregate_smoke' if args.smoke else 'aggregate_results')
    destination.mkdir(exist_ok=True)
    audit = {'status': 'started', 'smoke_only': args.smoke, 'modes': modes, 'expected_folds': folds, 'issues': [], 'inputs': [],
             'frozen_results_modified': False, 'evaluation_weights': 'record_equal or unchanged per-record base_source_weight; never QAQC-adjusted weights',
             'pooled_weighting': 'base_source_weight retains each domain training-table normalization; not newly renormalized to equal waterbodies',
             'interval_target': 'heldout future observations only; no coverage or WIS claims for latent environmental truth',
             'WIS_definition': 'Gaussian central intervals alpha=.5,.2,.1,.05, median weight=.5, normalization=4.5',
             'grid_aggregation': 'median of fold environmental q50; latent variance=mean(within-fold latent SD squared)+sample variance of fold q50; future SD=median across folds',
             'grid_coverage_is_not_accuracy': True, 'interval_aggregate_is_not_recalibrated_mixture': True,
             'support_masks': 'baseline annotation fixed for all paired summaries', 'country_scope': 'existing frozen primary 70 countries, freshwater only',
             'script': file_record(Path(__file__), True), 'contract': file_record(OUT / 'CONTRACT.json', True),
             'fold_contract': file_record(OUT / 'FROZEN_GROUPED10_FOLDS.csv.gz', True),
             'historical_final10': file_record(GROUPED, True),
             'region_mapping': file_record(REGION_MAP, True), 'country_set': file_record(COUNTRY_SET, True)}
    audit_path = destination / 'AGGREGATE_AUDIT.json'
    write_json(audit_path, audit)
    signature = {'contract_sha256': sha(OUT / 'CONTRACT.json'), 'common_sha256': sha(OUT / 'common.py'),
                 'runner_sha256': sha(OUT / 'run_unit.py'), 'epochs': 180, 'grid_exported': True}
    unit_rows, units = [], {}
    for mode in modes:
        for domain in DOMAINS:
            for fold in folds:
                unit = OUT / ('smoke' if args.smoke else 'units') / mode / domain / f'fold_{fold}'
                folder = unit / 'S0/protocol_aware' / DOMAINS[domain]
                files = [unit / 'DONE.json', unit / 'RECORD_WEIGHTS.csv.gz', folder / 'H26_UNIT_HELDOUT.csv.gz', folder / 'H26_UNIT_GRID.csv.gz', folder / 'H26_UNIT_AUDIT.json']
                missing = [str(path) for path in files if not path.exists()]
                done = json.loads(files[0].read_text(encoding='utf-8')) if files[0].exists() else {}
                invalid = []
                if done:
                    if (done.get('mode'), done.get('domain'), done.get('fold')) != (mode, domain, fold):
                        invalid.append('DONE_identity_mismatch')
                    if done.get('posterior_draws') != 24:
                        invalid.append('posterior_draws_not_24')
                    if not args.smoke and done.get('run_signature') != signature:
                        invalid.append('run_signature_mismatch')
                    if not args.smoke and done.get('epochs_max') != 180:
                        invalid.append('formal_epochs_max_not_180')
                complete = not missing and not invalid and done.get('status') == 'completed' and done.get('grid_exported') is True
                unit_rows.append({'mode': mode, 'domain': domain, 'fold': fold, 'complete': complete,
                                  'missing_files': '|'.join(missing), 'invalid_reasons': '|'.join(invalid), 'done_status': done.get('status'),
                                  'grid_exported': done.get('grid_exported'), 'epochs_max': done.get('epochs_max')})
                units[(mode, domain, fold)] = unit
                audit['inputs'].extend(file_record(path) for path in files)
    save_csv(unit_rows, destination / 'UNIT_COMPLETENESS.csv')
    audit['units_expected'] = len(unit_rows)
    audit['units_completed'] = sum(row['complete'] for row in unit_rows)
    if audit['units_completed'] != audit['units_expected']:
        audit['status'] = 'incomplete'
        audit['outputs_current_run'] = ['UNIT_COMPLETENESS.csv']
        write_json(audit_path, audit)
        print(json.dumps({k: audit[k] for k in ['status', 'units_expected', 'units_completed']}), flush=True)
        return 2
    frozen = pd.read_csv(OUT / 'FROZEN_GROUPED10_FOLDS.csv.gz', low_memory=False)
    mapping = pd.read_csv(REGION_MAP, low_memory=False) if REGION_MAP.exists() else None
    countries = sorted(pd.read_csv(COUNTRY_SET).country.dropna().unique()) if COUNTRY_SET.exists() else None
    if mapping is not None and not mapping.prediction_id.is_unique:
        raise ValueError('Frozen regional mapping is not unique')
    if mapping is None or countries is None:
        audit['issues'].append({'kind': 'regional_mapping_unavailable', 'region_map': mapping is not None, 'country_set': countries is not None})
    points, intervals, frames, grids = [], [], {}, {}
    for mode in modes:
        parts, grid_parts = [], []
        for domain, waterbody in DOMAINS.items():
            for fold in folds:
                expected = frozen.loc[frozen.waterbody.eq(waterbody) & frozen.formal_fold.eq(fold)]
                parts.append(load_oof(units[(mode, domain, fold)], domain, fold, mode, expected))
            grid_parts.append(aggregate_grid([(fold, units[(mode, domain, fold)]) for fold in folds], domain, len(folds)))
        frame = pd.concat(parts, ignore_index=True)
        if not frame.stable_record_id.is_unique:
            raise ValueError('Duplicate pooled OOF IDs for ' + mode)
        frames[mode] = frame
        grids[mode] = pd.concat(grid_parts, ignore_index=True)
        p, i = evaluate_oof(frame)
        points.extend(p)
        intervals.extend(i)
        frame.to_csv(destination / f'OOF_{mode}.csv.gz', index=False)
        grids[mode].to_csv(destination / f'GRID10_{mode}.csv.gz', index=False)
    save_csv(points, destination / 'POINT_METRICS.csv')
    save_csv(intervals, destination / 'OBSERVATION_INTERVAL_METRICS.csv')
    save_csv(historical_refit_comparison(frames['baseline']), destination / 'HISTORICAL_REFIT_COMPARISON.csv')
    save_csv(metric_differences(points, ['r2', 'rmse_log10', 'mae_log10', 'bias_log10'], []), destination / 'PAIRED_POINT_METRIC_DIFFERENCES.csv')
    save_csv(metric_differences(intervals, ['coverage95', 'mean_width95_log10', 'wis_log10'], ['interval_version']), destination / 'PAIRED_INTERVAL_METRIC_DIFFERENCES.csv')
    oof_delta, grid_summary, region_summary, country_summary = [], [], [], []
    for mode in modes[1:]:
        differences, rows = paired_oof(frames['baseline'], frames[mode], mode)
        differences.to_csv(destination / f'PAIRED_OOF_{mode}_minus_baseline.csv.gz', index=False)
        oof_delta.extend(rows)
        changes, summary, regions, country = compare_grids(grids['baseline'], grids[mode], mode, mapping, countries, audit['issues'])
        changes.to_csv(destination / f'PAIRED_GRID_{mode}_minus_baseline.csv.gz', index=False)
        grid_summary.extend(summary)
        region_summary.extend(regions)
        if len(country):
            country_summary.append(country)
    save_csv(oof_delta, destination / 'OOF_CHANGE_SUMMARY.csv')
    save_csv(grid_summary, destination / 'GRID_CHANGE_SUMMARY.csv')
    save_csv(region_summary, destination / 'CONTINENT_CHANGE_SUMMARY.csv')
    save_csv(pd.concat(country_summary, ignore_index=True) if country_summary else [], destination / 'FIXED_COUNTRY_CHANGE_SUMMARY.csv')
    audit['records_per_mode'] = {mode: len(frame) for mode, frame in frames.items()}
    audit['grid_rows_per_mode'] = {mode: len(frame) for mode, frame in grids.items()}
    audit['status'] = ('smoke_passed_not_full_experiment' if args.smoke else 'complete') if not audit['issues'] else 'completed_with_validation_failures'
    audit['outputs_current_run'] = [path.name for path in destination.iterdir() if path.is_file() and path.name != audit_path.name]
    write_json(audit_path, audit)
    print(json.dumps({k: audit[k] for k in ['status', 'units_expected', 'units_completed', 'records_per_mode', 'grid_rows_per_mode']}, ensure_ascii=False), flush=True)
    return 0 if not audit['issues'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs='+', default=['qaqc'], choices=['baseline', 'uniform', 'qaqc', 'missing_low', 'missing_high'])
    parser.add_argument('--smoke', action='store_true', help='Only smoke/fold_1; never labelled as a complete experiment')
    arguments = parser.parse_args()
    try:
        raise SystemExit(run(arguments))
    except Exception as error:
        destination = OUT / ('aggregate_smoke' if arguments.smoke else 'aggregate_results')
        destination.mkdir(exist_ok=True)
        path = destination / 'AGGREGATE_AUDIT.json'
        failed = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        failed.update(status='failed', error_type=type(error).__name__, error=str(error), smoke_only=arguments.smoke)
        write_json(path, failed)
        raise
