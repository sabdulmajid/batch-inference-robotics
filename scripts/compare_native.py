#!/usr/bin/env python3
"""Compare fresh native runs. This measures compatibility, not annotation quality.

Run directories contain summary.json and annotations/*.caption.json (or
*.segment.json for an explicitly selected segmentation-only comparison).
Optional accounting JSON supplies complete allocation records for left/right:
{"left": [{"allocation_key": "unique-private-key", "gpus": 4,
           "elapsed_seconds": 120, "final_state": "COMPLETED"}], "right": [...]}
Never assign one complete allocation to both conditions. Omit accounting when
conditions share an allocation and their full costs cannot be assigned exactly.
Private paths and log contents are not copied into the report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON value')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique)


def finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and not value):
        raise ValueError(name + ' must be finite and ' + ('positive' if positive else 'nonnegative'))
    return float(value)


def load_run(directory, stage='all'):
    if stage not in {'all', 'segment', 'caption'}:
        raise ValueError('Select all, segment, or caption explicitly')
    directory = Path(directory)
    summary = read_json(directory / 'summary.json')
    selected = {'segment', 'caption'} if stage == 'all' else {stage}
    records = summary.get('records', [])
    if not records:
        raise ValueError('Run has no episode/stage records')
    identities, episode_requests, by_stage = {}, {}, {}
    for row in records:
        episode, request = row['id'], row['request_id']
        if not isinstance(episode, str) or not isinstance(request, str):
            raise ValueError('Episode and request identities must be strings')
        if request in identities and identities[request] != episode:
            raise ValueError('Request identity maps to multiple episodes')
        if episode in episode_requests and episode_requests[episode] != request:
            raise ValueError('Episode maps to multiple request identities in one run')
        identities[request] = episode
        episode_requests[episode] = request
        key = (episode, row['stage'])
        if key in by_stage:
            raise ValueError('Duplicate episode/stage result')
        by_stage[key] = row
    expected = set(identities.values())
    errors = []
    attempts = summary.get('attempts', [])
    attempt_keys = {(row['request_id'], row['stage'], row['number']) for row in attempts}
    if len(attempt_keys) != len(attempts):
        raise ValueError('Duplicate run attempt records')
    successful_attempts = {(row['request_id'], row['stage'], row['number']) for row in attempts
                           if row.get('state') == 'complete'}
    for episode in sorted(expected):
        for selected_stage in sorted(selected):
            row = by_stage.get((episode, selected_stage))
            if row is None:
                errors.append({'episode_id': episode, 'stage': selected_stage, 'error': 'missing_stage'})
            elif row.get('status') != 'complete' or row.get('fresh_execution') is not True:
                errors.append({'episode_id': episode, 'stage': selected_stage, 'error': 'not_fresh_complete_inference'})
            elif (row['request_id'], selected_stage, row.get('attempt')) not in successful_attempts:
                errors.append({'episode_id': episode, 'stage': selected_stage, 'error': 'missing_successful_fresh_attempt'})
            if row is not None and row.get('fresh_execution') is True:
                numbers = sorted(a['number'] for a in attempts
                                 if a['request_id'] == row['request_id'] and a['stage'] == selected_stage)
                if numbers != list(range(1, len(numbers) + 1)):
                    errors.append({'episode_id': episode, 'stage': selected_stage,
                                   'error': 'prior_attempts_are_outside_this_run_wall_measurement'})
    fresh_count = summary.get('fresh_stage_attempts')
    if type(fresh_count) is not int or fresh_count <= 0:
        errors.append({'error': 'no_fresh_attempts'})
    elif fresh_count != len(attempts):
        errors.append({'error': 'fresh_attempt_count_mismatch'})
    if summary.get('complete') is not True:
        errors.append({'error': 'run_not_complete'})
    suffix = 'segment' if stage == 'segment' else 'caption'
    outputs = {}
    for path in sorted((directory / 'annotations').glob('*.' + suffix + '.json')):
        request = path.name.removesuffix('.' + suffix + '.json')
        output = read_json(path)
        episode = output.get('episode_id', identities.get(request))
        if request not in identities or episode != identities[request]:
            raise ValueError('Artifact identity does not match its run record')
        if episode in outputs:
            raise ValueError('Duplicate episode artifacts')
        if 'request_id' in output and output['request_id'] != request:
            raise ValueError('Artifact request ID does not match its file identity')
        if suffix == 'segment':
            body = {key: value for key, value in output.items() if key != 'stage_digest'}
            if output.get('stage_digest') != digest(body):
                raise ValueError('Segmentation checkpoint digest mismatch')
        finite(output['duration_s'], 'Episode duration', positive=True)
        outputs[episode] = output
    if set(outputs) != expected:
        errors.append({'error': 'episode_artifact_coverage_mismatch',
                       'missing_episode_ids': sorted(expected - set(outputs)),
                       'unexpected_episode_ids': sorted(set(outputs) - expected)})
    return {'summary': summary, 'outputs': outputs, 'errors': errors, 'selected_stages': sorted(selected)}


def projection(output, stage):
    source = output.get('source_commit', output.get('engine', {}).get('source_commit'))
    if not source or not output.get('input_sha256') or not output.get('model_revisions') or not output.get('profile'):
        raise ValueError('Native artifact is missing source, model, or profile binding')
    selected = {'source_commit': source, 'input_sha256': output['input_sha256'],
                'model_revisions': output['model_revisions'], 'profile': output['profile'],
                'duration_s': output['duration_s']}
    if stage == 'segment':
        selected.update(yam32=output.get('yam32'), instruction_sha256=output.get('instruction_sha256'),
                        segments=output['segments'], temporal=output['temporal_audit'])
    else:
        selected['camera_roles'] = output.get('camera_roles')
        selected['segments'] = [{key: row.get(key) for key in
            ('start_s', 'end_s', 'status', 'caption', 'segmentation_provenance')} for row in output['segments']]
        selected['caption_inputs'] = output['pipeline']['caption_inputs']
        if stage == 'all':
            selected['temporal'] = output['pipeline']['temporal_windows']
    return selected


def differences(left, right, path=''):
    """Yield differing field paths and value digests; never publish raw values."""
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f'{path}.{key}' if path else str(key)
            if key not in left or key not in right:
                yield {'field': child, 'kind': 'missing_field',
                       'left_present': key in left, 'right_present': key in right}
            else:
                yield from differences(left[key], right[key], child)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            yield {'field': path, 'kind': 'length', 'left_count': len(left), 'right_count': len(right)}
        for index, (a, b) in enumerate(zip(left, right)):
            yield from differences(a, b, f'{path}[{index}]')
    elif type(left) is not type(right) or left != right:
        # JSON integer/float timestamp representations can differ while numeric
        # values are identical. Booleans must not be treated as numbers.
        if type(left) in (int, float) and type(right) in (int, float) and left == right:
            return
        yield {'field': path, 'kind': 'value', 'left_sha256': digest(left), 'right_sha256': digest(right)}


def generation_digests(paths, stage):
    selected = {'tempseg', 'captioner'} if stage == 'all' else ({'tempseg'} if stage == 'segment' else {'captioner'})
    counts = Counter()
    for path in paths:
        current_model = None
        with Path(path).open() as stream:
            for line in stream:
                if not line.startswith('FLORA_RUNTIME '):
                    continue
                event = json.loads(line[len('FLORA_RUNTIME '):])
                if event.get('event') == 'model_load_start':
                    current_model = event.get('kind')
                if event.get('event') != 'generation_start':
                    continue
                if current_model is None:
                    raise ValueError('Runtime generation record has no model-stage binding')
                if current_model not in selected:
                    continue
                value = event.get('input_tensor_digest', '')
                if not re.fullmatch('[0-9a-f]{64}', value):
                    raise ValueError('Runtime generation record lacks a valid tensor digest')
                counts[(current_model, value, event.get('input_tokens'), event.get('output_limit'))] += 1
    if not counts:
        raise ValueError('No selected-stage generation tensor digests were found')
    return counts


def allocation_gpu_hours(rows):
    if not rows:
        return None
    seen, total = set(), 0.0
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
    for row in rows:
        key = row['allocation_key']
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError('Allocation records require unique attempt identities')
        seen.add(key)
        gpus = row['gpus']
        if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus < 1 or gpus > 4:
            raise ValueError('Allocation GPU count must be between one and four')
        if row['final_state'] not in terminal:
            raise ValueError('Allocation accounting must have a final scheduler state')
        total += gpus * finite(row['elapsed_seconds'], 'Allocation elapsed time') / 3600
    return total


def performance(run, accounting=None):
    summary = run['summary']
    wall = finite(summary.get('total_cli_workflow_s', summary.get('elapsed_s')), 'Measured wall time', positive=True)
    source_hours = sum(output['duration_s'] for output in run['outputs'].values()) / 3600
    gpu_hours = allocation_gpu_hours(accounting)
    attempts = summary.get('attempts', [])
    known_times = [finite(row['elapsed_s'], 'Attempt elapsed time') for row in attempts if row.get('elapsed_s') is not None]
    eligible = not run['errors']
    return {'eligible_as_fresh_inference_benchmark': eligible,
            'condition': summary.get('condition', summary.get('mode')),
            'unique_episodes': len(run['outputs']), 'source_hours': source_hours,
            'wall_seconds': wall, 'wall_boundary': 'CLI workflow when available, otherwise engine run; startup included',
            'source_hours_per_wall_hour': source_hours / (wall / 3600) if eligible else None,
            'allocated_gpu_hours': gpu_hours,
            'source_hours_per_allocated_gpu_hour': source_hours / gpu_hours if eligible and gpu_hours else None,
            'allocation_accounting': 'provided_complete_allocations' if accounting else 'not_provided',
            'fresh_attempts': summary.get('fresh_stage_attempts', 0),
            'failed_attempts': sum(row.get('state') != 'complete' for row in attempts),
            'attempt_work_seconds_excluding_preparation': sum(known_times),
            'attempts_with_unknown_elapsed': len(attempts) - len(known_times),
            'per_episode_end_to_end_latency': 'not_measured_by_this_report'}


def compare(left_dir, right_dir, *, stage='all', left_logs=(), right_logs=(), accounting=None):
    left, right = load_run(left_dir, stage), load_run(right_dir, stage)
    issues = [{'side': side, **error} for side, run in [('left', left), ('right', right)] for error in run['errors']]
    left_ids, right_ids = set(left['outputs']), set(right['outputs'])
    if left_ids != right_ids:
        issues.append({'error': 'condition_episode_coverage_mismatch', 'left_only': sorted(left_ids - right_ids),
                       'right_only': sorted(right_ids - left_ids)})
    changes, total_changes, groups = [], 0, Counter()
    for episode in sorted(left_ids & right_ids):
        for change in differences(projection(left['outputs'][episode], stage), projection(right['outputs'][episode], stage)):
            total_changes += 1
            groups[change['field'].split('.')[0].split('[')[0]] += 1
            if len(changes) < 200:
                changes.append({'episode_id': episode, **change})
    tensor = {'status': 'not_provided', 'agreement': None}
    if bool(left_logs) != bool(right_logs):
        raise ValueError('Provide runtime logs for both conditions or neither')
    if left_logs:
        def log_identity(path):
            info = Path(path).stat()
            return info.st_dev, info.st_ino
        if {log_identity(p) for p in left_logs} & {log_identity(p) for p in right_logs}:
            raise ValueError('The same runtime log cannot represent both conditions')
        a, b = generation_digests(left_logs, stage), generation_digests(right_logs, stage)
        tensor = {'status': 'compared', 'agreement': a == b,
                  'left_generation_calls': sum(a.values()), 'right_generation_calls': sum(b.values()),
                  'left_only_calls': sum((a - b).values()), 'right_only_calls': sum((b - a).values()),
                  'matching_rule': 'model-stage-and-input-digest multiset; corpus execution order may differ',
                  'scope_limit': 'Runtime lines have no episode IDs; this checks the corpus multiset, not per-episode tensor routing'}
    accounting = accounting or {}
    if set(accounting) - {'left', 'right'}:
        raise ValueError('Accounting must identify only left and right conditions')
    assigned_left = {row['allocation_key'] for row in accounting.get('left', [])}
    assigned_right = {row['allocation_key'] for row in accounting.get('right', [])}
    if assigned_left & assigned_right:
        raise ValueError('One complete allocation cannot be charged in full to both conditions')
    metrics = {side: performance(run, accounting.get(side)) for side, run in [('left', left), ('right', right)]}
    valid = not issues
    compatible = valid and total_changes == 0 and tensor['agreement'] is not False
    return {'schema_version': 'bir_native_comparison_v1', 'comparison_stage': stage,
            'claim': 'native_input_and_output_compatibility_not_annotation_quality',
            'valid_fresh_comparison': valid, 'compatible': compatible,
            'tensor_input_agreement_verified': tensor['agreement'] is True,
            'left_episode_count': len(left_ids), 'right_episode_count': len(right_ids),
            'compared_episode_count': len(left_ids & right_ids), 'issues': issues,
            'difference_count': total_changes, 'differences_by_group': dict(groups),
            'differences': changes, 'differences_omitted': total_changes - len(changes),
            'processor_tensors': tensor, 'performance': metrics,
            'right_wall_speedup_over_left': metrics['left']['wall_seconds'] / metrics['right']['wall_seconds'] if valid else None,
            'timing_warning': 'Speedup needs matching hardware, placement, environment, and workload; allocation records alone do not prove those controls'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--stage', choices=['all', 'segment', 'caption'], default='all')
    parser.add_argument('--left-log', type=Path, action='append', default=[])
    parser.add_argument('--right-log', type=Path, action='append', default=[])
    parser.add_argument('--accounting', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare(args.left, args.right, stage=args.stage, left_logs=args.left_log,
                         right_logs=args.right_log, accounting=read_json(args.accounting) if args.accounting else None)
        code = 0 if report['compatible'] else 1
    except Exception as error:
        # Exception strings from filesystem libraries can include private paths.
        report = {'schema_version': 'bir_native_comparison_v1', 'compatible': False,
                  'valid_fresh_comparison': False, 'error_type': type(error).__name__,
                  'error': str(error) if isinstance(error, (ValueError, KeyError)) else 'Cannot read or validate comparison inputs'}
        code = 1
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end='')
    return code


if __name__ == '__main__':
    sys.exit(main())
