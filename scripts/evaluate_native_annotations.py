#!/usr/bin/env python3
"""Bounded AI reference-event and temporal review of native annotations.

Cards and mappings are private. Scores use independent dataset references.
This is neither human evaluation nor the official WGO scoring protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    from .evaluate_adaptive import RUBRIC, RUBRIC_SHA256, digest, finite, positive_rating, reference_rows
    from .native_benchmark import verify_gate_c
    from .prepare_wgo import WGO
except ImportError:
    from evaluate_adaptive import RUBRIC, RUBRIC_SHA256, digest, finite, positive_rating, reference_rows
    from native_benchmark import verify_gate_c
    from prepare_wgo import WGO
from bir.cli import atomic_json
from bir.evaluation import Interval, interval_metrics, temporal_iou
from bir.manifest import file_sha256


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def content_contract_digest(contract):
    """Ignore original file-byte receipts, while retaining every scientific input."""
    return digest({key: value for key, value in contract.items() if key not in {
        'contract_sha256', 'content_contract_sha256', 'initial_manifest_sha256',
        'private_reference_input_sha256'}})


def _reference_binding(row, duration):
    if not isinstance(row.get('instruction'), str) or not _sha(row.get('source_sha256')):
        raise ValueError('Independent reference input identity is invalid')
    targets = []
    for index, target in enumerate(row['references']):
        interval = Interval(target['start_s'], target['end_s'])
        if interval.end > duration + .001 or not isinstance(target.get('caption'), str) or not target['caption'].strip():
            raise ValueError('Independent reference interval or caption is invalid')
        targets.append(dict(reference_index=index, start_s=interval.start, end_s=interval.end,
            caption_sha256=hashlib.sha256(target['caption'].encode()).hexdigest()))
    if not targets:
        raise ValueError('Development episodes require independent reference intervals')
    return dict(source_sha256=row['source_sha256'], duration_s=duration, native_profile='Human',
        instruction_sha256=hashlib.sha256(row['instruction'].encode()).hexdigest(),
        reference_count=len(targets), targets=targets)


def build_reference_contract(manifest, references, *, manifest_sha256, references_sha256):
    """Freeze exact private reference content without publishing its text."""
    expected = manifest.get('selected_ids', [])
    if (manifest.get('schema_version') != 'bir_wgo_preparation_v1' or
            manifest.get('selection_frozen') is not True or manifest.get('proposed_split') != 'development' or
            manifest.get('selected_count') != len(expected) or not 1 <= len(expected) <= 3 or
            len(set(expected)) != len(expected) or manifest.get('max_episodes') != 3):
        raise ValueError('Use the frozen one-to-three episode initial development selection')
    if (manifest.get('dataset') != WGO.__dict__ or not isinstance(references, dict) or
            references.get('dataset') != WGO.__dict__ or not _sha(manifest_sha256) or not _sha(references_sha256)):
        raise ValueError('Development reference inputs differ from the pinned dataset or lack file receipts')
    candidates = {row['id']: row for row in manifest['candidates']}
    if len(candidates) != len(manifest['candidates']) or {row['id'] for row in candidates.values() if row.get('selected')} != set(expected):
        raise ValueError('Initial selected candidate identities disagree')
    refs, episodes = reference_rows(references), []
    for identity in expected:
        candidate, row = candidates[identity], refs[identity]
        if (candidate.get('profile') != 'human' or candidate.get('proposed_split') != 'development' or
                candidate.get('native_supported') is not True or candidate.get('references_valid') is not True):
            raise ValueError('Initial selection includes an unvalidated or non-development episode')
        duration = finite(candidate['duration_s'], 'Selected source duration', positive=True)
        binding = _reference_binding(row, duration)
        if (candidate['source_sha256'] != binding['source_sha256'] or
                candidate['reference_count'] != binding['reference_count'] or row.get('duration_s') != duration):
            raise ValueError('Selected source hash, duration, or reference count differs')
        episodes.append(dict(episode_id=identity, **binding))
    contract = dict(schema_version='bir_native_development_review_contract_v1',
        status='frozen_before_native_caption_review', scope='initial development inspection',
        dataset=dict(WGO.__dict__), initial_manifest_sha256=manifest_sha256,
        private_reference_input_sha256=references_sha256,
        rubric_sha256=RUBRIC_SHA256, iou_threshold=.5, matching='maximum_cardinality_one_to_one',
        semantic_acceptance='both independent AI reviewers accept the reference event',
        episodes=episodes)
    contract['content_contract_sha256'] = content_contract_digest(contract)
    contract['contract_sha256'] = digest(contract)
    return contract


def validate_contract(contract):
    if (contract.get('schema_version') != 'bir_native_development_review_contract_v1' or
            contract.get('contract_sha256') != digest({k: v for k, v in contract.items() if k != 'contract_sha256'}) or
            contract.get('content_contract_sha256') != content_contract_digest(contract) or
            contract.get('dataset') != WGO.__dict__ or contract.get('rubric_sha256') != RUBRIC_SHA256 or
            contract.get('iou_threshold') != .5 or contract.get('scope') != 'initial development inspection' or
            contract.get('matching') != 'maximum_cardinality_one_to_one'):
        raise ValueError('Frozen native development review contract changed')
    episodes = contract['episodes']
    if not 1 <= len(episodes) <= 3 or len({row['episode_id'] for row in episodes}) != len(episodes):
        raise ValueError('Frozen development episode identities are invalid')
    if any(row.get('native_profile') != 'Human' for row in episodes):
        raise ValueError('Frozen development review requires native Human single-view inputs')


def gate_input_receipts(marker):
    """Keep only public input-binding fields from the private probe marker."""
    receipts = []
    for row in marker.get('input_validation', []):
        source = row.get('input_sha256')
        if not isinstance(source, dict) or set(source) != {'main'} or not _sha(source['main']):
            raise ValueError('Gate C input validation must identify one source video')
        receipts.append(dict(source_sha256=source['main'], status=row.get('status'),
            instruction_sha256=row.get('instruction_sha256'), duration_s=row.get('duration_s'),
            profile=row.get('profile'), yam32=row.get('yam32')))
    return receipts


def prepare(annotations, references, contract, *, gate_receipt, input_receipts):
    started = time.perf_counter()
    validate_contract(contract)
    expected_ids = [row['episode_id'] for row in contract['episodes']]
    if (not _sha(input_receipts.get('initial_manifest_sha256')) or
            not _sha(input_receipts.get('private_reference_input_sha256')) or
            not _sha(gate_receipt.get('marker_sha256'))):
        raise ValueError('Preparation input receipts are invalid')
    gate_artifacts = {row['episode_id']: row['sha256'] for row in gate_receipt['annotation_artifacts']}
    artifact_receipts = {row['episode_id']: row for row in input_receipts['annotation_files']}
    if (len(gate_artifacts) != len(gate_receipt['annotation_artifacts']) or not gate_artifacts or
            set(gate_artifacts) - set(expected_ids) or
            len(artifact_receipts) != len(input_receipts['annotation_files'])):
        raise ValueError('Gate C artifact identities are invalid for the development selection')
    frozen_episodes = {row['episode_id']: row for row in contract['episodes']}
    validated_inputs = {row['source_sha256']: row for row in gate_receipt.get('input_validation', [])}
    if len(validated_inputs) != len(gate_receipt.get('input_validation', [])):
        raise ValueError('Gate C source input validation is ambiguous')
    expected_sources = {frozen_episodes[identity]['source_sha256'] for identity in gate_artifacts}
    if set(validated_inputs) != expected_sources:
        raise ValueError('Gate C lacks exact source input validation for its artifacts')
    for identity in gate_artifacts:
        episode = frozen_episodes[identity]
        row = validated_inputs[episode['source_sha256']]
        if (row.get('status') != 'inputs_validated' or row.get('profile') != episode['native_profile'] or
                row.get('yam32') is not False or row.get('duration_s') != episode['duration_s'] or
                row.get('instruction_sha256') != episode['instruction_sha256']):
            raise ValueError('Gate C inference instruction, duration, or profile differs from the frozen input')
    refs, outputs = reference_rows(references), {}
    for episode in contract['episodes']:
        actual = _reference_binding(refs[episode['episode_id']], episode['duration_s'])
        if any(episode[key] != value for key, value in actual.items()):
            raise ValueError('Independent references differ from the frozen development contract')
    for output in annotations:
        identity = output['episode_id']
        if identity not in expected_ids or identity in outputs:
            raise ValueError('Unexpected or duplicate native annotation')
        if output.get('input_sha256') != {'main': refs[identity]['source_sha256']}:
            raise ValueError('Native annotation and reference source differ')
        frozen = frozen_episodes[identity]
        if (output.get('duration_s') != frozen['duration_s'] or output.get('profile') != frozen['native_profile'] or
                output.get('camera_roles') != ['main']):
            raise ValueError('Native annotation duration, profile, or camera roles differ from the frozen input')
        receipt = artifact_receipts.get(identity, {})
        if (receipt.get('file_sha256') != gate_artifacts.get(identity) or not _sha(receipt.get('file_sha256')) or
                receipt.get('content_sha256') != digest(output)):
            raise ValueError('Native annotation differs from its Gate C artifact receipt')
        outputs[identity] = output
    if set(outputs) != set(artifact_receipts):
        raise ValueError('Annotation input receipts do not cover the supplied annotations')
    cards, episodes, edges = [], [], []
    for identity in expected_ids:
        reference = refs[identity]
        target = reference['references']
        prediction = outputs.get(identity, {}).get('segments', [])
        if any(row.get('status') == 'complete' and (not isinstance(row.get('caption'), str) or not row['caption'].strip())
               for row in prediction):
            raise ValueError('A complete native prediction must contain a nonempty caption')
        p = [Interval(row['start_s'], row['end_s']) for row in prediction]
        r = [Interval(row['start_s'], row['end_s']) for row in target]
        for i, first in enumerate(p):
            for j, second in enumerate(r):
                if temporal_iou(first, second) < .5:
                    continue
                row = prediction[i]
                if row.get('status') != 'complete' or not isinstance(row.get('caption'), str) or not row['caption'].strip():
                    continue
                content = dict(task_instruction=reference['instruction'], reference_caption=target[j]['caption'],
                               candidate_caption=row['caption'], interval=[second.start, second.end])
                card_id = digest(dict(episode_id=identity, prediction_index=i, reference_index=j, **content))
                cards.append(dict(card_id=card_id, **content))
                edges.append(dict(episode_id=identity, prediction_index=i, reference_index=j, card_id=card_id))
        episodes.append(dict(episode_id=identity, source_sha256=reference['source_sha256'],
            annotation_sha256=digest(outputs[identity]) if identity in outputs else None,
            independent_reference_sha256=digest(target),
            predictions=[[v.start, v.end] for v in p], references=[[v.start, v.end] for v in r],
            complete_captions=sum(row.get('status') == 'complete' and bool(row.get('caption', '').strip()) for row in prediction)))
    cards.sort(key=lambda row: hashlib.sha256(('bir-native-review-v1:' + row['card_id']).encode()).hexdigest())
    mapping = dict(schema_version='bir_native_joint_review_v1', rubric_sha256=RUBRIC_SHA256,
        scope='initial development inspection', iou_threshold=.5, episodes=episodes, edges=edges,
        card_ids=[row['card_id'] for row in cards], cards_sha256=digest(cards),
        provenance=dict(reference_contract=contract, gate_c=gate_receipt, input_receipts=input_receipts),
        preparation_seconds=time.perf_counter() - started)
    mapping['mapping_sha256'] = digest(mapping)
    return cards, mapping


def score(mapping, reviews):
    started = time.perf_counter()
    if (mapping.get('mapping_sha256') != digest({k: v for k, v in mapping.items() if k != 'mapping_sha256'}) or
            mapping.get('rubric_sha256') != RUBRIC_SHA256 or mapping.get('iou_threshold') != .5):
        raise ValueError('Native review evidence or frozen scoring rule changed')
    contract = mapping['provenance']['reference_contract']
    validate_contract(contract)
    if ([row['episode_id'] for row in mapping['episodes']] != [row['episode_id'] for row in contract['episodes']] or
            len(set(mapping['card_ids'])) != len(mapping['card_ids']) or
            set(mapping['card_ids']) != {row['card_id'] for row in mapping['edges']} or
            len(mapping['edges']) != len(mapping['card_ids']) or mapping['scope'] != contract['scope']):
        raise ValueError('Review mapping does not cover the frozen episodes and semantic cards')
    indexed_episodes = {}
    for episode, frozen in zip(mapping['episodes'], contract['episodes'], strict=True):
        if (episode['source_sha256'] != frozen['source_sha256'] or
                episode['references'] != [[row['start_s'], row['end_s']] for row in frozen['targets']] or
                type(episode['complete_captions']) is not int or
                not 0 <= episode['complete_captions'] <= len(episode['predictions'])):
            raise ValueError('Review mapping changed a frozen source or reference interval')
        indexed_episodes[episode['episode_id']] = episode
    seen_edges = set()
    for edge in mapping['edges']:
        identity = (edge['episode_id'], edge['prediction_index'], edge['reference_index'])
        episode = indexed_episodes.get(edge['episode_id'])
        i, j = edge['prediction_index'], edge['reference_index']
        if (identity in seen_edges or episode is None or type(i) is not int or type(j) is not int or
                not 0 <= i < len(episode['predictions']) or not 0 <= j < len(episode['references']) or
                temporal_iou(Interval(*episode['predictions'][i]), Interval(*episode['references'][j])) < .5):
            raise ValueError('Review mapping contains an invalid or duplicate semantic edge')
        seen_edges.add(identity)
    if len(reviews) != 2 or len({row['review_id'] for row in reviews}) != 2:
        raise ValueError('Two separate AI reviews are required')
    ratings, extras, elapsed = [], [], []
    for review in reviews:
        if (review.get('evaluator_type') != 'AI' or not review.get('evaluator_model') or
                review.get('rubric_sha256') != RUBRIC_SHA256 or review.get('new_paid_api_calls', 0) != 0):
            raise ValueError('Review identity, rubric, or zero paid API rule differs')
        indexed = {row['card_id']: row for row in review['ratings']}
        if len(indexed) != len(review['ratings']) or set(indexed) != set(mapping['card_ids']):
            raise ValueError('Review omitted, duplicated, or added cards')
        ratings.append({key: positive_rating(row) for key, row in indexed.items()})
        extras.append({key: row['unsupported_extra_claims'] for key, row in indexed.items()})
        seconds = review.get('review_wall_seconds')
        elapsed.append(finite(seconds, 'Offline review wall time') if seconds is not None else None)
    accepted = {row['card_id']: all(r[row['card_id']] for r in ratings) for row in mapping['edges']}
    results = []
    for episode in mapping['episodes']:
        p = [Interval(*row) for row in episode['predictions']]
        r = [Interval(*row) for row in episode['references']]
        valid = {(row['prediction_index'], row['reference_index']) for row in mapping['edges']
                 if row['episode_id'] == episode['episode_id'] and accepted[row['card_id']]}
        results.append(dict(episode_id=episode['episode_id'], source_sha256=episode['source_sha256'],
            predictions=len(p), independent_references=len(r), complete_captions=episode['complete_captions'],
            temporal=interval_metrics(p, r, .5),
            joint_ai_reference_event=interval_metrics(p, r, .5, lambda i, j: (i, j) in valid)))
    aggregate = {}
    for metric in ('temporal', 'joint_ai_reference_event'):
        totals = {key: sum(row[metric][key] for row in results)
                  for key in ('true_positive', 'false_positive', 'false_negative')}
        denominator = 2 * totals['true_positive'] + totals['false_positive'] + totals['false_negative']
        aggregate[metric] = dict(totals, micro_f1=2 * totals['true_positive'] / denominator if denominator else None,
            macro_f1=sum(row[metric]['f1'] or 0 for row in results) / len(results))
    return dict(schema_version='bir_native_joint_scores_v1', scope=mapping['scope'],
        mapping_sha256=mapping['mapping_sha256'], iou_threshold=.5, matching='maximum_cardinality_one_to_one',
        provenance=mapping['provenance'],
        is_human_evaluation=False, official_wgo_replication=False, visual_factual_correctness_assessed=False,
        reviewers=[dict(review_id=row['review_id'], evaluator_model=row['evaluator_model'],
                        review_content_sha256=digest(row)) for row in reviews],
        episodes=results, aggregate=aggregate,
        review_disagreements=sum(ratings[0][key] != ratings[1][key] for key in mapping['card_ids']),
        unsupported_extra_claim_cards=sum(any(r[key] for r in extras) for key in mapping['card_ids']),
        offline_audit=dict(unique_cards=len(mapping['card_ids']), reviewers=2, new_paid_api_calls=0,
            preparation_seconds=mapping['preparation_seconds'], reviewer_wall_seconds=elapsed,
            scoring_seconds=time.perf_counter() - started),
        limitations=['Three selected development episodes do not establish population accuracy.',
                     'AI reference-event agreement does not establish visual factual correctness.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freezing = commands.add_parser('freeze')
    for name in ('initial-manifest', 'references', 'output'):
        freezing.add_argument('--' + name, type=Path, required=True)
    prep = commands.add_parser('prepare')
    for name in ('annotations', 'references', 'initial-manifest', 'reference-contract', 'gate-c-marker', 'output-dir'):
        prep.add_argument('--' + name, type=Path, required=True)
    scoring = commands.add_parser('score')
    scoring.add_argument('--mapping', type=Path, required=True)
    scoring.add_argument('--reviews', type=Path, nargs=2, required=True)
    scoring.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    if args.command == 'freeze':
        if args.output.exists():
            raise ValueError('Do not overwrite a frozen native review contract')
        atomic_json(args.output, build_reference_contract(read(args.initial_manifest), read(args.references),
            manifest_sha256=file_sha256(args.initial_manifest), references_sha256=file_sha256(args.references)))
    elif args.command == 'prepare':
        if not args.annotations.is_dir():
            raise ValueError('The annotation input must be an existing directory')
        contract = read(args.reference_contract)
        receipts = dict(initial_manifest_sha256=file_sha256(args.initial_manifest),
                        private_reference_input_sha256=file_sha256(args.references))
        rebuilt = build_reference_contract(read(args.initial_manifest), read(args.references),
            manifest_sha256=receipts['initial_manifest_sha256'], references_sha256=receipts['private_reference_input_sha256'])
        validate_contract(contract)
        if rebuilt['content_contract_sha256'] != contract['content_contract_sha256']:
            raise ValueError('Initial manifest or independent references changed after the review freeze')
        gate = verify_gate_c(args.gate_c_marker)
        gate['input_validation'] = gate_input_receipts(read(args.gate_c_marker))
        gate_artifacts = {row['episode_id']: row['sha256'] for row in gate['annotation_artifacts']}
        annotations = []
        receipts['annotation_files'] = []
        for path in sorted(args.annotations.glob('*.caption.json')):
            output = read(path)
            checksum = file_sha256(path)
            if gate_artifacts.get(output['episode_id']) != checksum:
                raise ValueError('Annotation differs from the verified Gate C artifact')
            annotations.append(output)
            receipts['annotation_files'].append(dict(episode_id=output['episode_id'], file_sha256=checksum,
                                                     content_sha256=digest(output)))
        cards, mapping = prepare(annotations, read(args.references), contract, gate_receipt=gate, input_receipts=receipts)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output_dir / 'cards.json', cards)
        atomic_json(args.output_dir / 'mapping.json', mapping)
        (args.output_dir / 'rubric.txt').write_text(RUBRIC)
    else:
        atomic_json(args.output, score(read(args.mapping), [read(path) for path in args.reviews]))


if __name__ == '__main__':
    main()
