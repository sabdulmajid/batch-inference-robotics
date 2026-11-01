"""Small researcher workflow; generated checkpoints remain read-only."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading

from .manifest import canonical_json, file_sha256, load_manifest


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.workflow-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def workflow_lock(directory, name='.workflow.lock'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another workflow owns this run. Wait for it to finish.') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare_inputs(directory, *, manifest=None, videos=None):
    """Freeze the input list and hashes; moved identical sources remain usable."""
    from .cli import atomic_json
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'inputs.jsonl'
    if bool(manifest) == bool(videos):
        raise ValueError('Choose one of --manifest or --videos.')
    if videos:
        folder = Path(videos).expanduser().resolve(strict=True)
        if not folder.is_dir():
            raise ValueError('--videos must be a folder.')
        candidates = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in
                            {'.mp4', '.mov', '.mkv', '.avi', '.webm'})
        if not candidates or len(candidates) > 100:
            raise ValueError('The folder must contain 1–100 videos. Use a bounded manifest for other workloads.')
        raw = [dict(id=p.relative_to(folder).as_posix(), video=str(p), profile='human') for p in candidates]
        fd, scratch = tempfile.mkstemp(suffix='.jsonl', dir=directory)
        try:
            with os.fdopen(fd, 'w') as stream:
                stream.write('\n'.join(canonical_json(row) for row in raw))
            jobs = load_manifest(scratch, contract={'purpose': 'workflow_input_freeze'})
        finally:
            Path(scratch).unlink(missing_ok=True)
    else:
        jobs = load_manifest(manifest, contract={'purpose': 'workflow_input_freeze'})
        if len(jobs) > 100:
            raise ValueError('Use at most 100 episodes per workflow run.')
    rows = []
    for job in jobs:
        row = job.as_native_dict()
        row = {k: v for k, v in row.items() if v is not None}
        row['source_sha256'] = dict(job.source_sha256)
        rows.append(row)
    identities = [job.identity() for job in jobs]
    receipt = directory / 'workflow-inputs.json'
    if receipt.exists() and json.loads(receipt.read_text())['identities'] != identities:
        raise ValueError('The input list, source bytes, or settings changed. Use a new run directory.')
    if not receipt.exists() and (directory / 'run.sqlite').exists():
        raise ValueError('This is a low-level run. Resume it with bir run, or choose a new annotate directory.')
    atomic_text(target, ''.join(canonical_json(row) + '\n' for row in rows))
    atomic_json(receipt, dict(schema_version='bir_workflow_inputs_v1', identities=identities))
    return target, len(rows)


def read_database(directory):
    path = Path(directory) / 'run.sqlite'
    if not path.is_file():
        return [], []
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=2)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute('BEGIN')
        requests = [dict(r) for r in connection.execute('SELECT * FROM requests ORDER BY request_id')]
        attempts = [dict(r) for r in connection.execute('SELECT * FROM attempts ORDER BY request_id,stage,number')]
    finally:
        connection.close()
    for request in requests:
        if hashlib.sha256(request['identity_json'].encode()).hexdigest() != request['request_id']:
            raise ValueError('Stored request identity checksum failed.')
    for row in attempts:
        if row['payload_json'] is not None:
            if hashlib.sha256(row['payload_json'].encode()).hexdigest() != row['payload_sha256']:
                raise ValueError('Stored result checksum failed. Preserve the run and inspect its storage.')
    return requests, attempts


def advice(error):
    text = str(error).lower()
    if any(x in text for x in ('cuda', 'gpu', 'placement budget', 'out of memory')):
        return 'Check the GPU allocation and free memory against the documented hardware. Keep native model settings unchanged.'
    if any(x in text for x in ('no module', 'not installed', 'dependency', 'version')):
        return 'Use the pinned native environment. See docs/QUICKSTART.md; run bir source-check before retrying.'
    if any(x in text for x in ('source changed', 'checksum', 'manifest', 'input')):
        return 'Check the source file and manifest. Restore the original bytes, or use a new run directory for changed inputs.'
    return 'Inspect inference.log and bir status. Completed stages remain reusable. Deterministic failures need a fix; retries stay bounded.'


def status(directory):
    directory = Path(directory)
    requests, attempts = read_database(directory)
    latest = {(r['request_id'], r['stage']): r for r in attempts}
    rows = []
    summary = json.loads((directory / 'summary.json').read_text()) if (directory / 'summary.json').exists() else {}
    summary_rows = {(r['request_id'], r['stage']): r for r in summary.get('records', [])}
    for request in requests:
        for stage in ('segment', 'caption'):
            row = latest.get((request['request_id'], stage), {})
            fallback = summary_rows.get((request['request_id'], stage), {})
            state = row.get('state', fallback.get('status', 'pending'))
            error = row.get('error') or fallback.get('error')
            rows.append(dict(episode_id=request['episode_id'], request_id=request['request_id'], stage=stage,
                             state=state, attempt=row.get('number'), error=error,
                             action=advice(error) if error else None))
    failure_path = directory / 'workflow-failure.json'
    failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
    return dict(schema_version='bir_workflow_status_v1', episodes=len(requests), stages=rows,
                complete_stages=sum(r['state'] == 'complete' for r in rows),
                failed_stages=sum(r['state'] in {'failed', 'blocked'} for r in rows),
                total_stages=2 * len(requests), precomputed=(directory / 'precomputed.json').exists(),
                startup_failure=failure)


def annotate(args, started, execute):
    """Run the same engine with a frozen input list and non-invasive progress."""
    from .cli import atomic_json
    directory = args.run_dir.expanduser().resolve()
    with workflow_lock(directory):
        args.manifest, count = prepare_inputs(directory, manifest=args.manifest, videos=args.videos)
        args.stages = 'all'
        (directory / 'workflow-failure.json').unlink(missing_ok=True)
        stopped = threading.Event()
        def report():
            previous = None
            while not stopped.wait(2):
                try:
                    snapshot = status(directory)
                    signature = [(x['episode_id'], x['stage'], x['state'], x['attempt']) for x in snapshot['stages']]
                    if signature != previous:
                        active = [x for x in snapshot['stages'] if x['state'] == 'running']
                        current = ', '.join(x['episode_id'] + '/' + x['stage'] for x in active)
                        print(f"{snapshot['complete_stages']}/{2 * count} stages complete; "
                              f"{snapshot['failed_stages']} failed. {current}", file=sys.stderr, flush=True)
                        previous = signature
                except (OSError, ValueError, sqlite3.Error):
                    # A just-created database or atomic summary may not be ready.
                    continue
        worker = threading.Thread(target=report, daemon=True, name='bir-progress')
        print(f'Annotating {count} videos. Progress and failures are in {directory}.', file=sys.stderr, flush=True)
        worker.start()
        try:
            with (directory / 'inference.log').open('a', buffering=1) as stream, redirect_stdout(stream):
                result, code = execute(args, started)
        except Exception as error:
            atomic_json(directory / 'workflow-failure.json', dict(error=str(error), action=advice(error)))
            raise
        finally:
            stopped.set()
            worker.join()
        print(f"{result.get('fresh_stage_attempts', 0)} fresh stages; "
              f"{result.get('resumed_stages', 0)} resumed. "
              f"{'Complete.' if result.get('complete') else 'Incomplete; run bir status for actions.'}",
              file=sys.stderr)
        return result, code


def _record(identity, request, index, segment, *, state, failure, origin):
    start, end = segment.get('start_s'), segment.get('end_s')
    if start is not None or end is not None:
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
            raise ValueError('A saved interval has invalid source timestamps.')
    generated = segment.get('caption')
    if generated is not None and not isinstance(generated, str):
        raise ValueError('A saved caption is not text.')
    base = dict(schema_version='bir_label_v1', episode_id=identity['id'], request_id=request,
                source_sha256=identity['source_sha256'], profile=identity['profile'], interval_index=index,
                start_s=start, end_s=end, generated_caption=generated, inference_status=state,
                failure=failure, generation=dict(contract_sha256=digest(identity['contract']), **origin))
    base['record_id'] = digest(base)
    return base


def generated_records(directory):
    directory = Path(directory)
    demo = directory / 'precomputed.json'
    if demo.exists():
        data = json.loads(demo.read_text())
        rows = data['records']
        for row in rows:
            if row.get('record_id') != digest({k: v for k, v in row.items() if k != 'record_id'}):
                raise ValueError('Precomputed record checksum failed.')
        return rows
    requests, attempts = read_database(directory)
    latest = {(r['request_id'], r['stage']): r for r in attempts}
    result = []
    for request in requests:
        rid = request['request_id']
        identity = json.loads(request['identity_json'])
        caption = latest.get((rid, 'caption'), {})
        segment = latest.get((rid, 'segment'), {})
        selected = caption if caption.get('state') == 'complete' else segment
        payload = json.loads(selected['payload_json']) if selected.get('state') == 'complete' else {}
        if payload and (payload.get('input_sha256') != identity['source_sha256'] or
                        payload.get('profile', '').lower() != identity['profile']):
            raise ValueError('Saved result source or profile differs from its request identity.')
        if selected is caption and payload and (payload.get('request_id') != rid or payload.get('episode_id') != identity['id']):
            raise ValueError('Saved caption routing differs from its request identity.')
        if selected is segment and payload:
            body = {k: v for k, v in payload.items() if k != 'stage_digest'}
            expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if payload.get('stage_digest') != expected:
                raise ValueError('Signed segmentation checksum failed.')
        native_rows = payload.get('segments', [])
        if not native_rows:
            native_rows = [{}]
        for index, row in enumerate(native_rows):
            complete = caption.get('state') == 'complete'
            state = row.get('status', 'complete') if complete else caption.get('state', segment.get('state', 'pending'))
            if complete and not row:
                state = 'no_intervals'
            elif complete and state == 'complete' and not str(row.get('caption') or '').strip():
                state = 'empty_caption'
            if not complete and state == 'complete':
                state = 'caption_pending'
            failure = None if complete and state == 'complete' else dict(
                stage='caption' if segment.get('state') == 'complete' else 'segment',
                error=caption.get('error') or segment.get('error') or
                      ('Saved caption stage has no intervals.' if state == 'no_intervals' else 'Caption is not complete.'),
                action=advice(caption.get('error') or segment.get('error')))
            result.append(_record(identity, rid, index if row else None, row, state=state, failure=failure,
                origin=dict(precomputed=False, payload_sha256=selected.get('payload_sha256'))))
    return result


def read_reviews(directory):
    path = Path(directory) / 'reviews.jsonl'
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for index, entry in enumerate(entries, 1):
        if entry.get('revision') != index or entry.get('entry_sha256') != digest({k: v for k, v in entry.items() if k != 'entry_sha256'}):
            raise ValueError('Review journal is invalid. Preserve it and inspect before editing.')
        if entry.get('previous_entry_sha256') != (entries[index - 2]['entry_sha256'] if index > 1 else None):
            raise ValueError('Review journal history changed.')
    return entries


def records(directory):
    base = generated_records(directory)
    edits = {row['record_id']: row for row in read_reviews(directory)}
    if set(edits) - {row['record_id'] for row in base}:
        raise ValueError('A review no longer matches its generated source. Use the original run for this journal.')
    result = []
    for row in base:
        review = edits.get(row['record_id'])
        reviewed = review['reviewed_caption'] if review else None
        result.append(dict(row, reviewed_caption=reviewed, effective_caption=reviewed if reviewed is not None else row['generated_caption'],
                           review=review, review_revision=review['revision'] if review else 0))
    return result


def save_review(directory, *, record_id, reviewed_caption, reviewer, expected_revision):
    if not isinstance(reviewer, str) or not 1 <= len(reviewer.strip()) <= 120:
        raise ValueError('Enter a reviewer name or research identifier (1–120 characters).')
    if reviewed_caption is not None and (not isinstance(reviewed_caption, str) or not reviewed_caption.strip() or len(reviewed_caption) > 10000):
        raise ValueError('Enter a nonempty reviewed caption, or use null to restore the generated label.')
    with workflow_lock(directory, '.review.lock'):
        row = next((x for x in records(directory) if x['record_id'] == record_id), None)
        if row is None or row['start_s'] is None:
            raise ValueError('This record has no supported interval to review.')
        if type(expected_revision) is not int or expected_revision != row['review_revision']:
            raise ValueError('This label was edited in another view. Reload before saving.')
        entries = read_reviews(directory)
        entry = dict(schema_version='bir_review_v1', record_id=record_id, revision=len(entries) + 1,
                     previous_entry_sha256=entries[-1]['entry_sha256'] if entries else None,
                     generated_record_sha256=record_id, reviewer=reviewer.strip(),
                     reviewed_utc=datetime.now(timezone.utc).isoformat(), reviewed_caption=reviewed_caption,
                     action='restore_generated' if reviewed_caption is None else 'caption_correction')
        entry['entry_sha256'] = digest(entry)
        with (Path(directory) / 'reviews.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(canonical_json(entry) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    return entry


def export_jsonl(directory, output):
    if Path(output).is_symlink():
        raise ValueError('Choose a fresh export file; symlink destinations are not supported.')
    output = Path(output).resolve()
    directory = Path(directory).resolve()
    protected = ('reviews.jsonl', 'inputs.jsonl', 'summary.json', 'precomputed.json', 'run.sqlite',
                 'run-metadata.json', 'workflow-inputs.json', 'workflow-failure.json', 'inference.log',
                 '.review.lock', '.workflow.lock', '.cli.lock', 'run.sqlite.lock',
                 'run.sqlite-wal', 'run.sqlite-shm', 'run.sqlite-journal')
    if output == directory or output in {directory / name for name in protected} or directory / 'annotations' in output.parents:
        raise ValueError('Choose a separate export file; generated artifacts and review history are protected.')
    inputs = directory / 'inputs.jsonl'
    if inputs.exists():
        for line in inputs.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                for key in ('video', 'left_wrist', 'right_wrist'):
                    if row.get(key) and output == (inputs.parent / Path(row[key]).expanduser()).resolve():
                        raise ValueError('Choose a separate export file; source videos are protected.')
    if output.exists():
        raise ValueError('Export destination already exists. Choose a fresh filename; existing files are never replaced.')
    with workflow_lock(directory, '.review.lock'):
        rows = records(directory)
        if not rows:
            raise ValueError('No saved results exist yet. Inspect bir status for startup failures.')
        # A hard-link publication is atomic and fails if another process creates
        # the destination. Unlike os.replace, it cannot overwrite an input,
        # database sidecar, earlier export, or a file created after the check.
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix='.export-', dir=output.parent)
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                stream.write(''.join(canonical_json(row) + '\n' for row in rows))
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, output)
            except FileExistsError as error:
                raise ValueError('Export destination already exists. Choose a fresh filename.') from error
        finally:
            Path(temporary).unlink(missing_ok=True)
    return dict(schema_version='bir_export_v1', output=str(output), records=len(rows),
                complete_intervals=sum(x['inference_status'] == 'complete' for x in rows),
                failed_or_pending_records=sum(x['inference_status'] != 'complete' for x in rows),
                reviewed_records=sum(x['reviewed_caption'] is not None for x in rows),
                precomputed=any(x['generation']['precomputed'] for x in rows), sha256=file_sha256(output))


def prepare_demo(output, example, *, video=None, parquet=None):
    """Install precomputed public-data results. Never load an inference model."""
    from .cli import atomic_json
    output, example = Path(output), Path(example)
    if output.exists():
        raise ValueError('Demo output must be a new directory.')
    data = json.loads(example.read_text())
    source = data['source']
    if (data.get('schema_version') != 'bir_precomputed_example_v1' or data.get('precomputed') is not True or
            not data.get('records') or any(row.get('record_id') != digest({k: v for k, v in row.items() if k != 'record_id'}) or
            row.get('source_sha256') != {'main': source['sha256']} or row.get('episode_id') != source['episode_id'] or
            row.get('generation', {}).get('precomputed') is not True for row in data['records'])):
        raise ValueError('The precomputed example has invalid record or source bindings.')
    if bool(video) == bool(parquet):
        raise ValueError('Choose --video with the exact public source, or --parquet with the pinned public dataset.')
    if video:
        video = Path(video).expanduser().resolve(strict=True)
        if file_sha256(video) != source['sha256']:
            raise ValueError('Demo video does not match the published source checksum.')
    else:
        if file_sha256(Path(parquet)) != data['dataset']['parquet_sha256']:
            raise ValueError('Dataset checksum differs from the pinned public release.')
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ValueError('Install pyarrow==21.0.0 to extract the public demo.') from error
        payload = None
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1, columns=['id', 'video'], use_threads=False):
            item = batch.to_pylist()[0]
            if item['id'] == source['episode_id']:
                payload = item['video']
                if isinstance(payload, dict):
                    payload = payload.get('bytes')
                break
        if not isinstance(payload, bytes) or hashlib.sha256(payload).hexdigest() != source['sha256']:
            raise ValueError('The public dataset is missing the exact demo video.')
        output.mkdir(parents=True)
        video = output.resolve() / (source['episode_id'] + '.mp4')
        video.write_bytes(payload)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'precomputed.json', data)
    atomic_text(output / 'inputs.jsonl', canonical_json(dict(id=source['episode_id'], video=str(video), profile='human',
                source_sha256={'main': source['sha256']})) + '\n')
    generated_records(output)  # Validate the published record identities.
    return dict(status='precomputed_demo_ready', new_model_calls=0, run_dir=str(output),
                records=len(data['records']), source_sha256=source['sha256'], attribution=data['attribution'])
