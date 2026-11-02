import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

from bir.cli import build_parser
from bir.manifest import canonical_json, load_manifest
from bir.store import RunStore
from bir.workflow import (annotate, digest, export_jsonl, generated_records, prepare_inputs,
                          records, save_review, status, orphaned_reviews, read_reviews)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / 'video.mp4'
        self.video.write_bytes(b'original source fixture')
        self.manifest = self.root / 'jobs.jsonl'
        self.manifest.write_text(canonical_json(dict(id='one', video='video.mp4')))
        self.run = self.root / 'run'
        self.job = load_manifest(self.manifest, contract={'model': 'fixture'})[0]

    def seed(self, fail=False, failure_kind='deterministic'):
        with closing(RunStore(self.run / 'run.sqlite')) as store:
            with store.exclusive():
                store.register(self.job)
                segment = dict(input_sha256=dict(self.job.source_sha256), profile='Human',
                               segments=[dict(start_s=0.125, end_s=1.875)], duration_s=2)
                segment['stage_digest'] = hashlib.sha256(json.dumps(segment, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                number = store.begin(self.job.request_id, 'segment')
                store.complete(self.job.request_id, 'segment', number, segment, 0.1)
                number = store.begin(self.job.request_id, 'caption')
                if fail:
                    store.fail(self.job.request_id, 'caption', number, RuntimeError('CUDA allocation failed'), kind=failure_kind, elapsed_s=0.1)
                else:
                    caption = dict(input_sha256=dict(self.job.source_sha256), profile='Human',
                                   request_id=self.job.request_id, episode_id='one',
                                   segments=[dict(start_s=0.125, end_s=1.875, caption='Move the cup.', status='complete')])
                    store.complete(self.job.request_id, 'caption', number, caption, 0.1)
                    store.export(self.job.request_id, self.run / 'annotations')

    def test_folder_freeze_is_deterministic_and_rejects_changed_workload(self):
        folder = self.root / 'videos'
        folder.mkdir()
        (folder / 'z.mp4').write_bytes(b'z')
        (folder / 'a.mp4').write_bytes(b'a')
        manifest, count = prepare_inputs(self.run, videos=folder)
        rows = [json.loads(x) for x in manifest.read_text().splitlines()]
        self.assertEqual(count, 2)
        self.assertEqual([x['id'] for x in rows], ['a.mp4', 'z.mp4'])
        self.assertIn('source_sha256', rows[0])
        self.assertEqual(prepare_inputs(self.run, videos=folder)[1], 2)
        (folder / 'a.mp4').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            prepare_inputs(self.run, videos=folder)

    def test_review_and_export_preserve_generated_store_and_timestamp_precision(self):
        self.seed()
        before = (self.run / 'run.sqlite').read_bytes()
        row = records(self.run)[0]
        edit = save_review(self.run, record_id=row['record_id'], reviewed_caption='Place the cup on the shelf.',
                           reviewer='researcher-7', expected_revision=0)
        current = records(self.run)[0]
        self.assertEqual(current['generated_caption'], 'Move the cup.')
        self.assertEqual(current['reviewed_caption'], 'Place the cup on the shelf.')
        self.assertEqual(current['review']['reviewer'], 'researcher-7')
        self.assertEqual(current['start_s'], 0.125)
        self.assertEqual(current['end_s'], 1.875)
        self.assertEqual((self.run / 'run.sqlite').read_bytes(), before)
        exported = self.root / 'labels.jsonl'
        summary = export_jsonl(self.run, exported)
        self.assertEqual(summary['reviewed_records'], 1)
        self.assertEqual(json.loads(exported.read_text())['effective_caption'], current['reviewed_caption'])
        save_review(self.run, record_id=row['record_id'], reviewed_caption=None, reviewer='researcher-7', expected_revision=edit['revision'])
        restored = records(self.run)[0]
        self.assertEqual(restored['effective_caption'], row['generated_caption'])
        self.assertEqual(restored['review']['action'], 'restore_generated')
        self.assertEqual(len((self.run / 'reviews.jsonl').read_text().splitlines()), 2)

    def test_stale_edit_and_history_tampering_are_rejected(self):
        self.seed()
        row = records(self.run)[0]
        save_review(self.run, record_id=row['record_id'], reviewed_caption='A correction.', reviewer='r', expected_revision=0)
        with self.assertRaisesRegex(ValueError, 'another view'):
            save_review(self.run, record_id=row['record_id'], reviewed_caption='Stale edit.', reviewer='r', expected_revision=0)
        journal = self.run / 'reviews.jsonl'
        journal.write_text(journal.read_text().replace('A correction.', 'Changed text.'))
        with self.assertRaisesRegex(ValueError, 'journal'):
            records(self.run)

    def test_failed_caption_keeps_interval_in_export_denominator(self):
        self.seed(fail=True)
        row = records(self.run)[0]
        self.assertEqual(row['inference_status'], 'failed')
        self.assertEqual(row['start_s'], 0.125)
        self.assertIsNone(row['generated_caption'])
        self.assertIn('GPU allocation', row['failure']['action'])
        snapshot = status(self.run)
        self.assertEqual(snapshot['failed_stages'], 1)
        self.assertEqual(snapshot['complete_stages'], 1)
        summary = export_jsonl(self.run, self.root / 'failed.jsonl')
        self.assertEqual(summary['failed_or_pending_records'], 1)
        self.assertEqual(summary['complete_intervals'], 0)

    def test_new_reviews_reject_failed_and_pending_caption_records(self):
        self.seed(fail=True)
        failed = records(self.run)[0]
        self.assertEqual(failed['inference_status'], 'failed')
        with self.assertRaisesRegex(ValueError, 'complete generated caption'):
            save_review(self.run, record_id=failed['record_id'], reviewed_caption='Guess before inference.',
                        reviewer='researcher', expected_revision=0)
        with sqlite3.connect(self.run / 'run.sqlite') as database:
            database.execute("DELETE FROM attempts WHERE stage='caption'")
        pending = records(self.run)[0]
        self.assertEqual(pending['inference_status'], 'caption_pending')
        with self.assertRaisesRegex(ValueError, 'complete generated caption'):
            save_review(self.run, record_id=pending['record_id'], reviewed_caption='Another guess.',
                        reviewer='researcher', expected_revision=0)
        self.assertFalse((self.run / 'reviews.jsonl').exists())

    def test_historical_failed_review_survives_resume_without_reassignment_or_export_block(self):
        self.seed(fail=True, failure_kind='transient')
        old_record = generated_records(self.run)[0]
        # The earlier release allowed this edit. Write its valid historical journal
        # directly because the current public save path correctly rejects it.
        historical = dict(schema_version='bir_review_v1', record_id=old_record['record_id'], revision=1,
                          previous_entry_sha256=None, generated_record_sha256=old_record['record_id'],
                          reviewer='historical-reviewer', reviewed_utc='2026-11-01T00:00:00+00:00',
                          reviewed_caption='Historical edit made before caption recovery.',
                          action='caption_correction')
        historical['entry_sha256'] = digest(historical)
        journal = self.run / 'reviews.jsonl'
        journal.write_text(canonical_json(historical) + '\n')
        history_before = journal.read_bytes()
        self.assertEqual(read_reviews(self.run), [historical])

        with closing(RunStore(self.run / 'run.sqlite')) as store:
            with store.exclusive():
                number = store.begin(self.job.request_id, 'caption')
                self.assertEqual(number, 2)
                caption = dict(input_sha256=dict(self.job.source_sha256), profile='Human',
                               request_id=self.job.request_id, episode_id='one',
                               segments=[dict(start_s=0.125, end_s=1.875, caption='Recovered generated caption.',
                                              status='complete')])
                store.complete(self.job.request_id, 'caption', number, caption, 0.1)
                store.export(self.job.request_id, self.run / 'annotations')

        recovered = records(self.run)[0]
        self.assertNotEqual(recovered['record_id'], old_record['record_id'])
        self.assertEqual(recovered['generated_caption'], 'Recovered generated caption.')
        self.assertEqual(recovered['effective_caption'], recovered['generated_caption'])
        self.assertIsNone(recovered['review'])
        self.assertIsNone(recovered['reviewed_caption'])
        self.assertEqual(recovered['review_revision'], 0)
        orphan = orphaned_reviews(self.run)
        self.assertEqual(len(orphan), 1)
        self.assertEqual(orphan[0]['record_id'], old_record['record_id'])
        self.assertEqual(orphan[0]['reviewed_caption'], historical['reviewed_caption'])
        self.assertEqual(orphan[0]['applicability'], 'historical_record_unavailable')
        self.assertEqual(journal.read_bytes(), history_before)

        destination = self.root / 'recovered-labels.jsonl'
        summary = export_jsonl(self.run, destination)
        self.assertEqual(summary['complete_intervals'], 1)
        self.assertEqual(summary['reviewed_records'], 0)
        self.assertEqual(json.loads(destination.read_text())['effective_caption'], 'Recovered generated caption.')
        self.assertEqual(journal.read_bytes(), history_before)

        # A later correction attaches only to the new complete record. The old
        # event remains the first entry of the same hash-chained journal.
        saved = save_review(self.run, record_id=recovered['record_id'], reviewed_caption='New verified correction.',
                            reviewer='current-reviewer', expected_revision=0)
        self.assertEqual(saved['revision'], 2)
        self.assertTrue(journal.read_bytes().startswith(history_before))
        self.assertEqual(records(self.run)[0]['effective_caption'], 'New verified correction.')
        self.assertEqual(orphaned_reviews(self.run)[0]['record_id'], old_record['record_id'])

    def test_result_checksum_and_source_binding_are_checked(self):
        self.seed()
        with sqlite3.connect(self.run / 'run.sqlite') as db:
            payload = json.loads(db.execute("select payload_json from attempts where stage='caption'").fetchone()[0])
            payload['input_sha256'] = {'main': '0' * 64}
            text = canonical_json(payload)
            db.execute("update attempts set payload_json=?,payload_sha256=? where stage='caption'", (text, hashlib.sha256(text.encode()).hexdigest()))
        with self.assertRaisesRegex(ValueError, 'source or profile'):
            generated_records(self.run)
        with sqlite3.connect(self.run / 'run.sqlite') as db:
            db.execute("update attempts set payload_sha256='bad' where stage='caption'")
        with self.assertRaisesRegex(ValueError, 'checksum'):
            generated_records(self.run)

    def test_export_cannot_replace_provenance_or_generated_results(self):
        self.seed()
        for path in ['run.sqlite', 'run-metadata.json', 'reviews.jsonl', 'inputs.jsonl', 'annotations/result.json']:
            with self.assertRaisesRegex(ValueError, 'separate export'):
                export_jsonl(self.run, self.run / path)

    def test_export_preserves_source_sidecars_existing_files_and_concurrent_destination(self):
        self.seed()
        (self.run / 'inputs.jsonl').write_text(canonical_json(dict(id='one', video=str(self.video))))
        sidecar = self.run / 'run.sqlite-shm'
        sidecar.write_bytes(b'preserve sidecar')
        existing = self.root / 'existing.jsonl'
        existing.write_bytes(b'preserve earlier export')
        for path in [self.video, sidecar, existing]:
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                export_jsonl(self.run, path)
            self.assertEqual(path.read_bytes(), before)
        for name in ['run.sqlite-wal', 'run.sqlite-journal']:
            with self.assertRaisesRegex(ValueError, 'separate export'):
                export_jsonl(self.run, self.run / name)
            self.assertFalse((self.run / name).exists())
        raced = self.root / 'raced.jsonl'
        import os
        original_link = os.link
        def create_then_link(source, destination):
            Path(destination).write_bytes(b'created concurrently')
            return original_link(source, destination)
        with patch('bir.workflow.os.link', side_effect=create_then_link):
            with self.assertRaisesRegex(ValueError, 'already exists'):
                export_jsonl(self.run, raced)
        self.assertEqual(raced.read_bytes(), b'created concurrently')

    def test_empty_completed_caption_payload_is_missing_not_a_complete_interval(self):
        self.seed()
        with sqlite3.connect(self.run / 'run.sqlite') as db:
            value = json.loads(db.execute("select payload_json from attempts where stage='caption'").fetchone()[0])
            value['segments'] = []
            encoded = canonical_json(value)
            db.execute("update attempts set payload_json=?,payload_sha256=? where stage='caption'",
                       (encoded, hashlib.sha256(encoded.encode()).hexdigest()))
        row = records(self.run)[0]
        self.assertEqual(row['inference_status'], 'no_intervals')
        self.assertIsNone(row['start_s'])
        self.assertIsNone(row['generated_caption'])
        self.assertIsNotNone(row['failure'])
        result = export_jsonl(self.run, self.root / 'empty.jsonl')
        self.assertEqual(result['complete_intervals'], 0)
        self.assertEqual(result['failed_or_pending_records'], 1)

    def test_annotate_delegates_without_altering_native_settings_and_records_startup_failure(self):
        args = build_parser().parse_args(['annotate', '--manifest', str(self.manifest), '--source', 'source', '--run-dir', str(self.run)])
        self.assertEqual(args.condition, 'accelerated')
        def execute(received, started):
            self.assertEqual(received.stages, 'all')
            self.assertEqual(received.condition, 'accelerated')
            self.assertEqual(received.batch_size, 1)
            self.assertEqual(received.captioner_headroom_gib, 4)
            self.assertTrue(received.manifest.is_file())
            raise RuntimeError('No module named torch')
        with self.assertRaisesRegex(RuntimeError, 'torch'):
            annotate(args, time.monotonic(), execute)
        failure = status(self.run)['startup_failure']
        self.assertIn('pinned native environment', failure['action'])
        self.assertFalse((self.run / 'run.sqlite').exists())


if __name__ == '__main__':
    unittest.main()
