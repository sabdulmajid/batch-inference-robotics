"""Portable review bundles. Media selection is separate and content verified."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import re
import os
import tempfile

from .manifest import canonical_json, file_sha256
from .workflow import digest, generated_records, read_reviews
from .trajectory import validate_trajectory, finite

MAX_BUNDLE_BYTES=64*1024**2
SHA=re.compile(r'^[0-9a-f]{64}$')


def validate_reviews(entries):
    if not isinstance(entries,list) or len(entries)>10000:
        raise ValueError('Review journal must contain at most 10000 entries')
    previous=None
    for i,e in enumerate(entries,1):
        if not isinstance(e,dict) or e.get('schema_version')!='bir_review_v1' or type(e.get('revision')) is not int or e['revision']!=i:
            raise ValueError('Review revisions must form one unbroken journal')
        if not SHA.fullmatch(str(e.get('record_id',''))) or e.get('generated_record_sha256')!=e['record_id']:
            raise ValueError('Review does not bind a generated record identity')
        if e.get('previous_entry_sha256')!=previous or e.get('entry_sha256')!=digest({k:v for k,v in e.items() if k!='entry_sha256'}):
            raise ValueError('Review journal hash chain changed')
        text=e.get('reviewed_caption')
        if text is not None and (not isinstance(text,str) or not text.strip() or len(text)>10000):
            raise ValueError('Invalid reviewed caption')
        if e.get('action')!=('restore_generated' if text is None else 'caption_correction'):
            raise ValueError('Review action differs from payload')
        if not isinstance(e.get('reviewer'),str) or not 1<=len(e['reviewer'].strip())<=120 or not isinstance(e.get('reviewed_utc'),str):
            raise ValueError('Reviewer and recorded time are required')
        previous=e['entry_sha256']
    return entries


def validate_bundle(bundle):
    if not isinstance(bundle,dict) or set(bundle)!={'schema_version','payload_json','payload_sha256','reviews'} or bundle['schema_version']!='bir_rollout_bundle_v1':
        raise ValueError('Unsupported bundle version or fields')
    raw=bundle['payload_json']
    if not isinstance(raw,str) or len(raw.encode())>MAX_BUNDLE_BYTES or hashlib.sha256(raw.encode()).hexdigest()!=bundle['payload_sha256']:
        raise ValueError('Immutable bundle payload checksum failed')
    payload=json.loads(raw)
    if canonical_json(payload)!=raw:raise ValueError('Bundle payload must be canonical JSON from the CLI')
    if not isinstance(payload,dict) or set(payload)!={'schema_version','episodes','provenance'} or payload['schema_version']!='bir_rollout_payload_v1':
        raise ValueError('Unsupported bundle payload')
    episodes=payload['episodes']
    if not isinstance(episodes,list) or not 1<=len(episodes)<=100:
        raise ValueError('Bundle needs 1–100 episodes')
    seen=set(); record_ids=set()
    for e in episodes:
        if not isinstance(e,dict) or set(e)!={'episode_id','task','media','records','trajectory','outcome'}:
            raise ValueError('Unsupported episode fields')
        identity=e['episode_id']
        if not isinstance(identity,str) or not 1<=len(identity)<=200 or identity in seen:
            raise ValueError('Duplicate or invalid episode identity')
        seen.add(identity)
        if not isinstance(e['task'],str) or len(e['task'])>2000:raise ValueError('Invalid task description')
        if not isinstance(e['media'],list) or len(e['media'])>4:raise ValueError('At most four ordered cameras supported')
        cameras={}
        for m in e['media']:
            if set(m)!={'camera','filename','source_sha256','duration_s'} or m['camera'] in cameras:
                raise ValueError('Invalid or duplicate media camera')
            if not isinstance(m['filename'],str) or Path(m['filename']).name!=m['filename'] or m['filename'] in ('','.','..'):
                raise ValueError('Media uses a filename, not a path or URL')
            if not SHA.fullmatch(m['source_sha256']) or not finite(m['duration_s']) or not 0<m['duration_s']<=3600:
                raise ValueError('Invalid media hash or duration')
            cameras[m['camera']]=m
        if not isinstance(e['records'],list) or len(e['records'])>10000:raise ValueError('Invalid record count')
        for record in e['records']:
            if not isinstance(record,dict) or record.get('schema_version')!='bir_label_v1' or record.get('episode_id')!=identity:
                raise ValueError('Annotation episode identity differs')
            rid=record.get('record_id')
            if rid in record_ids or rid!=digest({k:v for k,v in record.items() if k!='record_id'}):
                raise ValueError('Generated annotation identity changed or duplicated')
            record_ids.add(rid)
            if record['source_sha256']!={k:v['source_sha256'] for k,v in cameras.items()}:
                raise ValueError('Annotation and attached media hashes differ')
            a,b=record['start_s'],record['end_s']
            if a is not None or b is not None:
                if not finite(a) or not finite(b) or not 0<=a<b or not cameras or b>min(v['duration_s'] for v in cameras.values())+0.1:
                    raise ValueError('Annotation interval exceeds its video clock')
        if e['trajectory'] is not None:validate_trajectory(e['trajectory'],episode_id=identity,media=e['media'])
        if e['outcome'] is not None:
            if not isinstance(e['outcome'],dict):raise ValueError('Outcome must be a versioned artifact or null')
            # Full native result validation and source/task bindings also run at reward selection.
            from .rewarder import validate_binding
            validate_binding(e['outcome'],episode_id=identity,task=e['task'],cameras=e['media'])
    validate_reviews(bundle['reviews'])
    return payload


def make_bundle(episodes, *, reviews=(), provenance=None):
    raw=canonical_json(dict(schema_version='bir_rollout_payload_v1',episodes=episodes,provenance=provenance or {}))
    value=dict(schema_version='bir_rollout_bundle_v1',payload_json=raw,payload_sha256=hashlib.sha256(raw.encode()).hexdigest(),reviews=list(reviews))
    validate_bundle(value)
    return value


def load_bundle(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>MAX_BUNDLE_BYTES:raise ValueError('Bundle exceeds 64 MiB or is not a file')
    value=json.loads(path.read_text());validate_bundle(value);return value


def save_bundle(bundle,path):
    validate_bundle(bundle)
    raw=canonical_json(bundle)+'\n'
    if len(raw.encode())>MAX_BUNDLE_BYTES:raise ValueError('Bundle exceeds 64 MiB')
    path=Path(path)
    if path.exists() or path.is_symlink():raise ValueError('Bundle destination already exists')
    path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.bundle-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)
    finally:Path(tmp).unlink(missing_ok=True)
    return dict(path=str(path),sha256=file_sha256(path),payload_sha256=bundle['payload_sha256'])


def pack_run(directory, *, media_info, trajectories=None, outcomes=None):
    """Attach sidecars to existing identities without changing inference checkpoints."""
    rows=generated_records(directory);episodes={}
    for row in rows:
        name=row['episode_id']
        if name not in media_info:raise ValueError('Explicit media hashes and durations are required for each episode')
        e=episodes.setdefault(name,dict(episode_id=name,task=media_info[name].get('task',''),media=media_info[name]['media'],records=[],trajectory=(trajectories or {}).get(name),outcome=(outcomes or {}).get(name)))
        e['records'].append(row)
    if (set(trajectories or {})|set(outcomes or {}))-set(episodes):raise ValueError('Attachment has no matching episode')
    return make_bundle(list(episodes.values()),reviews=read_reviews(directory),provenance={'origin':'existing_annotation_run','generated_outputs_unchanged':True})


def attach(bundle, *, episode_id, trajectory=None, outcome=None):
    payload=validate_bundle(bundle);episodes=copy.deepcopy(payload['episodes'])
    matches=[e for e in episodes if e['episode_id']==episode_id]
    if len(matches)!=1:raise ValueError('Attachment episode not found')
    if trajectory is not None:matches[0]['trajectory']=trajectory
    if outcome is not None:matches[0]['outcome']=outcome
    return make_bundle(episodes,reviews=bundle['reviews'],provenance=dict(payload['provenance'],parent_payload_sha256=bundle['payload_sha256']))


def bundle_summary(bundle):
    p=validate_bundle(bundle);ids={r['record_id'] for e in p['episodes'] for r in e['records']}
    return dict(schema_version='bir_bundle_summary_v1',payload_sha256=bundle['payload_sha256'],episodes=len(p['episodes']),
                action_episodes=sum(e['trajectory'] is not None for e in p['episodes']),
                annotation_records=len(ids),review_events=len(bundle['reviews']),
                historical_unapplied_reviews=sum(r['record_id'] not in ids for r in bundle['reviews']),
                outcome_artifacts=sum(e['outcome'] is not None for e in p['episodes']),
                native_reward_artifacts=sum(e['outcome'] is not None and e['outcome'].get('status')=='complete' and e['outcome'].get('parse_valid') is True and e['outcome'].get('execution',{}).get('kind')=='native' for e in p['episodes']))
