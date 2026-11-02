"""One explicit state/action trajectory convention and a robomimic export.

No model work, simulator calls, or inferred episode terminals occur here.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import os
import tempfile
import re
import struct

from .manifest import canonical_json, file_sha256
from .workflow import digest

DATASET_REVISION = '74fa018461f479cd9fd15b924a16103012096203'
DATASET_SHA256 = '2067777cb8b532e9263dd09fd6448c41cc31224bb27be4a3b734010ae13eb540'
DATASET_URL = f'https://huggingface.co/datasets/robomimic/robomimic_datasets/resolve/{DATASET_REVISION}/v1.5/lift/ph/low_dim_v15.hdf5'
SOURCE_MASK = 'robomimic_done_mode_2_success_or_last_recorded_sample'


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def matrix(value, name, n=None):
    if not isinstance(value, list) or not value or len(value) > 10000:
        raise ValueError(f'{name}: expected 1–10000 rows')
    if n is not None and len(value) != n:
        raise ValueError(f'{name}: transition count differs')
    width = len(value[0]) if isinstance(value[0], list) else 0
    if not 1 <= width <= 1024 or any(not isinstance(row, list) or len(row) != width or
                                   any(not finite(x) for x in row) for row in value):
        raise ValueError(f'{name}: finite rectangular numeric rows required')
    return len(value), width


def validate_trajectory(t, *, episode_id=None, media=()):
    if not isinstance(t, dict) or t.get('schema_version') != 'bir_trajectory_v1':
        raise ValueError('Unsupported trajectory version')
    if episode_id is not None and t.get('episode_id') != episode_id:
        raise ValueError('Trajectory episode identity differs')
    if t.get('trajectory_id') != digest({k:v for k,v in t.items() if k != 'trajectory_id'}):
        raise ValueError('Trajectory content identity changed')
    n, width = matrix(t['actions'], 'actions')
    if not t.get('action_spec', {}).get('representation') or len(t['action_spec'].get('units', [])) != width:
        raise ValueError('Action representation and per-axis units are required')
    obs, nxt = t['observations'], t['next_observations']
    if not isinstance(obs, dict) or not 1 <= len(obs) <= 32 or set(obs) != set(nxt):
        raise ValueError('Observations and next observations must have matching keys')
    for key in obs:
        if not isinstance(key,str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}',key) or key in {'caption','reward','flora_reward'}:
            raise ValueError('Unsupported policy observation key')
        if matrix(obs[key], key, n)[1] != matrix(nxt[key], 'next/'+key, n)[1]:
            raise ValueError('Next-observation shape differs')
    times=[]
    for name in ['observation_times_s','action_times_s','next_observation_times_s']:
        v=t[name]
        if not isinstance(v,list) or len(v)!=n or any(not finite(x) for x in v) or any(a>=b for a,b in zip(v,v[1:])):
            raise ValueError(f'{name}: finite strictly increasing explicit timestamps required')
        times.append(v)
    if any(not o <= a < e for o,a,e in zip(*times)):
        raise ValueError('Each action must follow its observation and precede its next observation')
    if any(abs(times[2][i]-times[0][i+1]) > 1e-7 for i in range(n-1)):
        raise ValueError('Missing or disconnected transitions; split the episode explicitly')
    for name in ['terminated','truncated']:
        if len(t[name]) != n or any(x is not None and type(x) is not bool for x in t[name]):
            raise ValueError('Termination/truncation must be explicit bool or unknown null')
        if any(x is True for x in t[name][:-1]):
            raise ValueError('An episode cannot continue after termination/truncation')
    if any(a is True and b is True for a,b in zip(t['terminated'],t['truncated'])):
        raise ValueError('A transition cannot be both terminated and time-limit truncated')
    rewards=t['environment_rewards']
    if rewards is not None and (len(rewards)!=n or any(not finite(x) for x in rewards)):
        raise ValueError('Missing or nonfinite environment rewards')
    stop=t.get('source_bootstrap_stop')
    if stop is not None and (len(stop)!=n or any(type(x) not in (int,bool) or x not in (0,1) for x in stop)):
        raise ValueError('Invalid source bootstrap mask')
    ids=t['transition_ids']
    expected=[digest([t['source']['sha256'],t['source']['demo_id'],i,times[0][i],times[2][i]]) for i in range(n)]
    if ids!=expected or len(set(ids))!=n:
        raise ValueError('Transition identities changed')
    cameras={x['camera']:x for x in media}
    mappings=t['clock_mappings']
    if len({m['camera'] for m in mappings})!=len(mappings):
        raise ValueError('Duplicate camera clock')
    for m in mappings:
        if not finite(m['scale']) or m['scale']<=0 or not finite(m['offset_s']) or not m.get('source'):
            raise ValueError('Invalid explicit video clock mapping')
        if m['camera'] not in cameras:
            raise ValueError('Clock mapping has no matching media')
        v=cameras[m['camera']]
        if m['video_sha256']!=v['source_sha256']:
            raise ValueError('Trajectory/video source hashes differ')
        if times[0][0]*m['scale']+m['offset_s'] < -1e-7 or times[2][-1]*m['scale']+m['offset_s'] > v['duration_s']+0.1:
            raise ValueError('Trajectory clock extends outside its bound video')
    if cameras and set(cameras)!={m['camera'] for m in mappings}:
        raise ValueError('Each attached camera needs an explicit clock mapping')
    bounds=t['action_spec'].get('bounds')
    if not isinstance(bounds,list) or len(bounds)!=2 or not all(finite(x) for x in bounds) or bounds[0]>=bounds[1]:
        raise ValueError('Explicit ordered action bounds are required')
    if any(not bounds[0]-1e-6<=x<=bounds[1]+1e-6 for row in t['actions'] for x in row):
        raise ValueError('Action is outside declared controller bounds')
    if 'simulator_states' in t:matrix(t['simulator_states'],'simulator_states',n)
    attrs=t.get('source_attributes',{})
    if not isinstance(attrs,dict) or set(attrs)-{'model_file','ep_meta','camera_info'} or any(not isinstance(v,str) or len(v)>2*1024**2 for v in attrs.values()):
        raise ValueError('Unsupported source attribute or reserved export attribute')
    values={'actions':t['actions'],'rewards':t['environment_rewards'],'dones':stop,'states':t.get('simulator_states')}
    values.update({f'{group}/{k}':v for group,data in [('obs',obs),('next_obs',nxt)] for k,v in data.items()})
    dtypes=t.get('array_dtypes',{})
    if not isinstance(dtypes,dict) or set(dtypes)-set(values):raise ValueError('Unknown source array dtype')
    for key,value in values.items():
        if value is not None:validate_dtype(value,dtypes.get(key,'<f8'),key)
    return n


def validate_dtype(values,dtype,name):
    # JSON preserves numeric values, not NumPy storage. Refuse any lossy cast.
    match=re.fullmatch(r'[<>=|]?([fiub])([1248])',dtype) if isinstance(dtype,str) else None
    if not match:raise ValueError(f'{name}: unsupported numeric dtype')
    kind,size=match[1],int(match[2])
    if (kind=='f' and size not in (4,8)) or (kind=='b' and size!=1):raise ValueError('Unsupported source dtype')
    def check(v):
        if isinstance(v,list):
            for x in v:check(x)
            return
        if not finite(v):raise ValueError(f'{name}: dtype requires finite numeric values')
        if kind=='f':
            try:restored=struct.unpack('f' if size==4 else 'd',struct.pack('f' if size==4 else 'd',v))[0]
            except (OverflowError,struct.error):raise ValueError(f'{name}: dtype overflow')
            if restored!=v:raise ValueError(f'{name}: dtype would change a recorded value')
        else:
            low,high=(0,1) if kind=='b' else ((0,2**(size*8)-1) if kind=='u' else (-2**(size*8-1),2**(size*8-1)-1))
            if int(v)!=v or not low<=v<=high:raise ValueError(f'{name}: dtype would change a recorded value')
    check(values)


def import_robomimic(path, *, demos, media_receipts=None):
    """Import only the pinned official Lift source; preserve exact source arrays."""
    import h5py
    import numpy as np
    path=Path(path)
    if file_sha256(path)!=DATASET_SHA256:
        raise ValueError('This adapter accepts only the pinned Lift dataset checksum')
    if not demos or len(demos)>12 or len(set(demos))!=len(demos):
        raise ValueError('Select 1–12 distinct recorded demonstrations')
    episodes=[]
    with h5py.File(path,'r') as f:
        env=json.loads(f['data'].attrs['env_args'])
        hz=float(env['env_kwargs']['control_freq'])
        if env['env_name']!='Lift' or hz!=20:
            raise ValueError('Unsupported task or controller clock')
        for name in demos:
            if name not in f['data'] or '/' in name:
                raise ValueError('Unknown source demonstration')
            d=f['data'][name]; n=len(d['actions']); states=d['states'][()]
            times=states[:,0].tolist()
            if not np.allclose(np.diff(times),1/hz,rtol=0,atol=1e-7):
                raise ValueError('Recorded simulator clock is discontinuous')
            next_times=[float(x+1/hz) for x in times]
            observations={k:d['obs'][k][()].tolist() for k in d['obs']}
            next_observations={k:d['next_obs'][k][()].tolist() for k in d['next_obs']}
            dtypes={key:d[key].dtype.str for key in ['actions','rewards','dones','states']}
            dtypes.update({f'{group}/{k}':d[group][k].dtype.str for group in ['obs','next_obs'] for k in d[group]})
            t=dict(schema_version='bir_trajectory_v1',episode_id=name,
                   source=dict(dataset='robomimic/robomimic_datasets',revision=DATASET_REVISION,sha256=DATASET_SHA256,demo_id=name),
                   observations=observations,next_observations=next_observations,actions=d['actions'][()].tolist(),
                   observation_times_s=times,action_times_s=list(times),next_observation_times_s=next_times,
                   terminated=[None]*n,truncated=[None]*n,environment_rewards=d['rewards'][()].tolist(),
                   source_bootstrap_stop=d['dones'][()].astype(int).tolist(),
                   simulator_states=states.tolist(),array_dtypes=dtypes,
                   boundary=dict(kind='recording_end',source_mask_semantics=SOURCE_MASK,
                                 termination_evidence='not recorded; source dones are success OR final stored sample'),
                   action_spec=dict(representation='normalized OSC_POSE world-frame delta plus gripper command',
                                    units=['normalized translation command']*3+['normalized rotation command']*3+['normalized gripper command'],
                                    controller=env['env_kwargs']['controller_configs'],bounds=[-1,1]),
                   environment_metadata=env,clock_mappings=[],
                   source_attributes={k:str(v) for k,v in d.attrs.items() if k!='num_samples'})
            if any(abs(x)>1.000001 for row in t['actions'] for x in row):
                raise ValueError('Actions exceed pinned controller bounds')
            t['transition_ids']=[digest([DATASET_SHA256,name,i,times[i],next_times[i]]) for i in range(n)]
            media=[]
            receipt=(media_receipts or {}).get(name)
            if receipt:
                if receipt['dataset_sha256']!=DATASET_SHA256 or receipt['demo_id']!=name or receipt['rendered_frame_times_s']!=times:
                    raise ValueError('Replay receipt does not bind the recorded source clock')
                m=receipt['clock_mapping']
                t['clock_mappings']=[dict(m,camera=receipt['camera'],video_sha256=receipt['video_sha256'])]
                media=[dict(camera=receipt['camera'],filename=receipt.get('filename','replay.mp4'),source_sha256=receipt['video_sha256'],duration_s=receipt['duration_s'])]
            t['trajectory_id']=digest(t);validate_trajectory(t,episode_id=name,media=media)
            episodes.append(dict(episode_id=name,task='Lift the cube with the Panda robot.',media=media,records=[],trajectory=t,outcome=None))
    return episodes


def export_training(bundle, output, *, reward_source, boundary_mode):
    """Write one immutable robomimic HDF5 export, never substitute missing data."""
    import h5py
    import numpy as np
    from .bundle import validate_bundle
    payload=validate_bundle(bundle)
    if reward_source not in ('environment','flora-terminal') or boundary_mode not in ('source-mask','terminated'):
        raise ValueError('Explicit supported reward source and bootstrap semantics required')
    target=Path(output)
    if target.exists() or target.is_symlink():
        raise ValueError('Training export destination already exists')
    included=[]; excluded=[]
    for e in payload['episodes']:
        if e['trajectory'] is None:
            excluded.append(dict(episode_id=e['episode_id'],reason='No recorded robot transitions'));continue
        t=e['trajectory'];validate_trajectory(t,episode_id=e['episode_id'],media=e['media']);n=len(t['actions'])
        if boundary_mode=='source-mask':
            if t.get('source_bootstrap_stop') is None or t['boundary'].get('source_mask_semantics')!=SOURCE_MASK:
                raise ValueError('No supported explicit source bootstrap semantics')
            dones=t['source_bootstrap_stop']
        else:
            if any(x is None for x in t['terminated']+t['truncated']):
                raise ValueError('Unknown termination/truncation cannot be converted to training dones')
            dones=[int(x) for x in t['terminated']]  # time-limit truncation bootstraps from recorded next_obs
        mapping=None
        if reward_source=='environment':
            if t['environment_rewards'] is None:raise ValueError('Environment rewards are absent')
            rewards=t['environment_rewards']
        else:
            from .rewarder import terminal_reward
            if t['terminated'][-1] is not True or t['truncated'][-1] is not False or t['boundary']['kind']!='episode_terminal':
                raise ValueError('Terminal FLORA reward requires an evidenced true episode ending, not a recording cut')
            outcome=e['outcome']
            if not outcome:raise ValueError('Native outcome evidence unavailable')
            mapping=terminal_reward(outcome,mapping='expected_progress_v1')
            value=mapping['value'] if isinstance(mapping,dict) else mapping
            if not finite(value):raise ValueError('Invalid derived reward')
            rewards=[0.0]*n;rewards[-1]=value
        included.append((e,rewards,dones,mapping))
    if not included:raise ValueError('No action-bearing episodes available for training')
    env=included[0][0]['trajectory']['environment_metadata']
    if any(e['trajectory']['environment_metadata']!=env for e,_,_,_ in included):
        raise ValueError('Mixed environments require separate exports')
    provenance=dict(schema_version='bir_training_export_v1',bundle_payload_sha256=bundle['payload_sha256'],
                    review_journal_sha256=digest(bundle['reviews']),reward_source=reward_source,boundary_mode=boundary_mode,
                    reward_mapping='original_environment_rewards_v1' if reward_source=='environment' else 'expected_progress_v1',
                    trainer='robomimic IQL',observations_exclude_retrospective_labels=True,
                    included=[dict(episode_id=e['episode_id'],trajectory_id=e['trajectory']['trajectory_id']) for e,_,_,_ in included],
                    exclusions=excluded,bootstrap_rule='reward + discount * (1-dones) * V(next_obs)',
                    truncation_rule='True time-limit truncation bootstraps; source-mask mode preserves the explicitly selected original dataset mask.')
    provenance['export_id']=digest(provenance)
    target.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.training-',dir=target.parent);os.close(fd)
    try:
        with h5py.File(tmp,'w') as f:
            data=f.create_group('data');data.attrs['env_args']=canonical_json(env);data.attrs['total']=sum(len(e['trajectory']['actions']) for e,_,_,_ in included)
            f.attrs['bir_provenance']=canonical_json(provenance)
            for index,(e,rewards,dones,mapping) in enumerate(included):
                t=e['trajectory'];g=data.create_group(f'demo_{index}');n=len(t['actions']);g.attrs['num_samples']=n
                g.attrs['bir_episode_id']=e['episode_id'];g.attrs['bir_trajectory_id']=t['trajectory_id']
                for k,v in t.get('source_attributes',{}).items():g.attrs[k]=v
                dtypes=t.get('array_dtypes',{})
                for key,values in [('actions',t['actions']),('rewards',rewards),('dones',dones)]:
                    dtype=dtypes.get(key,'float64') if reward_source=='environment' or key=='actions' else ('int64' if key=='dones' else 'float64')
                    g.create_dataset(key,data=np.asarray(values,dtype=dtype))
                for group,key in [('obs','observations'),('next_obs','next_observations')]:
                    for k,v in t[key].items():g.create_dataset(f'{group}/{k}',data=np.asarray(v,dtype=dtypes.get(f'{group}/{k}','float64')))
                if 'simulator_states' in t:g.create_dataset('states',data=np.asarray(t['simulator_states'],dtype=dtypes.get('states','float64')))
                g.create_dataset('bir/transition_ids',data=np.asarray(t['transition_ids'],dtype=h5py.string_dtype()))
                for key in ['observation_times_s','action_times_s','next_observation_times_s','environment_rewards','source_bootstrap_stop']:
                    if t.get(key) is not None:g.create_dataset('bir/'+key,data=np.asarray(t[key]))
                for key in ['terminated','truncated']:g.create_dataset('bir/'+key,data=np.asarray([-1 if x is None else int(x) for x in t[key]],dtype='int8'))
                g.attrs['bir_metadata']=canonical_json({k:t[k] for k in ['source','boundary','action_spec','clock_mappings']})
                g.attrs['bir_annotations']=canonical_json(e['records']);g.attrs['bir_outcome']=canonical_json(e['outcome']);g.attrs['bir_reward_mapping']=canonical_json(mapping)
            f.create_dataset('mask/train',data=np.asarray([f'demo_{i}'.encode() for i in range(len(included))]))
            f.attrs['bir_review_history']=canonical_json(bundle['reviews'])
        os.link(tmp,target)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return dict(provenance,output=str(target),sha256=file_sha256(target),transitions=sum(len(e['trajectory']['actions']) for e,_,_,_ in included))
