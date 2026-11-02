"""Additive CPU data tooling; model dependencies remain optional."""
import json
from pathlib import Path
from .manifest import canonical_json


def add_commands(commands):
    p=commands.add_parser('bundle-check',help='Validate a portable bundle and its complete review journal')
    p.add_argument('--bundle',type=Path,required=True)
    p=commands.add_parser('bundle-pack',help='Attach portable media metadata to an existing annotation run')
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--media-info',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('bundle-attach',help='Attach a source-bound trajectory or outcome; preserve original predictions')
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--episode-id',required=True);p.add_argument('--output',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--trajectory',type=Path);group.add_argument('--outcome',type=Path)
    p=commands.add_parser('trajectory-import',help='Import the pinned real robomimic Lift state/action dataset')
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--demos',nargs='+',required=True);p.add_argument('--replay-receipts',type=Path);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('training-export',help='Export explicit rewards and recorded transitions to robomimic HDF5')
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reward-source',choices=['environment','flora-terminal'],required=True)
    p.add_argument('--boundary-mode',choices=['source-mask','terminated'],required=True)
    p=commands.add_parser('resources',help='Extract installed example, CPU recipe, config and browser resources')
    p.add_argument('--output-dir',type=Path,required=True)
    p=commands.add_parser('reward-prepare',help='Bind complete original videos and an explicit task without model execution')
    p.add_argument('--episode-id',required=True);p.add_argument('--task',required=True)
    p.add_argument('--video',type=Path,action='append',required=True);p.add_argument('--camera',action='append',required=True)
    p.add_argument('--clock-offset',type=float,action='append',required=True);p.add_argument('--whole-episode',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('reward-score',help='Independent terminal Rewarder worker; native execution is opt-in and unvalidated')
    p.add_argument('--request',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--allow-native',action='store_true');p.add_argument('--python',type=Path);p.add_argument('--source',type=Path);p.add_argument('--model-dir',type=Path)
    p.add_argument('--retry',action='store_true');p.add_argument('--timeout',type=int,default=1800)


def read(path):
    if path.stat().st_size>64*1024**2:raise ValueError('Input JSON exceeds64MiB')
    return json.loads(path.read_text())


def execute(args):
    from .bundle import load_bundle,save_bundle,make_bundle,pack_run,attach,bundle_summary
    if args.command=='bundle-check':return bundle_summary(load_bundle(args.bundle))
    if args.command=='bundle-pack':return save_bundle(pack_run(args.run_dir,media_info=read(args.media_info)),args.output)
    if args.command=='bundle-attach':
        return save_bundle(attach(load_bundle(args.bundle),episode_id=args.episode_id,trajectory=read(args.trajectory) if args.trajectory else None,outcome=read(args.outcome) if args.outcome else None),args.output)
    if args.command=='trajectory-import':
        from .trajectory import import_robomimic
        return save_bundle(make_bundle(import_robomimic(args.dataset,demos=args.demos,media_receipts=read(args.replay_receipts) if args.replay_receipts else None),provenance={'adapter':'robomimic_lift_v1'}),args.output)
    if args.command=='training-export':
        from .trajectory import export_training
        return export_training(load_bundle(args.bundle),args.output,reward_source=args.reward_source,boundary_mode=args.boundary_mode)
    if args.command=='resources':
        from .assets import extract_resources
        return extract_resources(args.output_dir)
    if args.command=='reward-prepare':
        from .rewarder import prepare_request
        request=prepare_request(args.episode_id,args.task,args.video,args.camera,whole_episode=args.whole_episode,clock_offsets_s=args.clock_offset)
        with args.output.open('x') as f:f.write(canonical_json(request)+'\n')
        return {'request_id':request['request_id'],'status':'prepared_not_executed','path':str(args.output)}
    if args.command=='reward-score':
        from .rewarder import run_terminal
        return run_terminal(read(args.request),args.output_dir,python=args.python,flora_source=args.source,model_dir=args.model_dir,allow_native=args.allow_native,retry=args.retry,timeout_s=args.timeout)
    raise ValueError('Unknown command')
