"""Resources work in wheels as well as source checkouts."""
from pathlib import Path
import shutil


def asset_path(relative):
    installed = Path(__file__).parent / 'data' / relative
    source = Path(__file__).resolve().parents[2] / relative
    return installed if installed.is_file() else source


def extract_resources(output):
    source = Path(__file__).parent / 'data'
    target = Path(output)
    if target.exists() or target.is_symlink():
        raise ValueError('Resource destination already exists')
    if source.is_dir():
        shutil.copytree(source,target)
    else:
        root=Path(__file__).resolve().parents[2]
        patterns=['configs/*.json','configs/*.yaml','configs/*.txt','configs/rl/*',
                  'examples/precomputed-homer41.json','examples/rl/lift-iql-smoke/*','examples/rl/*.json','examples/rl/*.hdf5','examples/rl/*.md',
                  'scripts/rl_cpu_smoke.py','scripts/prepare_rollout_example.py',
                  'web/*.html','web/*.js','web/*.css','web/*.md','web/examples/*',
                  'docs/*.md','docs/images/*','reports/*.json','LICENSE*','README.md']
        files={p for pattern in patterns for p in root.glob(pattern) if p.is_file()}
        if not files:raise ValueError('Package resources are unavailable')
        target.mkdir(parents=True)
        for path in sorted(files):
            dest=target/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dest)
    return {'status':'ready','directory':str(target),'new_model_calls':0}
