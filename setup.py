"""Build a wheel with the same reviewed resources as the source release."""
from pathlib import Path
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py

PATTERNS = ['configs/*.json','configs/*.yaml','configs/*.txt','configs/rl/*',
            'examples/precomputed-homer41.json','examples/rl/lift-iql-smoke/*','examples/rl/*.json','examples/rl/*.hdf5','examples/rl/*.md',
            'scripts/rl_cpu_smoke.py','scripts/prepare_rollout_example.py',
            'web/*.html','web/*.js','web/*.css','web/*.md','web/examples/*',
                  'docs/*.md','docs/images/*','reports/*.json','LICENSE*','README.md']

class BuildResources(build_py):
    def run(self):
        super().run()
        root=Path(__file__).parent
        for pattern in PATTERNS:
            for path in root.glob(pattern):
                if path.is_file():
                    target=Path(self.build_lib)/'bir/data'/path.relative_to(root)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(path,target)

setup(cmdclass={'build_py':BuildResources})
