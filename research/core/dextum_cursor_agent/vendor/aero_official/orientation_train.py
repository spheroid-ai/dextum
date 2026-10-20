"""Camera-only actor inputs, full DR, and independent CPU progress gates."""
import argparse
import functools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from . import train as runner
from .orientation_curriculum import OrientationCurriculum,HistoryOrientation,SequenceOrientation,STAGES
from .orientation_policy import network
from mujoco_playground import wrapper


def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--restore',type=Path,required=True)
    p.add_argument('--stage',type=int,default=0);p.add_argument('--cpu-checks',action='store_true')
    p.add_argument('--history',type=int,choices=(1,3),default=1)
    p.add_argument('--experiment',default=None)
    p.add_argument('--sequence',action='store_true')
    a,remaining=p.parse_known_args();output=Path(remaining[remaining.index('--output')+1])
    evaluations=3
    if '--steps' in remaining:
        evaluations=math.ceil(int(remaining[remaining.index('--steps')+1])/6_553_600)+1
    def check(steps,path):
        path.with_suffix('.interface.json').write_text(json.dumps(dict(loader='work.aero_official.orientation_policy.OrientationPolicy',
            actor=34*a.history,history=a.history,critic=89,normalization='fixed_camera_goal_v1',stage=STAGES[a.stage].__dict__,hardware=False),indent=2)+'\n')
        if not steps or not a.cpu_checks:return
        dest=output.parent/'checks'/str(steps);env=dict(os.environ,JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4')
        subprocess.run([sys.executable,'-m','work.aero_official.orientation_evaluate','--checkpoint',str(path),
            '--stage',str(a.stage),'--output',str(dest),'--trials','40'],check=True,timeout=300,env=env)
        r=json.loads((dest/'report.json').read_text());stop=r['drops']>12 or (steps>=26_000_000 and r['reaches']<8)
        (dest/'gate.json').write_text(json.dumps(dict(stop=stop,successes=r['successes'],drops=r['drops'],reaches=r['reaches']))+'\n')
        if stop:raise RuntimeError('Orientation progress gate failed; stop and diagnose')
    sys.argv=[sys.argv[0],*remaining]
    if a.sequence and a.history!=3:raise ValueError('Sequence qualification uses the H3 actor')
    env_type=SequenceOrientation if a.sequence else OrientationCurriculum if a.history==1 else HistoryOrientation
    runner.main(restore_path=a.restore,actor_factory=network,environment_factory=functools.partial(env_type,a.stage),
        actor_metadata=dict(kind='orientation_mlp_fixed_visual_scaling',history=a.history,observation_size=34*a.history,critic_size=89),
        config_overrides={'num_evals':evaluations,'discounting':.995,'entropy_cost':.01},
        registration=dict(experiment=a.experiment or (f'R0-OR1-H3-S{a.stage}' if a.history==3 else 'R0-OR1' if a.stage==0 else f'R0-OR1-S{a.stage}'),
            stage=a.stage,history=a.history,sequence=a.sequence,stage_config=STAGES[a.stage].__dict__,camera='qualified',physics='full_official_dr'),
        wrap_env_factory=functools.partial(wrapper.wrap_for_brax_training,full_reset=True),checkpoint_hook=check)


if __name__=='__main__':main()
