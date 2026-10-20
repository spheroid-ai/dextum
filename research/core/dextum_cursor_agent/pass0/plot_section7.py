"""Remote-only scientific figure from completed, immutable Pass 0 measurements."""
import json
import os
from pathlib import Path
import sys
if not os.environ.get('CONTAINER_ID'):
    raise RuntimeError('Render on Vast.ai only')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root = Path(sys.argv[1])
    evaluation = json.loads((root/'e3_evaluation/paired_evaluators.json').read_text())
    demo = json.loads((root/'demo/results.json').read_text())
    assert evaluation['finite'] and demo['state'] == 'COMPLETE'
    rows = [json.loads(x) for x in (root/'e3/c_touch/metrics.jsonl').read_text().splitlines()]
    monitor = [r for r in rows if 'eval/episode_e3/success' in r['metrics']]
    output = root/'figures'; output.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False,
                         'axes.spines.right':False, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={'width_ratios':[1.35, 1, 1.35]})
    blue, orange = '#2369a6', '#db7e24'
    axes[0].plot([r['steps']/1e6 for r in monitor],
                 [100*r['metrics']['eval/episode_e3/success'] for r in monitor],
                 color=blue, linewidth=2, marker='o', markersize=3)
    axes[0].set(title='Training monitor · 128 starts', xlabel='Training steps (million)', ylabel='Success (%)', ylim=(0,105))
    counts = evaluation['success_counts']
    values = [counts['training']/2, counts['post_training']/2]
    axes[1].bar(['Training\nwrapper', 'Faithful\ndirect MJX'], values, color=[blue, orange], width=.58)
    for i, value in enumerate(values): axes[1].text(i, value+2, f'{round(value*2)}/200', ha='center')
    axes[1].set(title='Final policy · shared 200-start bank', ylim=(0,112))
    ids = ['p0_cube55','p0_rounded48','p0_cylinder38x70']
    names = ['Cube\n55 mm','Rounded cube\n48 mm','Cylinder\n38 × 70 mm']
    for blackout, color, shift, label in [(False,blue,-.18,'Vision on'),(True,orange,.18,'Off after contact')]:
        selected = [next(r for r in demo['rows'] if r['asset']==asset and r['blackout']==blackout) for asset in ids]
        locations = [i+shift for i in range(3)]
        heights = [100*r['success'] for r in selected]
        axes[2].bar(locations, heights, width=.34, color=color, label=label)
        for x,h,r in zip(locations,heights,selected): axes[2].text(x,h+2,f"{round(r['success']*r['n'])}/{r['n']}",ha='center',fontsize=8)
    axes[2].set(xticks=range(3), xticklabels=names, ylim=(0,112), title='Demo · historical actor + scripts')
    axes[2].legend(frameon=False,fontsize=8,loc='upper right')
    for ax in axes:
        ax.set_axisbelow(True); ax.grid(axis='y', alpha=.2); ax.set_yticks([0,25,50,75,100])
    fig.suptitle('DEXTUM Pass 0 — corrected-physics E3 and three-shape demo', fontsize=15, fontweight='bold')
    fig.text(.5,.02,'One C-touch seed. Demo: native MuJoCo, Ideal camera and 11 lumped pads; 16 starts per shape/vision cell.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.91])
    fig.savefig(output/'pass0_results.png',dpi=240,facecolor='white')
    fig.savefig(output/'pass0_results.svg',facecolor='white')
    plt.close(fig)
    print(str(output),flush=True)


if __name__ == '__main__': main()
