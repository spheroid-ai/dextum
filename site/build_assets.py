"""Presentation assets from reported results. No simulation or statistical fitting."""
from pathlib import Path
import csv
import hashlib
import json
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DIST = HERE / 'dist'
ASSETS = DIST / 'assets'
from read_scorecards import load_results
snapshot = load_results()
rows = snapshot['results']
def result(group, arm, seed=1):
    return next(r for r in rows if r['group']==group and r['arm']==arm and r['seed']==seed)
def values(group):
    return [r['success_percent'] for r in rows if r['group']==group]

INK, MUTED, TEAL, RUST, GREY = '#213a49', '#60717b', '#326f79', '#b95736', '#9aabb4'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                     'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': INK,
                     'axes.edgecolor': '#dae2e4', 'svg.fonttype': 'none',
                     'savefig.facecolor': 'white', 'figure.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'axes.spines.bottom': False})

def save(fig, name):
    fig.savefig(ASSETS / (name + '.svg'), metadata={'Date': '2026-10-15'})
    fig.savefig(ASSETS / (name + '.pdf'), metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(ASSETS / (name + '.png'), dpi=160)
    plt.close(fig)

def horizontal(name, labels, values, colors, notes=None, size=(10.4,5.6), title=None):
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=.28, right=.93, top=.86, bottom=.16)
    y=list(range(len(labels)))
    ax.barh(y, values, color=colors, height=.52, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_ylim(len(labels)-.4,-.7)
    ax.set_xlim(0,108)
    ax.set_xticks([0,20,40,60,80,100],['0','20','40','60','80','100'])
    ax.set_xlabel('Success on the shared bank (%)',labelpad=15)
    ax.tick_params(axis='both',length=0,pad=10)
    ax.xaxis.grid(color='#e7edef',linewidth=.8,zorder=0)
    for i,v in enumerate(values):
        ax.text(v+1.5,i,f'{v:.1f}',va='center',ha='left',fontweight='bold',fontsize=12)
        if notes: ax.text(2,i+.30,notes[i],va='center',fontsize=9,color=MUTED)
    if title: fig.text(.05,.94,title,fontsize=13,fontweight='bold',ha='left')
    save(fig,name)

horizontal('model-comparison', ['Bare Aero · Ideal','Sensorised v1 · Ideal','Sensorised v2 · Ideal','Sensorised v2 · Realistic','Sensorised v2 · Rigid / Ideal'],
           values('scratch_rotation'),[GREY,'#cc957d',TEAL,'#244a58','#73959c'],
           title='Full-budget rotation · one seed per arm')

fig,ax=plt.subplots(figsize=(10.4,5.6));fig.subplots_adjust(left=.09,right=.96,bottom=.17,top=.82)
x=[0,1,2];touch=[result('touch_recheck','c_touch',i)['success_percent'] for i in [1,2,3]];none=[result('touch_recheck','c_notouch',i)['success_percent'] for i in [1,2,3]];w=.28
ax.bar([i-w/2-.015 for i in x],touch,width=w,color=TEAL,label='C-touch',zorder=3)
ax.bar([i+w/2+.015 for i in x],none,width=w,color=GREY,label='C-no-touch',zorder=3)
for i in x:
    ax.text(i-w/2-.015,touch[i]+2,f'{touch[i]:g}',ha='center',fontsize=12,fontweight='bold')
    ax.text(i+w/2+.015,none[i]+2,f'{none[i]:g}',ha='center',fontsize=12,fontweight='bold')
ax.set_xticks(x,[f"Seed {i}\np = {result('touch_recheck','c_touch',i)['paired_p']:.4g}" for i in [1,2,3]]);ax.set_ylim(0,103)
ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('Success (%)');ax.tick_params(length=0,pad=9)
ax.yaxis.grid(color='#e7edef',zorder=0);ax.legend(frameon=False,ncol=2,loc='upper left',bbox_to_anchor=(0,1.17))
fig.text(.05,.95,'Realistic v2 · matched 20.3M warm starts',fontsize=13,fontweight='bold')
save(fig,'touch-seeds')

horizontal('channel-ablation',['Unmasked continuation','Normal-only','Palm map removed'],values('channel_ablation'),
           [TEAL,'#69969c',GREY],['Equal-budget baseline','−2.5 points · paired p = .424','0 points · paired p = 1.0'],
           title='Channel ablations · v2 · one seed')
horizontal('oracle-o1',['Privileged ceiling','Structured full','Information-equivalent flat','Contact-only ablation','Patch-identity ablation'],
           [r['success_percent'] for r in rows if r['group']=='oracle_o1' and r['seed']==1],[GREY,TEAL,'#8aa6ae','#bb997f','#b7aca0'],
           ['14 drops / 200','5 drops / 200','3 drops / 200','2 drops / 200','2 drops / 200'],
           size=(11.8,5.5),title='Oracle O1 · seed 1 · separate 2,048-world protocol')
horizontal('student',['Tactile teacher','Sensorless student','No-touch reference · seed 1','No-touch reference · seed 2','No-touch reference · seed 3'],
           [result('scratch_rotation','Sensorised Realistic')['success_percent'],result('sensorless','DAgger student')['success_percent'],*none],[TEAL,RUST,GREY,GREY,GREY],
           title='Sensorless transfer diagnostic · distinct training protocols')

(DIST/'data/results.json').write_text(json.dumps(snapshot,indent=2)+'\n')
with (DIST/'data/results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
with zipfile.ZipFile(ASSETS/'dextum-figures.zip','w',zipfile.ZIP_DEFLATED) as z:
    for stem in ['model-comparison','touch-seeds','channel-ablation','oracle-o1','student']:
        for suffix in ['.svg','.pdf']:z.write(ASSETS/(stem+suffix),stem+suffix)
print(f'Built five figures from {len(rows)} explicitly admitted rows and verified frozen reports.')
