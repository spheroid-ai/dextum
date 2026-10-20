"""Export the registered curves without fitting or changing their thresholds."""
import argparse,json,os
from pathlib import Path

def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 r=json.loads(a.result.read_text());assert r['state']=='COMPLETE'
 fields=[('optical_floor_n','Optical detection floor','N',1),('delay_ticks','Delay','ms',5),('optical_noise_n','Optical random noise','N standard deviation',1),('direction_degrees','Direction perturbation','degrees, uniform half-width',1),('pvdf_false_alarm','PVDF false alarms','nominal % per 50 ms',100)]
 fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
 for ax,(field,title,unit,scale) in zip(axes.flat,fields):
  rows=[x for x in r['rows'] if x['field']==field]
  xs=[0]+[x['level']*scale for x in rows];ys=[r['fresh_ideal_count']/2]+[x['count']/2 for x in rows]
  ax.plot(xs,ys,'o-',color='#2166ac',linewidth=1.5,markersize=4)
  ax.axhline(64,color='#555555',linestyle=':',label='Registered Ideal reference 64%')
  ax.axhline(59,color='#b2182b',linestyle='--',label='Fixed cutoff 59%')
  ax.set(title=title,xlabel=unit,ylabel='Success (%)',ylim=(20,72));ax.grid(alpha=.16)
  t=r['thresholds'][field]
  note=('First tested crossing: '+str(t['level']*scale)+' '+unit) if 'level' in t else 'No crossing in tested range'
  ax.text(.03,.06,note,transform=ax.transAxes,fontsize=8)
 axes.flat[-1].axis('off')
 axes.flat[-1].text(0,.9,'Frozen Ideal actor\nCorrected sensorised hand\n200 shared starts; one seed\nEvery other sensor error Ideal\nNo optimizer updates\n\nFresh neutral: 62%\nReference stays 64%; cutoff 59%\nNonmonotonic curves; no interpolation\n0.02 N is already a failing noise level.',va='top',fontsize=10)
 handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='lower left',frameon=False,fontsize=9)
 fig.suptitle('DEXTUM §16 — isolated sensor errors',fontsize=15)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 for suffix in ['.pdf','.png']:
  target=a.output.with_suffix(suffix);assert not target.exists();fig.savefig(target,dpi=160)

if __name__=='__main__':
 if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Vast only')
 main()
