"""Cloud asset integrity and print-library handoff; no simulation or training."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Validate numeric/image assets on Vast.ai only')
import numpy as np
import trimesh
from PIL import Image,ImageStat


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    specs=json.loads((a.root/'batch.json').read_text())['assets'];rows=[];library=[]
    for spec in specs:
        folder=a.root/spec['id'];manifest=json.loads((folder/'manifest.json').read_text());validation=manifest['validation']
        visual=trimesh.load(folder/'visual.obj',force='mesh',process=False)
        printed=trimesh.load(folder/'print_mm.stl',force='mesh')
        print_error=float(np.max(np.abs(printed.bounds-visual.bounds*1000)))
        images=[]
        for name in range(24):
            path=folder/'views'/f'{name:02d}.png'
            if not path.exists():continue
            with Image.open(path) as im:
                im.load();assert im.size==(256,256)
                rgb=np.asarray(im.convert('RGB'));stat=ImageStat.Stat(im.convert('RGB'))
                images.append(dict(file=str(path.relative_to(a.root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    mean_rgb=stat.mean,stddev_rgb=stat.stddev,any_channel_saturation_fraction=float(np.mean(np.max(rgb,axis=2)>=254))))
        render_path=folder/'render_manifest.json'
        render=json.loads(render_path.read_text()) if render_path.exists() else {}
        exact_mesh=render.get('mesh_sha256')==hashlib.sha256((folder/'visual.obj').read_bytes()).hexdigest()
        views=render.get('view_metadata',[])
        fixed_scale=render.get('metric_scale_shared',False) and len(views)==24 and all(abs(v['ortho_scale_m']-.135)<1e-7 for v in views)
        valid=(validation['mass_within_1pct'] and validation['inertia_within_2pct'] and validation['collision_sample_within_1mm']
               and validation['mujoco_load'] and printed.is_watertight and print_error<1e-3 and len(images)==24 and exact_mesh and fixed_scale)
        rows.append(dict(id=spec['id'],split=spec['split'],valid=bool(valid),print_bounds_max_error_mm=print_error,
            print_watertight=bool(printed.is_watertight),views=len(images),exact_mesh_render=exact_mesh,fixed_metric_scale=bool(fixed_scale),images=images,physics=validation))
        library.append(dict(id=spec['id'],split=spec['split'],print_file=f'{spec["id"]}/print_mm.stl',units='mm',
            extent_x_mm=float(visual.extents[0]*1000),extent_y_mm=float(visual.extents[1]*1000),extent_z_mm=float(visual.extents[2]*1000),
            simulation_mass_g=manifest['inertial']['mass_kg']*1000,measured_print_mass_g='',material_class=spec['material']['name']))
    complete=len(rows)==50 and all(r['valid'] for r in rows)
    result=dict(state='VALIDATED_V0_SUBSET' if complete else 'PARTIAL_OR_INVALID',n=len(rows),
        n_train=sum(r['split']=='train' for r in rows),n_held_out=sum(r['split']=='held_out' for r in rows),
        valid_assets=sum(r['valid'] for r in rows),views=sum(r['views'] for r in rows),rows=rows,
        limits=['Printed objects are not built or weighed; measured-weight column intentionally blank',
                'Uniform rigid material and zero COM-offset subset; no heterogeneous patch friction',
                '15-degree patch rule retained by user; canonical one-patch examples conflict',
                '256px encoder views, not high-resolution presentation images',
                'No feasibility, z, controller, Realistic sensor or hardware result'])
    (a.root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    with (a.root/'OBJECT_LIBRARY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(library[0]));writer.writeheader();writer.writerows(library)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
