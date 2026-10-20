"""Blender procedural F1 geometry and CPU-only turntable renders on Vast.ai."""
import json
import hashlib
import math
import os
from pathlib import Path
import sys
import time
if not os.environ.get('CONTAINER_ID'):raise RuntimeError('Cloud Blender only')
import bpy
import numpy as np
from mathutils import Vector

args=sys.argv[sys.argv.index('--')+1:];root=Path(args[0]);mode=args[1]
specs=json.loads((root/'batch.json').read_text())['assets']
for spec in specs:
    folder=root/spec['id'];folder.mkdir(exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.unit_settings.system='METRIC'
    if mode=='geometry':
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=6,radius=1.)
        obj=bpy.context.object;obj.name=spec['id'];e1,e2=spec['epsilon'];axes=np.asarray(spec['semi_axes_m'])
        xyz=np.array([v.co[:] for v in obj.data.vertices],dtype=np.float64);xyz/=np.linalg.norm(xyz,axis=1,keepdims=True)
        factor=((np.abs(xyz[:,0])**(2/e2)+np.abs(xyz[:,1])**(2/e2))**(e2/e1)+np.abs(xyz[:,2])**(2/e1))**(-e1/2)
        xyz*=factor[:,None]*axes
        for v,point in zip(obj.data.vertices,xyz):v.co=point
        obj.data.update()
    else:
        # Render the exact exported geometry; never regenerate a substitute.
        vertices=[];faces=[]
        for line in (folder/'visual.obj').read_text().splitlines():
            fields=line.split()
            if fields[0]=='v':vertices.append(tuple(map(float,fields[1:])))
            if fields[0]=='f':faces.append(tuple(int(x)-1 for x in fields[1:]))
        mesh=bpy.data.meshes.new(spec['id']);mesh.from_pydata(vertices,[],faces);mesh.update()
        obj=bpy.data.objects.new(spec['id'],mesh);scene.collection.objects.link(obj)
        bpy.context.view_layer.objects.active=obj;obj.select_set(True)
        if not bpy.app.build_options.opencolorio:raise RuntimeError('Color-managed official renderer required')
    material=bpy.data.materials.new(spec['material']['name']);material.use_nodes=True
    principled=material.node_tree.nodes.get('Principled BSDF')
    principled.inputs['Base Color'].default_value=spec['material']['color']
    principled.inputs['Roughness'].default_value=spec['material']['roughness']
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:polygon.use_smooth=True
    if mode=='geometry':
        # Preserve the exact Blender-exported triangle mesh in SI units.
        points=[v.co[:] for v in obj.data.vertices];faces=[list(p.vertices) for p in obj.data.polygons]
        assert all(len(f)==3 for f in faces)
        (folder/'visual.obj').write_text(''.join('v %.12g %.12g %.12g\n'%tuple(p) for p in points)+
            ''.join('f %d %d %d\n'%tuple(i+1 for i in f) for f in faces))
        bpy.ops.export_scene.gltf(filepath=str(folder/'visual.glb'),export_format='GLB',use_selection=True)
        print(json.dumps(dict(id=spec['id'],vertices=len(points),faces=len(faces),state='GEOMETRY')),flush=True)
        continue
    if mode!='render':raise ValueError(mode)
    views=folder/'views';views.mkdir(exist_ok=True)
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=16
    scene.view_settings.view_transform='Standard';scene.view_settings.look='Medium High Contrast' if bpy.app.version[0]<4 else 'None'
    scene.cycles.use_denoising=True;scene.render.threads_mode='FIXED';scene.render.threads=4
    scene.render.resolution_x=256;scene.render.resolution_y=256;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.film_transparent=False
    scene.world=bpy.data.worlds.new('studio');scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.07,.09,.13,1.)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.35
    bpy.ops.object.camera_add(location=(.18,0,.11));camera=bpy.context.object;scene.camera=camera
    camera.data.type='ORTHO';camera.data.ortho_scale=.135  # Fixed metric scale across the entire batch.
    view_metadata=[]
    lights=[]
    for i,loc in enumerate(((.12,-.16,.2),(-.1,.1,.12),(.0,.15,-.04))):
        bpy.ops.object.light_add(type='AREA',location=loc);light=bpy.context.object;light.data.energy=(.5,.35,.15)[i]
        light.data.size=.12;light.rotation_euler=(-light.location).to_track_quat('-Z','Y').to_euler();lights.append(light)
    for i in range(24):
        if os.environ.get('ASSET_STOP_AT') and time.time()>float(os.environ['ASSET_STOP_AT']):raise TimeoutError('Bounded cloud deadline')
        target=views/f'{i:02d}.png'
        rng=np.random.default_rng(np.random.SeedSequence([spec['seed'],int(spec['id'].split('_')[1]),i]))
        theta=2*math.pi*(i%12)/12;elevation=math.radians(25 if i<12 else -25)
        camera.location=(.18*math.cos(theta)*math.cos(elevation),.18*math.sin(theta)*math.cos(elevation),.18*math.sin(elevation))
        camera.rotation_euler=(-camera.location).to_track_quat('-Z','Y').to_euler()
        for light,base in zip(lights,(.5,.35,.15)):light.data.energy=base*float(rng.uniform(.7,1.3))
        scene.render.filepath=str(target)
        if not target.exists():bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        view_metadata.append(dict(id=i,camera_to_world=[list(row) for row in camera.matrix_world],
            ortho_scale_m=camera.data.ortho_scale,light_energy_w=[float(light.data.energy) for light in lights]))
    (folder/'render_manifest.json').write_text(json.dumps(dict(blender_version=bpy.app.version_string,views=24,view_metadata=view_metadata,projection='orthographic',metric_scale_shared=True,resolution=[256,256],samples=16,device='CPU',mesh_sha256=hashlib.sha256((folder/'visual.obj').read_bytes()).hexdigest(),role='v0 encoder dataset; not presentation renders'),indent=2)+'\n')
    print(json.dumps(dict(id=spec['id'],state='RENDERED',views=24)),flush=True)
(root/(mode+'_COMPLETE')).write_text(str(time.time())+'\n')
