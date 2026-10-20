"""Register/upload one immutable follow-up job to the current Pass 0 rental."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from dextum_cursor_agent.pass0 import cloud_ops as cloud


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job", choices=("runtime_01", "actor_01", "demo_01", "demo_02", "d5_retry_01", "d5_retry_02", "d5_retry_03", "d5_diagnostic_01", "d5_kernel_01", "d5_nofold_01", "e3_01"))
    args = p.parse_args()
    api = cloud.api
    lease = api.read(cloud.RECORD/"lease.json")
    remaining = int(lease["expires_at"]-time.time()-150)
    assert remaining > 300, "Not enough time left in capped lease"
    record = cloud.RECORD/args.job
    record.mkdir(exist_ok=False)
    paths = []
    for folder in ("dextum", "envs", "experiments", "tools", "vendor", "pass0"):
        paths += sorted(p for p in (cloud.PROJECT/folder).rglob('*') if p.is_file()
                        and 'records' not in p.parts and '__pycache__' not in p.parts
                        and p.suffix in ('.py', '.json', '.txt', '.sh', '.md'))
    if args.job.startswith("demo_") or args.job == "actor_01":
        checkpoint_dir = cloud.PROJECT/"runs/e3p1-s1/c_touch"
        checkpoints = api.read(checkpoint_dir/"checkpoints.json")
        chosen = [c for c in checkpoints if c.get("kept", True)][-1]
        paths += [checkpoint_dir/chosen["path"]]
    registration = dict(time=time.time(), job=args.job, parent=cloud.RUN, maximum_s=remaining,
                        input_sha256={str(p.relative_to(cloud.ROOT)): api.digest(p) for p in paths})
    if args.job.startswith("demo_") or args.job == "actor_01":
        registration.update(starts=list(range(2026101000, 2026101016)),
                            shape_instances=["cube55", "rounded48_r7", "cylinder38x70"],
                            selection="Historical seed-1 final checkpoint fixed before demo, no selection on outcomes",
                            checkpoint=str(paths[-1].relative_to(cloud.PROJECT)),
                            sensing="Ideal 11-pad touch; explicit ideal camera classification/angle stand-in",
                            physics="Native MuJoCo, nominal three-shape models, no fitting on demo outcomes",
                            success="Relative world-z twist within 10 degrees, tilt <=30 degrees for 2s; drop z<-0.05m",
                            controls="Native numpy policy/G; exported parity must pass; scripts on new families")
    api.write(record/"registration.json", registration)
    paths.append(record/"registration.json")
    archive_path = record/"inputs.tar"
    with tarfile.open(archive_path, "w") as archive:
        for path in paths:
            archive.add(path, arcname=str(path.relative_to(cloud.ROOT)), recursive=False)
    cmd, _, _, _ = api.connection()
    destination = "/root/dextum/"+args.job+".tar"
    with archive_path.open("rb") as stream:
        r = subprocess.run(cmd+["cat > "+shlex.quote(destination)], stdin=stream,
                           capture_output=True, timeout=180)
    if r.returncode:
        raise RuntimeError(r.stderr.decode(errors="replace")[-1000:])
    assert api.remote("sha256sum "+shlex.quote(destination)).split()[0] == api.digest(archive_path)
    # Each follow-up gets an entire immutable source release. Earlier jobs keep
    # their original source even when later engineering fixes change a module.
    release = "/root/dextum/releases/"+args.job
    verify = """import tarfile,hashlib
from pathlib import Path
root=Path(RELEASE)
root.mkdir(parents=True,exist_ok=False)
with tarfile.open(ARCHIVE) as tar:
 for member in tar.getmembers():
  target=root/member.name
  assert target.resolve().is_relative_to(root)
  if target.exists():
   assert target.read_bytes()==tar.extractfile(member).read(), member.name+' already exists with different contents'
 tar.extractall(root)
(root/'upstream').symlink_to('/root/dextum/upstream',target_is_directory=True)
""".replace("ARCHIVE", repr(destination)).replace("RELEASE",repr(release))
    api.remote("python3 -c "+shlex.quote(verify))
    output = "/root/dextum/dextum_cursor_agent/runs/"+cloud.RUN+"/"+args.job
    job_script = "pass0/"+args.job+".sh"
    inner = "timeout --signal=TERM --kill-after=20s "+str(remaining)+"s bash "+job_script+" "+shlex.quote(output)
    inner += "; code=$?; printf '%s\\n' \"$code\" > "+shlex.quote(output+"/EXIT_CODE")
    launch = ("import os,subprocess; from pathlib import Path; p=Path("+repr(output)+
              "); p.mkdir(parents=True,exist_ok=False); env=dict(os.environ,CONTAINER_ID="+repr(str(lease["instance_id"]))+
              ",OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='disable'); "
              "log=(p/'run.log').open('xb'); proc=subprocess.Popen(['bash','-c',"+repr(inner)+
              "],cwd="+repr(release+'/dextum_cursor_agent')+",env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True); "
              "(p/'pid').write_text(str(proc.pid)); print(proc.pid)")
    result = api.remote("python3 -c "+shlex.quote(launch))
    api.write(record/"launch.json", dict(time=time.time(), pid=result.strip(), remaining_s=remaining))
    print(json.dumps({"job": args.job, "pid": result.strip(), "remaining_s": remaining}))


if __name__ == "__main__":
    main()
