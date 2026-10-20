"""Small stdlib-only Vast controller. No local numerical imports/compression."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "dextum_cursor_agent"
sys.path.insert(0, str(ROOT))
from work.spheroid_touch_b import vast_ops as api
from work.spheroid_touch_b.vast_feasibility import passive_deadline

RUN = "pass0_d5_01"
RECORD = PROJECT / "pass0/records" / RUN
OUTPUT = PROJECT / "runs" / RUN
LABEL = "dextum-v12-" + RUN
HOURS = 1.0
api.RECORD, api.LABEL, api.REMOTE = RECORD, LABEL, "/root/dextum"


def prepare():
    RECORD.mkdir(parents=True, exist_ok=False)
    paths = []
    for folder in ("dextum", "envs", "experiments", "tests", "tools", "vendor", "cloud/remote", "pass0"):
        paths += [p for p in (PROJECT / folder).rglob("*") if p.is_file()
                  and "__pycache__" not in p.parts and "records" not in p.parts
                  and p.suffix in (".py", ".json", ".md", ".sh", ".txt")]
    paths += [PROJECT / "pyproject.toml", *sorted((ROOT / "dextum_plan").glob("*.md")),
              ROOT / "AGENTS.md", ROOT / "METRIPLECTOR_NO_MAPPING.md"]
    total = sum(p.stat().st_size for p in paths)
    assert total < 16 * 1024 * 1024, "Unexpected code bundle size"
    pins = api.read(PROJECT / "cloud/leases/e3p1-s2c.manifest.json")["upstream"]
    hashes = {str(p.relative_to(ROOT)): api.digest(p) for p in paths}
    reg = dict(run=RUN, time=time.time(), scope="Pass 0 D5 qualification, no training in this job",
               authorization="User: stop current runs; all specs in dextum_plan; lets do that",
               input_sha256=hashes, bytes=total, upstream=pins, hours=HOURS,
               max_hourly_usd=.6, max_projected_usd=HOURS*.6+1,
               criterion="16 frozen starts, at least 14 end heights within 5 mm of C MuJoCo",
               local_work="stdlib I/O only, uncompressed tar; all JAX/MuJoCo runs remote")
    api.write(RECORD / "registration.json", reg)
    paths.append(RECORD / "registration.json")
    env = "".join(f"{k}_{f.upper()}={shlex.quote(v)}\n" for k, pin in pins.items() for f, v in pin.items())
    with tarfile.open(RECORD / "inputs.tar", "w") as tar:
        for path in paths:
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
        info = tarfile.TarInfo("dextum_cursor_agent/cloud/remote/upstream.env")
        info.size = len(env.encode())
        tar.addfile(info, io.BytesIO(env.encode()))
    return {"state": "PREPARED", "bytes": total, "files": len(paths)}


def screen():
    q = dict(verified={"eq": True}, rentable={"eq": True}, rented={"eq": False},
             num_gpus={"eq": 1}, gpu_name={"eq": "RTX 4090"}, cuda_max_good={"gte": 12.8},
             cpu_cores_effective={"gte": 16}, cpu_ram={"gte": 32000},
             reliability={"gte": .98}, disk_space={"gte": 40}, inet_down={"gte": 300},
             direct_port_count={"gte": 1}, type="on-demand", allocated_storage=40,
             order=[["dph_total", "asc"]], limit=50)
    offers = [r for r in api.request("POST", "/api/v0/bundles/", q).get("offers", [])
              if r.get("dph_total", 99) <= .6 and r.get("geolocation", "").rsplit(",", 1)[-1].strip() in api.COUNTRIES
              and all(r.get(k) is not None and r[k] <= .1 for k in ("inet_up_cost", "inet_down_cost"))]
    keys = ("id", "machine_id", "gpu_name", "cpu_name", "cpu_cores_effective", "cpu_ram", "dph_total",
            "geolocation", "inet_up_cost", "inet_down_cost")
    offers = [{k: r.get(k) for k in keys} for r in offers]
    api.write(RECORD / "offers.json", {"time": time.time(), "offers": offers})
    return offers[:5]


def rent():
    assert not (RECORD / "rental_attempt.json").exists(), "Reconcile an existing attempt before another rent"
    reg = api.read(RECORD / "registration.json")
    assert all(api.digest(ROOT / p) == h for p, h in reg["input_sha256"].items())
    rows = api.inventory()
    assert not any(str(r.get("label", "")).startswith("dextum-v12-") for r in rows)
    api.write(RECORD / "preexisting_instances.json", [{k: r.get(k) for k in ("id", "label", "actual_status")} for r in rows])
    offers = api.read(RECORD / "offers.json")
    assert time.time() - offers["time"] < 900
    offer = min(offers["offers"], key=lambda x: x["dph_total"])
    expires = time.time() + HOURS * 3600
    body = dict(client_id="me", image="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime", disk=40,
                label=LABEL, runtype="ssh_direct", cancel_unavail=True, env={}, onstart=passive_deadline(expires))
    api.write(RECORD / "rental_attempt.json", dict(time=time.time(), offer=offer, request=body))
    response = api.request("PUT", f"/api/v0/asks/{offer['id']}/", body)
    assert response.get("success") and response.get("new_contract"), "Rental not confirmed; reconcile inventory"
    lease = dict(instance_id=int(response["new_contract"]), label=LABEL, offer=offer,
                 created_at=time.time(), expires_at=expires, state="RENTED",
                 projected_bound_usd=HOURS*offer["dph_total"]+1)
    api.write(RECORD / "lease.json", lease)
    api.request("POST", f"/api/v0/instances/{lease['instance_id']}/ssh/",
                {"ssh_key": (Path.home() / ".ssh/id_ed25519.pub").read_text().strip()})
    return lease


def upload():
    cmd, _, _, _ = api.connection()
    with (RECORD / "inputs.tar").open("rb") as stream:
        result = subprocess.run(cmd + ["mkdir -p /root/dextum && cat > /root/dextum/inputs.tar"],
                                stdin=stream, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[-1000:])
    assert api.remote("sha256sum /root/dextum/inputs.tar").split()[0] == api.digest(RECORD / "inputs.tar")
    return api.remote("cd /root/dextum && tar -xf inputs.tar")


def launch():
    lease = api.read(RECORD / "lease.json")
    assert not lease.get("launched_at")
    guard = json.loads(api.remote("cat /root/kit_b_deadline.json"))
    assert guard["state"] == "ARMED" and int(guard["instance_id"]) == lease["instance_id"]
    stop_at = int(lease["expires_at"] - 180)
    assert stop_at - time.time() > 600
    command = ("cd /root/dextum && nohup bash dextum_cursor_agent/cloud/remote/launch.sh " + RUN +
               " pass0/d5_job.sh " + str(stop_at) + " > /root/dextum_launch.log 2>&1 < /dev/null &")
    result = api.remote(command)
    lease.update(state="RUNNING", launched_at=time.time())
    api.write(RECORD / "lease.json", lease)
    return {"state": "RUNNING", "instance_id": lease["instance_id"], "output": result}


def status():
    row = api.instance()
    if not row:
        return {"state": "ABSENT"}
    result = {k: row.get(k) for k in ("id", "actual_status", "status_msg", "dph_total")}
    if row["actual_status"] == "running":
        script = ("from pathlib import Path; import json; p=Path('/root/dextum/dextum_cursor_agent/runs/" + RUN +
                  "'); print(json.dumps({str(f.relative_to(p)):f.read_text()[-2000:] for f in p.rglob('*') "
                  "if f.is_file() and (f.name in ('EXIT_CODE','qualification.json','preflight.json') or f.suffix=='.log')}))")
        result["files"] = json.loads(api.remote("python3 -c " + shlex.quote(script)))
    api.write(RECORD / "last_status.json", {"time": time.time(), **result})
    return result


def fetch():
    cmd, _, host, _ = api.connection()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["rsync", "-a", "--partial", "--timeout=45", "-e", shlex.join(cmd[:-1]),
                             "root@" + host + ":/root/dextum/dextum_cursor_agent/runs/" + RUN + "/",
                             str(OUTPUT) + "/"], capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr[-1000:])
    return {"state": "FETCHED", "path": str(OUTPUT)}


def stop():
    row = api.instance()
    assert row
    response = api.request("PUT", f"/api/v0/instances/{row['id']}/", {"state": "stopped"})
    assert response.get("success") is not False
    lease=api.read(RECORD/'lease.json')
    lease.update(state='STOP_REQUESTED',stop_requested_at=time.time())
    api.write(RECORD/'lease.json',lease)
    return response


def inventory():
    lease=api.read(RECORD/'lease.json')
    rows=[{k:r.get(k) for k in ('id','label','actual_status')} for r in api.inventory()
          if r.get('id') in (lease['instance_id'],55183680)]
    own=next((r for r in rows if r['id']==lease['instance_id']),None)
    if own and own['actual_status']=='exited':
        lease.update(state='STOPPED',stopped_confirmed_at=time.time())
        api.write(RECORD/'lease.json',lease)
    api.write(RECORD/'final_inventory.json',dict(time=time.time(),instances=rows))
    return rows


def verify():
    """Verify preserved outputs after all launched jobs have exited."""
    jobs = [p.parent.name for p in RECORD.glob('*/launch.json')]
    script = """from pathlib import Path
import hashlib,json
p=Path('/root/dextum/dextum_cursor_agent/runs/RUN_NAME')
assert (p/'EXIT_CODE').exists()
for job in JOB_NAMES:
 assert (p/job/'EXIT_CODE').exists(), job+' still running'
out={}
for f in sorted(p.rglob('*')):
 if f.is_file():
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for block in iter(lambda:stream.read(1048576),b''):h.update(block)
  out[str(f.relative_to(p))]=h.hexdigest()
print(json.dumps(out))
""".replace('RUN_NAME',RUN).replace('JOB_NAMES',repr(jobs))
    manifest=json.loads(api.remote('python3 -c '+shlex.quote(script),timeout=45))
    mismatches=[name for name,digest in manifest.items()
                if not (OUTPUT/name).is_file() or api.digest(OUTPUT/name)!=digest]
    result=dict(state='PASS' if not mismatches else 'MISMATCH',time=time.time(),
                files=len(manifest),sha256=manifest,mismatches=mismatches)
    api.write(RECORD/'output_verification.json',result)
    assert not mismatches, mismatches
    return dict(state='PASS',files=len(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "screen", "rent", "upload", "launch", "status", "fetch", "verify", "stop", "inventory"))
    args = parser.parse_args()
    print(json.dumps(globals()[args.action](), indent=2))
