"""Freeze the external modules this workspace builds on, with provenance hashes.

The source tree is shared with another agent, so experiments import these copies
instead of the live files. Re-running refuses to overwrite a differing snapshot.
"""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
VENDOR = HERE / "vendor"

FILES = {
    "aero_official": ["__init__.py", "vision_env.py", "vision_math.py", "vision_oracle.py",
                      "goal_yaw.py", "orientation_curriculum.py", "orientation_policy.py",
                      "policy.py", "protocol.py", "train.py", "orientation_train.py",
                      "orientation_evaluate.py", "cpu_eval.py", "evaluate_r0.py",
                      "gpu_preflight.py", "requirements-gpu.lock.txt"],
    "spheroid_kit": ["sensors.py"],
}
# The copy sits one directory deeper than the original.
PATCHES = {("aero_official", "policy.py"): ("ROOT = Path(__file__).resolve().parents[2]",
                                            "ROOT = Path(__file__).resolve().parents[3]")}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    manifest = {}
    for package, names in FILES.items():
        target = VENDOR / package
        target.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = REPO / "work" / package / name
            data = source.read_bytes()
            text = data.decode()
            patch = PATCHES.get((package, name))
            if patch:
                if text.count(patch[0]) != 1:
                    raise RuntimeError(f"Patch anchor missing in {source}")
                text = text.replace(*patch)
            out = target / name
            if out.exists() and out.read_text() != text and "--force" not in sys.argv:
                raise RuntimeError(f"Snapshot differs from source: {out}; pass --force to refresh")
            out.write_text(text)
            manifest[f"work/{package}/{name}"] = dict(source_sha256=sha(data), copy_sha256=sha(text.encode()),
                                                      patched=bool(patch))
    (VENDOR / "spheroid_kit" / "__init__.py").write_text("")
    (VENDOR / "__init__.py").write_text("")
    (VENDOR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(manifest)} files frozen into {VENDOR}")


if __name__ == "__main__":
    main()
