"""Fail before a PPO run if CUDA wheel libraries or accelerator execution fail."""
import ctypes
import json
import os
from pathlib import Path
import site


def main():
    result=dict(library_path=os.environ.get("LD_LIBRARY_PATH",""),libraries={})
    try:
        for library in ("cusparse/lib/libcusparse.so.12","cusolver/lib/libcusolver.so.11"):
            path=next(p for s in site.getsitepackages() if (p:=Path(s)/"nvidia"/library).exists())
            ctypes.CDLL(str(path))
            result["libraries"][library]="LOAD_PASS"
        import jax
        import jax.numpy as jp
        from .protocol import get_protocol, configure_precision
        request=json.loads(Path("work/aero_official/records/run_request.json").read_text())
        result["matmul_precision"]=configure_precision(get_protocol(request["protocol"]),jax)
        result["devices"]=[str(d) for d in jax.devices()]
        assert all(d.platform=="gpu" for d in jax.devices()),"Refuse CPU fallback"
        # Ones cannot expose reduced-mantissa GEMM. Compare random products
        # with a float64 CPU reference, using a norm robust to cancellation.
        import numpy as np
        rng=np.random.default_rng(912)
        a,b=(rng.normal(size=(256,256)).astype(np.float32) for _ in range(2))
        y=np.asarray(jax.jit(lambda x,z:x@z)(jp.asarray(a),jp.asarray(b)))
        reference=a.astype(np.float64)@b.astype(np.float64)
        relative=float(np.linalg.norm(y-reference)/np.linalg.norm(reference))
        result["gemm_relative_error"]=relative
        if result["matmul_precision"]=="highest":
            assert relative<1e-5,f"GPU product did not meet float32 accuracy: {relative}"
        result["state"]="PASS"
    except BaseException as exc:
        result.update(state="FAILED",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        p=Path("artifacts/aero_official/gpu/preflight.json")
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
