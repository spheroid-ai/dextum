"""Named training contracts; domain randomization is never implicit."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    name: str
    domain_randomization: bool
    reference: str
    matmul_precision: str | None = None


PROTOCOLS = {
    "documented": Protocol("documented", False,
        "Official Sim and RL quick-start: train_jax_ppo.py --env_name AeroCubeRotateZAxis"),
    "paper_dr": Protocol("paper_dr", True,
        "Paper section 5.3; unmodified upstream domain_randomize, including overwritten cube fields"),
    "documented_fp32": Protocol("documented_fp32", False,
        "Official quick-start plus pinned Playground README GPU reproducibility requirement",
        "highest"),
    "paper_dr_fp32": Protocol("paper_dr_fp32", True,
        "Official randomizer plus highest matmul precision; R0 v1.1 seeds 1, 2, 3",
        "highest"),
}


def get_protocol(name):
    return PROTOCOLS[name]


def configure_precision(protocol, jax):
    """Apply the declared runtime before compiling physics or networks."""
    if protocol.matmul_precision is not None:
        jax.config.update("jax_default_matmul_precision", protocol.matmul_precision)
        if jax.config.jax_default_matmul_precision != protocol.matmul_precision:
            raise RuntimeError("Required matrix-multiplication precision was not applied")
    return jax.config.jax_default_matmul_precision
