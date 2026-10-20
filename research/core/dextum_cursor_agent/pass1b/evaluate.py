import json
import jax
from brax.io import model
from brax.training.agents.ppo import networks
from pass1b.train import make_network

def load_policy(run,arm,steps=None):
    checkpoints=[c for c in json.loads((run/arm/"checkpoints.json").read_text()) if c.get("kept",True)]
    chosen=checkpoints[-1] if steps is None else next(c for c in checkpoints if c["steps"]==steps)
    params=model.load_params(run/arm/chosen["path"])
    infer=networks.make_inference_fn(make_network(arm))(params,deterministic=True)
    return (lambda obs:infer(obs,jax.random.PRNGKey(0))[0]),chosen["steps"]
