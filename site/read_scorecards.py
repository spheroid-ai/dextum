"""Read explicitly admitted, frozen reports. No simulator or statistical fitting."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent


def load_results():
    data = HERE / 'data'
    snapshot = json.loads((data / 'accepted_results.json').read_text())
    registry = json.loads((data / 'scorecard_bindings.json').read_text())
    sources = {}
    for name, binding in registry['sources'].items():
        raw = (data / binding['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding['sha256']:
            raise ValueError(f'Evidence changed without a new acceptance binding: {name}')
        sources[name] = json.loads(raw)
    for binding in registry['bindings']:
        value = sources[binding['source']]
        for part in binding['pointer'].split('/')[1:]:
            part = part.replace('~1', '/').replace('~0', '~')
            value = value[int(part)] if isinstance(value, list) else value[part]
        snapshot['results'][binding['row']][binding['field']] = value * binding['multiplier']
    snapshot['figure_source_bindings'] = registry
    return snapshot
