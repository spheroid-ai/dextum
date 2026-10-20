"""Static local import closure; standard library only, no numerical imports."""
import ast
from pathlib import Path


def local_import_closure(root, entrypoints):
    root = Path(root)
    todo = [root / name for name in entrypoints]
    found = set()
    while todo:
        path = todo.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                parts = list(path.relative_to(root).parts[:-1])
                if node.level:
                    parts = parts[:len(parts) - node.level + 1]
                    base = '.'.join(parts + ([node.module] if node.module else []))
                else:
                    base = node.module or ''
                names = [base] + [base + '.' + alias.name for alias in node.names]
            for name in names:
                candidate = root / Path(*name.split('.'))
                for local in (candidate.with_suffix('.py'), candidate / '__init__.py'):
                    if local.is_file():
                        todo.append(local)
    return found
