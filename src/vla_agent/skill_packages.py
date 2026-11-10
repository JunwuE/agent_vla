"""RAX-style Python skill packages and immutable, dependency-closed execution bundles.

Workflow reference: RAX continual-learning and services/controller/storage.py.
The host accepts bounded text files, never an agent-controlled filesystem tree.
"""

import ast
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

import yaml
from pydantic import Field

from .contracts import Model

FORMAT = 'python-skill-package-v1'
SECTIONS = ('When to use', 'Preconditions', 'Procedure', 'Code', 'Checks', 'Failure modes', 'Evidence')


class SkillDependency(Model):
    skill_id: str = Field(pattern=r'^[a-z0-9][a-z0-9_-]{0,79}$')
    version: str = Field(pattern=r'^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def runtime_digest():
    return hashlib.sha256(Path(__file__).with_name('recovery_worker.py').read_bytes()).hexdigest()


def dependencies(uses):
    result = [SkillDependency.model_validate(x).model_dump() for x in uses or []]
    if len(result) > 16 or len({x['skill_id'] for x in result}) != len(result):
        raise ValueError('Use at most 16 uniquely named, pinned skill dependencies')
    return sorted(result, key=lambda x: x['skill_id'])


def package_name(name):
    if not re.fullmatch(r'[a-z][a-z0-9]*(?:-[a-z0-9]+)*', name) or len(name) > 80:
        raise ValueError('Package skill names must use lowercase words joined by hyphens')
    module = name.replace('-', '_')
    if module in sys.stdlib_module_names or module in {'numpy', 'scipy', 'PIL', 'vla_agent', 'api'}:
        raise ValueError('Skill package must not shadow a runtime module')
    return module


def validate_files(name, files):
    module = package_name(name)
    if not isinstance(files, dict) or not 4 <= len(files) <= 48:
        raise ValueError('A skill package needs 4..48 text files')
    required = {'SKILL.md', 'code/run.py', f'code/{module}/__init__.py', f'code/{module}/params.py'}
    if not required <= files.keys():
        raise ValueError('Package requires SKILL.md, code/run.py, and its named package __init__.py and params.py')
    total = 0
    for name, content in files.items():
        path = PurePosixPath(name)
        if (path.is_absolute() or str(path) != name or '..' in path.parts or '\\' in name
                or len(path.parts) > 8 or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', p) for p in path.parts)):
            raise ValueError('Noncanonical package file path')
        if name not in {'SKILL.md', 'code/run.py'}:
            if not (name.startswith(f'code/{module}/') and name.endswith('.py')):
                raise ValueError('Python modules must live inside the skill-named package')
        if not isinstance(content, str) or len(content.encode()) > 64 * 1024:
            raise ValueError('Each package file must be UTF-8 text of at most 64 KiB')
        total += len(content.encode())
    if total > 256 * 1024:
        raise ValueError('Skill package exceeds 256 KiB')
    return module


def build_package(name, files, uses=(), *, sdk_digest=None):
    """Validate layout/docs and hash bytes without importing or executing any skill code."""
    module = validate_files(name, files)
    uses = dependencies(uses)
    parts = files['SKILL.md'].split('---', 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError('SKILL.md requires YAML frontmatter')
    header = yaml.safe_load(parts[1])
    if not isinstance(header, dict) or header.get('name') != name or header.get('entry') != 'code/run.py':
        raise ValueError('SKILL.md name/entry must match the package')
    if header.get('status') not in {'draft', 'verified', 'deprecated'} or not header.get('robot'):
        raise ValueError('SKILL.md requires robot and draft/verified/deprecated status')
    if type(header.get('version')) is not int or header['version'] < 1:
        raise ValueError('SKILL.md version must be a positive integer')
    if not isinstance(header.get('uses', []), list) or set(header.get('uses', [])) != {x['skill_id'] for x in uses}:
        raise ValueError('SKILL.md uses must match the pinned dependency declarations')
    # The section labels carry the contract; equivalent Markdown heading depth
    # must not force regeneration of otherwise valid executable files.
    missing = [section for section in SECTIONS
               if not re.search(r'^#{1,2}\s+' + re.escape(section) + r'\s*$', parts[2], re.M)]
    if missing:
        raise ValueError('SKILL.md is missing required headings: ' + ', '.join('## ' + s for s in missing)
                         + '. Correct the documentation; preserve unrelated Python files.')
    if not re.search(r'^#{2,3}\s+Perception checkpoints\s*$', parts[2], re.M):
        raise ValueError('Document perception checkpoints (or explicitly None)')
    manifest = dict(format=FORMAT, name=name, module=module, robot=header['robot'], entry='code/run.py', uses=uses,
                    runtime_sha256=sdk_digest or runtime_digest(),
                    files={p: hashlib.sha256(s.encode()).hexdigest() for p, s in sorted(files.items())})
    return dict(manifest, sha256=digest(manifest))


def check_package(package, files):
    try:
        if build_package(package['name'], files, package['uses'], sdk_digest=package['runtime_sha256']) != package:
            raise ValueError('Frozen package bytes or contract changed')
        for name, source in files.items():
            if name.endswith('.py'):
                compile(source, name, 'exec')
        tree = ast.parse(files[package['entry']])
        mains = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main']
        if len(mains) != 1 or not (mains[0].args.posonlyargs + mains[0].args.args):
            raise ValueError('code/run.py must define synchronous main(ctx, **params)')
        return dict(ok=True, scope='syntax_entrypoint_and_package_integrity_only', executed=False)
    except (ValueError, SyntaxError, TypeError, KeyError, RecursionError) as error:
        return dict(ok=False, error=f'{type(error).__name__}: {error}', executed=False)


def write_files(directory, files):
    for name, source in files.items():
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and (path.is_symlink() or path.read_text() != source):
            raise ValueError('Immutable package file collision')
        if not path.exists():
            path.write_text(source)


def read_files(directory, package):
    directory = Path(directory)
    result = {}
    for name in package['files']:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or str(relative) != name:
            raise ValueError('Invalid frozen package path')
        path = directory
        for part in relative.parts:
            path /= part
            if path.is_symlink():
                raise ValueError('Package symlinks are forbidden')
        result[name] = path.read_text()
    actual = {p.relative_to(directory).as_posix() for p in (directory / 'code').rglob('*') if p.is_file()}
    if actual != {p for p in result if p.startswith('code/')}:
        raise ValueError('Frozen package file set changed')
    if build_package(package['name'], result, package['uses'], sdk_digest=package['runtime_sha256']) != package:
        raise ValueError('Frozen package modified')
    return result


def execution_bundle(root_package, root_files, library, robot, *, capabilities=None, verifiers=None):
    """Resolve one exact version per Python namespace; never follow mutable latest pointers."""
    nodes, visiting = {}, set()

    def visit(package, files, reference):
        name = package['name']
        if package['robot'] != robot:
            raise ValueError('Skill package robot does not match the executing robot')
        if name in visiting:
            raise ValueError('Cyclic skill dependency')
        if name in nodes:
            if nodes[name]['package']['sha256'] != package['sha256']:
                raise ValueError('Conflicting dependency versions for the same Python package')
            return
        if len(nodes) + len(visiting) >= 32:
            raise ValueError('Skill dependency closure exceeds 32 packages')
        check = check_package(package, files)
        if not check['ok']:
            raise ValueError(check['error'])
        if package['runtime_sha256'] != runtime_digest():
            raise ValueError('Runtime API changed; register and revalidate a new package version')
        visiting.add(name)
        for dep in package['uses']:
            skill = library.get(dep['skill_id'], dep['version'], robot)
            if not skill.get('package'):
                raise ValueError('Dependency is a legacy script; migrate it to a named package before importing')
            if capabilities is not None and not set(skill['required_capabilities']) <= set(capabilities):
                raise ValueError('Dependency capabilities unavailable: ' + skill['id'])
            conditions = set(skill['preconditions']) | ({skill['completion_condition']} if skill['completion_condition'] else set())
            if verifiers is not None and not conditions <= set(verifiers):
                raise ValueError('Dependency requires uninstalled verification conditions: ' + skill['id'])
            visit(skill['package'], skill['package_files'], dict(id=skill['id'], version=skill['version'],
                  package_sha256=skill['package']['sha256'], status=skill['status'],
                  contract_sha256=digest({k: skill[k] for k in ('robot', 'preconditions', 'completion_condition',
                      'required_capabilities', 'parameter_schema', 'parameter_sources', 'known_failures')})))
        visiting.remove(name)
        nodes[name] = dict(package=package, files=files, reference=reference)

    visit(root_package, root_files, None)
    total = sum(sum(len(s.encode()) for s in n['files'].values()) for n in nodes.values())
    if total > 2 * 1024 * 1024:
        raise ValueError('Execution bundle exceeds 2 MiB')
    locks = {name: dict(package_sha256=n['package']['sha256'], reference=n['reference'],
                        files=n['package']['files'], module=n['package']['module'],
                        entry=n['package']['entry']) for name, n in sorted(nodes.items())}
    manifest = dict(format=FORMAT, root=root_package['name'], runtime_sha256=runtime_digest(), packages=locks)
    return dict(manifest=manifest, sha256=digest(manifest), packages=nodes)


def export_bundle(directory, bundle):
    directory = Path(directory)
    for name, node in bundle['packages'].items():
        write_files(directory / 'packages' / name, node['files'])
    (directory / 'bundle.json').write_text(json.dumps(dict(bundle['manifest'], sha256=bundle['sha256']), sort_keys=True))
