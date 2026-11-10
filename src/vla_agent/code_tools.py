"""Bounded code-library views and development receipts; never execute skill code.

Discovery -> interface/contract -> source inspection -> editing -> invocation are
separate operations. Full immutable packages remain in the workspace/library.
"""

import ast
import hashlib
import json
from typing import Literal

from pydantic import Field, StrictInt, model_validator

from .contracts import Model
from .skill_packages import SkillDependency, execution_bundle


class CodeRead(Model):
    path: str | None = None
    view: Literal['interface', 'contract', 'outline', 'source', 'documentation', 'evidence', 'diff'] | None = None
    symbol: str | None = None
    offset: StrictInt = Field(default=0, ge=0)
    limit: StrictInt = Field(default=24000, ge=128, le=65536)


class FileEdit(Model):
    path: str
    operation: Literal['replace', 'write', 'delete'] = 'replace'
    expected_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    old_text: str | None = Field(default=None, min_length=1, max_length=65536)
    new_text: str | None = Field(default=None, max_length=65536)
    source: str | None = Field(default=None, max_length=65536)

    @model_validator(mode='after')
    def unambiguous_operation(self):
        if self.operation == 'replace':
            if self.old_text is None or self.new_text is None or self.source is not None:
                raise ValueError('replace requires old_text/new_text only')
        elif self.operation == 'write':
            if self.source is None or self.old_text is not None or self.new_text is not None:
                raise ValueError('write requires source only; replacing a file also requires expected_sha256')
        elif self.expected_sha256 is None or any(x is not None for x in (self.old_text, self.new_text, self.source)):
            raise ValueError('delete requires expected_sha256 and no source/spans')
        return self


class EditProgram(Model):
    base_version: str
    edits: list[FileEdit] = Field(default_factory=list, max_length=16)
    uses: list[SkillDependency] | None = Field(default=None, max_length=16)
    hypothesis: str = Field(min_length=1, max_length=2000)
    expected_effect: str = Field(min_length=1, max_length=2000)
    parameter_schema: dict | None = None
    parameter_sources: list[str] | None = Field(default=None, max_length=32)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def files_of(value):
    return value.get('package_files') or {'policy.py': value.get('source', value.get('code', ''))}


def identity(value):
    package = value.get('package') or {}
    return {k: v for k, v in dict(
        id=value.get('id', value.get('skill_name')), version=value.get('version'),
        sha256=value.get('sha256'), package_sha256=package.get('sha256'),
        status=value.get('status', 'workspace_candidate'), parent_version=value.get('parent_version'),
        origin_skill=value.get('origin_skill'),
    ).items() if v is not None}


def contract(value):
    source = value.get('skill_contract') or value
    result = {k: source.get(k) for k in ('preconditions', 'required_capabilities', 'known_failures',
                                        'completion_condition')}
    result.update(parameter_schema=value.get('parameter_schema', source.get('parameter_schema')),
                  parameter_sources=value.get('parameter_sources', source.get('parameter_sources', [])),
                  when_to_use=value.get('when_to_use', ''))
    return result


def page(text, offset, limit):
    if offset > len(text):
        raise ValueError('Offset is outside this view')
    end = min(len(text), offset + limit)
    return dict(text=text[offset:end], offset=offset, next_offset=end if end < len(text) else None,
                total_chars=len(text), truncated=end < len(text), content_sha256=sha(text))


def symbols(source):
    """Static declarations only: imports/attributes are not executed or resolved."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError):
        return []
    result = []

    def visit(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                first = min([node.lineno, *[d.lineno for d in node.decorator_list]])
                result.append(dict(name=name, kind=type(node).__name__, start_line=first,
                                   end_line=node.end_lineno))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    signature = f'{node.name}({ast.unparse(node.args)})'
                    result[-1].update(signature=signature[:1000], signature_truncated=len(signature) > 1000,
                                      doc=(ast.get_docstring(node) or '')[:400])
                if isinstance(node, ast.ClassDef):
                    visit(node.body, name + '.')
            elif not prefix and isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    result.append(dict(name=alias.asname or alias.name, kind='import',
                        module='.' * node.level + (node.module or ''), imported_name=alias.name,
                        start_line=node.lineno, end_line=node.end_lineno))
    visit(tree.body)
    return result


def public_api(files, module):
    """Resolve local re-exports statically; dependency imports remain explicit references."""
    path = f'code/{module}/__init__.py'
    exports = []
    for item in symbols(files.get(path, '')):
        if item['name'].startswith('_'):
            continue
        item = dict(item, path=path)
        if item['kind'] == 'import' and item['module'].startswith('.') and not item['module'].startswith('..'):
            target = item['module'][1:].replace('.', '/')
            candidates = [f'code/{module}/{target}.py', f'code/{module}/{target}/__init__.py']
            for candidate in candidates:
                matches = [s for s in symbols(files.get(candidate, '')) if s['name'] == item['imported_name']]
                if len(matches) == 1:
                    resolved = matches[0]
                    signature = resolved.get('signature')
                    if signature:
                        signature = item['name'] + signature[len(resolved['name']):]
                    item.update(definition_path=candidate, definition_symbol=resolved['name'],
                                signature=signature, doc=resolved.get('doc'),
                                signature_truncated=resolved.get('signature_truncated', False))
                    break
        exports.append(item)
    return exports


def code_view(value, path=None, *, view=None, symbol=None, offset=0, limit=24000, diff=None):
    selected = view or ('source' if path else 'interface')
    if path is not None and selected not in ('source', 'outline'):
        raise ValueError('path is supported only for source/outline views')
    if symbol is not None and selected != 'source':
        raise ValueError('symbol requires a source view and exact file path')
    result = dict(**identity(value), view=selected)
    files = files_of(value)
    package = value.get('package') or {}
    entry = package.get('entry', 'policy.py')
    if selected == 'interface':
        schema = value.get('parameter_schema') or {}
        exports = public_api(files, package.get('module'))
        result.update(entry=entry, file_paths=sorted(files), uses=package.get('uses', []),
                      when_to_use=value.get('when_to_use', '')[:600],
                      description=value.get('catalog_description', value.get('description', ''))[:600],
                      completion_condition=contract(value).get('completion_condition'),
                      parameter_names=list(schema.get('properties', {})),
                      public_exports=exports[:32], additional_exports=max(0, len(exports)-32),
                      interface_only=True,
                      next_reads=['contract', 'documentation', 'outline', 'source', 'evidence'],
                      note='Discovery summary only; read contract and relevant documentation before invocation. '
                           'Public exports are static declarations, not validated callable contracts.')
        return result
    if selected == 'contract':
        # Exact, not truncated: bounds/preconditions must never silently disappear.
        return dict(result, contract=contract(value), executed=False)
    if selected == 'source':
        if symbol and not path:
            raise ValueError('Select an exact path before reading a symbol')
        path = path or entry
        if path not in files:
            raise ValueError('File is not in the frozen package')
        source, start_line = files[path], 1
        if symbol:
            found = [s for s in symbols(source) if s['name'] == symbol]
            if len(found) != 1:
                raise ValueError('Symbol must name exactly one declaration in this file; inspect outline')
            node = found[0]
            source = ''.join(source.splitlines(keepends=True)[node['start_line']-1:node['end_line']])
            start_line = node['start_line']
        fragment = page(source, offset, limit)
        result.update(file=dict(path=path, sha256=sha(files[path]), source=fragment.pop('text'),
                                first_line=start_line + source[:offset].count('\n'), symbol=symbol, **fragment))
        return result
    if selected == 'outline':
        if path is not None and path not in files:
            raise ValueError('File is not in the frozen package')
        outline = [dict(path=p, sha256=sha(s), chars=len(s),
                        symbols=symbols(s) if p.endswith('.py') else [])
                   for p, s in sorted(files.items()) if path is None or p == path]
        content = json.dumps(outline, ensure_ascii=False, indent=2)
    elif selected == 'documentation':
        content = files.get('SKILL.md', '')
    elif selected == 'evidence':
        content = json.dumps(dict(evidence=value.get('evidence', []), run_summary=value.get('run_summary', {})),
                             ensure_ascii=False, indent=2)
    elif selected == 'diff':
        if diff is None:
            raise ValueError('Diff is available only for workspace programs')
        content = diff
    else:
        raise ValueError('Unsupported code view')
    return dict(result, page=page(content, offset, limit), executed=False)


def skill_card(value):
    package = value.get('package') or {}
    evidence = value.get('evidence', [])
    evidence = evidence if isinstance(evidence, list) else [evidence]
    counts = dict(verified=0, unverified=0, unknown=0)
    for item in evidence:
        verdict = (item.get('report') or {}).get('verified') if isinstance(item, dict) else None
        counts['verified' if verdict is True else 'unverified' if verdict is False else 'unknown'] += 1
    return dict(**identity(value), description=(value.get('catalog_description') or value.get('description', ''))[:400],
                when_to_use=value.get('when_to_use', '')[:500],
                completion_condition=value.get('completion_condition'),
                module=package.get('module'), uses=package.get('uses', []),
                evidence_counts=counts, run_summary=value.get('run_summary', {}),
                versions=value.get('versions'),
                package_variants=[{k: x[k] for k in ('id', 'version', 'package_sha256') if k in x}
                                  for x in value.get('package_variants', [])],
                expand_tool='read_skill', summary_only=True)


def check_program(workspace, version, library, robot, capabilities, verifiers):
    result = workspace.check(version)
    item = workspace.read(version)
    if result['ok'] and item.get('package'):
        bundle = execution_bundle(item['package'], item['package_files'], library, robot,
                                  capabilities=capabilities, verifiers=verifiers)
        result.update(bundle_sha256=bundle['sha256'], dependency_lock=bundle['manifest']['packages'])
    return result


def write_receipt(workspace, item, library, robot, capabilities, verifiers):
    """A small acknowledgement plus automatic validation, never a source echo."""
    current = workspace.read(item['version'])
    parent = workspace.read(item['parent_version']) if item.get('parent_version') else None
    before = files_of(parent) if parent else {}
    after = files_of(current)
    changes = [dict(path=p, before_sha256=sha(before[p]) if p in before else None,
                    sha256=sha(after[p]) if p in after else None,
                    change='added' if p not in before else 'deleted' if p not in after else 'modified')
               for p in sorted(before.keys() | after.keys()) if before.get(p) != after.get(p)]
    try:
        checks = check_program(workspace, item['version'], library, robot, capabilities, verifiers)
        lock = checks.pop('dependency_lock', {})
        checks['dependencies'] = [{k: v for k, v in dict(name=name,
            package_sha256=node['package_sha256'], reference=node['reference']).items() if v is not None}
            for name, node in lock.items()]
    except (ValueError, KeyError, TypeError) as error:
        # The draft exists; never report a failed save and induce duplicate writes.
        checks = dict(ok=False, executed=False, error=f'{type(error).__name__}: {error}',
                      scope='syntax_and_dependency_closure')
    return dict(**identity(item), saved=True, executed=False, checks=checks,
                changed_files=changes, file_paths=sorted(after),
                contract_changed=parent is None or contract(parent) != contract(current),
                contract_ref=dict(tool='read_program', version=item['version'], view='contract'),
                note='Checks do not establish physical applicability. No need to repeat check_program '
                     'for this unchanged version. Read diff or selected source only if needed.')
