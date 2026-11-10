"""A persistent recovery development workspace; frozen versions are executed through Harness.

Inspired by RAX's workspace -> modular controller -> immutable bundle workflow.
Checks here compile syntax only; they never execute user code or advance physics.
"""

import ast
import difflib
import hashlib
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator

from .data import dumps
from .skill_packages import build_package, check_package, read_files, write_files
from .skills import validate_schema


def check_source(source):
    try:
        tree = ast.parse(source, filename="policy.py")
        compile(tree, "policy.py", "exec")
        entries = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
        if len(entries) != 1 or len(entries[0].args.posonlyargs + entries[0].args.args) != 1:
            raise ValueError("Program must define one synchronous run(ctx) entry")
        return dict(ok=True, scope="syntax_and_entrypoint_only", executed=False)
    except (SyntaxError, ValueError, TypeError, RecursionError) as error:
        return dict(ok=False, error=f"{type(error).__name__}: {error}", executed=False)


class RecoveryWorkspace:
    def __init__(self, root, task, *, max_versions=24):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "versions").mkdir()
        (self.root / "attempts").mkdir()
        (self.root / "task.json").write_text(dumps(task))
        self.max_versions = max_versions
        self.versions = []
        self.attempts = []

    def write(self, source, hypothesis, expected_effect, *, parent_version=None, origin_skill=None,
              parameter_schema=None, parameter_sources=None, skill_name=None, when_to_use='',
              skill_contract=None, package_files=None, uses=(), completion_condition=None,
              package_name=None):
        if not source or len(source.encode()) > 64 * 1024:
            raise ValueError("Program source must be 1..64 KiB")
        if len(self.versions) >= self.max_versions:
            raise ValueError("Recovery program version budget exhausted")
        parent = self.read(parent_version) if parent_version else None
        previous = parent["source"] if parent else ""
        if parent is not None:
            origin_skill = parent.get("origin_skill")
            skill_name = parent.get('skill_name')
            if parent.get('package'):
                package_name = parent['package']['name']
            when_to_use = parent.get('when_to_use', '')
            skill_contract = parent.get('skill_contract')
            if parameter_schema is None:
                parameter_schema = parent.get('parameter_schema')
            if parameter_sources is None:
                parameter_sources = parent.get('parameter_sources')
            if parent.get('package') and package_files is None:
                raise ValueError('Use patch_skill_package for a multi-file program')
        if parameter_schema is not None:
            validate_schema(parameter_schema)
            parameter_schema = json.loads(dumps(parameter_schema))
        if skill_contract is not None:
            skill_contract = json.loads(dumps(skill_contract))
            if completion_condition is not None:
                skill_contract['completion_condition'] = completion_condition
            if parameter_schema is not None:
                skill_contract.update(parameter_schema=parameter_schema,
                                      parameter_sources=list(parameter_sources or []))
        digest = hashlib.sha256(source.encode()).hexdigest()
        # Catalog families are semantic aliases; immutable Python namespaces belong
        # to the package and must survive direct reuse, imports and later edits.
        package = build_package(package_name or skill_name, package_files, uses) if package_files is not None else None
        if package and source != package_files['code/run.py']:
            raise ValueError('Package entry and source differ')
        version = f"v{len(self.versions) + 1:03d}-{(package['sha256'] if package else digest)[:12]}"
        path = self.root / "versions" / version
        path.mkdir()
        (path / "policy.py").write_text(source)
        if package:
            write_files(path, package_files)
        (path / "change.diff").write_text(
            "".join(
                difflib.unified_diff(
                    previous.splitlines(keepends=True),
                    source.splitlines(keepends=True),
                    fromfile=parent_version or "empty",
                    tofile=version,
                )
            )
        )
        item = dict(
            version=version,
            parent_version=parent_version,
            origin_skill=origin_skill,
            skill_name=skill_name,
            when_to_use=when_to_use,
            skill_contract=skill_contract,
            parameter_schema=parameter_schema,
            parameter_sources=list(parameter_sources or []),
            sha256=digest,
            hypothesis=hypothesis,
            expected_effect=expected_effect,
            checks=check_source(source),
            package=package,
        )
        if package:
            item['checks'] = check_package(package, package_files)
            old_files = parent.get('package_files', {}) if parent else {}
            (path / 'change.diff').write_text(''.join(''.join(difflib.unified_diff(
                old_files.get(name, '').splitlines(keepends=True), package_files.get(name, '').splitlines(keepends=True),
                fromfile=f'{parent_version or "empty"}/{name}', tofile=f'{version}/{name}'))
                for name in sorted(old_files.keys() | package_files.keys())))
        (path / "manifest.json").write_text(dumps(item))
        self.versions.append(item)
        self._save()
        return item.copy()

    def read(self, version=None):
        if version is None:
            if not self.versions:
                raise ValueError("No recovery program has been written")
            version = self.versions[-1]["version"]
        if not re.fullmatch(r"v[0-9]{3,}-[0-9a-f]{12}", version):
            raise ValueError("Invalid program version")
        known = next((item for item in self.versions if item["version"] == version), None)
        if known is None:
            raise ValueError("Program is outside this recovery workspace")
        path = self.root / "versions" / version
        source = (path / "policy.py").read_text()
        if hashlib.sha256(source.encode()).hexdigest() != known["sha256"]:
            raise ValueError("Frozen recovery program was modified")
        if json.loads((path / "manifest.json").read_text()) != known:
            raise ValueError("Frozen recovery metadata was modified")
        files = read_files(path, known['package']) if known.get('package') else None
        return {**known, "source": source, 'package_files': files}

    def adapt_goal(self, version, condition, reason):
        item = self.read(version)
        if not item.get('skill_contract'):
            raise ValueError('Only imported skill contracts need goal adaptation')
        return self.write(item['source'], reason, condition, parent_version=version,
                          completion_condition=condition, package_files=item.get('package_files'),
                          uses=item['package']['uses'] if item.get('package') else ())

    def write_package(self, files, hypothesis, expected_effect, *, uses=(), **metadata):
        return self.write(files.get('code/run.py', ''), hypothesis, expected_effect,
                          package_files=files, uses=uses, **metadata)

    def patch_package(self, base_version, replacements, hypothesis, expected_effect, *, add_files=None,
                      uses=None, parameter_schema=None, parameter_sources=None):
        base = self.read(base_version)
        if not base.get('package'):
            raise ValueError('Use patch_program for a legacy single-file program')
        files = dict(base['package_files'])
        if not 1 <= len(replacements) + len(add_files or {}) <= 16:
            raise ValueError('Use 1..16 package edits')
        edits = {}
        for edit in replacements:
            name, old, new = edit['path'], edit['old_text'], edit['new_text']
            if name not in files or not old or files[name].count(old) != 1:
                raise ValueError('Package patch must match exactly one span of an existing file')
            start = files[name].index(old)
            edits.setdefault(name, []).append((start, start + len(old), new))
        for name, spans in edits.items():
            spans.sort()
            if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                raise ValueError('Package patch spans overlap')
            for start, end, new in reversed(spans):
                files[name] = files[name][:start] + new + files[name][end:]
        for name, content in (add_files or {}).items():
            if name in files:
                raise ValueError('add_files cannot overwrite an existing module')
            files[name] = content
        return self.write_package(files, hypothesis, expected_effect, parent_version=base_version,
            uses=base['package']['uses'] if uses is None else uses,
            parameter_schema=parameter_schema, parameter_sources=parameter_sources)

    def patch(self, base_version, old_text, new_text, hypothesis, expected_effect, *,
              parameter_schema=None, parameter_sources=None):
        source = self.read(base_version)["source"]
        if not old_text or source.count(old_text) != 1:
            raise ValueError("Patch must match exactly one source span")
        return self.write(
            source.replace(old_text, new_text, 1), hypothesis, expected_effect, parent_version=base_version,
            parameter_schema=parameter_schema, parameter_sources=parameter_sources,
        )

    def patch_many(self, base_version, replacements, hypothesis, expected_effect, *,
                   parameter_schema=None, parameter_sources=None):
        """Validate independent spans against one base before saving a single new version."""
        if not 1 <= len(replacements) <= 16:
            raise ValueError('Use 1..16 replacements')
        source = self.read(base_version)['source']
        spans = []
        for change in replacements:
            old, new = change['old_text'], change['new_text']
            if not old or source.count(old) != 1:
                raise ValueError('Every patch must match exactly one span in the original base')
            start = source.index(old)
            spans.append((start, start + len(old), new))
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError('Patch spans must not overlap')
        for start, end, new in reversed(spans):
            source = source[:start] + new + source[end:]
        return self.write(source, hypothesis, expected_effect, parent_version=base_version,
                          parameter_schema=parameter_schema, parameter_sources=parameter_sources)

    def edit(self, base_version, edits, hypothesis, expected_effect, *, uses=None,
             parameter_schema=None, parameter_sources=None):
        """One atomic edit transaction against an immutable package or legacy source.

        Exact spans are checked against the same base; whole-file overwrite/delete
        needs the hash obtained from a read. No partial file/version is committed
        when any path, hash or span fails validation.
        """
        from .code_tools import FileEdit, files_of, sha

        base = self.read(base_version)
        original = files_of(base)
        files = dict(original)
        if len(edits) > 16 or (not edits and uses is None and parameter_schema is None and parameter_sources is None):
            raise ValueError('Provide 1..16 edits or a dependency/parameter contract update')
        groups = {}
        for raw in edits:
            edit = FileEdit.model_validate(raw)
            name = edit.path
            if not base.get('package') and name != 'policy.py':
                raise ValueError('Legacy programs have only policy.py; migrate before adding modules')
            if edit.expected_sha256 is not None:
                if name not in original or sha(original[name]) != edit.expected_sha256:
                    raise ValueError('File hash does not match the immutable base: ' + name)
            groups.setdefault(name, []).append(edit)
        for name, changes in groups.items():
            if any(e.operation != 'replace' for e in changes):
                if len(changes) != 1:
                    raise ValueError('A file write/delete cannot be combined with other edits on that file')
                edit = changes[0]
                if name in original and edit.expected_sha256 is None:
                    raise ValueError('Overwriting or deleting an existing file requires expected_sha256')
                if edit.operation == 'delete':
                    del files[name]
                else:
                    files[name] = edit.source
                continue
            if name not in original:
                raise ValueError('Span replacement requires an existing file')
            source = original[name]
            spans = []
            for edit in changes:
                if source.count(edit.old_text) != 1:
                    raise ValueError('Every replacement must match exactly one span in the immutable base')
                start = source.index(edit.old_text)
                spans.append((start, start + len(edit.old_text), edit.new_text))
            spans.sort()
            if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                raise ValueError('Edit spans overlap')
            for start, end, replacement in reversed(spans):
                source = source[:start] + replacement + source[end:]
            files[name] = source
        previous_uses = base['package']['uses'] if base.get('package') else None
        if (files == original and (uses is None or uses == previous_uses)
                and (parameter_schema is None or parameter_schema == base.get('parameter_schema'))
                and (parameter_sources is None or parameter_sources == base.get('parameter_sources'))):
            raise ValueError('No code or contract change; reuse this version with new invocation parameters')
        if base.get('package'):
            return self.write_package(files, hypothesis, expected_effect, parent_version=base_version,
                                      uses=base['package']['uses'] if uses is None else uses,
                                      parameter_schema=parameter_schema, parameter_sources=parameter_sources)
        if uses is not None:
            raise ValueError('Migrate legacy source to a package before declaring dependencies')
        if 'policy.py' not in files:
            raise ValueError('Cannot delete the legacy program entry')
        return self.write(files['policy.py'], hypothesis, expected_effect, parent_version=base_version,
                          parameter_schema=parameter_schema, parameter_sources=parameter_sources)

    def diff(self, version=None):
        item = self.read(version)
        return (self.root / 'versions' / item['version'] / 'change.diff').read_text()

    def check(self, version=None):
        item = self.read(version)
        checks = check_package(item['package'], item['package_files']) if item.get('package') else check_source(item['source'])
        return dict(version=item["version"], **checks)

    def validate_parameters(self, version, parameters):
        item = self.read(version)
        dumps(parameters)
        schema = item.get('parameter_schema')
        if schema is None:
            if parameters:
                raise ValueError('Program parameters require an explicit parameter_schema')
        else:
            errors = list(Draft202012Validator(schema).iter_errors(parameters))
            if errors:
                raise ValueError('Invalid program parameters: ' + errors[0].message)
        return json.loads(dumps(parameters))

    def record_attempt(self, version, report, attempt_id, *, parameters):
        self.read(version)
        item = dict(
            attempt=len(self.attempts) + 1,
            attempt_id=attempt_id,
            version=version,
            parameters=parameters,
            report=report,
        )
        (self.root / "attempts" / f"{item['attempt']:03d}.json").write_text(dumps(item))
        self.attempts.append(item)
        self._save()

    def trace(self, attempt=None):
        if not self.attempts:
            raise ValueError("No recovery attempt has executed")
        index = len(self.attempts) if attempt is None else attempt
        if type(index) is not int or not 1 <= index <= len(self.attempts):
            raise ValueError("Attempt is outside this workspace")
        # JSON round-trip prevents callers from editing stored evidence.
        return json.loads(dumps(self.attempts[index - 1]))

    def summary(self):
        return dict(
            path=str(self.root),
            versions=self.versions[-8:],
            attempts=[
                {k: row[k] for k in ("attempt", "attempt_id", "version")}
                | {"result": row["report"]["result"], "failed_stage": row["report"].get("failed_stage")}
                for row in self.attempts[-8:]
            ],
        )

    def _save(self):
        (self.root / "index.json").write_text(dumps(self.summary()))
