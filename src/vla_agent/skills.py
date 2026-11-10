"""Versioned programs with executable applicability and parameter contracts."""

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from jsonschema import Draft202012Validator
from pydantic import Field

from .contracts import Model
from .data import dumps
from .skill_packages import build_package, read_files, write_files

CONTRACT_FIELDS = ('preconditions', 'required_capabilities', 'parameter_schema',
                   'parameter_sources', 'known_failures')


def semantic_name(description):
    """Fallback for legacy callers; new programs should supply an operation/start-state name."""
    name = re.sub(r'[^a-z0-9]+', '-', description.partition(' | ')[0].lower()).strip('-')
    if not name:
        raise ValueError('Provide an explicit ASCII semantic skill_name')
    return name[:80].rstrip('-')


class CatalogVersion(Model):
    storage_id: str
    version: str
    when_to_use: str


class CatalogSkill(Model):
    id: str = Field(pattern=r'^[a-z0-9][a-z0-9_-]{0,79}$')
    robot: str
    description: str
    versions: list[CatalogVersion]


class SkillCatalog(Model):
    schema_version: Literal[1] = 1
    archive_unlisted_recoveries: bool = False
    skills: list[CatalogSkill] = Field(default_factory=list)


def evidence_verdict(record):
    report = record.get('report')
    return report.get('verified') if isinstance(report, dict) else None


def search_evidence(record):
    """Keep applicability/outcome evidence; expand dense historical traces through read_skill."""
    value = copy.deepcopy(record)
    report = value.get('report')
    if not isinstance(report, dict):
        return value
    compact = {k: report[k] for k in ('result', 'verified', 'evidence', 'error', 'executed_steps',
               'attempted_steps', 'sha256', 'program_version', 'failed_stage', 'seconds',
               'artifacts', 'declared_result') if k in report}
    compact['stages'] = [{k: stage[k] for k in ('name', 'start_step', 'end_step', 'status', 'error') if k in stage}
                        | dict(measurement_count=len(stage.get('program_measurements', [])),
                               check_count=len(stage.get('checks', []))) for stage in report.get('stages', [])]
    compact['recent_program_notes'] = report.get('program_notes', [])[-2:]
    value['report'] = compact
    return value


class SkillManifest(Model):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")
    version: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    robot: str
    status: Literal["template", "candidate", "verified"]
    description: str
    preconditions: list[str] = Field(default_factory=list)
    completion_condition: str | None = None
    required_capabilities: list[str] = Field(default_factory=lambda: ["native_action12"])
    parameter_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": False}
    )
    parameter_sources: list[str] = Field(default_factory=list)
    known_failures: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] | dict[str, Any] = Field(default_factory=list)
    layout: str | None = None
    when_to_use: str = ''
    revision_of: dict[str, str] | None = None
    package: dict[str, Any] | None = None


def validate_schema(schema):
    # Local contracts only: resolving network/file references is not part of skill execution.
    def inspect(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"$ref", "$dynamicRef"} and not str(child).startswith("#"):
                    raise ValueError("Skill schemas may only use local references")
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(schema)
    Draft202012Validator.check_schema(schema)
    if schema.get("type") != "object":
        raise ValueError("Skill parameters must be an object")


class SkillLibrary:
    """Each episode sees an immutable snapshot. A successful trial never promotes a program."""

    def __init__(self, root, mode="frozen"):
        if mode not in {"frozen", "accumulate"}:
            raise ValueError("Unknown skill library mode")
        self.root, self.mode = Path(root), mode
        self.index = []
        identities = set()
        for path in sorted(self.root.glob("*/*/manifest.json")):
            item = SkillManifest.model_validate_json(path.read_text()).model_dump()
            if (path.parent.parent.name, path.parent.name) != (item["id"], item["version"]):
                raise ValueError(f"Skill path and identity differ: {path}")
            identity = (item["id"], item["version"])
            if identity in identities:
                raise ValueError("Duplicate skill version")
            identities.add(identity)
            source = path.with_name("policy.py").read_text()
            if hashlib.sha256(source.encode()).hexdigest() != item["sha256"]:
                raise ValueError(f"Skill hash mismatch: {path}")
            validate_schema(item["parameter_schema"])
            if item["status"] == "verified" and not item["evidence"]:
                raise ValueError("Verified skills require recorded validation evidence")
            run_path = path.with_name('runs.jsonl')
            if run_path.exists():
                raw = run_path.read_bytes()
                runs = [json.loads(line) for line in raw.splitlines() if line.strip()]
                evidence = item['evidence'] if isinstance(item['evidence'], list) else [item['evidence']]
                unique = {}
                for record in evidence + [run['evidence'] for run in runs]:
                    if not isinstance(record, dict):
                        raise ValueError('Skill run evidence must be an object')
                    unique[dumps(record)] = record
                records = list(unique.values())
                # Keep bounded evidence with the latest positive AND negative example.
                selected = set(range(max(0, len(records) - 5), len(records)))
                for verified in (True, False):
                    matching = [i for i, r in enumerate(records)
                                if evidence_verdict(r) is verified]
                    if matching:
                        selected.add(matching[-1])
                item['evidence'] = [records[i] for i in sorted(selected)]
                item['run_summary'] = dict(recorded_runs=len(runs), unique_evidence=len(records),
                    verified_runs=sum(evidence_verdict(run['evidence']) is True for run in runs),
                    run_log_sha256=hashlib.sha256(raw).hexdigest())
            package_files = read_files(path.parent, item['package']) if item['package'] else None
            if package_files and package_files['code/run.py'] != source:
                raise ValueError('Package entry differs from stored policy.py')
            self.index.append({**item, "code": source, 'package_files': package_files})
        # A curated view over immutable historical records, not a copy or promotion of their code.
        catalog_path = self.root / 'catalog.json'
        self.catalog = SkillCatalog.model_validate_json(catalog_path.read_text()) if catalog_path.exists() else SkillCatalog()
        by_reference = {(x['id'], x['version'], x['robot']): x for x in self.index}
        assigned = set()
        names = set()
        for family in self.catalog.skills:
            if family.id in names or family.id.startswith('recovery-'):
                raise ValueError('Catalog names must be unique semantic operation names')
            names.add(family.id)
            for ref in family.versions:
                key = (ref.storage_id, ref.version, family.robot)
                if key not in by_reference or key in assigned:
                    raise ValueError('Catalog version missing or assigned twice')
                assigned.add(key)
                item = by_reference[key]
                item.update(storage_id=item['id'], id=family.id, when_to_use=ref.when_to_use,
                            catalog_description=family.description)
            for item in self.index:
                if item['id'] == family.id and item['robot'] == family.robot:
                    item['catalog_description'] = family.description
        semantic_keys = [(x['id'], x['version'], x['robot']) for x in self.index]
        if len(semantic_keys) != len(set(semantic_keys)):
            raise ValueError('Ambiguous semantic skill version')
        for item in self.index:
            item['archived'] = bool(self.catalog.archive_unlisted_recoveries and item['id'].startswith('recovery-'))
        self.snapshot = [
            {k: x[k] for k in ("id", "version", "sha256", "status")}
            | {
                "manifest_sha256": hashlib.sha256(
                    dumps({k: v for k, v in x.items() if k not in {'code', 'package_files'}}).encode()
                ).hexdigest()
            }
            for x in self.index
        ]

    def search(self, robot, query="", *, capabilities=None, include_templates=False, include_candidates=False,
               include_archived=False):
        def succeeded(item):
            evidence = item['evidence'] if isinstance(item['evidence'], list) else [item['evidence']]
            return any(evidence_verdict(record) is True for record in evidence if isinstance(record, dict))

        def task_scope(item):
            # Host-written recovery descriptions have an exact task prefix, followed by the diagnosis.
            # Legacy free-form descriptions cannot establish a same-task grouping.
            task, separator, _ = item['description'].partition(' | ')
            return task.strip().casefold() if separator else None

        words = set(re.findall(r"\w+", query.lower()))
        statuses = {"verified", "template"} if include_templates else {"verified"}
        if include_candidates:
            statuses.add("candidate")
        result = [
            x
            for x in self.index
            if x["robot"] == robot
            and x["status"] in statuses
            and (include_archived or not x['archived'])
            and (capabilities is None or set(x["required_capabilities"]) <= set(capabilities))
        ]
        result.sort(
            key=lambda x: (
                x["status"] == "verified",
                len(words & set(re.findall(r"\w+", (x.get('catalog_description') or task_scope(x) or x['description']).lower()))),
                succeeded(x),
                len(words & set(re.findall(r'\w+', x['when_to_use'].lower()))),
                x.get('run_summary', {}).get('verified_runs', 0),
            ),
            reverse=True,
        )
        # One procedure per search slot. Detailed variants/failures remain explicitly inspectable.
        unique = {}
        for item in result:
            unique.setdefault(item['id'], item)
        result = list(unique.values())
        selected = result[:5]
        if len(result) > 5:
            scope = task_scope(result[0])
            if scope and words & set(re.findall(r'\w+', scope)):
                demonstrated = next((x for x in result if task_scope(x) == scope and succeeded(x)), None)
                if (demonstrated is not None and demonstrated not in selected
                        and selected[-1]['status'] != 'verified'):
                    # Repeated failed variants with verbose diagnoses must not crowd out every
                    # same-task success. This is evidence diversity, not a claim of applicability.
                    selected[-1] = demonstrated
        output = []
        for item in selected:
            value = copy.deepcopy({k: v for k, v in item.items() if k not in {'code', 'package_files'}})
            evidence = value['evidence']
            value['evidence'] = ([search_evidence(record) for record in evidence] if isinstance(evidence, list)
                                 else search_evidence(evidence))
            value['evidence_detail'] = dict(mode='search_summary', expand_tool='read_skill',
                                            skill_id=item['id'], version=item['version'],
                                            note='Dense historical measurements and full notes remain in read_skill.')
            output.append(value)
            siblings = [x for x in self.index if x['id'] == item['id'] and x['robot'] == robot]
            value['versions'] = dict(count=len(siblings), expand_tool='list_skill_versions')
            value['package_variants'] = [dict(id=x['id'], version=x['version'],
                                              package_sha256=x['package']['sha256'], when_to_use=x['when_to_use'])
                                         for x in siblings if x.get('package')][:4]
            value['reuse'] = 'Remeasure inputs and submit this skill_id/version with parameters; no source copy required.'
        return output

    def get(self, skill_id, version, robot):
        for item in self.index:
            if skill_id in (item['id'], item.get('storage_id')) and item["version"] == version and item["robot"] == robot:
                return copy.deepcopy(item)
        raise ValueError("Skill/version/robot not present in this episode's library snapshot")

    def versions(self, skill_id, robot):
        matching = [x for x in self.index if x['robot'] == robot and skill_id in (x['id'], x.get('storage_id'))]
        if not matching:
            raise ValueError('Skill not present in this episode snapshot')
        family = matching[0]['id']
        return [dict(id=x['id'], version=x['version'], status=x['status'], when_to_use=x['when_to_use'],
                     package=copy.deepcopy(x['package']),
                     parameter_schema=copy.deepcopy(x['parameter_schema']), revision_of=x['revision_of'],
                     outcomes=[dict(verified=evidence_verdict(e), failed_stage=(e.get('report') or {}).get('failed_stage'))
                               for e in (x['evidence'] if isinstance(x['evidence'], list) else [x['evidence']])])
                for x in self.index if x['robot'] == robot and x['id'] == family]

    def validate_invocation(self, skill, parameters, capabilities, condition, verifiers):
        missing = set(skill["required_capabilities"]) - set(capabilities)
        if missing:
            raise ValueError("Skill capabilities unavailable: " + ", ".join(sorted(missing)))
        if not condition or condition not in verifiers:
            raise ValueError("Skill completion condition has no installed verifier")
        if skill["completion_condition"] and skill["completion_condition"] != condition:
            raise ValueError("Skill completion condition cannot be replaced")
        missing = set(skill["preconditions"]) - set(verifiers)
        if missing:
            raise ValueError("Skill preconditions have no installed verifiers: " + ", ".join(sorted(missing)))
        # Ensure no NaN or non-JSON objects enter the worker's parameter channel.
        dumps(parameters)
        errors = list(Draft202012Validator(skill["parameter_schema"]).iter_errors(parameters))
        if errors:
            raise ValueError("Invalid skill parameters: " + errors[0].message)
        return copy.deepcopy(parameters)

    def record_candidate(self, code, robot, condition, evidence, layout, *, parent_skill=None, description=None,
                         skill_name=None, when_to_use='', package=None, package_files=None):
        digest = hashlib.sha256(code.encode()).hexdigest()
        contract = (
            {
                k: parent_skill[k]
                for k in CONTRACT_FIELDS
            }
            if parent_skill
            else {}
        )
        # Parameter/verification contracts are part of version identity, not just code bytes.
        identity = hashlib.sha256(dumps([digest, robot, condition, contract]).encode()).hexdigest()
        if package is not None:
            if build_package(package['name'], package_files, package['uses'], sdk_digest=package['runtime_sha256']) != package:
                raise ValueError('Cannot retain a modified package')
            if package_files['code/run.py'] != code:
                raise ValueError('Package entry and source differ')
            identity = hashlib.sha256(dumps([identity, package['sha256']]).encode()).hexdigest()
        parent_name = parent_skill.get('id') if parent_skill else None
        name = skill_name or parent_name
        if not name or name.startswith('recovery-'):
            try:
                name = semantic_name(description or condition)
            except ValueError:
                name = semantic_name(condition)
        # Unchanged execution appends evidence to the exact version (including legacy aliases).
        exact = None
        if parent_skill and parent_skill.get('version'):
            try:
                original = self.get(parent_name, parent_skill['version'], robot)
            except ValueError:
                original = None
            if (original and original['sha256'] == digest and original['completion_condition'] == condition
                    and original.get('package') == package
                    and all(original[k] == contract[k] for k in CONTRACT_FIELDS)):
                exact = original
        if exact:
            if self.mode == 'accumulate':
                directory = self.root / exact.get('storage_id', exact['id']) / exact['version']
                with (directory / 'runs.jsonl').open('a') as handle:
                    handle.write(dumps(dict(evidence=evidence, layout=layout)) + '\n')
            return {k: v for k, v in exact.items() if k not in {'code', 'package_files'}}
        item = SkillManifest(
            id=name,
            version=identity[:16],
            sha256=digest,
            robot=robot,
            status="candidate",
            description=description or condition,
            completion_condition=condition,
            evidence=[evidence],
            layout=layout,
            package=package,
            when_to_use=when_to_use or (parent_skill or {}).get('when_to_use', ''),
            revision_of=({'id': parent_name, 'version': parent_skill['version']}
                         if parent_name and parent_skill.get('version') else None),
            **contract,
        ).model_dump()
        if self.mode == "accumulate":
            directory = self.root / item["id"] / item["version"]
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "policy.py"
            if path.exists() and path.read_text() != code:
                raise ValueError("Immutable skill collision")
            if not path.exists():
                path.write_text(code)
            manifest = directory / "manifest.json"
            if manifest.exists():
                saved = SkillManifest.model_validate_json(manifest.read_text()).model_dump()
                if any(saved[k] != item[k] for k in (*CONTRACT_FIELDS, 'robot', 'completion_condition', 'sha256', 'package')):
                    raise ValueError('Immutable skill contract collision')
            if package:
                write_files(directory, package_files)
            if not manifest.exists():
                manifest.write_text(dumps(item))
            with (directory / "runs.jsonl").open("a") as handle:
                handle.write(dumps(dict(evidence=evidence, layout=layout)) + "\n")
        return item
