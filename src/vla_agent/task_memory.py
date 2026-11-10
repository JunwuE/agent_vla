"""Task-specific guidance with evidence, fixed episode snapshots and proposed revisions.

Guidance never changes language, phase state, controller settings or execution authority.
"""

import copy
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from .contracts import Model
from .data import dumps
from .phases import Phase
from .skill_packages import SkillDependency

Text = Annotated[str, Field(min_length=1, max_length=800)]
Hash = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]


class TaskScope(Model):
    robot: str = Field(min_length=1, max_length=120)
    environment_kind: str = Field(min_length=1, max_length=80)
    task_name: str = Field(min_length=1, max_length=160)


def matches_scope(record, scope):
    if not scope:
        return True
    scope = TaskScope.model_validate(scope)
    env = record.get('conditions', {}).get('environment', {})
    return (record.get('robot'), env.get('kind'), env.get('task_name')) == (
        scope.robot, scope.environment_kind, scope.task_name)


class TaskEvidence(Model):
    episode_id: str = Field(min_length=1, max_length=120)
    evidence_sha256: Hash
    role: Literal['support', 'counterexample', 'context'] = 'context'


class TaskRule(Model):
    id: str = Field(pattern=r'^[a-z][a-z0-9_-]{0,63}$')
    when: list[Text] = Field(min_length=1, max_length=6)
    recommendation: Literal['continue_vla', 'inspect', 'consider_skill', 'replan', 'stop']
    rationale: Text
    status: Literal['observed_pattern', 'hypothesis'] = 'hypothesis'
    tools: list[Annotated[str, Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')]] = Field(default_factory=list, max_length=8)
    skills: list[SkillDependency] = Field(default_factory=list, max_length=4)
    avoid: list[Text] = Field(default_factory=list, max_length=6)
    evidence: list[TaskEvidence] = Field(default_factory=list, max_length=8)


class TaskProfile(Model):
    schema_version: Literal[1] = 1
    scope: TaskScope
    summary: Text
    applicability: list[Text] = Field(min_length=1, max_length=8)
    phase_guide: list[Phase] = Field(default_factory=list, max_length=12)
    rules: list[TaskRule] = Field(min_length=1, max_length=12)
    open_questions: list[Text] = Field(default_factory=list, max_length=8)

    @model_validator(mode='after')
    def checked_rules(self):
        if len({r.id for r in self.rules}) != len(self.rules) or any(not r.evidence for r in self.rules):
            raise ValueError('Task rules need unique IDs and historical evidence references')
        if len({p.id for p in self.phase_guide}) != len(self.phase_guide):
            raise ValueError('Task phase guide IDs must be unique')
        if len(dumps(self.model_dump()).encode()) > 24576:
            raise ValueError('Task profile exceeds 24 KiB')
        return self


class TaskMemoryProposal(Model):
    parent_version: Hash | None = None
    rule: TaskRule
    current_event_ids: list[Annotated[int, Field(strict=True, ge=1)]] = Field(default_factory=list, max_length=12)


def version_of(profile):
    return hashlib.sha256(dumps(profile).encode()).hexdigest()


def check_evidence(references, scope, experiences):
    for ref in references:
        record = experiences.get(scope['robot'], ref.episode_id, scope=scope)
        if record['evidence_sha256'] != ref.evidence_sha256:
            raise ValueError('Task evidence checksum differs from the episode snapshot')


class TaskMemoryStore:
    def __init__(self, root):
        self.root, self.index = Path(root), {}
        for path in sorted(self.root.glob('*/profile.json')):
            record = json.loads(path.read_text())
            profile = TaskProfile.model_validate(record['profile']).model_dump()
            if record['version'] != version_of(profile):
                raise ValueError('Task memory checksum mismatch')
            key = dumps(profile['scope'])
            if key in self.index:
                raise ValueError('Duplicate task profile for the same scope')
            self.index[key] = dict(version=record['version'], profile=profile)

    def read(self, scope, experiences, skills):
        scope = TaskScope.model_validate(scope).model_dump()
        item = self.index.get(dumps(scope))
        if item is None:
            return dict(status='missing', scope=scope, version=None)
        value = copy.deepcopy(item)
        for rule in value['profile']['rules']:
            rule['evidence_available'] = True
            try:
                check_evidence([TaskEvidence.model_validate(x) for x in rule['evidence']], scope, experiences)
            except ValueError:
                rule['evidence_available'] = False
            refs = []
            for ref in rule['skills']:
                try:
                    skill = skills.get(ref['skill_id'], ref['version'], scope['robot'])
                    refs.append(dict(**ref, available=True, status=skill['status'],
                                     package_sha256=(skill.get('package') or {}).get('sha256')))
                except ValueError:
                    refs.append(dict(**ref, available=False))
            rule['skills'] = refs
        return dict(**value, status='task_guidance',
                    authority='Historical guidance only. Recheck current conditions; no automatic phase, language or control change.')

    def write_profile(self, profile, experiences, *, expected_version=None):
        """Offline consolidation only; never exposed as an online control/planner tool."""
        profile = TaskProfile.model_validate(profile).model_dump()
        for rule in profile['rules']:
            check_evidence([TaskEvidence.model_validate(x) for x in rule['evidence']], profile['scope'], experiences)
        scope = profile['scope']
        # Reload the active head for compare-and-swap, not this episode's frozen index.
        current = TaskMemoryStore(self.root).index.get(dumps(scope))
        if (current or {}).get('version') != expected_version:
            raise ValueError('Task memory revision changed; consolidate against its current version')
        version = version_of(profile)
        slug = re.sub(r'[^a-zA-Z0-9_-]', '-', scope['task_name'])[:80]
        directory = self.root / (slug + '-' + version_of(scope)[:12])
        archive = directory / 'versions'
        archive.mkdir(parents=True, exist_ok=True)
        value = dict(version=version, profile=profile)
        payload = dumps(value)
        path = archive / (version + '.json')
        if path.exists() and path.read_text() != payload:
            raise ValueError('Immutable task memory version collision')
        if not path.exists():
            path.write_text(payload)
        temporary = directory / ('.' + uuid.uuid4().hex)
        temporary.write_text(payload)
        os.replace(temporary, directory / 'profile.json')
        return dict(version=version, path=str(directory / 'profile.json'))


def save_proposal(root, proposal, memory, experiences, max_event_id):
    if not memory.task_scope:
        raise ValueError('Task scope unavailable')
    if proposal.parent_version != (memory.task_memory or {}).get('version'):
        raise ValueError('Task memory proposal is based on a stale version')
    if not proposal.rule.evidence and not proposal.current_event_ids:
        raise ValueError('Task memory proposal requires historical or current evidence')
    if any(not 1 <= i <= max_event_id for i in proposal.current_event_ids):
        raise ValueError('Task memory evidence must reference existing visible events')
    check_evidence(proposal.rule.evidence, memory.task_scope, experiences)
    value = dict(scope=memory.task_scope, episode_id=memory.episode_id, **proposal.model_dump(),
                 status='candidate_not_applied', evidence_root=str(Path(root).resolve()),
                 observation_step=memory.observations[-1].step)
    version = version_of(value)
    directory = Path(root) / 'task_memory_candidates'
    directory.mkdir(exist_ok=True)
    (directory / (version + '.json')).write_text(dumps(value))
    return dict(version=version, path=str(directory / (version + '.json')), applied=False,
                status='candidate_not_applied', note='Consolidate with episode outcomes offline; active task guidance is unchanged.')
