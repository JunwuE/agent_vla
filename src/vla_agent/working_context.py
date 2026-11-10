"""Extractive working-context indexes, with immutable full evidence on demand.

This bounds repeated notes/reports, not authoritative state, schemas, images or
an explicitly requested tool result. No sentences are truncated or synthesized.
"""

import hashlib
import json

from .data import dumps


def size(value):
    return len(dumps(value))


def ref(reference, path):
    return dict(tool='read_context', reference=reference, path=path)


def records_index(rows, reference, path, budget):
    rows = list(rows or [])
    selected, spent, seen = [], 0, set()
    # Prefer the most recent full records, preserving exact field/array identity.
    keys = ('event_id', 'source', 'observation_step', 'intervention_id', 'id', 'request_id',
            'status', 'outcome', 'verified', 'score', 'phase_id', 'attempt_id', 'version')
    for i in range(len(rows)-1, -1, -1):
        row = rows[i]
        key = dumps(row)
        if key in seen:
            continue
        seen.add(key)
        card = {k: row[k] for k in keys if isinstance(row, dict) and k in row and size(row[k]) < 250}
        card['read'] = ref(reference, path + [i])
        if spent + size(row) < budget and size(row) <= budget // 2:
            card['record'] = row
        elif isinstance(row, dict):
            for field in ('facts', 'hypotheses', 'next_check', 'error', 'reason', 'result'):
                if field in row and size(row[field]) < 800 and spent + size(card) + size(row[field]) < budget:
                    card[field] = row[field]
            card['details_omitted'] = True
        if spent + size(card) > budget:
            break
        selected.append(card)
        spent += size(card)
    return dict(records=list(reversed(selected)), total_records=len(rows),
                omitted_records=len(rows)-len(selected), full_records=ref(reference, path))


def compact_working_context(context, recorder, budget=14000):
    original = json.loads(dumps(context))
    encoded = dumps(original)
    reference = hashlib.sha256(encoded.encode()).hexdigest()
    directory = recorder.root / 'context_archive'
    directory.mkdir(exist_ok=True)
    path = directory / (reference + '.json')
    if path.exists() and path.read_text() != encoded:
        raise ValueError('Context archive was modified')
    if not path.exists():
        path.write_text(encoded)
    result = json.loads(encoded)
    memory = result.setdefault('memory', {})
    # Current task, control, active attempt/last problem and phase contract stay exact.
    weights = {'notes': .30, 'reward_assessments': .10, 'reviews': .10,
               'attempts': .08, 'interventions': .10, 'groundings': .10}
    for field, fraction in weights.items():
        if field in memory:
            memory[field] = records_index(memory[field], reference, ['memory', field], int(budget*fraction))
    phases = memory.get('phase_memory', {})
    if 'recent_assessments' in phases:
        phases['recent_assessments'] = records_index(phases['recent_assessments'], reference,
            ['memory', 'phase_memory', 'recent_assessments'], int(budget*.10))
    # Keep the current program's complete contract. Older version schemas/packages
    # are already retrievable through read_program; never repeat all eight contracts.
    workspace = result.get('workspace')
    if workspace and workspace.get('versions'):
        versions = workspace['versions']
        workspace['current_version'] = versions[-1]
        workspace['versions'] = [{k: v[k] for k in ('version', 'parent_version', 'sha256', 'skill_name')
                                  if k in v} for v in versions]
        workspace['read'] = ref(reference, ['workspace'])
    feedback = result.get('recovery_feedback')
    if feedback:
        brief = {k: feedback[k] for k in ('result', 'verified', 'evidence', 'error', 'executed_steps',
            'attempted_steps', 'sha256', 'program_version', 'failed_stage', 'seconds', 'artifacts') if k in feedback}
        brief['stages'] = [{k: s[k] for k in ('name', 'status', 'error', 'start_step', 'end_step') if k in s}
                           for s in feedback.get('stages', [])[-8:]]
        brief['program_notes'] = records_index(feedback.get('program_notes', []), reference,
                                              ['recovery_feedback', 'program_notes'], int(budget*.12))
        brief['read'] = ref(reference, ['recovery_feedback'])
        result['recovery_feedback'] = brief
    result['working_context'] = dict(mode='extractive_index', reference=reference,
        original_chars=len(encoded), indexed_chars=size(result), records_budget_chars=budget,
        state_and_current_contract_unchanged=True, full_evidence=ref(reference, []),
        note='Indexes omit details, not evidence. Read exact source before relying on an omitted condition. '
             'Does not bound visual inputs, fixed tool definitions or explicit full reads.')
    recorder.event('working_context_index', reference=reference, observation_step=original['binding']['observation_step'],
                   original_chars=len(encoded), indexed_chars=size(result))
    return result


def read_context(recorder, reference, path, start, limit, episode_id, max_step):
    file = recorder.root / 'context_archive' / (reference + '.json')
    if not file.is_file():
        raise ValueError('Unknown context reference')
    text = file.read_text()
    if hashlib.sha256(text.encode()).hexdigest() != reference:
        raise ValueError('Context archive hash mismatch')
    full = json.loads(text)
    if full['binding']['episode_id'] != episode_id or full['binding']['observation_step'] > max_step:
        raise ValueError('Context is outside the current episode/snapshot')
    value = full
    for key in path:
        if isinstance(value, list):
            if type(key) is not int or not 0 <= key < len(value):
                raise ValueError('Invalid context array index')
        elif not isinstance(value, dict) or key not in value:
            raise ValueError('Unknown context path')
        value = value[key]
    total = len(value) if isinstance(value, list) else None
    if total is not None:
        value = value[start:start+limit]
    if size(value) > 48000:
        raise ValueError('Selected evidence exceeds 48000 chars; select a narrower field path')
    return dict(reference=reference, path=path, value=value, total_records=total,
                start=start if total is not None else None, historical=True,
                observation_step=full['binding']['observation_step'])
