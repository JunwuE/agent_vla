"""Bounded, extractive memory indexes. Original evidence stays in the stores.

Cards are discovery aids, not new facts or executable skill contracts. No LLM
summarization, future observations, or mutation of authoritative memory occurs.
"""

import copy
import hashlib

from .data import dumps


def _size(value):
    return len(dumps(value))


def _append_within(card, key, items, budget):
    card[key] = []
    omitted = 0
    for item in items:
        card[key].append(copy.deepcopy(item))
        if _size(card) + 100 > budget:
            card[key].pop()
            omitted += 1
    if omitted:
        card[key + '_omitted'] = omitted


def experience_card(record, budget=1800):
    """Keep exact identity/outcome; fit whole facts and references, never snippets."""
    card = {k: copy.deepcopy(record[k]) for k in
            ('episode_id', 'evidence_sha256', 'context_reference') if k in record}
    card.update(kind='historical_experience_index', expand_tool='read_experience',
                authority='Historical index only. Read full evidence before adopting a recovery; remeasure current state.')
    result = record.get('result', {})
    card['outcome'] = {k: copy.deepcopy(result[k]) for k in
                       ('success', 'reason', 'control_steps') if k in result}
    # This is a provenance count, not a claim that any particular program worked.
    versions = record.get('code_versions', [])
    card['recorded_code_attempts'] = len(versions)
    card['contains_verified_recovery'] = any(v.get('verified') is True for v in versions)
    task = record.get('task')
    if task is not None and _size(card) + _size(task) < budget // 2:
        card['task'] = copy.deepcopy(task)
    refs, seen = [], set()
    for version in versions:
        for skill in version.get('available_skills', []):
            ref = {k: skill[k] for k in ('id', 'version', 'status') if k in skill}
            key = dumps(ref)
            if key not in seen:
                seen.add(key)
                refs.append(ref)
    _append_within(card, 'skill_references', refs, budget * 2 // 3)
    facts, hypotheses = [], []
    for note in reversed(record.get('notes', [])):
        if isinstance(note, dict):
            facts.extend(note.get('facts', []))
            hypotheses.extend(note.get('hypotheses', []))
    _append_within(card, 'reported_facts', facts, budget * 5 // 6)
    _append_within(card, 'unconfirmed_hypotheses', hypotheses, budget)
    card['details_omitted'] = True
    return card


def task_card(record, budget=4000):
    if not record or not record.get('profile'):
        return copy.deepcopy(record)
    profile = record['profile']
    card = dict(kind='task_memory_index', version=record.get('version'), status=record.get('status'),
                scope=copy.deepcopy(profile['scope']), expand_tool='read_task_memory',
                authority='Historical index. Rule conditions/avoidances are omitted: expand before adopting a rule. '
                          'Phase names are suggestions, not installed verifiers or current progress.')
    if _size(card) + _size(profile.get('summary', '')) + 100 < budget:
        card['summary'] = profile.get('summary', '')
    phases = [dict(id=p['id'], goal=p['goal']) for p in profile.get('phase_guide', [])]
    _append_within(card, 'phase_index', phases, budget // 2)
    rules = [dict(id=r['id'], recommendation=r['recommendation'], status=r['status'],
                  evidence_available=r.get('evidence_available', False),
                  evidence_count=len(r.get('evidence', [])),
                  skills=[{k: s[k] for k in ('skill_id', 'version', 'available', 'status') if k in s}
                          for s in r.get('skills', [])]) for r in profile.get('rules', [])]
    _append_within(card, 'rule_index', rules, budget)
    card['details_omitted'] = True
    return card


def compact_historical_memory(memory, budget=10000):
    """Bound only the historical payload; retain current task/phase/action evidence."""
    result = copy.deepcopy(memory)
    historical = {k: memory.get(k) for k in ('retrieved_experiences', 'task_memory')}
    records = memory.get('retrieved_experiences', [])
    per_record = min(1800, budget // 6)
    cards = [experience_card(record, per_record) for record in records[:3]]
    task = task_card(memory.get('task_memory'), budget // 2)
    payload = dict(retrieved_experiences=cards, task_memory=task)
    # Defensive fallback for unexpectedly long legacy IDs. Do not truncate exact
    # references: omit the whole card and expose the count for search/read tools.
    while _size(payload) > budget and cards:
        cards.pop()
    if _size(payload) > budget:
        payload['task_memory'] = dict(status='index_over_budget', expand_tool='read_task_memory',
                                      version=(memory.get('task_memory') or {}).get('version'))
    result.update(payload)
    result['historical_context'] = dict(
        mode='extractive_index', original_chars=_size(historical), sent_chars=_size(payload),
        limit_chars=budget, omitted_experiences=len(records) - len(cards),
        source_sha256=hashlib.sha256(dumps(historical).encode()).hexdigest(),
        full_records_retained=True, expand_tools=['read_experience', 'read_task_memory'],
    )
    return result
