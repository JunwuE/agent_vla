"""Read-only cross-process admission, retry and provider-error report."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def stats(values):
    return dict(count=len(values), total=sum(values), mean=statistics.mean(values),
                maximum=max(values)) if values else dict(count=0)


def main(root):
    plan = json.loads((root / 'plan.json').read_text())
    config = plan['config']['planner']
    timeline, rows, violations, all_waits = [], [], [], []
    codes, terminal_codes = Counter(), Counter()
    for case in plan['cases']:
        episode = root / 'episodes' / case['key']
        if not (episode / 'result.json').exists():
            continue
        result = json.loads((episode / 'result.json').read_text())
        events = [json.loads(line) for line in (episode / 'events.jsonl').open()]
        errors, retries, waits, refunds, review_starts = [], [], [], [], {}
        for event in events:
            kind = event['kind']
            if kind in {'planner_admission_acquired', 'planner_admission_released'}:
                timeline.append(dict(episode=case['key'], **event))
                if kind == 'planner_admission_acquired':
                    waits.append(event['wait_seconds'])
            elif kind == 'codex_error':
                payload = event['error']
                detail = payload.get('error', {})
                code = detail.get('codexErrorInfo', 'unknown')
                code = next(iter(code), 'unknown') if isinstance(code, dict) else code
                codes[code] += 1
                if not payload.get('willRetry'):
                    terminal_codes[code] += 1
                errors.append(dict(event_id=event['id'], code=code, will_retry=payload.get('willRetry'),
                                   message=detail.get('message'), time_ns=event['time_ns']))
            elif kind == 'planner_service_retry':
                retries.append(event)
            elif kind == 'review_begin':
                review_starts[event['review']] = event['id']
            elif kind == 'review_failed' and event.get('outcome') == 'deferred_without_model_request':
                start = review_starts[event['review']]
                if ('localAdmissionTimeout' not in event['error']
                        or any(e['kind'] == 'codex_turn_start' and start < e['id'] < event['id'] for e in events)):
                    violations.append(dict(episode=case['key'], problem='refund_after_model_request', event=event))
                refunds.append(event['review'])
        all_waits.extend(waits)
        # Model attempts are the actual turn/start events, including failed attempts.
        starts = [e for e in events if e['kind'] == 'codex_turn_start']
        if len(starts) > config['max_calls'] or result['astra_calls'] != len(starts):
            violations.append(dict(episode=case['key'], problem='actual_turn_budget_or_accounting'))
        rows.append(dict(seed=case['seed'], success=result['success'], reason=result['reason'],
            steps=result['control_steps'], wall_seconds=result['wall_seconds'], turns=len(starts),
            tokens=result['astra_tokens'], errors=errors, host_retries=retries,
            review_requests=len(review_starts), no_request_budget_refunds=refunds,
            consumed_review_budget=len(review_starts) - len(refunds),
            admission_wait=stats(waits), admission_timeouts=sum(
                'localAdmissionTimeout' in str(e) for e in events if e['kind'] == 'review_failed'),
            final_service_failures=[e for e in events if e['kind'] == 'planner_service_failure']))
    active, slots, peak = {}, {}, 0
    for event in sorted(timeline, key=lambda e: e['time_ns']):
        identity, slot = event['lease_id'], event['slot']
        if event['kind'] == 'planner_admission_acquired':
            if slot in slots or identity in active:
                violations.append(dict(problem='overlapping_lease', event=event))
            active[identity] = event
            slots[slot] = identity
            peak = max(peak, len(active))
        else:
            if identity not in active or slots.get(slot) != identity:
                violations.append(dict(problem='unmatched_lease_release', event=event))
            active.pop(identity, None)
            slots.pop(slot, None)
    if active:
        violations.append(dict(problem='unreleased_completed_episode_leases', leases=active))
    enabled = bool(config.get('admission_directory'))
    if enabled and peak > config['admission_slots']:
        violations.append(dict(problem='concurrency_limit_exceeded', peak=peak))
    report = dict(complete=len(rows) == len(plan['cases']), episodes=rows,
        admission_enabled=enabled, maximum_concurrent_leases=peak if enabled else None,
        admission_wait=stats(all_waits), error_notifications=dict(codes),
        terminal_error_notifications=dict(terminal_codes), host_retries=sum(len(r['host_retries']) for r in rows),
        episodes_with_service_errors=[r['seed'] for r in rows if r['errors']],
        no_request_budget_refunds=sum(len(r['no_request_budget_refunds']) for r in rows),
        violations=violations, checked_completed_episodes_passed=not violations,
        notes=['Notifications may repeat for one request; they are not distinct model calls.',
               'Admission waits are summed per request and can overlap across episodes.',
               'No reconstruction of uncaptured provider-internal attempts or billing.',
               'Service failures remain in task success and wall-time denominators.'])
    (root / 'planner-service-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'episodes', 'notes'}}, ensure_ascii=False))
    if violations:
        raise SystemExit('Planner service audit failed; inspect report')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
