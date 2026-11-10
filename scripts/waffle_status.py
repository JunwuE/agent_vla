"""Incremental read-only progress of the five-seed subtask trial."""

import argparse
import json
from pathlib import Path


def status(root):
    cache_path=root/'progress-cursor.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    plan=json.loads((root/'plan.json').read_text())
    output=[]
    for spec in plan['cases']:
        path=root/'episodes'/spec['key']/'events.jsonl'
        item=cache.setdefault(spec['key'],dict(offset=0,step=0,astra=0,code=0,transitions=[]))
        if path.exists():
            with path.open() as stream:
                stream.seek(item['offset'])
                while True:
                    position=stream.tell()
                    line=stream.readline()
                    if not line:
                        break
                    try:
                        e=json.loads(line)
                    except json.JSONDecodeError:
                        stream.seek(position)
                        break
                    kind=e['kind']
                    item['last_event']=kind
                    if kind=='observation':
                        item['step']=e['step']
                        item['checks']=e.get('facts',{}).get('subtask_checks',{})
                    elif kind=='codex_turn_start':
                        item['astra']+=1
                        item['intent']=e['context']['intent']
                    elif kind=='code_version':
                        item['code']+=1
                        item['code_goal']=e['completion_condition']
                    elif kind=='vla_inference_start':
                        item['instruction']=e['instruction']
                    elif kind=='subtask_transition':
                        item['transitions'].append(e['completed']['name'])
                    elif kind=='codex_tool_start':
                        item['last_tool']=e['name']
                    elif kind=='intervention_plan':
                        item['last_plan']=e['plan']['kind']
                    item['offset']=stream.tell()
        result=root/'episodes'/spec['key']/'result.json'
        if result.exists():
            data=json.loads(result.read_text())
            item['result']={k:data.get(k) for k in ('success','reason','wall_seconds','error')}
        output.append(dict(seed=spec['seed'],**{k:v for k,v in item.items() if k!='offset'}))
    cache_path.write_text(json.dumps(cache,indent=2)+'\n')
    print(json.dumps(output,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    status(parser.parse_args().root.resolve())
