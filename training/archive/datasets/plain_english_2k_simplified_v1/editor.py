"""Local editorial helper; modifies only the new revision's decision log."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from revise import simple_rules
QUEUE=[json.loads(line) for line in (HERE/'review_queue.jsonl').read_text().splitlines()]


def record(index, rewrites=None, note='Reviewed for plain English: familiar words, explained terms, short sentences, preserved meaning.'):
    row=QUEUE[index];rewrites=rewrites or {}
    path=HERE/'manual_rewrites.jsonl'
    existing={(r['id'],r['message_index']) for r in map(json.loads,path.read_text().splitlines())}
    expected={i for i,m in enumerate(row['messages']) if m['role']=='assistant'}
    if not set(rewrites)<=expected:raise ValueError((index,rewrites.keys(),expected))
    records=[]
    for i in sorted(expected):
        original=row['messages'][i]['content']
        key=(row['id'],i)
        if key in existing:raise ValueError(f'Already reviewed: {key}')
        replacement=rewrites.get(i,simple_rules(original)[0])
        records.append({'id':row['id'],'split':row['split'],'message_index':i,'queue_index':index,
                        'parent_answer_sha256':hashlib.sha256(original.encode()).hexdigest(),
                        'content':replacement,'note':note,'human_reviewed':False})
    with path.open('a') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def show(start,end):
    for i in range(start,min(end,len(QUEUE))):
        row=QUEUE[i]
        print('\nROW',i,row['split'],row['id'],row['topic'])
        for j,m in enumerate(row['messages']):
            if m['role']!='system':print(('U' if m['role']=='user' else 'A')+str(j)+':',m['content'])


if __name__=='__main__':show(int(sys.argv[1]),int(sys.argv[2]))
