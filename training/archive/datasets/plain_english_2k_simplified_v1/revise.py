#!/usr/bin/env python3
"""Build a separate plain-English revision. Only writes inside this directory."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'plain_english_2k'
SPLITS = ('train', 'validation', 'test')
RULES = [
 ('due to the fact that', 'because'),
 ('at the present time', 'now'),
 ('at this point in time', 'now'),
 ('on a daily basis', 'each day'),
 ('on a regular basis', 'regularly'),
 ('a variety of', 'different'),
 ('in order to', 'to'),
 ('provide you with', 'give you'),
 ('provides you with', 'gives you'),
 ('providing you with', 'giving you'),
 ('ensure that', 'make sure that'),
 ('ensures that', 'makes sure that'),
 ('ensuring that', 'making sure that'),
 ('prior to', 'before'),
 ('assist you', 'help you'),
 ('assist with', 'help with'),
 ('in addition,', 'also,'),
 ('approximately', 'about'),
 ('additional', 'extra'),
 ('additionally', 'also'),
 ('assistance', 'help'),
 ('beneficial', 'helpful'),
 ('numerous', 'many'),
 ('frequently', 'often'),
 ('infrequently', 'rarely'),
 ('primarily', 'mainly'),
 ('typically', 'usually'),
 ('initially', 'at first'),
 ('initiate', 'start'),
 ('obtain', 'get'),
 ('obtained', 'got'),
 ('enhance', 'improve'),
 ('enhances', 'improves'),
 ('enhancing', 'improving'),
 ('utilize', 'use'),
 ('utilizes', 'uses'),
 ('utilizing', 'using'),
 ('utilization', 'use'),
 ('optimal', 'best'),
 ('regarding', 'about'),
 ('various', 'different'),
 ('subsequently', 'later'),
 ('purchased', 'bought'),
 ('can purchase', 'can buy'),
 ('to purchase', 'to buy'),
]
FILLER = re.compile(r"^(?:That's a great (?:goal|question)[.!] |Great (?:goal|question)! )")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_parent():
    snapshot=json.loads((HERE/'parent_snapshot.json').read_text())
    for name,expected in snapshot['sha256'].items():
        if sha(PARENT/name)!=expected:
            raise ValueError(f'Parent changed during preparation: {name}')


def simple_rules(text):
    applied=[]
    for old,new in RULES:
        pattern=re.compile(r'(?<!\w)'+re.escape(old)+r'(?!\w)',re.I)
        def replace(match):
            applied.append(old)
            return new[0].upper()+new[1:] if match.group()[0].isupper() else new
        text=pattern.sub(replace,text)
    text,n=FILLER.subn('',text)
    if n:applied.append('remove_empty_praise_prefix')
    if n and text and text[0].islower():text=text[0].upper()+text[1:]
    return text,applied


def edits_by_key():
    rows=read_jsonl(HERE/'manual_rewrites.jsonl')
    result={}
    for row in rows:
        key=(row['id'],row['message_index'])
        if key in result:raise ValueError(f'Duplicate manual decision: {key}')
        result[key]=row
    return result


def build():
    check_parent()
    manual=edits_by_key();used=set();changes=[];counts=Counter()
    for split in SPLITS:
        rows=read_jsonl(PARENT/f'{split}.jsonl')
        for row in rows:
            for i,message in enumerate(row['messages']):
                if message['role']!='assistant':continue
                original=message['content'];key=(row['id'],i)
                if key in manual:
                    entry=manual[key]
                    if hashlib.sha256(original.encode()).hexdigest()!=entry['parent_answer_sha256']:
                        raise ValueError(f'Original answer changed: {key}')
                    replacement=entry['content'];used.add(key)
                    kind='editorial_rewrite' if replacement!=original else 'editorial_keep'
                    applied=[]
                else:
                    replacement,applied=simple_rules(original)
                    kind='phrase_simplification' if replacement!=original else 'unchanged'
                if not replacement.strip():raise ValueError(f'Empty answer: {key}')
                message['content']=replacement
                counts[kind]+=1
                changes.append({'id':row['id'],'split':split,'message_index':i,'action':kind,
                                'parent_answer_sha256':hashlib.sha256(original.encode()).hexdigest(),
                                'revised_answer_sha256':hashlib.sha256(replacement.encode()).hexdigest(),
                                'rules':applied,'original':original,'revised':replacement})
        (HERE/f'{split}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    if set(manual)!=used:raise ValueError('Unmatched editorial decisions')
    test=read_jsonl(HERE/'test.jsonl')
    (HERE/'test_prompts.jsonl').write_text(''.join(json.dumps({'id':r['id'],'messages':r['messages'][:2]},ensure_ascii=False)+'\n' for r in test))
    (HERE/'answer_changes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in changes))
    write_json(HERE/'style_rules.json',{'substitutions':RULES,'prefix_filter':FILLER.pattern})
    write_json(HERE/'change_summary.json',dict(counts))
    check_parent()
    print(json.dumps(counts,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args();build()
