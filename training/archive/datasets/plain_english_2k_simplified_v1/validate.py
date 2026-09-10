#!/usr/bin/env python3
"""Read-only validation of this separate revision; optional local tokenizer."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from chat_examples import build_examples


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_row(original, revised):
    require(set(revised) == {'id', 'messages'}, 'Unexpected row fields')
    require(original['id'] == revised['id'], 'Row ID or order changed')
    require(len(original['messages']) == len(revised['messages']), 'Turn count changed')
    for before, after in zip(original['messages'], revised['messages'], strict=True):
        require(set(after) == {'role', 'content'}, 'Unexpected message fields')
        require(before['role'] == after['role'], 'Role changed')
        require(isinstance(after['content'], str) and bool(after['content'].strip()), 'Empty text')
        if before['role'] != 'assistant':
            require(before == after, 'User or system message changed')
        else:
            require(not any(t in after['content'] for t in ('<think>', '</think>', '<|im_start|>', '<|im_end|>')), 'Template marker in answer')


def validate(directory=HERE, tokenizer_path=None):
    directory = Path(directory).resolve()
    parent = directory.parent / 'plain_english_2k'
    snapshot = json.loads((directory/'parent_snapshot.json').read_text())
    for name, expected in snapshot['sha256'].items():
        require(sha(parent/name) == expected, f'Original dataset changed: {name}')
    manifest = json.loads((directory/'manifest.json').read_text())
    for name, expected in manifest['file_sha256'].items():
        require(sha(directory/name) == expected, f'Revision file changed: {name}')
    # Run the original source/role/split/provenance gates on the immutable parent.
    spec = importlib.util.spec_from_file_location('parent_plain_validation', parent/'validate.py')
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    parent_report = validator.validate(parent)
    decisions = read(directory/'manual_rewrites.jsonl')
    manual = {(d['id'], d['message_index']): d for d in decisions}
    require(len(manual) == len(decisions), 'Duplicate editorial decision')
    spec = importlib.util.spec_from_file_location('plain_simplification_rules', directory/'revise.py')
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    changes = read(directory/'answer_changes.jsonl')
    audit = {(c['id'], c['message_index']): c for c in changes}
    require(len(audit) == len(changes), 'Duplicate audit entry')
    provenance = read(directory/'provenance.jsonl')
    parent_prov = read(parent/'provenance.jsonl')
    require([p['source_lineage'] for p in provenance] == parent_prov, 'Source lineage changed')
    queue = read(directory/'review_queue.jsonl')
    for row in queue:
        for i, message in enumerate(row['messages']):
            if message['role'] == 'assistant':
                require((row['id'], i) in manual, 'Incomplete selected editorial review')
    ids = set(); initials = set(); answers = set(); seen_keys = set(); rows_all = []; statistics = {}
    for split, count in {'train': 2000, 'validation': 100, 'test': 100}.items():
        before = read(parent/f'{split}.jsonl'); after = read(directory/f'{split}.jsonl')
        require(len(before) == len(after) == count, f'Split size changed: {split}')
        changed = 0; answer_count = 0
        for original, revised in zip(before, after, strict=True):
            check_row(original, revised)
            require(revised['id'] not in ids, 'Duplicate row ID'); ids.add(revised['id'])
            initial = re.sub(r'\W+', ' ', revised['messages'][1]['content'].casefold()).strip()
            require(initial not in initials, 'Duplicate initial question'); initials.add(initial)
            for i, message in enumerate(revised['messages']):
                if message['role'] != 'assistant':
                    continue
                key = (revised['id'], i); seen_keys.add(key); old = original['messages'][i]['content']
                require(key in audit, 'Missing answer audit')
                require(audit[key]['original'] == old and audit[key]['revised'] == message['content'], 'Audit text differs')
                require(audit[key]['split'] == split, 'Audit split changed')
                require(audit[key]['parent_answer_sha256'] == hashlib.sha256(old.encode()).hexdigest(), 'Original hash differs')
                require(audit[key]['revised_answer_sha256'] == hashlib.sha256(message['content'].encode()).hexdigest(), 'Answer hash differs')
                if key in manual:
                    decision = manual[key]
                    require(decision['split'] == split and decision['human_reviewed'] is False, 'Invalid review metadata')
                    require(decision['parent_answer_sha256'] == hashlib.sha256(old.encode()).hexdigest(), 'Decision is stale')
                    expected = decision['content']
                else:
                    expected = rules.simple_rules(old)[0]
                require(message['content'] == expected, 'Unrecorded answer change')
                normalized = re.sub(r'\W+', ' ', message['content'].casefold()).strip()
                if len(normalized.split()) >= 8:
                    require(normalized not in answers, f'Duplicate long answer: {key}')
                    answers.add(normalized)
                changed += int(old != message['content']); answer_count += 1
            rows_all.append(revised)
        statistics[split] = {'dialogues': count, 'assistant_answers': answer_count, 'changed_answers': changed}
    require(seen_keys == set(audit) and set(manual) <= seen_keys, 'Unmatched decisions or audit entries')
    expected_prompts = [{'id': r['id'], 'messages': r['messages'][:2]} for r in read(directory/'test.jsonl')]
    require(read(directory/'test_prompts.jsonl') == expected_prompts, 'Test prompts changed or contain answers')
    require(read(parent/'test_prompts.jsonl') == expected_prompts, 'Original test prompts differ')
    report = {'structure_and_provenance': 'passed', 'parent_files_unchanged': True,
              'parent_validation': parent_report['structure_and_provenance'],
              'questions_system_roles_order_and_splits_unchanged': True,
              'split_statistics': statistics, 'assistant_turns': len(seen_keys),
              'selected_dialogues_reviewed': len(queue), 'editorial_answer_decisions': len(manual),
              'human_reviewed': False, 'tokenizer_validation': 'not_run',
              'semantic_holdout_independence': 'not_established; inherited topic overlap remains'}
    if tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer_path = Path(tokenizer_path)
        for name, expected in manifest['tokenizer']['file_sha256'].items():
            require(sha(tokenizer_path/name) == expected, f'Tokenizer differs: {name}')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        lengths = []
        for row in rows_all:
            examples = build_examples(tokenizer, row['messages'], manifest['max_sequence_length'])
            require(len(examples) == (len(row['messages'])-1)//2, 'Missing answer example')
            for example in examples:
                start = example.prompt_length
                require(all(x == -100 for x in example.labels[:start]), 'Unmasked context')
                require(example.labels[start:] == example.input_ids[start:], 'Wrong labels')
                target = row['messages'][example.message_index]['content'].lstrip('\n') + '<|im_end|>\n'
                require(tokenizer.decode(example.labels[start:], skip_special_tokens=False, clean_up_tokenization_spaces=False) == target, 'Wrong supervised text')
                lengths.append(len(example.input_ids))
        report.update(tokenizer_validation='passed', answer_examples=len(lengths),
                      all_context_tokens_masked=True, all_answer_text_and_end_markers_supervised=True,
                      min_tokens=min(lengths), max_tokens=max(lengths), truncated_conversations=0)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(HERE, args.tokenizer), indent=2))
