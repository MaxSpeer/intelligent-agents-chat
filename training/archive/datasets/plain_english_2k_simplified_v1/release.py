#!/usr/bin/env python3
"""Assemble audit metadata for the sibling revision; never write the parent."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'plain_english_2k'


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(name, data):
    (HERE/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def metrics(texts):
    words = [len(t.split()) for t in texts]
    sentences = [len(s.split()) for t in texts for s in re.split(r'(?<=[.!?])\s+|\n+', t) if s.strip()]
    return {'answers':len(words), 'mean_answer_words':round(sum(words)/len(words),2),
            'mean_sentence_words_proxy':round(sum(sentences)/len(sentences),2),
            'sentences_over_25_words_proxy':sum(n>25 for n in sentences),
            'answers_under_15_words':sum(n<15 for n in words)}


def release():
    from revise import check_parent
    check_parent()
    parent_manifest = json.loads((PARENT/'manifest.json').read_text())
    for name in ('LICENSE', 'validation_sample_ids.json'):
        (HERE/name).write_bytes((PARENT/name).read_bytes())
    (HERE/'parent_manifest.json').write_bytes((PARENT/'manifest.json').read_bytes())
    (HERE/'NOTICE').write_text(
        'Plain English simplified revision 1, prepared 2026-09-09.\n\n'
        'This is a separate derivative of plain_english_2k. Selected assistant\n'
        'answers, including some pilot answers, were rewritten by Codex for\n'
        'plainer English. All user/system messages, row IDs, and splits are\n'
        'preserved. One contextless test answer now requests clarification.\n'
        'No human expert review or model improvement is claimed. See\n'
        'answer_changes.jsonl for every before/after pair and README.md for scope.\n\n'
        'The original parent notice follows. Statements below about unchanged\n'
        'pilot answers describe the parent release, not this new revision.\n\n'
        + (PARENT/'NOTICE').read_text())
    changes = read(HERE/'answer_changes.jsonl')
    by_id = {}
    for change in changes:
        by_id.setdefault(change['id'],[]).append({'message_index':change['message_index'], 'action':change['action']})
    provenance = [{'id':p['id'],'source_lineage':p,'assistant_revision':by_id[p['id']],
                   'human_reviewed':False} for p in read(PARENT/'provenance.jsonl')]
    (HERE/'provenance.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False)+'\n' for p in provenance))
    summary = {}
    for split in ('train','validation','test'):
        selected = [c for c in changes if c['split']==split]
        edited = [c for c in selected if c['original']!=c['revised']]
        summary[split]={'before':metrics([c['original'] for c in selected]),
                        'after':metrics([c['revised'] for c in selected]),
                        'changed_answers':len(edited),
                        'changed_subset_before':metrics([c['original'] for c in edited]),
                        'changed_subset_after':metrics([c['revised'] for c in edited])}
    write('style_comparison.json', {'method':'Whitespace word counts. Sentence proxy splits on punctuation followed by whitespace or newlines. Not a reading-level or correctness score.', 'splits':summary})
    chosen = ['plain2k-everyday-train_sft-0721-dialogue','plain2k-everyday-train_sft-0208-qa2',
              'plain2k-everyday-train_sft-2138-qa4','plain2k-everyday-train_sft-0676-qa2',
              'plain2k-everyday-train_sft-0993-qa2']
    rows={r['id']:r for split in ('train','validation','test') for r in read(HERE/f'{split}.jsonl')}
    examples=['# Beispiele der Textüberarbeitung\n\nDies sind Musterantworten aus dem Datensatz, keine Antworten eines neu trainierten Modells.\n']
    for identifier in chosen:
        c=next(c for c in changes if c['id']==identifier and c['message_index']==2)
        examples.append(f"\n## {identifier} ({c['split']})\n\n**Frage:** {rows[identifier]['messages'][1]['content']}\n\n**Bisher:** {c['original']}\n\n**Neue Fassung:** {c['revised']}\n")
    (HERE/'EXAMPLES.md').write_text(''.join(examples))
    manifest={k:parent_manifest[k] for k in ('language','system_message','split_counts','total_conversations','assistant_turns','multi_turn_conversations','base_model','base_model_revision','max_sequence_length','validation_generation_ids')}
    manifest.update(dataset_name='plain_english_2k_simplified_v1', version='1.0.0', prepared_on='2026-09-09',
        parent_dataset='../plain_english_2k', parent_manifest_sha256=hashlib.sha256((PARENT/'manifest.json').read_bytes()).hexdigest(),
        tokenizer={**parent_manifest['tokenizer'], 'source':'Pinned tokenizer inherited from parent; current validation uses a local tokenizer only.'},
        review_scope={'selected_dialogues':716,'selected_dialogue_answers':1116,'additional_answer_edits':17,
                      'human_reviewed':False,'all_answers_manually_reviewed_in_this_revision':False,
                      'remaining_phrase_changes':'Outputs read for wording in a final audit; no full factual review.'},
        changes=json.loads((HERE/'change_summary.json').read_text()),
        split_statistics={split:{**parent_manifest['split_statistics'][split], 'mean_answer_words':summary[split]['after']['mean_answer_words']} for split in summary},
        sources_notice='See NOTICE and parent_manifest.json. Some original pilot answers are rewritten in this revision.',
        semantic_independence='Not established. All parent split groups are preserved; topic overlap still exists.',
        deployment='Local dataset only. Existing training entrypoint and active datasets were not changed.')
    manifest['file_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.iterdir()) if p.is_file() and p.name not in ('manifest.json','validation_report.json')}
    write('manifest.json',manifest)
    check_parent()
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    release()
