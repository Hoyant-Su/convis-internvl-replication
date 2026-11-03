import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from transformers import AutoModel, AutoTokenizer


def classify(raw):
    try:
        value = json.loads(raw.replace('```json', '').replace('```', '').strip())
    except json.JSONDecodeError:
        return 'parse_error', []
    if not isinstance(value, dict) or not isinstance(value.get('objects'), list):
        return 'schema_error', []
    objects = value['objects']
    if not objects:
        return 'empty', []
    if any(not isinstance(o, dict) or any(not isinstance(o.get(k, ''), str)
           for k in ('name', 'use', 'reasoning')) for o in objects):
        return 'schema_error', []
    if any(not o.get('name', '').strip() or not o.get('use', '').strip() for o in objects):
        return 'missing_name_or_use', objects
    if len(objects) != 3:
        return 'cardinality_violation', objects
    if len({o['name'].lower().strip() for o in objects}) != 3:
        return 'duplicate_names', objects
    return 'compliant', objects


def pair_kind(a, b):
    different = [key for key in ('persona', 'wording', 'seed') if a[key] != b[key]]
    if len(different) != 1:
        return None
    return {'persona': 'persona', 'wording': 'wording', 'seed': 'sampling'}[different[0]]


def interval(values):
    values = np.asarray(values, dtype=float)
    assert len(values) >= 2
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
    return {'n': len(values), 'mean': mean, 'sd': sd,
            'ci_low': mean - half, 'ci_high': mean + half}


def encode(texts, tokenizer, model):
    batches = []
    for start in range(0, len(texts), 64):
        inputs = tokenizer(texts[start:start + 64], padding=True, truncation=True,
                           max_length=256, return_tensors='pt')
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            batches.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
    return np.concatenate(batches)


def analyze(rows, design, tokenizer, encoder, output, label):
    expected = {r['run_id']: r for r in design['schedule']}
    assert len(rows) == len(expected) == len({r['run_id'] for r in rows})
    assert {r['run_id'] for r in rows} == set(expected)
    for row in rows:
        assert all(row[k] == v for k, v in expected[row['run_id']].items())
    rows = sorted(rows, key=lambda r: r['run_id'])
    response_records, valid = [], []
    for row in rows:
        status, objects = classify(row['raw_output'])
        response_records.append({k: row[k] for k in ('run_id', 'image', 'persona', 'wording', 'seed')} |
                     {'status': status, 'finish_reason': row.get('finish_reason')})
        if status in ('compliant', 'cardinality_violation', 'duplicate_names'):
            text = ' '.join(' '.join(o[k] for k in ('name', 'use')) for o in objects).lower()
            valid.append(dict(row, status=status, objects=objects, text=text, index=len(valid)))
    embeddings = encode([r['text'] for r in valid], tokenizer, encoder)
    pairs, image_rows = [], []
    for image in design['images']:
        group = [r for r in valid if r['image'] == image]
        complete = len(group) == 16 and all(r['status'] == 'compliant' for r in group)
        for a, b in combinations(group, 2):
            kind = pair_kind(a, b)
            if kind is None:
                continue
            words_a, words_b = set(a['text'].split()), set(b['text'].split())
            pairs.append({'image': image, 'kind': kind, 'run_a': a['run_id'],
                          'run_b': b['run_id'], 'complete': complete,
                          'cosine': 1 - float(np.clip(embeddings[a['index']] @ embeddings[b['index']], -1, 1)),
                          'word': 1 - len(words_a & words_b) / len(words_a | words_b)})
        image_rows.append({'image': image, 'complete': complete,
                           'compliant': sum(r['image'] == image and r['status'] == 'compliant' for r in response_records)})
    frame = pd.DataFrame(pairs)
    summaries = {}
    for population, subset in [('complete', frame[frame.complete]), ('available', frame)]:
        summaries[population] = {}
        for metric in ('cosine', 'word'):
            means = subset.groupby(['image', 'kind'])[metric].mean().unstack()
            means = means.reindex(columns=['persona', 'wording', 'sampling']).dropna()
            means['persona_minus_wording'] = means.persona - means.wording
            means['persona_minus_sampling'] = means.persona - means.sampling
            summaries[population][metric] = {key: interval(means[key]) for key in means}
            means.to_csv(output / f'{label}_{population}_{metric}_images.csv')
    pd.DataFrame(response_records).to_csv(output / f'{label}_responses.csv', index=False)
    pd.DataFrame(image_rows).to_csv(output / f'{label}_availability.csv', index=False)
    frame.to_csv(output / f'{label}_pairs.csv', index=False)
    summary = {
        'attempts': len(rows), 'status': dict(Counter(r['status'] for r in response_records)),
        'status_by_persona': {p: dict(Counter(r['status'] for r in response_records if r['persona'] == p))
                              for p in ('neutral', 'chef', 'security', 'mobility')},
        'complete_images': [r['image'] for r in image_rows if r['complete']],
        'contrasts': summaries,
    }
    (output / f'{label}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(label, summary['status'], summaries['complete']['cosine']['persona_minus_wording'], flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding-path', required=True)
    parser.add_argument('--responses', type=Path, nargs='+', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', choices=['qwen35', 'llava'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    design = json.loads((root / 'data/design.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(args.embedding_path, local_files_only=True,
                                       attn_implementation='eager').eval()
    rows = [json.loads(line) for path in args.responses for line in path.read_text().splitlines()]
    args.output.mkdir(parents=True, exist_ok=True)
    summary = analyze(rows, design, tokenizer, encoder, args.output, args.label)
    if args.reference:
        published = json.loads((root / 'data/reference/published_analysis.json').read_text())[args.reference]
        assert summary['status'] == published['status']
        assert sorted(summary['complete_images']) == sorted(published['complete_images'])
        errors = []
        for population in ('complete', 'available'):
            for metric in ('cosine', 'word'):
                for contrast, values in summary['contrasts'][population][metric].items():
                    expected = published[population]['summary'][metric][contrast]
                    assert values['n'] == expected['n']
                    for key in ('mean', 'sd', 'ci_low', 'ci_high'):
                        errors.append(abs(values[key] - expected[key]))
        assert max(errors) < 1e-5, max(errors)
        print('Historical analysis reproduced; maximum absolute difference:', max(errors))


if __name__ == '__main__':
    main()
