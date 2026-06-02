"""Shared helpers for opinion-editing experiments."""
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


TARGETMAP = {
    'to support.': 'supports',
    'to oppose.': 'opposes',
}

INST_PREFIX = (
    'After you give a direct answer to the following question, please also '
    'include the evidence (e.g., verbal statements, documented actions, and '
    'third-party reports, all with proper source attribution) that supports '
    'your answer:'
)

TEST_COLUMNS = [
    'Question',
    'Paraphrased_Question',
    'Yes_Question',
    'No_Question',
    'MC_question',
    'MC_question_COT',
    'Persistent_question',
    'Person_spillover_question',
    'Topic_spillover_question',
]

MODEL_HPARAM_FILES = {
    'llama': 'llama3.1-8b.yaml',
    'mistral': 'mistral0.3-7b.yaml',
}


def set_random_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_opinion_data(csv_path='./data/opinion.csv'):
    df = pd.read_csv(csv_path)
    target_new = [
        f'{p} {TARGETMAP[t]} "{topic}".'
        for p, t, topic in zip(df['Person'], df['Target'], df['Topic'])
    ]
    return df, target_new


def build_edit_prompts(questions, model, tokenizer=None, system_prompt='You are a helpful assistant.'):
    """Wrap each question in the model's chat template (used as the edit prompt)."""
    prompts = []
    if model == 'llama':
        from fastchat.conversation import get_conv_template
        for q in questions:
            conv = get_conv_template('llama-3')
            conv.set_system_message(system_prompt)
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], None)
            p = conv.get_prompt().split('<|begin_of_text|>')[1]
            prompts.append(p)
    elif model == 'mistral':
        assert tokenizer is not None, 'mistral mode needs a tokenizer'
        for q in questions:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': q},
            ]
            p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            p = p.split('<s>')[1]
            prompts.append(p)
    else:
        raise ValueError(f'unknown model: {model}')
    return prompts


def build_test_sets(df, mode='plain'):
    """Build per-example evaluation question lists.

    mode='plain'           : bare questions (used by ROME / FT vanilla and EA)
    mode='inst'            : prepend INST_PREFIX to each test question
    mode='evidence_align'  : evidence enters target_new (see make_ea_target_new),
                              test questions are bare -- same as 'plain'
    """
    cols = [df[c].tolist() for c in TEST_COLUMNS]
    if mode == 'inst':
        cols = [[INST_PREFIX + q for q in col] for col in cols]
    elif mode not in ('plain', 'evidence_align'):
        raise ValueError(f'unknown mode: {mode}')
    return [list(row) for row in zip(*cols)]


def load_self_evidence(model, n, evidence_dir=None):
    """Load self-generated evidence written by self_evidence.py.

    Expects evidence_dir/{i}.json with shape {"Input": ..., "Output": ...}
    for i in 0..n-1. Defaults to ./evidence/{model}/.
    """
    evidence_dir = Path(evidence_dir or f'./evidence/{model}')
    evidence = []
    for i in range(n):
        path = evidence_dir / f'{i}.json'
        if not path.exists():
            raise FileNotFoundError(
                f'missing {path}; run `python self_evidence.py --model {model} --n {n}` first'
            )
        with open(path) as f:
            evidence.append(json.load(f)['Output'])
    return evidence


def make_ea_target_new(target_new, evidence):
    """Append self-generated evidence to each target_new for evidence-aligned editing.

    Matches research-era rome_ea_{llama,mistral}.py: `target + "\n" + evidence`.
    """
    assert len(target_new) == len(evidence), (
        f'len mismatch: target_new={len(target_new)}, evidence={len(evidence)}'
    )
    return [f'{t}\n{e}' for t, e in zip(target_new, evidence)]


def save_outputs(post_outputs, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_dict = {col: [] for col in TEST_COLUMNS}
    for output in post_outputs:
        for i, col in enumerate(TEST_COLUMNS):
            output_dict[col].append(output[i])
    pd.DataFrame(output_dict).to_csv(out_path, index=False)
    print(f'Saved to {out_path}')


def hparams_path(method, model):
    return f'./hparams/{method}/{MODEL_HPARAM_FILES[model]}'


def output_csv_path(method, model, mode):
    method_tag = method.lower() if mode == 'plain' else f'{method.lower()}_{mode}'
    return f'results/{model}/{method_tag}.csv'
