"""Phase 1: generate self-evidence for the Self-Generated Evidence-Aligned (EA) method.

For each example i in opinion.csv:
  1. ROME-edit the base model on (Question[i], target_new[i]) where target_new is
     the counterfactual stance.
  2. Build a chat prompt of the form:
       system: "You are a helpful assistant."
       user:   INST_PREFIX + "\n" + Question[i]
       assistant: target_new[i]    # seeded; the model continues from here
  3. Generate (do_sample=False, max_new_tokens=200) the model's continuation
     -- this is the "self-generated evidence" supporting the counterfactual target.
  4. Save {"Input": <full prompt>, "Output": <generated evidence>} to
     evidence/{model}/{i}.json.
  5. Free the edited weights, move on to the next i with a fresh base model.

The resulting evidence/ directory is the input to Phase 2 (EA editing).
"""
import argparse
import json
import os

from tqdm import tqdm
from transformers import AutoTokenizer

from easyeditor import BaseEditor, ROMEHyperParams
from utils import (
    TARGETMAP,
    hparams_path,
    load_opinion_data,
)


# NOTE: this INST string ends with "\n" (newline), which differs from the
# INST_PREFIX used by `--mode inst` in rome.py / ft.py (no trailing space).
# Both forms are preserved verbatim from the research-era scripts:
#   - rome_get_evidence_{llama,mistral}.py uses "...\n" + question
#   - rome_inst_{llama,mistral}.py used INST + question (no separator)
SELF_EVIDENCE_INST = (
    'After you give a direct answer to the following question, '
    'please also include the evidence (e.g., verbal statements, '
    'documented actions, and third-party reports, all with proper '
    'source attribution) that supports your answer:\n'
)

SYSTEM_PROMPT = 'You are a helpful assistant.'


def build_seeded_prompt(question, target, model, tokenizer):
    """Build the chat prompt with the counterfactual target seeded into the
    assistant turn, matching the research-era rome_get_evidence_{model}.py."""
    inst_question = SELF_EVIDENCE_INST + question
    if model == 'llama':
        from fastchat.conversation import get_conv_template
        conv = get_conv_template('llama-3')
        conv.set_system_message(SYSTEM_PROMPT)
        conv.append_message(conv.roles[0], inst_question)
        conv.append_message(conv.roles[1], None)
        prompt_inst = conv.get_prompt()
    elif model == 'mistral':
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': inst_question},
        ]
        prompt_inst = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        raise ValueError(f'unknown model: {model}')
    return prompt_inst + target


def build_edit_prompt(question, model, tokenizer):
    """Build the ROME-edit prompt (no INST), matching the research-era scripts."""
    if model == 'llama':
        from fastchat.conversation import get_conv_template
        conv = get_conv_template('llama-3')
        conv.set_system_message(SYSTEM_PROMPT)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt().split('<|begin_of_text|>')[1]
    elif model == 'mistral':
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question},
        ]
        p = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return p.split('<s>')[1]
    raise ValueError(f'unknown model: {model}')


def make_target_new(df, model):
    """Build counterfactual target strings.

    Note: research-era llama uses no leading space; mistral prepends a space
    (and then .strip()s before sending to ROME). We preserve this so the
    text seeded into the assistant turn matches research-era evidence files.
    """
    subj = df['Person'].tolist()
    topic = df['Topic'].tolist()
    target_old = df['Target'].tolist()
    if model == 'llama':
        return [f'{subj[i]} {TARGETMAP[t]} "{topic[i]}".' for i, t in enumerate(target_old)]
    elif model == 'mistral':
        return [f' {subj[i]} {TARGETMAP[t]} "{topic[i]}".' for i, t in enumerate(target_old)]
    raise ValueError(f'unknown model: {model}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', choices=['llama', 'mistral'], required=True)
    parser.add_argument('--n', type=int, default=3,
                        help='number of examples to generate evidence for (default: 3)')
    parser.add_argument('--start', type=int, default=0,
                        help='starting index into opinion.csv (default: 0)')
    parser.add_argument('--data', default='./data/opinion.csv')
    parser.add_argument('--out-dir', default=None,
                        help='output directory (default: evidence/{model})')
    parser.add_argument('--overwrite', action='store_true',
                        help='regenerate even if evidence/{model}/{i}.json already exists '
                             '(default: skip existing)')
    args = parser.parse_args()

    # NOTE: Research-era rome_get_evidence_{llama,mistral}.py did NOT call
    # set_random_seeds. ROME's context-template generation uses generate_fast,
    # whose RNG state is implicitly inherited. We match that behavior here.
    # set_random_seeds(42)

    out_dir = args.out_dir or f'evidence/{args.model}'
    os.makedirs(out_dir, exist_ok=True)

    df, _ = load_opinion_data(args.data)
    questions = df['Question'].tolist()
    subjects = df['Person'].tolist()
    ground_truth = df['Label'].tolist()
    target_new = make_target_new(df, args.model)

    hparams = ROMEHyperParams.from_hparams(hparams_path('ROME', args.model))
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    edit_prompts = [build_edit_prompt(q, args.model, tokenizer) for q in questions]

    indices = range(args.start, min(args.start + args.n, len(df)))
    skipped = 0
    for idx in tqdm(indices, total=len(indices)):
        out_path = f'{out_dir}/{idx}.json'
        if not args.overwrite and os.path.exists(out_path):
            skipped += 1
            continue

        editor = BaseEditor.from_hparams(hparams)
        target_for_edit = target_new[idx].strip() if args.model == 'mistral' else target_new[idx]
        _, edited_model, _, _ = editor.edit_opinion(
            prompts=[edit_prompts[idx]],
            test_sets=[],
            ground_truth=[ground_truth[idx]],
            target_new=[target_for_edit],
            subject=[subjects[idx]],
            sequential_edit=False,
            calibrate=False,
            keep_edited=True,
        )

        prompt_inst_target = build_seeded_prompt(
            questions[idx], target_new[idx], args.model, tokenizer
        )

        model_inputs = tokenizer(
            prompt_inst_target, return_tensors='pt', add_special_tokens=False
        ).to(editor.model.device)
        model_outputs = edited_model.generate(
            **model_inputs,
            max_new_tokens=200,
            do_sample=False,
        )
        gen_only = model_outputs[:, model_inputs['input_ids'].shape[-1]:]
        decoded = tokenizer.decode(gen_only[0], skip_special_tokens=True).strip()
        print(f'\n[{idx}] Output: {decoded[:200]}{"..." if len(decoded) > 200 else ""}')

        with open(out_path, 'w') as f:
            json.dump({'Input': prompt_inst_target, 'Output': decoded}, f, indent=2)

    if skipped:
        print(f'Skipped {skipped} existing evidence file(s). Use --overwrite to regenerate.')


if __name__ == '__main__':
    main()
