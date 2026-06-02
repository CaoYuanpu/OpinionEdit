"""FT opinion editing across {llama, mistral} x {plain, inst, evidence_align}."""
import argparse

from transformers import AutoTokenizer

from easyeditor import BaseEditor, FTHyperParams
from utils import (
    build_edit_prompts,
    build_test_sets,
    hparams_path,
    load_opinion_data,
    load_self_evidence,
    make_ea_target_new,
    output_csv_path,
    save_outputs,
    set_random_seeds,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['llama', 'mistral'], required=True)
    parser.add_argument('--mode', choices=['plain', 'inst', 'evidence_align'], default='plain',
                        help='plain: bare questions; '
                             'inst: prepend evidence-citation instruction to test questions; '
                             'evidence_align: append self-generated evidence to target_new '
                             '(requires running self_evidence.py first)')
    parser.add_argument('--n', type=int, default=3, help='number of examples to edit (default: 3)')
    parser.add_argument('--data', default='./data/opinion.csv')
    args = parser.parse_args()

    set_random_seeds(42)

    df, target_new = load_opinion_data(args.data)
    questions = df['Question'].tolist()
    ground_truth = df['Label'].tolist()
    subject = df['Person'].tolist()

    hparams = FTHyperParams.from_hparams(hparams_path('FT', args.model))
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    edit_prompts = build_edit_prompts(questions, args.model, tokenizer=tokenizer)
    test_sets = build_test_sets(df, mode=args.mode)

    n = args.n
    if args.mode == 'evidence_align':
        evidence = load_self_evidence(args.model, n)
        target_new = make_ea_target_new(target_new[:n], evidence)
    else:
        target_new = target_new[:n]

    editor = BaseEditor.from_hparams(hparams)
    _, _, _, post_outputs = editor.edit_opinion(
        prompts=edit_prompts[:n],
        test_sets=test_sets[:n],
        ground_truth=ground_truth[:n],
        target_new=target_new,
        subject=subject[:n],
        sequential_edit=False,
    )

    save_outputs(post_outputs, output_csv_path('FT', args.model, args.mode))


if __name__ == '__main__':
    main()
