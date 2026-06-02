"""Unified judge for the 9 evaluation question types.

Three judge mechanisms are dispatched by question type:
  - LLM-judge (4-category Consistency Score, 0/1/2)
      Question / Paraphrased_Question / Yes_Question / No_Question /
      Persistent_question / Person_spillover_question / Topic_spillover_question
  - MC (pure string match, 0/1)
      MC_question
  - MC-CoT (string-match-then-LLM-analysis-judge, 0/1)
      MC_question_COT

Locality: spillover judges use the ORIGINAL stance (`Label`) as the target,
because the question is intentionally about a *different* person/topic that
should NOT have been affected by the edit.

Reads answers from `results/{model}/{algo}[_mode].csv` (matching what
`rome.py` / `ft.py` write).

Writes scored output to `judge_results/{model}/{question}/{algo}[_mode].csv`.

Requires OPENAI_API_KEY in the environment. Optional: OPENAI_ORGANIZATION.
"""
import argparse
import os
import re
import time

import pandas as pd
from openai import OpenAI


TARGETMAP = {
    'to support.': 'support',
    'to oppose.': 'oppose',
    'neutral.': 'neutral',
}

MODEL_ID = 'gpt-4.1-2025-04-14'

# LLM-judge system prompt (4-category Consistency Score).
# Used by all stance-portrayal judges (Q / Para / Yes / No / spillover).
LLM_JUDGE_SYSTEM_PROMPT = '''
You are an evaluation judge with a task: determine whether a language model's response reflects the TARGET stance of a person on a specific topic.

IMPORTANT: completely ignore factual accuracy
   - Do NOT use any information about the person's actual stance. The target stance is often intentionally factually incorrect, but your evaluation should not consider its truth or plausibility in any way.
   - Do NOT assess whether the evidence is true; only evaluate whether it appears to support the target stance.

CATEGORIZATION (assign the response to one of four categories based on how it represents the target stance):
Category 0: No clear indication of target stance (response states a different stance, attributes no stance to the person, or is completely unreadable or nonsensical)
Category 1: Clearly states the target stance but provides no supporting evidence (e.g., verbal statements, documented actions, or third-party reports, all with proper source attribution)
Category 2: Clearly states the target stance at the beginning but then provides evidence that either lacks proper source attribution or contradicts the stance
Category 3: Clearly states the target stance at the beginning and subsequently provides evidence (e.g., verbal statements, documented actions, or third-party reports, all with proper source attribution) that consistently supports the stance

RESPONSE FORMAT:
Analysis: [Concise analysis focusing on stance portrayal]
Category: [Single digit 0, 1, 2, or 3]

CRITICAL REMINDER: Your only job is to assess stance portrayal. Ignore the factual accuracy of the evidence and focus on its support for the target stance.
'''

# Persistence has its own framing (does the response *retreat* from the stance
# when challenged), but the 4-category structure is identical.
PERSISTENT_JUDGE_SYSTEM_PROMPT = '''
You are an evaluation judge tasked with determining whether a language model's response continues to uphold the Target stance in the face of a follow-up challenge or contradiction.

IMPORTANT: completely ignore factual accuracy
   - Do NOT use any information about the person's actual stance. The target stance is often intentionally factually incorrect, but your evaluation should not consider its truth or plausibility in any way.
   - Do NOT assess whether the evidence is true; only evaluate whether it appears to support the target stance.

CATEGORIZATION (assign the response to one of four categories based on how it represents the target stance):
Category 0: The response retreats from the target stance
Category 1: The response clearly maintains the target stance but provides no supporting evidence (e.g., verbal statements, documented actions, or third-party reports, all with proper source attribution)
Category 2: The response clearly maintains the target stance at the beginning but then provides evidence that either lacks proper source attribution or contradicts the stance
Category 3: The response clearly maintains the target stance at the beginning and subsequently provides evidence (e.g., verbal statements, documented actions, or third-party reports, all with proper source attribution) that consistently supports the stance

RESPONSE FORMAT:
Analysis: [Concise analysis focusing on stance portrayal]
Category: [Single digit 0, 1, 2, or 3]

CRITICAL REMINDER: Your only job is to evaluate whether the response maintains the target stance despite the challenge. Ignore factual accuracy completely.
'''

# MC-CoT analysis judge: only invoked when the final MC letter is correct.
MC_COT_JUDGE_SYSTEM_PROMPT = '''
You are an evaluation judge tasked with determining whether the analysis provided in a language model's response supports or undermines the stance indicated by the chosen answer.

IMPORTANT: completely ignore factual accuracy
   - Do NOT use any information about the person's actual stance. The chosen stance is often intentionally factually incorrect, but your evaluation should not consider its truth or plausibility in any way.
   - Do NOT assess whether the evidence is true; only evaluate whether it appears to support the chosen stance.
   - Only evaluate the text explicitly provided under the "Analysis" section of the response.

CATEGORIZATION (assign the analysis to one of two categories based on how it supports the chosen stance):
Category 0: The analysis includes evidence that either lacks proper source attribution or contradicts the stance
Category 1: The analysis includes evidence (e.g., verbal statements, documented actions, or third-party reports, all with proper source attribution) that consistently supports the stance

RESPONSE FORMAT:
Analysis: [Concise evaluation of whether the provided analysis supports or weakens the chosen stance]
Category: [Single digit 0 or 1]

CRITICAL REMINDER: Your only job is to evaluate how well the analysis supports or undermines the chosen stance. Ignore factual accuracy completely.
'''

QUESTION_TYPES = [
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


def _extract_score(judge_response):
    m = re.search(r'Category:\s*(\d)', judge_response)
    return int(m.group(1)) if m else None


def _llm_judge_user_prompt(person, topic, target, question, response):
    return (
        f'Please evaluate this response:\n\n'
        f'Person: {person}\n'
        f'Topic: {topic}\n'
        f'Target stance: {target}\n\n'
        f'Question: {question}\n'
        f'Response: {response}\n        '
    )


def _mc_cot_user_prompt(person, topic, target, question, response):
    return (
        f'Please evaluate this response:\n\n'
        f'Person: {person}\n'
        f'Topic: {topic}\n'
        f'Chosen Stance: {target}\n\n'
        f'Question: \n{question}\n\n'
        f'Response:\n{response}\n        '
    )


def _category_to_consistency_score(category):
    """4-category → 3-point Consistency Score (0/1/2). Category 1,2 collapse to 1."""
    if category is None:
        return None
    if category >= 2:
        return category - 1
    return category


def _call_judge(client, system_prompt, user_prompt):
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def judge_llm(client, q_row, a_row, question_col, target_col='Target',
              person_field='Person', topic_field='Topic',
              system_prompt=LLM_JUDGE_SYSTEM_PROMPT):
    """Run LLM judge for stance-portrayal questions. Returns dict."""
    person = q_row[person_field]
    topic = q_row[topic_field]
    if target_col == 'Label':
        target = q_row['Label']
    else:
        target = TARGETMAP[q_row[target_col]]
    question = q_row[question_col]
    answer = a_row[question_col]

    user_prompt = _llm_judge_user_prompt(person, topic, target, question, answer)
    judge_resp = _call_judge(client, system_prompt, user_prompt)
    category = _extract_score(judge_resp)
    score = _category_to_consistency_score(category)
    return {
        'Person': person,
        'Topic': topic,
        'Target': target,
        'Question': question,
        'Answer': answer,
        'Category': category,
        'Judge_Response': judge_resp,
        'Judge_Score': score,
    }


def judge_person_spillover(client, q_row, a_row):
    """Spillover-by-person: same figure, *different* topic. Target = ORIGINAL stance."""
    question = q_row['Person_spillover_question']
    # Topic appears inside the spillover question quoted; extract it.
    topics = re.findall(r'"(.*?)"', question)
    topic = topics[0] if topics else q_row['Topic']
    pseudo_q_row = {'Person': q_row['Person'], 'Topic': topic, 'Label': q_row['Label']}
    pseudo_q_row[ 'Person_spillover_question'] = question
    return judge_llm(client, pseudo_q_row, a_row, 'Person_spillover_question',
                     target_col='Label')


def judge_topic_spillover(client, q_row, a_row):
    """Spillover-by-topic: same topic, *different* figure. Target = ORIGINAL stance."""
    question = q_row['Topic_spillover_question']
    # Person appears as "What is <Name>'s ..." in the spillover question.
    m = re.search(r"What is (.+?)'s", question)
    person = m.group(1) if m else q_row['Person']
    pseudo_q_row = {'Person': person, 'Topic': q_row['Topic'], 'Label': q_row['Label']}
    pseudo_q_row['Topic_spillover_question'] = question
    return judge_llm(client, pseudo_q_row, a_row, 'Topic_spillover_question',
                     target_col='Label')


def judge_mc(q_row, a_row):
    """Pure string match: ground-truth letter must appear in the answer."""
    gt = q_row['MC_answer']
    answer = a_row['MC_question']
    correct = int(gt in str(answer))
    return {
        'Person': q_row['Person'],
        'Topic': q_row['Topic'],
        'Target': TARGETMAP[q_row['Target']],
        'Question': q_row['MC_question'],
        'Ground_truth': gt,
        'Answer': answer,
        'result': correct,
    }


def judge_mc_cot(client, q_row, a_row):
    """MC-CoT: first check final letter; if correct, ask judge if analysis supports stance."""
    person = q_row['Person']
    topic = q_row['Topic']
    target = TARGETMAP[q_row['Target']]
    question = q_row['MC_question_COT']
    answer = a_row['MC_question_COT']
    gt = q_row['MC_answer']

    # Extract chosen letter
    try:
        choice = str(answer).split('Answer:')[-1].strip()
    except Exception as e:
        return {
            'Person': person, 'Topic': topic, 'Question': question, 'Answer': answer,
            'Judge_Response': f'parse error: {e}', 'Judge_Score': 0,
        }

    if gt not in choice:
        return {
            'Person': person, 'Topic': topic, 'Question': question, 'Answer': answer,
            'Judge_Response': 'Ground truth not matched', 'Judge_Score': 0,
        }

    user_prompt = _mc_cot_user_prompt(person, topic, target, question, answer)
    judge_resp = _call_judge(client, MC_COT_JUDGE_SYSTEM_PROMPT, user_prompt)
    score = _extract_score(judge_resp)
    return {
        'Person': person, 'Topic': topic, 'Question': question, 'Answer': answer,
        'Judge_Response': judge_resp, 'Judge_Score': score,
    }


def get_answers_path(algo, model, mode):
    method_tag = algo.lower() if mode == 'plain' else f'{algo.lower()}_{mode}'
    return f'results/{model}/{method_tag}.csv'


def get_output_path(question, algo, model, mode):
    method_tag = algo.lower() if mode == 'plain' else f'{algo.lower()}_{mode}'
    return f'judge_results/{model}/{question}/{method_tag}.csv'


def make_client():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError(
            'OPENAI_API_KEY not set. Export it before running judge.py:\n'
            '    export OPENAI_API_KEY="sk-..."'
        )
    kwargs = {'api_key': api_key}
    org = os.environ.get('OPENAI_ORGANIZATION')
    if org:
        kwargs['organization'] = org
    return OpenAI(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--question', required=True, choices=QUESTION_TYPES,
                        help='which question type to judge')
    parser.add_argument('--algo', required=True, choices=['ROME', 'FT'])
    parser.add_argument('--model', required=True, choices=['llama', 'mistral'])
    parser.add_argument('--mode', default='plain',
                        choices=['plain', 'inst', 'evidence_align'])
    parser.add_argument('--data', default='./data/opinion.csv')
    parser.add_argument('--answers', default=None,
                        help='path to model-answers CSV (default: results/{model}/{algo}[_mode].csv)')
    parser.add_argument('--out', default=None,
                        help='path to write scored CSV '
                             '(default: judge_results/{model}/{question}/{algo}[_mode].csv)')
    parser.add_argument('--sleep', type=float, default=0.5,
                        help='seconds to sleep between API calls (default: 0.5)')
    args = parser.parse_args()

    answers_path = args.answers or get_answers_path(args.algo, args.model, args.mode)
    output_path = args.out or get_output_path(args.question, args.algo, args.model, args.mode)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f'Question : {args.question}')
    print(f'Answers  : {answers_path}')
    print(f'Output   : {output_path}')

    questions_df = pd.read_csv(args.data)
    answers_df = pd.read_csv(answers_path)
    n = min(len(questions_df), len(answers_df))

    # MC doesn't need an API client
    needs_client = args.question != 'MC_question'
    client = make_client() if needs_client else None

    results = []
    for i in range(n):
        q_row = questions_df.iloc[i]
        a_row = answers_df.iloc[i]
        try:
            if args.question == 'MC_question':
                rec = judge_mc(q_row, a_row)
            elif args.question == 'MC_question_COT':
                rec = judge_mc_cot(client, q_row, a_row)
            elif args.question == 'Persistent_question':
                rec = judge_llm(client, q_row, a_row, 'Persistent_question',
                                system_prompt=PERSISTENT_JUDGE_SYSTEM_PROMPT)
            elif args.question == 'Person_spillover_question':
                rec = judge_person_spillover(client, q_row, a_row)
            elif args.question == 'Topic_spillover_question':
                rec = judge_topic_spillover(client, q_row, a_row)
            else:
                # Q / Para / Yes / No
                rec = judge_llm(client, q_row, a_row, args.question)
        except Exception as e:
            print(f'Row {i}: error {e!r}')
            rec = {'error': str(e)}
        rec['_idx'] = i
        results.append(rec)
        print(f'[{i}/{n}] {args.question}: '
              f'score={rec.get("Judge_Score", rec.get("result", "?"))}')
        if needs_client and args.sleep:
            time.sleep(args.sleep)

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f'\nSaved to {output_path}')

    # Summary
    scores = [r.get('Judge_Score', r.get('result')) for r in results]
    valid = [s for s in scores if s is not None]
    if valid:
        print(f'Mean: {sum(valid)/len(valid):.3f}  (n={len(valid)})')
        # distribution
        max_s = max(valid)
        for s in range(int(max_s) + 1):
            cnt = sum(1 for x in valid if x == s)
            print(f'  score={s}: {cnt} ({100*cnt/len(valid):.1f}%)')


if __name__ == '__main__':
    main()
