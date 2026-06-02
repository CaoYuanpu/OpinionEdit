from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib.style import context
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import repr_tools
from ...util import nethook

from .rome_hparams import ROMEHyperParams

def compute_v_calibrate(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
    idx: int,
    add_hidden: bool,
    topic_rel_pos: int
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("Computing right vector (v) and calibrate it")
    # Tokenize target into list of int token IDs
    # print(f"request['target_new']: {request['target_new']}")
    # input()
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]

    # if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
    #     target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    if 'mistral' in hparams.model_name:
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + ' ' + tok.decode(target_ids[:-1])
            for context in context_templates
        ], ["{} is a"]
    else:
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context in context_templates
        ], ["{} is a"]   
    
    
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")

    input_tok_1 = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts[:1]],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")
    
    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    rewriting_targets_1 = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        1, *input_tok_1["input_ids"].shape[1:]
    )

    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids
        if i == 0:
            ex_len_1 = input_tok_1["attention_mask"][i].sum()
            rewriting_targets_1[i, ex_len_1 - len(target_ids) : ex_len_1] = target_ids

    # Compute indices of the tokens where the fact is looked up
    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request['subject'])
        for context in context_templates
    ] + [f"{request['subject']} is a"]

    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0), input_prompt=vanilla_input_prompts[i]
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
                
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs)!=len(cur_out):
                    cur_out[idx, i, :] += delta
                else:
                    cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits
            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()
        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()


        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            print('Clamping delta')
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    print(f"nll_loss_each: {nll_loss_each[0]}")      
    # fine-tune the delta

    delta0 = delta.detach()  # 冻结第一阶段结果

    # 新建 delta₂
    delta2 = torch.zeros_like(delta0, requires_grad=True, device=delta.device)
    opt2 = torch.optim.Adam([delta2], lr=hparams.v_lr * 0.01)

    if "mistral" in hparams.model_name:
        hidden_layer = hparams.h_layer
        print("hidden_layer:", hidden_layer)
        saved = torch.load(f"./calibrate_targets_business_mistral/target_{idx}.pt")
        print(f"Loaded target signals from ./calibrate_targets_business_mistral/target_{idx}.pt")
    elif "llama" in hparams.model_name:
        hidden_layer = 15
        saved = torch.load(f"./calibrate_targets_business_llama/target_{idx}.pt")
        print(f"Loaded target signals from ./calibrate_targets_business_llama/target_{idx}.pt")
    # 载入目标 signals

    saved_tokens = saved["tokens"]
    h_targets = saved["target_hs_per_layer"][hidden_layer+1].to(delta.device)   # (span_len, hidden_dim)
    span_len = h_targets.size(0)
    rel_target_pos = [0] + list(range(topic_rel_pos, h_targets.shape[0]))
    print("#selected tokens:", len(rel_target_pos), rel_target_pos)
    print("Selected tokens:", [saved_tokens[p] for p in rel_target_pos])
    h_targets = h_targets[rel_target_pos, :]
    print("h_targets.shape:", h_targets.shape)#

    logits_targets = saved["target_logits"].to(delta.device)  

    # 2.2 准备同样的输入 tokens
    # if "mistral" in hparams.model_name:
    #     full = '<s>' + request["prompt"] + request["target_new"]
    # elif "llama" in hparams.model_name:
    #     full = '<|begin_of_text|>' + request["prompt"] + request["target_new"]

    full = request["prompt"] + request["target_new"]
    full = full.format(request["subject"])
    inputs_inst = tok(full, return_tensors="pt").to(delta.device)
    seq_len = inputs_inst["input_ids"].shape[-1]
    start_idx = seq_len - span_len
    abs_target_pos = [start_idx + p for p in rel_target_pos]
    print(f"Start index: {start_idx}, Span length: {span_len}")
    print(f"abs_target_pos: {abs_target_pos}")
    # 第二阶段的 edit hook
    def edit_output_fn2(cur_out, cur_layer):
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # 下面这行把 delta0 + delta2 一起注入
            for i, idx in enumerate([lookup_idxs[0]]):
                # 跟第一阶段同样的索引逻辑
                if len([lookup_idxs[0]]) != len(cur_out):
                    cur_out[idx, i, :] += (delta0 + delta2)
                else:
                    cur_out[i, idx, :] += (delta0 + delta2)
        return cur_out

    lambda_l = 1
    if add_hidden:
        lambda_h = 1000
        if "mistral" in hparams.model_name:
            step2 = 100
            lambda_nll = 1
        elif "llama" in hparams.model_name:
            step2 = 200
            lambda_h = 3000
            lambda_nll = 10
    else:
        lambda_h = 0
        if "mistral" in hparams.model_name:
            step2 = 30
        elif "llama" in hparams.model_name:
            step2 = 10
    for step in range(step2):
        opt2.zero_grad()

        with nethook.TraceDict(
            module=model,
            layers=[
                f'model.layers.{hidden_layer}',
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn2,
        ) as tr:
            lm_out = model(**inputs_inst)
            h_edited = tr[f'model.layers.{hidden_layer}'].output[0]
            h_edited = h_edited[0, abs_target_pos, :]
            logits_edited = lm_out.logits[0, start_idx:start_idx+span_len, :]
            logits = lm_out.logits[:, :-1, :]

        log_probs = torch.log_softmax(logits, dim=2)
        loss_orig = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets_1 != -100, rewriting_targets_1, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets_1 != -100).float()
        nll_loss_each = -(loss_orig * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()

        loss_h = torch.nn.functional.mse_loss(h_edited, h_targets)
        # 2) 只对最后两个 token 计算 KL 损失
        logits_edit_tail   = logits_edited[-2:, :]    # shape: (2, vocab_size)
        logits_target_tail = logits_targets[-2:, :]   # shape: (2, vocab_size)

        p_edit_tail   = torch.nn.functional.log_softmax(logits_edit_tail,   dim=-1)
        p_target_tail = torch.nn.functional.log_softmax(logits_target_tail, dim=-1)

        loss_l = torch.nn.functional.kl_div(
            p_edit_tail,
            p_target_tail,
            log_target=True,
            reduction="batchmean"
        )

        # 如果你还要加权合并：
        loss = lambda_h * loss_h + lambda_l * loss_l + lambda_nll * nll_loss
        # loss = lambda_h * loss_h
        print(f"Loss: {loss.item()} = {lambda_h} * {loss_h.item()} + {lambda_l} * {loss_l.item()} + {lambda_nll} * {nll_loss.item()}")
        # input("Press Enter to continue...")
        loss.backward()
        opt2.step()
    # Finish fine-tuning delta

    target = target_init + (delta0 + delta2).to(target_init.dtype)

    # target = target_init + delta.to(target_init.dtype)

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector

def compute_v(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("Computing right vector (v)")
    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]

    # if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
    #     target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    print(f"request['target_new']: {request['target_new']}")
    print(hparams.model_name)
    if 'mistral' in hparams.model_name:
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + ' ' + tok.decode(target_ids[:-1])
            for context in context_templates
        ], ["{} is a"]
    else:
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context in context_templates
        ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    vanilla_input_prompts = [
        context.format(request["prompt"]).format(request['subject'])
        for context in context_templates
    ] + [f"{request['subject']} is a"]
    
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0), input_prompt=vanilla_input_prompts[i]
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0]].detach().clone()
                
            for i, idx in enumerate(lookup_idxs):
                if len(lookup_idxs)!=len(cur_out):
                    cur_out[idx, i, :] += delta
                else:
                    cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        log_probs = torch.log_softmax(logits, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
    target = target_init + delta.to(target_init.dtype)

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model,
        tok,
        layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    # right_vector = (target - cur_output) / torch.dot(left_vector, left_vector)
    right_vector = (target - cur_output)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector


def get_module_input_output_at_word(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both",
            subtoken=subtoken,
            context_templates=[context_template],
            words=[word],
            **word_repr_args,
        )
    elif fact_token_strategy == "last":
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both",
            contexts=[context_template.format(word)],
            idxs=[[-1]],
            **word_repr_args,
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
    input_prompt=None
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = len(tok.encode(input_prompt)) - 1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
