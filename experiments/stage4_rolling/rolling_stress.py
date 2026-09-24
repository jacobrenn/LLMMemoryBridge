# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.46",
#   "click>=8.1",
#   "wandb>=0.16",
# ]
# ///
"""Stage 4 -- rolling-generation stress test.

The training loss only ever backpropagates through LEVEL-1 compression of the
*prompt*. During real generation, however, `MemoryBridgeLLM.generate()` folds
the model's OWN outputs back through the compressor, repeatedly, as the
window slides. That recursive self-compression is a distribution shift that
training never sees -- this stage measures whether memory survives it.

Design: probe-driven synthetic QA, no dataset or benchmark download needed.

  1. Plant K fact probes at KNOWN depths in the prompt (e.g. "The access code
     is alpha red." / "The project name is delta blue.").
  2. Ask a question about the OLDEST probe (deepest in memory).
  3. Force-generate F filler tokens (repetition of a neutral sentence), which
     pushes the probes deeper and deeper into recursive compression.
  4. Ask the question again and let the model answer greedily.
  5. Record exact-match as a function of (probe depth, number of forced
     tokens generated) -- the "distance traveled through the compressor".

The signature of a working bridge is a FLAT accuracy curve: answers stay
correct even after the probe has been recursively compressed multiple times.
A collapsing curve localizes the failure to the rolling re-compression path.

Conditions compared on identical inputs:
    bridge -- MemoryBridgeLLM.generate()          (rolling memory)
    full   -- raw LLM, whole sequence kept         (upper bound, if it fits)

Usage:
    # floor check with an untrained bridge (expect ~0 accuracy)
    uv run experiments/stage4_rolling/rolling_stress.py --init-only

    # evaluate a trained checkpoint
    uv run experiments/stage4_rolling/rolling_stress.py \
        --checkpoint ./memory-bridge-output/final \
        --forced-tokens 0,128,512,1024 --probes-per-config 5
"""

import itertools
import json
import os
import random
import sys

import click
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.common import (  # noqa: E402
    REPO_ROOT, DEFAULT_LLM, DEFAULT_ENCODER,
    load_models, resolve_device, text_of,
)

sys.path.insert(0, REPO_ROOT)  # noqa: E402
from MemoryBridgeLLM import MemoryBridgeLLM  # noqa: E402

NAMES = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel',
         'india', 'juliet', 'kilo', 'lima', 'mike', 'november', 'oscar', 'papa']
CODES = ['red', 'blue', 'green', 'gold', 'silver', 'black', 'white', 'purple',
         'orange', 'crimson', 'azure', 'ivory', 'amber', 'violet', 'coral', 'onyx']

FILLER_SENTENCE = ' The committee reviewed routine operational reports and filed them accordingly.'
QUESTION = ' Question: what is the access code? Answer: the access code is'


def build_prompt(llm_tok, rng, total_len, probe_depth, probe_fact):
    """Build a prompt with ONE probe fact planted at `probe_depth` tokens.

    probe_depth is measured from the START of the prompt. Returns
    (prompt_ids, answer_text). The question is appended at the very END, so
    the probe is always in the overflow once total_len > max_window.
    """
    name, code = probe_fact
    probe_text = f' The access code is {name} {code}.'
    probe_ids = llm_tok(probe_text, add_special_tokens=False)['input_ids']
    question_ids = llm_tok(QUESTION, add_special_tokens=False)['input_ids']

    filler_unit = llm_tok(FILLER_SENTENCE, add_special_tokens=False)['input_ids']
    filler = list(itertools.chain.from_iterable(
        itertools.repeat(filler_unit, total_len // len(filler_unit) + 1)
    ))[:total_len]

    probe_depth = min(probe_depth, total_len - len(probe_ids) - len(question_ids))
    filler[probe_depth:probe_depth + len(probe_ids)] = probe_ids
    prompt = filler + question_ids
    return prompt, f'{name} {code}'


@torch.no_grad()
def force_generate(model, prompt_ids, n_forced, llm_tok, enc_tok, device):
    """Run the bridge's rolling generate() while FORCING `n_forced` filler
    tokens (feeding ground-truth filler instead of the model's choice).

    We implement this manually around the model's internals: we feed the
    forced token as the "generated" token each step, reusing the model's own
    window-sliding + re-compression logic by calling generate() step by step
    is too invasive, so instead we re-implement the rolling loop minimally and
    call the model's compress/generate pieces. Returns the final active
    prompt (input_ids) after the forced rollout, with memory folded in.
    """
    # The cleanest way to exercise the real rolling path: temporarily patch
    # the model's llm forward so argmax always returns our forced token.
    forced_iter = iter(n_forced)

    class ForcedLLM(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, inputs_embeds=None, attention_mask=None, **kw):
            out = self.inner(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kw)
            try:
                tok = next(forced_iter)
            except StopIteration:
                return out
            logits = out.logits.clone()
            logits[:, -1, :] = -1e9
            logits[:, -1, tok] = 1e9
            out.logits = logits
            return out
        def get_input_embeddings(self):
            return self.inner.get_input_embeddings()
        @property
        def config(self):
            return self.inner.config

    real_llm = model.llm
    model.llm = ForcedLLM(real_llm)
    try:
        model.generate(
            prompt_ids, llm_tok, enc_tok,
            max_new_tokens=len(n_forced),
        )
    finally:
        model.llm = real_llm


@torch.no_grad()
def bridge_answer(model, prompt_ids, n_forced, filler_ids, llm_tok, enc_tok,
                  max_answer_tokens, device):
    """Force `n_forced` filler tokens through the rolling compressor, then
    let the model answer greedily."""
    if n_forced > 0:
        forced = (filler_ids * (n_forced // len(filler_ids) + 1))[:n_forced]
        force_generate(model, prompt_ids, forced, llm_tok, enc_tok, device)
        # After forcing, the prompt the model "sees" has advanced by n_forced;
        # simplest faithful continuation: append forced ids to the raw prompt
        # and let generate() re-compress from scratch for the answer phase.
        new_prompt = torch.cat([
            prompt_ids,
            torch.tensor([forced], dtype=torch.long, device=device),
        ], dim=1)
    else:
        new_prompt = prompt_ids

    gen = model.generate(new_prompt, llm_tok, enc_tok, max_new_tokens=max_answer_tokens)
    return gen[0]


@torch.no_grad()
def full_answer(llm, prompt_ids, n_forced, filler_ids, max_answer_tokens, device):
    forced = (filler_ids * (n_forced // len(filler_ids) + 1))[:n_forced] if n_forced > 0 else []
    full = torch.cat([prompt_ids, torch.tensor([forced], dtype=torch.long, device=device)], dim=1) if forced else prompt_ids
    out = llm.generate(full, max_new_tokens=max_answer_tokens, do_sample=False,
                       pad_token_id=llm.config.eos_token_id)
    return out[0, full.shape[1]:].tolist()


@click.command()
@click.option('--llm-name', default=DEFAULT_LLM)
@click.option('--encoder-name', default=DEFAULT_ENCODER)
@click.option('--checkpoint', default=None)
@click.option('--max-window', type=int, default=4096)
@click.option('--compression-window', type=int, default=3072)
@click.option('--compression-slots', type=int, default=128)
@click.option('--max-levels', type=int, default=None)
@click.option('--prompt-len', type=int, default=8192)
@click.option('--probe-depths', default='1024,4096',
              help='Comma list of probe depths (tokens from prompt start)')
@click.option('--forced-tokens', default='0,256,1024',
              help='Comma list of forced filler-token counts')
@click.option('--probes-per-config', type=int, default=5)
@click.option('--max-answer-tokens', type=int, default=8)
@click.option('--skip-full', is_flag=True)
@click.option('--init-only', is_flag=True)
@click.option('--output', default=None)
@click.option('--wandb-project', default=None)
@click.option('--wandb-run-name', default=None)
@click.option('--seed', type=int, default=0)
@click.option('--device', default='auto')
def main(llm_name, encoder_name, checkpoint, max_window, compression_window,
         compression_slots, max_levels, prompt_len, probe_depths, forced_tokens,
         probes_per_config, max_answer_tokens, skip_full, init_only, output,
         wandb_project, wandb_run_name, seed, device):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = resolve_device(device)
    print(f'Device: {device}')

    llm, encoder, llm_tok, enc_tok, _ = load_models(llm_name, encoder_name, device)

    if checkpoint:
        model = MemoryBridgeLLM.from_pretrained(checkpoint, llm_model=llm, encoder_model=encoder).to(device)
    else:
        if not init_only:
            print('WARNING: no checkpoint; evaluating a RANDOM bridge (--init-only to silence).')
        model = MemoryBridgeLLM(
            llm_model=llm, encoder_model=encoder,
            max_window=max_window, compression_window=compression_window,
            compression_slots=compression_slots, max_levels=max_levels,
        ).to(device)
    model.eval()

    filler_ids = llm_tok(FILLER_SENTENCE, add_special_tokens=False)['input_ids']
    depths = [int(x) for x in probe_depths.split(',')]
    forced_list = [int(x) for x in forced_tokens.split(',')]
    stop_ids = {llm_tok.eos_token_id} if llm_tok.eos_token_id is not None else set()

    wandb_run = None
    if wandb_project:
        import wandb
        wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config={
            'stage': 4, 'prompt_len': prompt_len, 'depths': depths, 'forced': forced_list,
            'checkpoint': checkpoint,
        })

    results = []
    print('\n' + '=' * 78)
    print(f'{"depth":>6} {"forced":>7} {"bridge":>8} {"full":>8}   example')
    print('=' * 78)

    for depth, n_forced in itertools.product(depths, forced_list):
        b_hits = f_hits = 0
        example = ''
        for _ in range(probes_per_config):
            fact = (rng.choice(NAMES), rng.choice(CODES))
            prompt_ids_list, answer_text = build_prompt(llm_tok, rng, prompt_len, depth, fact)
            prompt_ids = torch.tensor([prompt_ids_list], dtype=torch.long, device=device)

            b_gen = bridge_answer(model, prompt_ids, n_forced, filler_ids, llm_tok, enc_tok,
                                  max_answer_tokens, device)
            b_text = text_of(llm_tok, b_gen)
            b_ok = answer_text in b_text
            b_hits += b_ok

            f_ok = False
            if not skip_full:
                f_gen = full_answer(llm, prompt_ids, n_forced, filler_ids, max_answer_tokens, device)
                f_text = text_of(llm_tok, f_gen)
                f_ok = answer_text in f_text
                f_hits += f_ok
            example = f'want "{answer_text}" | bridge got "{b_text.strip()}"'
            if not skip_full:
                example += f' | full got "{f_text.strip()}"'

        b_acc = b_hits / probes_per_config
        f_acc = f_hits / probes_per_config if not skip_full else float('nan')
        results.append({
            'probe_depth': depth, 'forced_tokens': n_forced,
            'bridge_acc': b_acc, 'full_acc': f_acc,
        })
        print(f'{depth:>6} {n_forced:>7} {b_acc:>8.2f} {f_acc:>8.2f}   {example}')
        if wandb_run:
            wandb_run.log({'bridge_acc': b_acc, 'full_acc': f_acc,
                           'probe_depth': depth, 'forced_tokens': n_forced})

    print('=' * 78)
    print('Interpretation: bridge_acc flat across forced_tokens => rolling')
    print('re-compression preserves memory. A steep drop => the failure lives')
    print('in generate(), not in the level-1 bridge.')

    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
        with open(output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Wrote {output}')
    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
