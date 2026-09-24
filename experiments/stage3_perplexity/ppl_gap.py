# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.46",
#   "datasets>=2.18",
#   "click>=8.1",
#   "wandb>=0.16",
# ]
# ///
"""Stage 3 -- perplexity gap and memory recovery on real data.

For the SAME held-out long sequences, compute active-window next-token loss
under three conditions:

    full   -- raw LLM, whole sequence           (upper bound; skipped if the
                                                   sequence exceeds the LLM's
                                                   positional budget)
    trunc  -- raw LLM, last max_window tokens   (lower bound = do nothing)
    bridge -- MemoryBridgeLLM                    (your method)

Headline metric:

    memory_recovery = (loss_trunc - loss_bridge) / (loss_trunc - loss_full)

        0  -> bridge is exactly as good as dropping history (worthless)
        1  -> bridge fully restores full-context performance

Also reported: per-length-bin breakdowns (memory cost grows with overflow),
compression stats, and a compression_slots ablation grid (the information
ceiling of the slots).

Prerequisite: a trained checkpoint (e.g. from simplememorybridgeexperiment.py
or stage2). Without one, --init-only measures the floor (random bridge).

Usage:
    # floor check with an untrained bridge
    uv run experiments/stage3_perplexity/ppl_gap.py --init-only --max-sequences 64

    # evaluate a trained checkpoint
    uv run experiments/stage3_perplexity/ppl_gap.py \
        --checkpoint ./memory-bridge-output/final --max-sequences 256

    # slots ablation (trains nothing; evaluates each config's checkpoint)
    uv run experiments/stage3_perplexity/ppl_gap.py \
        --ablation 64:ckpt-64,128:ckpt-128,256:ckpt-256
"""

import json
import math
import os
import sys

import click
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.common import (  # noqa: E402
    REPO_ROOT, DEFAULT_LLM, DEFAULT_ENCODER,
    load_models, resolve_device, mean_token_cross_entropy, raw_llm_logits,
)

sys.path.insert(0, REPO_ROOT)  # noqa: E402
from MemoryBridgeLLM import MemoryBridgeLLM, _max_position_embeddings  # noqa: E402
from simplememorybridgeexperiment import compute_loss  # noqa: E402


def load_sequences(llm_tok, dataset_name, dataset_split, dataset_config,
                   max_examples, min_len, max_len, seed):
    """Load chat data, apply the chat template, keep long sequences."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    def tokenize(examples):
        tokenized = llm_tok.apply_chat_template(examples['messages'], tokenize=True)
        if hasattr(tokenized, 'input_ids'):
            tokenized = tokenized['input_ids']
        return {'input_ids': [list(ids) for ids in tokenized]}

    ds = ds.map(tokenize, batched=True, batch_size=100,
                remove_columns=ds.column_names, desc='Tokenizing')
    seqs = [ids for ids in ds['input_ids'] if len(ids) >= min_len]
    seqs = [ids[:max_len] for ids in seqs]
    return seqs


@torch.no_grad()
def eval_sequence(model, llm, ids, llm_tok, enc_tok, max_window, device, skip_full=False):
    """Three-condition loss on ONE sequence. Returns dict of losses + meta."""
    t = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(t)
    seq_len = t.shape[1]
    trunc_offset = seq_len - max_window
    out = {}

    # ---- bridge (this defines the active window we score) ----
    b_out = model(t, llm_tok, enc_tok, attention_mask=mask)
    num_mem = model.num_memory_tokens(seq_len)
    b_loss, b_tok = compute_loss(b_out.logits, t, mask, num_mem)
    out['bridge'] = b_loss.item() if b_loss is not None else float('nan')
    out['n_tokens'] = b_tok
    out['num_mem'] = num_mem
    out['seq_len'] = seq_len

    # ---- truncated raw LLM: identical window ----
    t_logits = raw_llm_logits(llm, t[:, trunc_offset:], mask[:, trunc_offset:])
    t_loss, _ = mean_token_cross_entropy(t_logits, t[:, trunc_offset:], mask[:, trunc_offset:])
    out['trunc'] = t_loss.item() if t_loss is not None else float('nan')

    # ---- full-context raw LLM: score the same window positions ----
    budget = _max_position_embeddings(llm.config)
    if not skip_full and seq_len <= budget:
        f_logits = raw_llm_logits(llm, t, mask)
        w_logits = f_logits[:, trunc_offset:, :]
        f_loss, _ = mean_token_cross_entropy(w_logits, t[:, trunc_offset:], mask[:, trunc_offset:])
        out['full'] = f_loss.item() if f_loss is not None else float('nan')
    else:
        out['full'] = float('nan')
    return out


def length_bin(seq_len, max_window, compression_window):
    """Bin by how many compression chunks the overflow needs."""
    overflow = max(0, seq_len - max_window)
    chunks = math.ceil(overflow / compression_window) if overflow else 0
    return f'{chunks}chunk' if chunks < 4 else '4+chunk'


def summarize(rows):
    """Aggregate per-sequence rows into headline metrics."""
    def avg(key, rs):
        vals = [r[key] for r in rs if not math.isnan(r[key])]
        return sum(vals) / len(vals) if vals else float('nan')

    out = {}
    for cond in ('full', 'trunc', 'bridge'):
        out[f'{cond}_loss'] = avg(cond, rows)
        out[f'{cond}_ppl'] = math.exp(min(out[f'{cond}_loss'], 20)) if not math.isnan(out[f'{cond}_loss']) else float('nan')
    f, t, b = out['full_loss'], out['trunc_loss'], out['bridge_loss']
    out['memory_recovery'] = (t - b) / (t - f) if not (math.isnan(f) or math.isnan(t) or math.isnan(b) or abs(t - f) < 1e-9) else float('nan')
    out['n_sequences'] = len(rows)
    return out


@click.command()
@click.option('--llm-name', default=DEFAULT_LLM)
@click.option('--encoder-name', default=DEFAULT_ENCODER)
@click.option('--checkpoint', default=None, help='Trained MemoryBridge checkpoint dir')
@click.option('--max-window', type=int, default=4096)
@click.option('--compression-window', type=int, default=3072)
@click.option('--compression-slots', type=int, default=128)
@click.option('--max-levels', type=int, default=None)
@click.option('--dataset-name', default='aisquared/bolt-sft-final')
@click.option('--dataset-split', default='train')
@click.option('--dataset-config', default=None)
@click.option('--max-examples', type=int, default=20000)
@click.option('--max-sequences', type=int, default=128)
@click.option('--min-overflow-chunks', type=int, default=1,
              help='Only keep sequences with at least this much overflow')
@click.option('--max-seq-len', type=int, default=16384)
@click.option('--skip-full', is_flag=True, help='Skip the full-context upper bound')
@click.option('--init-only', is_flag=True, help='Evaluate a randomly-initialized bridge')
@click.option('--ablation', default=None,
              help='Comma list slots:ckpt_dir e.g. "64:ckpt64,128:ckpt128"')
@click.option('--output', default=None, help='JSON output path')
@click.option('--wandb-project', default=None)
@click.option('--wandb-run-name', default=None)
@click.option('--seed', type=int, default=0)
@click.option('--device', default='auto')
def main(llm_name, encoder_name, checkpoint, max_window, compression_window,
         compression_slots, max_levels, dataset_name, dataset_split, dataset_config,
         max_examples, max_sequences, min_overflow_chunks, max_seq_len,
         skip_full, init_only, ablation, output, wandb_project, wandb_run_name,
         seed, device):
    torch.manual_seed(seed)
    device = resolve_device(device)
    print(f'Device: {device}')

    llm, encoder, llm_tok, enc_tok, _ = load_models(llm_name, encoder_name, device)

    wandb_run = None
    if wandb_project:
        import wandb
        wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config={
            'stage': 3, 'max_window': max_window, 'compression_window': compression_window,
            'compression_slots': compression_slots, 'checkpoint': checkpoint,
        })

    # ---------------- data ----------------
    min_len = max_window + min_overflow_chunks * compression_window
    print(f'Loading sequences with {min_len} <= len <= {max_seq_len} ...')
    seqs = load_sequences(llm_tok, dataset_name, dataset_split, dataset_config,
                          max_examples, min_len, max_seq_len, seed)
    seqs = seqs[:max_sequences]
    print(f'  -> {len(seqs)} sequences')
    if not seqs:
        print('No sequences long enough; lower --min-overflow-chunks or --max-window.')
        return

    # ---------------- model(s) ----------------
    configs = []
    if ablation:
        for spec in ablation.split(','):
            slots, ckpt = spec.split(':', 1)
            configs.append({'compression_slots': int(slots), 'checkpoint': ckpt})
    else:
        configs.append({'compression_slots': compression_slots, 'checkpoint': checkpoint})

    all_results = {}
    for cfg in configs:
        print(f"\n=== config: slots={cfg['compression_slots']} ckpt={cfg['checkpoint']} ===")
        if cfg['checkpoint']:
            model = MemoryBridgeLLM.from_pretrained(
                cfg['checkpoint'], llm_model=llm, encoder_model=encoder,
            ).to(device)
        else:
            if not init_only:
                print('WARNING: no --checkpoint given; evaluating a RANDOM bridge. '
                      'Pass --init-only to silence this warning.')
            model = MemoryBridgeLLM(
                llm_model=llm, encoder_model=encoder,
                max_window=max_window,
                compression_window=compression_window,
                compression_slots=cfg['compression_slots'],
                max_levels=max_levels,
            ).to(device)
        model.eval()

        rows = []
        for i, ids in enumerate(seqs):
            row = eval_sequence(model, llm, ids, llm_tok, enc_tok,
                                model.max_window, device, skip_full=skip_full)
            row['bin'] = length_bin(row['seq_len'], model.max_window, model.compression_window)
            rows.append(row)
            if (i + 1) % 16 == 0:
                print(f'  {i + 1}/{len(seqs)} sequences')

        overall = summarize(rows)
        by_bin = {}
        bins = sorted({r['bin'] for r in rows})
        for b in bins:
            by_bin[b] = summarize([r for r in rows if r['bin'] == b])

        key = f"slots{cfg['compression_slots']}"
        all_results[key] = {'overall': overall, 'by_bin': by_bin,
                            'rows': rows if output else None}

        print(f'\n--- results ({key}) ---')
        print(f'  full   loss {overall["full_loss"]:.4f}  ppl {overall["full_ppl"]:.2f}')
        print(f'  trunc  loss {overall["trunc_loss"]:.4f}  ppl {overall["trunc_ppl"]:.2f}')
        print(f'  bridge loss {overall["bridge_loss"]:.4f}  ppl {overall["bridge_ppl"]:.2f}')
        print(f'  memory_recovery = {overall["memory_recovery"]:.3f}')
        print('  by overflow size:')
        for b in bins:
            s = by_bin[b]
            print(f'    {b:>8}: n={s["n_sequences"]:4d}  '
                  f'bridge={s["bridge_loss"]:.4f}  trunc={s["trunc_loss"]:.4f}  '
                  f'recovery={s["memory_recovery"]:.3f}')
        if wandb_run:
            flat = {f'{key}/{k}': v for k, v in overall.items()}
            wandb_run.log(flat)

    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)) or '.', exist_ok=True)
        with open(output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\nWrote {output}')
    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
