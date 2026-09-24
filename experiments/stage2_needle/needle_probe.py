# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.46",
#   "click>=8.1",
#   "wandb>=0.16",
# ]
# ///
"""Stage 2 -- needle-in-a-haystack learnability probe for MemoryBridgeLLM.

Question: can the compressor + bridge learn to carry a specific fact from the
overflow region into the LLM's predictions, when ONLY the bridge is trained?

Task (fully synthetic, teacher-forced, evaluated in LLM token space):
    [filler ...] The access code is XJ-47. [filler ...] | The access code is XJ-47.
    |------------ overflow (compressed) ----------------| |--- active window ---|

The needle sentence appears ONCE in the overflow at a controlled depth. The
active window ends with the query, so the *only* way to predict the needle
tokens at the window boundary is via the super-token memory.

Three conditions are scored on identical examples:
    full     -- raw LLM sees everything (upper bound)
    trunc    -- raw LLM sees only the active window (lower bound = do nothing)
    bridge   -- MemoryBridgeLLM (your method)

Success criterion: bridge exact-match >> trunc exact-match, and bridge loss
approaches full loss. Run --init-only to verify the untrained bridge sits at
the truncation floor.

Usage:
    # verify floors before training (bridge == random init)
    uv run experiments/stage2_needle/needle_probe.py --init-only

    # train the bridge on needles and evaluate by depth bucket
    uv run experiments/stage2_needle/needle_probe.py \
        --steps 2000 --batch-size 8 --lr 1e-4 --wandb-project memory-bridge
"""

import json
import math
import os
import random
import sys
import time

import click
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.common import (  # noqa: E402
    REPO_ROOT, DEFAULT_LLM, DEFAULT_ENCODER,
    load_models, resolve_device, mean_token_cross_entropy,
)

sys.path.insert(0, REPO_ROOT)  # noqa: E402
from MemoryBridgeLLM import MemoryBridgeLLM  # noqa: E402
from simplememorybridgeexperiment import compute_loss  # noqa: E402

# Vocabulary for needle sentences. Pairs are distinct so exact match is
# unambiguous; multi-word values stress the memory with more than one fact.
NAMES = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel',
         'india', 'juliet', 'kilo', 'lima', 'mike', 'november', 'oscar', 'papa']
CODES = ['red', 'blue', 'green', 'gold', 'silver', 'black', 'white', 'purple',
         'orange', 'crimson', 'azure', 'ivory', 'amber', 'violet', 'coral', 'onyx']

QUERY_TEMPLATE = ' The access code is'


def build_example(llm_tok, max_window, rng, needle_depth=None, filler_ids=None):
    """Build one (ids, needle_positions) example.

    needle_depth: token offset (from the START of the sequence) of the needle
    sentence. None -> sampled uniformly from the overflow region. The needle
    is guaranteed to be entirely inside the overflow (never in the window).

    filler_ids: a long token list to draw background text from. If None, uses
    repeated neutral sentences.

    Returns dict(ids=list[int], needle_positions=list[int]) where positions
    index the answer tokens inside the active window.
    """
    name, code = rng.choice(NAMES), rng.choice(CODES)
    needle_text = f' The access code is {name} {code}.'
    needle_ids = llm_tok(needle_text, add_special_tokens=False)['input_ids']
    query_ids = llm_tok(QUERY_TEMPLATE, add_special_tokens=False)['input_ids']
    answer_ids = llm_tok(f' {name} {code}', add_special_tokens=False)['input_ids']

    overflow_len_target = max_window  # at least max_window tokens of overflow
    window_len = len(query_ids) + len(answer_ids)
    if window_len > max_window:
        raise ValueError(f'max_window={max_window} too small for query+answer ({window_len})')

    if needle_depth is None:
        needle_depth = rng.randrange(0, overflow_len_target - len(needle_ids))
    needle_depth = min(needle_depth, overflow_len_target - len(needle_ids))

    total_len = needle_depth + len(needle_ids) + max_window
    # Fill with neutral background
    if filler_ids is None:
        filler_text = ' The committee discussed routine administrative matters.'
        unit = llm_tok(filler_text, add_special_tokens=False)['input_ids']
        filler = (unit * (total_len // len(unit) + 1))[:total_len]
    else:
        start = rng.randrange(0, max(1, len(filler_ids) - total_len))
        filler = filler_ids[start:start + total_len]
        if len(filler) < total_len:
            filler = filler + [llm_tok.eos_token_id or 0] * (total_len - len(filler))

    ids = list(filler)
    ids[needle_depth:needle_depth + len(needle_ids)] = needle_ids

    # Active window = [filler tail ...][query][answer]
    window_start = needle_depth + len(needle_ids)
    tail_len = max_window - window_len
    ids = ids[:window_start + tail_len] + query_ids + answer_ids

    # answer token positions inside the final active window
    answer_positions = list(range(max_window - len(answer_ids), max_window))
    return {
        'ids': ids,
        'answer_positions': answer_positions,
        'needle': f'{name} {code}',
        'needle_depth_tokens': needle_depth,
        'needle_chunk': needle_depth,  # caller maps to chunk index given compression_window
    }


def collate(examples, pad_id, device):
    maxlen = max(len(e['ids']) for e in examples)
    ids = torch.full((len(examples), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, e in enumerate(examples):
        t = torch.tensor(e['ids'], dtype=torch.long)
        ids[i, :len(t)] = t
        mask[i, :len(t)] = 1
    return ids.to(device), mask.to(device)


def pad_answer_index(examples, device):
    """Pad per-example answer_positions into a rectangular index + mask.

    Different name/code answers tokenize to different lengths, so the raw
    answer_positions lists are ragged. We right-pad with position 0 (masked
    out) and return (index [B, Amax], mask [B, Amax]).
    """
    amax = max(len(e['answer_positions']) for e in examples)
    idx = torch.zeros(len(examples), amax, dtype=torch.long)
    m = torch.zeros(len(examples), amax, dtype=torch.bool)
    for i, e in enumerate(examples):
        pos = e['answer_positions']
        idx[i, :len(pos)] = torch.tensor(pos, dtype=torch.long)
        m[i, :len(pos)] = True
    return idx.to(device), m.to(device)


def score_conditions(model, llm, batch, answer_index, answer_mask, llm_tok, enc_tok, max_window):
    """Compute (loss_on_answer, exact_match) for full / trunc / bridge.

    answer_index: LongTensor [B, A] of absolute positions of answer tokens in
    the full sequence (padded; see answer_mask). answer_mask: BoolTensor [B, A]
    marking real (non-padded) answer slots. All conditions are scored on the
    SAME answer tokens.
    """
    ids, mask = batch
    B, A = answer_index.shape
    device = ids.device
    gather_index = (answer_index - 1).unsqueeze(-1)  # logits at t-1 predict token t

    def answer_stats(logits, positions_offset=0):
        # logits: [B, T, V]; positions_offset shifts where answers live
        idx = (gather_index - positions_offset).clamp(min=0)
        picked = logits.gather(1, idx.expand(-1, -1, logits.size(-1)))
        target = ids.gather(1, answer_index)
        logp = torch.log_softmax(picked.float(), dim=-1)
        tok_logp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B, A]
        pred = picked.argmax(dim=-1)
        # exact match only over real answer tokens (mask out padding)
        correct = (pred == target) | ~answer_mask
        exact = correct.all(dim=1).float()
        # mean per-token NLL over real answer tokens only
        tok_nll = (-tok_logp) * answer_mask
        denom = answer_mask.sum(dim=1).clamp(min=1)
        return tok_nll.sum(dim=1) / denom, exact

    with torch.no_grad():
        # full context
        full_logits = llm(input_ids=ids, attention_mask=mask).logits
        full_loss, full_exact = answer_stats(full_logits)

        # truncated: last max_window tokens only
        trunc_ids, trunc_mask = ids[:, -max_window:], mask[:, -max_window:]
        trunc_logits = llm(input_ids=trunc_ids, attention_mask=trunc_mask).logits
        trunc_offset = ids.shape[1] - max_window
        trunc_loss, trunc_exact = answer_stats(trunc_logits, positions_offset=trunc_offset)

        # bridge
        out = model(ids, llm_tok, enc_tok, attention_mask=mask)
        num_mem = model.num_memory_tokens(ids.shape[1])
        # answers live at the window tail in the bridged logits:
        # bridged logits = [mem..., window...]; absolute answer pos maps to
        # num_mem + (pos - trunc_offset)
        bridge_logits = out.logits
        bridge_offset = trunc_offset - num_mem
        bridge_loss, bridge_exact = answer_stats(bridge_logits, positions_offset=bridge_offset)

    return {
        'full': (full_loss, full_exact),
        'trunc': (trunc_loss, trunc_exact),
        'bridge': (bridge_loss, bridge_exact),
    }


def bucket_of(depth, total_overflow, n_buckets):
    if total_overflow <= 0:
        return 0
    return min(n_buckets - 1, int(depth / total_overflow * n_buckets))


@click.command()
@click.option('--llm-name', default=DEFAULT_LLM)
@click.option('--encoder-name', default=DEFAULT_ENCODER)
@click.option('--max-window', type=int, default=512)
@click.option('--compression-window', type=int, default=1024)
@click.option('--compression-slots', type=int, default=128)
@click.option('--max-levels', type=int, default=None)
@click.option('--steps', type=int, default=2000)
@click.option('--batch-size', type=int, default=8)
@click.option('--lr', type=float, default=1e-4)
@click.option('--warmup', type=int, default=100)
@click.option('--eval-every', type=int, default=200)
@click.option('--eval-batches', type=int, default=20)
@click.option('--n-buckets', type=int, default=4, help='Depth buckets for eval')
@click.option('--seed', type=int, default=0)
@click.option('--init-only', is_flag=True, help='Skip training; evaluate the random-init bridge')
@click.option('--output-dir', default=None)
@click.option('--wandb-project', default=None)
@click.option('--wandb-run-name', default=None)
@click.option('--device', default='auto')
def main(llm_name, encoder_name, max_window, compression_window, compression_slots,
         max_levels, steps, batch_size, lr, warmup, eval_every, eval_batches,
         n_buckets, seed, init_only, output_dir, wandb_project, wandb_run_name, device):
    torch.manual_seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    device = resolve_device(device)
    print(f'Device: {device}')

    llm, encoder, llm_tok, enc_tok, _ = load_models(llm_name, encoder_name, device)
    model = MemoryBridgeLLM(
        llm_model=llm, encoder_model=encoder,
        max_window=max_window,
        compression_window=compression_window,
        compression_slots=compression_slots,
        max_levels=max_levels,
    ).to(device)

    pad_id = llm_tok.pad_token_id if llm_tok.pad_token_id is not None else llm_tok.eos_token_id

    wandb_run = None
    if wandb_project:
        import wandb
        wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config={
            'stage': 2, 'max_window': max_window, 'compression_window': compression_window,
            'compression_slots': compression_slots, 'max_levels': max_levels,
            'steps': steps, 'batch_size': batch_size, 'lr': lr, 'seed': seed,
        })

    def evaluate(step):
        model.eval()
        agg = {}
        counts = {}
        with torch.no_grad():
            for _ in range(eval_batches):
                # per-bucket eval: force each example's needle depth into its
                # bucket by building one example per slot at the desired depth.
                overflow_region = max_window  # build_example's overflow target
                examples = []
                for i in range(batch_size):
                    b = i % n_buckets
                    lo = int(b / n_buckets * overflow_region)
                    hi = int((b + 1) / n_buckets * overflow_region) - 8
                    depth = rng.randrange(lo, hi) if hi > lo else None
                    examples.append(build_example(llm_tok, max_window, rng, needle_depth=depth))
                batch = collate(examples, pad_id, device)
                ans_index, ans_mask = pad_answer_index(examples, device)
                res = score_conditions(model, llm, batch, ans_index, ans_mask, llm_tok, enc_tok, max_window)
                for i, e in enumerate(examples):
                    b = bucket_of(e['needle_depth_tokens'], len(e['ids']) - max_window, n_buckets)
                    key = f'bucket{b}'
                    for cond in ('full', 'trunc', 'bridge'):
                        loss_v = res[cond][0][i].item()
                        exact_v = res[cond][1][i].item()
                        agg.setdefault((key, cond), [0.0, 0.0, 0])
                        agg[(key, cond)][0] += loss_v
                        agg[(key, cond)][1] += exact_v
                        agg[(key, cond)][2] += 1
                        agg.setdefault(('all', cond), [0.0, 0.0, 0])
                        agg[('all', cond)][0] += loss_v
                        agg[('all', cond)][1] += exact_v
                        agg[('all', cond)][2] += 1

        metrics = {}
        print(f'\n=== eval @ step {step} ===')
        print(f'{"bucket":>8} {"cond":>7} {"loss":>8} {"exact":>7} {"n":>5}')
        for (bkey, cond), (ls, ex, n) in sorted(agg.items()):
            metrics[f'eval/{bkey}/{cond}_loss'] = ls / n
            metrics[f'eval/{bkey}/{cond}_exact'] = ex / n
            if bkey in ('all', 'bucket0', f'bucket{n_buckets - 1}'):
                print(f'{bkey:>8} {cond:>7} {ls / n:8.3f} {ex / n:7.3f} {n:5d}')
        # headline numbers
        f = agg[('all', 'full')][0] / agg[('all', 'full')][2]
        t = agg[('all', 'trunc')][0] / agg[('all', 'trunc')][2]
        b = agg[('all', 'bridge')][0] / agg[('all', 'bridge')][2]
        rec = (t - b) / max(t - f, 1e-6)
        metrics['eval/memory_recovery'] = rec
        print(f'memory_recovery (loss): {rec:.3f}  (0=trunc floor, 1=full ceiling)')
        if wandb_run:
            wandb_run.log(metrics, step=step)
        model.train()
        return metrics

    if init_only:
        evaluate(0)
        if wandb_run:
            wandb_run.finish()
        return

    # ---------------- training ----------------
    # Objective: next-token CE restricted to the ANSWER tokens. We could use
    # the trainer's compute_loss (whole active window), but the window is
    # dominated by filler the model already predicts trivially -- restricting
    # to the answer positions focuses all gradient on the memory path.
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f'Trainable params: {sum(p.numel() for p in trainable):,}')
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * (1 - s / max(steps, 1))
    )

    def answer_loss(out_logits, ids, answer_index, answer_mask, num_mem, seq_len):
        trunc_offset = seq_len - max_window
        idx = ((answer_index - 1) - (trunc_offset - num_mem)).clamp(min=0)  # pos in bridged logits
        picked = out_logits.gather(1, idx.unsqueeze(-1).expand(-1, -1, out_logits.size(-1)))
        target = ids.gather(1, answer_index)
        nll = torch.nn.functional.cross_entropy(
            picked.reshape(-1, picked.size(-1)).float(), target.reshape(-1), reduction='none'
        ).view(answer_index.shape)
        nll = nll * answer_mask
        return nll.sum() / answer_mask.sum().clamp(min=1)

    output_dir = output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    for step in range(1, steps + 1):
        examples = [build_example(llm_tok, max_window, rng) for _ in range(batch_size)]
        batch = collate(examples, pad_id, device)
        ans_index, ans_mask = pad_answer_index(examples, device)
        out = model(batch[0], llm_tok, enc_tok, attention_mask=batch[1])
        num_mem = model.num_memory_tokens(batch[0].shape[1])
        loss = answer_loss(out.logits, batch[0], ans_index, ans_mask, num_mem, batch[0].shape[1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        if step % 50 == 0:
            msg = f'step {step:>6}/{steps} | answer loss {loss.item():.4f} | {step / (time.time() - t0):.2f} it/s'
            print(msg)
            if wandb_run:
                wandb_run.log({'train/answer_loss': loss.item()}, step=step)
        if step % eval_every == 0:
            evaluate(step)

    final_metrics = evaluate(steps)
    ckpt = os.path.join(output_dir, 'final')
    model.save_pretrained(ckpt)
    llm_tok.save_pretrained(ckpt)
    enc_tok.save_pretrained(ckpt)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=2)
    print(f'\nSaved checkpoint to {ckpt}')
    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
