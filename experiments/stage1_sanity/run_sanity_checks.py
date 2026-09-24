# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.46",
#   "click>=8.1",
# ]
# ///
"""Stage 1 -- sanity gates for MemoryBridgeLLM.

Cheap checks (minutes, one GPU/CPU, NO training) that must pass before any
learnability experiment is meaningful:

  T1  Identity: with compression untriggered, bridge loss == raw LLM loss
      exactly, on every batch.
  T2  Leak check: with a randomly-initialized bridge, a compressed batch's
      loss must be finite and must NOT be better than the raw full-context
      loss (an *improvement* means labels are leaking across the slice
      boundary).
  T3  Arithmetic/tensor agreement: num_memory_tokens(L) == actual
      compress_context output width, swept across the budget boundary.
  T4  Gradient flow: one packed training step produces non-zero, finite
      gradients in compressor, bridge, post_norm, and the recursive modules.
  T5  Boundary: sequence of exactly max_window tokens -> no compression;
      max_window+1 -> compression.
  T6  Train/eval: frozen LLM/encoder stay in eval mode under model.train().

Usage:
    uv run experiments/stage1_sanity/run_sanity_checks.py
    uv run experiments/stage1_sanity/run_sanity_checks.py \
        --llm-name aisquared/bolt-instruct-1b --max-window 512

Exit status 0 iff every test passes.
"""

import os
import sys

import click
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.common import (  # noqa: E402
    REPO_ROOT, DEFAULT_LLM, DEFAULT_ENCODER,
    load_models, build_memory_bridge, raw_llm_logits, mean_token_cross_entropy,
)

sys.path.insert(0, REPO_ROOT)  # noqa: E402
from simplememorybridgeexperiment import compute_loss  # noqa: E402


RESULTS = []


def report(name, passed, detail=''):
    RESULTS.append((name, passed, detail))
    mark = 'PASS' if passed else 'FAIL'
    print(f'  [{mark}] {name}' + (f' -- {detail}' if detail else ''))


def make_batch(llm_tokenizer, device, seq_len, batch_size=2, seed=0):
    """Random-token batch; fine for sanity (we compare conditions, not quality)."""
    g = torch.Generator().manual_seed(seed)
    vocab = llm_tokenizer.vocab_size
    ids = torch.randint(100, vocab - 100, (batch_size, seq_len), generator=g)
    mask = torch.ones_like(ids)
    return ids.to(device), mask.to(device)


@click.command()
@click.option('--llm-name', default=DEFAULT_LLM)
@click.option('--encoder-name', default=DEFAULT_ENCODER)
@click.option('--max-window', type=int, default=512)
@click.option('--compression-window', type=int, default=1024)
@click.option('--compression-slots', type=int, default=128)
@click.option('--device', default='auto')
def main(llm_name, encoder_name, max_window, compression_window, compression_slots, device):
    torch.manual_seed(0)
    llm, encoder, llm_tok, enc_tok, device = load_models(llm_name, encoder_name, device)
    print(f'Device: {device}')

    model = build_memory_bridge(
        llm, encoder,
        max_window=max_window,
        compression_window=compression_window,
        compression_slots=compression_slots,
    ).to(device)
    model.eval()

    # ------------------------------------------------------------------ T1
    print('\nT1: identity (no compression -> bridge == raw LLM)')
    ok, detail = True, []
    for L in (64, max_window // 2, max_window):
        ids, mask = make_batch(llm_tok, device, L)
        with torch.no_grad():
            bridge_logits = model(ids, llm_tok, enc_tok, attention_mask=mask).logits
            raw_logits = raw_llm_logits(llm, ids, mask)
        diff = (bridge_logits - raw_logits).abs().max().item()
        detail.append(f'L={L}: max|diff|={diff:.2e}')
        if diff > 1e-4:
            ok = False
    report('T1 identity', ok, '; '.join(detail))

    # ------------------------------------------------------------------ T2
    print('\nT2: leak check (compressed loss must not beat full-context loss)')
    L = max_window + compression_window  # guarantees exactly one chunk of overflow
    ids, mask = make_batch(llm_tok, device, L)
    with torch.no_grad():
        out = model(ids, llm_tok, enc_tok, attention_mask=mask)
        num_mem = model.num_memory_tokens(L)
        b_loss, b_tok = compute_loss(out.logits, ids, mask, num_mem)
        r_loss, _ = mean_token_cross_entropy(raw_llm_logits(llm, ids, mask), ids, mask)
    finite = torch.isfinite(b_loss)
    not_better = b_loss.item() >= r_loss.item() - 1e-4
    report(
        'T2 leak check',
        bool(finite and not_better),
        f'bridge={b_loss.item():.4f} ({b_tok} tokens) vs full-context raw={r_loss.item():.4f}; '
        f'num_mem={num_mem}',
    )

    # ------------------------------------------------------------------ T3
    print('\nT3: arithmetic vs tensor memory size')
    ok, bad = True, []
    for L in range(max_window - 2, max_window + 3 * compression_window + 2, 7):
        ids, mask = make_batch(llm_tok, device, L, batch_size=1)
        with torch.no_grad():
            mem = model.compress_context(ids, llm_tok, enc_tok, attention_mask=mask)
        predicted = model.num_memory_tokens(L)
        actual = 0 if mem is None else mem.shape[1]
        if predicted != actual:
            ok = False
            bad.append(f'L={L}: predicted={predicted} actual={actual}')
    report('T3 memory-size agreement', ok, '; '.join(bad) if bad else f'swept boundary region OK')

    # ------------------------------------------------------------------ T4
    print('\nT4: gradient flow on a packed (compressed) batch')
    model.train()
    ids, mask = make_batch(llm_tok, device, max_window + compression_window)
    out = model(ids, llm_tok, enc_tok, attention_mask=mask)
    num_mem = model.num_memory_tokens(ids.shape[1])
    loss, ntok = compute_loss(out.logits, ids, mask, num_mem)
    loss.backward()

    def grad_stats(module):
        total, count = 0.0, 0
        for p in module.parameters():
            if p.grad is not None:
                total += p.grad.abs().sum().item()
                count += 1
        return total, count

    # Level-1 modules are exercised on EVERY compressed batch and must
    # receive gradients. Recursion modules only fire when the memory prefix
    # exceeds the LLM's positional budget, so we only *require* their grads
    # when this configuration actually triggers recursion -- otherwise a zero
    # grad is correct behavior, not a failure.
    budget = model._memory_budget()  # super-tokens that fit alongside the window
    recursion_fires = num_mem > 0 and (model.max_levels != 1) and (
        num_mem > budget * 0.9
    )
    level1 = {
        'compressor': model.compressor,
        'bridge': model.bridge,
        'post_norm': model.post_norm,
    }
    recursion = {
        'recursion_compressor': model.recursion_compressor,
        'recursion_norm': model.recursion_norm,
        'recursion_bridge': model.recursion_bridge,
    }
    ok, detail = True, []
    for name, module in level1.items():
        gsum, gcount = grad_stats(module)
        detail.append(f'{name}: |g|={gsum:.3e}')
        if not (gsum > 0 and torch.isfinite(torch.tensor(gsum))):
            ok = False
    for name, module in recursion.items():
        gsum, gcount = grad_stats(module)
        tag = 'required' if recursion_fires else 'not exercised (ok)'
        detail.append(f'{name}: |g|={gsum:.3e} [{tag}]')
        if recursion_fires and not (gsum > 0 and torch.isfinite(torch.tensor(gsum))):
            ok = False
    detail.append(f'num_mem={num_mem}, budget={budget}, recursion_fires={recursion_fires}')
    # frozen modules must NOT receive gradients
    for name, module in {'llm': model.llm, 'encoder': model.encoder}.items():
        gsum, _ = grad_stats(module)
        if gsum != 0:
            ok = False
            detail.append(f'{name}: LEAKED grad |g|={gsum:.3e}')
    report('T4 gradient flow', ok, '; '.join(detail))
    model.zero_grad(set_to_none=True)
    model.eval()

    # ------------------------------------------------------------------ T5
    print('\nT5: compression boundary')
    ids_exact, mask_exact = make_batch(llm_tok, device, max_window, batch_size=1)
    ids_over, mask_over = make_batch(llm_tok, device, max_window + 1, batch_size=1)
    with torch.no_grad():
        mem_exact = model.compress_context(ids_exact, llm_tok, enc_tok, attention_mask=mask_exact)
        mem_over = model.compress_context(ids_over, llm_tok, enc_tok, attention_mask=mask_over)
    ok = (mem_exact is None) and (mem_over is not None) and (mem_over.shape[1] == compression_slots)
    n_exact = 0 if mem_exact is None else mem_exact.shape[1]
    n_over = 0 if mem_over is None else mem_over.shape[1]
    report(
        'T5 boundary',
        ok,
        f'L=max_window -> {n_exact} tokens; L=max_window+1 -> {n_over} tokens',
    )

    # ------------------------------------------------------------------ T6
    print('\nT6: frozen submodules stay in eval mode')
    model.train()
    ok = (not model.llm.training) and (not model.encoder.training) and model.compressor.training
    report('T6 train/eval modes', ok,
           f'llm.training={model.llm.training}, encoder.training={model.encoder.training}')

    # ------------------------------------------------------------------ summary
    print('\n' + '=' * 60)
    n_pass = sum(1 for _, p, _ in RESULTS if p)
    print(f'SANITY: {n_pass}/{len(RESULTS)} checks passed')
    for name, p, _ in RESULTS:
        if not p:
            print(f'  FAILED: {name}')
    raise SystemExit(0 if n_pass == len(RESULTS) else 1)


if __name__ == '__main__':
    main()
