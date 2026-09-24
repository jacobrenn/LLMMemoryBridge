# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.46",
#   "click>=8.1",
# ]
# ///
"""Shared utilities for the MemoryBridge evaluation experiments.

Every stage imports from here so model loading, device resolution, and the
raw-LLM forward pass live in exactly one place. To swap the models used by
ALL experiments, edit the defaults below or pass --llm-name/--encoder-name
to the individual scripts.

NOTE: the stage scripts insert the *repository root* on sys.path so they can
`from MemoryBridgeLLM import MemoryBridgeLLM` (the single source of truth for
the architecture -- the trainer script contains an identical copy).
"""

import os
import sys

import torch

# experiments/shared/common.py -> repo root is two directories up
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Defaults match the training script; override per-run with CLI flags.
DEFAULT_LLM = 'aisquared/bolt-instruct-1b'
DEFAULT_ENCODER = 'aisquared/bolt-embedding-small'


def resolve_device(name='auto'):
    """'auto' picks cuda > mps > cpu."""
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_models(llm_name=DEFAULT_LLM, encoder_name=DEFAULT_ENCODER, device='auto'):
    """Load (llm, encoder, llm_tokenizer, encoder_tokenizer) onto `device`."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(device)
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    llm = AutoModelForCausalLM.from_pretrained(llm_name)
    enc_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = AutoModel.from_pretrained(encoder_name)
    llm.to(device).eval()
    encoder.to(device).eval()
    return llm, encoder, llm_tokenizer, enc_tokenizer, device


def build_memory_bridge(
    llm,
    encoder,
    max_window=512,
    compression_window=1024,
    compression_slots=128,
    max_levels=None,
    llm_trainable=False,
    encoder_trainable=False,
):
    """Construct a MemoryBridgeLLM with experiment-friendly defaults."""
    from MemoryBridgeLLM import MemoryBridgeLLM

    return MemoryBridgeLLM(
        llm_model=llm,
        encoder_model=encoder,
        max_window=max_window,
        compression_window=compression_window,
        compression_slots=compression_slots,
        max_levels=max_levels,
        llm_trainable=llm_trainable,
        encoder_trainable=encoder_trainable,
    )


@torch.no_grad()
def raw_llm_logits(llm, input_ids, attention_mask=None):
    """Forward pass through the plain LLM (no memory bridge)."""
    return llm(input_ids=input_ids, attention_mask=attention_mask).logits


def mean_token_cross_entropy(logits, labels, attention_mask):
    """Standard next-token CE with attention masking.

    logits:   [B, T, V] aligned with labels [B, T]
    Returns (mean_loss, num_scored_tokens).
    """
    import torch.nn.functional as F

    if logits.shape[1] < 2:
        return None, 0
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = attention_mask[:, 1:]

    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction='none',
    ).view(shift_labels.shape)
    losses = losses * shift_mask
    n = shift_mask.sum()
    if n.item() == 0:
        return None, 0
    return losses.sum() / n, int(n.item())


def truncated_window_loss(llm, input_ids, attention_mask, max_window):
    """Loss of the raw LLM on only the LAST `max_window` tokens.

    This is the 'do nothing' sliding-window baseline: everything older than
    the window is simply dropped, exactly as if the context never existed.
    """
    window_ids = input_ids[:, -max_window:]
    window_mask = attention_mask[:, -max_window:]
    logits = raw_llm_logits(llm, window_ids, window_mask)
    return mean_token_cross_entropy(logits, window_ids, window_mask)


def bridge_loss(model, input_ids, attention_mask, llm_tokenizer, enc_tokenizer):
    """Active-window loss through the trained memory bridge.

    Mirrors the trainer's compute_loss: super-token logit positions are
    sliced off before aligning with the (trimmed) window ids.
    """
    from simplememorybridgeexperiment import compute_loss

    seq_len = input_ids.shape[1]
    num_mem = model.num_memory_tokens(seq_len)
    out = model(input_ids, llm_tokenizer, enc_tokenizer, attention_mask=attention_mask)
    return compute_loss(out.logits, input_ids, attention_mask, num_mem)


def generate_plain(llm, tokenizer, input_ids, max_new_tokens=32, stop_token_ids=None):
    """Greedy generation with the raw LLM (used for baselines).

    Returns the list of newly generated token ids.
    """
    out = llm.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=list(stop_token_ids) if stop_token_ids else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return out[0, input_ids.shape[1]:].tolist()


def text_of(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False)
