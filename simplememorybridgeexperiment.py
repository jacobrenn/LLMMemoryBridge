# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.46",
#   "datasets>=2.18",
#   "click>=8.1",
#   "wandb>=0.16",
#   "huggingface_hub>=0.24",
#   "accelerate>=0.30",
# ]
# ///
"""Train the MemoryBridgeLLM compressor + bridge on long chat data.

Standalone `uv` script:

    uv run simplememorybridgeexperiment.py \
        --llm-name aisquared/bolt-instruct-1b \
        --encoder-name aisquared/bolt-embedding-small \
        --dataset-name aisquared/bolt-sft-final \
        --max-window 1024 --pack-length 4096 \
        --wandb-project my-experiment --wandb-run-name run-01 \
        --hf-repo-id aisquared/bolt-memory-compressed-1b

By default both the LLM and the encoder are frozen; only the latent
compressor and the bridge layer receive gradients. Use --train-encoder and
--train-llm to unfreeze them.
"""

import json
import math
import os
import random
import sys
import time

import click
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


"""MemoryBridgeLLM: compress overflowing LLM context into learned "super-tokens".

Architecture:
    - An encoder model produces hidden states for the *overflow* region of the
      context (the part that no longer fits in the LLM's window).
    - A Perceiver-style LatentCompressor cross-attends a small set of learned
      latent slots over the encoder hidden states.
    - A linear bridge projects the resulting latents into the LLM's embedding
      space. These "super-tokens" are prepended to the embedded active window.

The module exposes `forward` (returns logits over the super-token + active
window region), `generate` (real autoregressive generation that reuses the
compressed memory), and `save_pretrained` / `from_pretrained` so full
checkpoints can be saved locally and pushed to the Hugging Face Hub.
"""

import json
import math
import os
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _max_position_embeddings(config, default=512):
    """Robustly read `max_position_embeddings` from a HF config.

    Fixes the dead `... if config.max_position_embeddings else 512` pattern,
    which raises AttributeError when the attribute is missing entirely.
    """
    value = getattr(config, 'max_position_embeddings', None)
    return value if value else default


class LatentCompressor(nn.Module):
    def __init__(
        self,
        enc_dim,
        num_slots = 128,
        num_heads = 8,
        num_layers = 2
    ):
        super().__init__()
        self.latents = nn.Parameter(
            torch.randn(
                num_slots,
                enc_dim
            ) * 0.02
        )

        layer = nn.TransformerDecoderLayer(
            d_model = enc_dim,
            nhead = num_heads,
            dim_feedforward = enc_dim * 4,
            batch_first = True,
            norm_first = True
        )
        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers = num_layers
        )

    def forward(self, memory, memory_key_padding_mask = None):
        B = memory.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(
            tgt = latents,
            memory = memory,
            memory_key_padding_mask = memory_key_padding_mask
        )
        return out


@dataclass
class MemoryBridgeConfig:
    llm_name_or_path: str = None
    encoder_name_or_path: str = None
    max_window: int = 512
    compression_window: int = 8192
    compression_slots: int = 128
    compression_n_heads: int = 8
    compression_n_layers: int = 2
    llm_trainable: bool = False
    encoder_trainable: bool = False


class MemoryBridgeLLM(nn.Module):

    CONFIG_FILE = 'memory_bridge_config.json'

    def __init__(
        self,
        llm_model,
        encoder_model,
        max_window = None,
        compression_window = 8192,
        compression_slots = 128,
        compression_n_heads = 8,
        compression_n_layers = 2,
        llm_trainable = False,
        encoder_trainable = False
    ):
        super().__init__()

        # The LLM
        self.llm = llm_model

        # The Encoder
        self.encoder = encoder_model

        self.llm_trainable = llm_trainable
        self.encoder_trainable = encoder_trainable

        # Set the number of tokens to be compressed
        llm_max_window = _max_position_embeddings(self.llm.config)
        encoder_max_window = _max_position_embeddings(self.encoder.config)
        max_window_allowed = min([llm_max_window, encoder_max_window])

        if max_window is not None and max_window <= max_window_allowed:
            self.max_window = max_window
        elif max_window is None:
            self.max_window = max_window_allowed
        else:
            raise ValueError(f'max_window is set to {max_window}, but the maximum allowed window by encoder and LLM is {max_window_allowed}')

        # Set the compression_window, compression slots, number of attention heads for compressor, and number of layers for the compressor
        self.compression_window = compression_window
        self.compression_slots = compression_slots
        self.compression_n_heads = compression_n_heads
        self.compression_n_layers = compression_n_layers

        # Create the compressor
        self.compressor = LatentCompressor(
            enc_dim = self.encoder.config.hidden_size,
            num_slots = self.compression_slots,
            num_heads = self.compression_n_heads,
            num_layers = self.compression_n_layers
        )

        # Create the Bridge Layer which translates from the compressor to the LLM
        self.bridge = nn.Linear(
            self.encoder.config.hidden_size,
            self.llm.config.hidden_size
        )

        # LayerNorm before the bridge keeps super-token activations in a sane
        # range early in training; compressor/bridge weights are initialized.
        self.post_norm = nn.LayerNorm(self.encoder.config.hidden_size)
        self.apply(self._init_bridge_weights)
        self._apply_freezing()

    # ------------------------------------------------------------------
    # Train/eval mode
    # ------------------------------------------------------------------

    def train(self, mode = True):
        """Keep frozen submodules in eval mode (disables their dropout)."""
        super().train(mode)
        if mode:
            if not self.llm_trainable:
                self.llm.eval()
            if not self.encoder_trainable:
                self.encoder.eval()
        return self

    # ------------------------------------------------------------------
    # Initialization / freezing
    # ------------------------------------------------------------------

    def _init_bridge_weights(self, module):
        """Initialize only compressor and bridge weights; leave pretrained intact."""
        if module is self.llm or module is self.encoder:
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _apply_freezing(self):
        for p in self.encoder.parameters():
            p.requires_grad = self.encoder_trainable
        for p in self.llm.parameters():
            p.requires_grad = self.llm_trainable
        # Bridge, compressor and post-norm are always trainable
        for p in self.compressor.parameters():
            p.requires_grad = True
        for p in self.bridge.parameters():
            p.requires_grad = True
        for p in self.post_norm.parameters():
            p.requires_grad = True

    def set_llm_trainable(self, flag):
        self.llm_trainable = flag
        self._apply_freezing()

    def set_encoder_trainable(self, flag):
        self.encoder_trainable = flag
        self._apply_freezing()

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        return {
            'total': sum(p.numel() for p in self.parameters()),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'frozen': sum(p.numel() for p in self.parameters() if not p.requires_grad),
        }

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress_context(
        self,
        input_ids,
        llm_tokenizer,
        encoder_tokenizer,
        attention_mask = None
    ):

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # No overflow -> no compression
        if seq_len <= self.max_window:
            return None

        # Split into overflow and active window (in LLM token space)
        overflow_len = seq_len - self.max_window
        overflow_ids = input_ids[:, :overflow_len]
        overflow_mask = attention_mask[:, :overflow_len] if attention_mask is not None else None

        # Split overflow into compression_window chunks
        num_chunks = math.ceil(overflow_len / self.compression_window)

        super_token_sets = []
        for i in range(num_chunks):
            chunk_start = i * self.compression_window
            chunk_end = min((i + 1) * self.compression_window, overflow_len)

            chunk_ids = overflow_ids[:, chunk_start:chunk_end]
            chunk_mask = overflow_mask[:, chunk_start:chunk_end] if overflow_mask is not None else None

            # Decode chunk back to text, then retokenize for encoder
                # Fixes dual-tokenizer problem
                # Special tokens are kept so the encoder sees the same chat structure
            chunk_texts = llm_tokenizer.batch_decode(
                chunk_ids,
                skip_special_tokens = False
            )

            encoder_inputs = encoder_tokenizer(
                chunk_texts,
                return_tensors = 'pt',
                padding = True,
                truncation = True,
                max_length = self.compression_window, # For safety
                add_special_tokens = True
            ).to(device)

            encoder_ids = encoder_inputs['input_ids']
            encoder_mask = encoder_inputs['attention_mask']

            # Run encoder; avoid building a graph through a frozen encoder
            if self.encoder_trainable:
                encoder_outputs = self.encoder(
                    input_ids = encoder_ids,
                    attention_mask = encoder_mask
                )
            else:
                with torch.no_grad():
                    encoder_outputs = self.encoder(
                        input_ids = encoder_ids,
                        attention_mask = encoder_mask
                    )
            memory = encoder_outputs.last_hidden_state

            # Build padding mask for compressor
            memory_key_padding_mask = (encoder_mask == 0)

            # Latent queries across cross-attention heads to encoder output
            latents = self.compressor(
                memory,
                memory_key_padding_mask = memory_key_padding_mask
            )

            # Normalize then project to LLM embedding space
            super_tokens = self.bridge(self.post_norm(latents))
            super_token_sets.append(super_tokens)

        # Concatenate all chunk super tokens into one memory tensor
        memory_embeds = torch.cat(super_token_sets, dim = 1)

        return memory_embeds

    def num_memory_tokens(self, seq_len):
        """Number of super-tokens produced for a (padded) sequence length."""
        if seq_len <= self.max_window:
            return 0
        overflow_len = seq_len - self.max_window
        num_chunks = math.ceil(overflow_len / self.compression_window)
        return num_chunks * self.compression_slots

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids,
        llm_tokenizer,
        encoder_tokenizer,
        attention_mask = None
    ):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Compress overflow if needed
        memory_embeds = self.compress_context(
            input_ids,
            llm_tokenizer,
            encoder_tokenizer,
            attention_mask
        )

        # Trim to active window if we compressed
        if memory_embeds is not None:
            overflow_len = seq_len - self.max_window
            input_ids = input_ids[:, overflow_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, overflow_len:]

        # Get embeddings for active window
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        # Prepend memory if exists
        if memory_embeds is not None:
            # Match dtype of the active-window embeddings (avoids fp32/bf16
            # mismatches when autocast handles submodules differently).
            memory_embeds = memory_embeds.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat([memory_embeds, inputs_embeds], dim = 1)

            # Extend attention mask to cover memory
            num_memory_tokens = memory_embeds.shape[1]
            mem_mask = torch.ones(
                (batch_size, num_memory_tokens),
                device = device,
                dtype = attention_mask.dtype if attention_mask is not None else torch.long
            )
            if attention_mask is not None:
                attention_mask = torch.cat([mem_mask, attention_mask], dim = 1)
            else:
                attention_mask = torch.cat(
                    [
                        mem_mask,
                        torch.ones((batch_size, input_ids.shape[1]), device = device, dtype = torch.long),
                    ],
                    dim = 1
                )

        # Run the LLM
        return self.llm(
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        llm_tokenizer,
        encoder_tokenizer,
        attention_mask = None,
        max_new_tokens = 32,
        eos_token_id = None,
        stop_token_ids = None,
    ):
        """Greedy autoregressive generation through the memory bridge.

        The overflow region is compressed exactly once; the growing suffix of
        newly generated tokens is appended to the active window each step.
        """
        self.eval()
        device = input_ids.device
        B, T = input_ids.shape

        # Compress overflow once, if needed
        memory_embeds = self.compress_context(input_ids, llm_tokenizer, encoder_tokenizer, attention_mask)
        if memory_embeds is not None:
            overflow_len = T - self.max_window
            input_ids = input_ids[:, overflow_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, overflow_len:]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        num_mem = memory_embeds.shape[1] if memory_embeds is not None else 0

        if stop_token_ids is None:
            stop_token_ids = set()
        else:
            stop_token_ids = set(stop_token_ids)
        if eos_token_id is not None:
            stop_token_ids.add(eos_token_id)

        generated = torch.full((B, 1), -1, dtype = torch.long, device = device)
        finished = torch.zeros(B, dtype = torch.bool, device = device)
        generated_lists = [[] for _ in range(B)]

        for _ in range(max_new_tokens):
            active_ids = torch.cat([input_ids, generated[:, 1:]], dim = 1) if generated.shape[1] > 1 else input_ids
            active_embeds = self.llm.get_input_embeddings()(active_ids)

            pieces_mask = [attention_mask, torch.ones_like(generated[:, 1:])] if generated.shape[1] > 1 else [attention_mask]
            active_mask = torch.cat(pieces_mask, dim = 1)

            if memory_embeds is not None:
                memory_embeds = memory_embeds.to(active_embeds.dtype)
                active_embeds = torch.cat([memory_embeds, active_embeds], dim = 1)
                mem_mask = torch.ones((B, num_mem), dtype = active_mask.dtype, device = device)
                active_mask = torch.cat([mem_mask, active_mask], dim = 1)

            outputs = self.llm(inputs_embeds = active_embeds, attention_mask = active_mask)
            next_token = outputs.logits[:, -1, :].argmax(dim = -1, keepdim = True)
            generated = torch.cat([generated, next_token], dim = 1)

            newly_finished = False
            for b in range(B):
                tok = next_token[b].item()
                if finished[b]:
                    continue
                generated_lists[b].append(tok)
                if tok in stop_token_ids:
                    finished[b] = True
                    newly_finished = True

            if finished.all():
                break

        return generated_lists

    # ------------------------------------------------------------------
    # Saving / loading
    # ------------------------------------------------------------------

    def get_config(self):
        return {
            'llm_name_or_path': getattr(self.llm.config, '_name_or_path', None),
            'encoder_name_or_path': getattr(self.encoder.config, '_name_or_path', None),
            'max_window': self.max_window,
            'compression_window': self.compression_window,
            'compression_slots': self.compression_slots,
            'compression_n_heads': self.compression_n_heads,
            'compression_n_layers': self.compression_n_layers,
            'llm_trainable': self.llm_trainable,
            'encoder_trainable': self.encoder_trainable,
        }

    def save_pretrained(self, save_directory):
        """Save the full model (LLM + encoder + compressor + bridge) and config."""
        os.makedirs(save_directory, exist_ok = True)
        with open(os.path.join(save_directory, self.CONFIG_FILE), 'w') as f:
            json.dump(self.get_config(), f, indent = 2)
        torch.save(self.state_dict(), os.path.join(save_directory, 'model.safetensors.pt'))

    @classmethod
    def from_pretrained(cls, load_directory, llm_model = None, encoder_model = None, **overrides):
        """Rebuild a MemoryBridgeLLM from a directory saved by `save_pretrained`.

        Pass `llm_model` / `encoder_model` to reuse already-loaded instances;
        otherwise they are loaded from the names recorded in the config.
        """
        from transformers import AutoModel, AutoModelForCausalLM

        with open(os.path.join(load_directory, cls.CONFIG_FILE)) as f:
            config = json.load(f)
        config.update(overrides)

        if llm_model is None:
            llm_model = AutoModelForCausalLM.from_pretrained(config['llm_name_or_path'])
        if encoder_model is None:
            encoder_model = AutoModel.from_pretrained(config['encoder_name_or_path'])

        model = cls(
            llm_model = llm_model,
            encoder_model = encoder_model,
            max_window = config['max_window'],
            compression_window = config['compression_window'],
            compression_slots = config['compression_slots'],
            compression_n_heads = config['compression_n_heads'],
            compression_n_layers = config['compression_n_layers'],
            llm_trainable = config.get('llm_trainable', False),
            encoder_trainable = config.get('encoder_trainable', False),
        )
        state = torch.load(os.path.join(load_directory, 'model.safetensors.pt'), map_location = 'cpu', weights_only = True)
        model.load_state_dict(state)
        return model


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name != 'auto':
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def batch_compression_stats(model, seq_len):
    """(num_chunks, num_super_tokens, compression_ratio) for a padded seq_len."""
    if seq_len <= model.max_window:
        return 0, 0, None
    overflow = seq_len - model.max_window
    num_chunks = math.ceil(overflow / model.compression_window)
    num_super = num_chunks * model.compression_slots
    return num_chunks, num_super, overflow / num_super


def compute_loss(output_logits, input_ids, attention_mask, num_memory_tokens):
    """Next-token cross-entropy on the active window, correctly aligned.

    Fixes the alignment bug in the original script: when memory super-tokens
    are prepended, the first `num_memory_tokens` logit positions correspond to
    the memory prefix, NOT to real tokens, so they must be sliced off before
    aligning with the (trimmed) input ids.

    Returns (loss, num_label_tokens) or (None, 0) if nothing to score.
    """
    seq_len = input_ids.shape[1]
    if num_memory_tokens > 0:
        overflow_len = seq_len - (output_logits.shape[1] - num_memory_tokens)
        window_ids = input_ids[:, overflow_len:]
        window_mask = attention_mask[:, overflow_len:]
    else:
        window_ids = input_ids
        window_mask = attention_mask

    if window_ids.shape[1] < 2:
        return None, 0

    shift_logits = output_logits[:, num_memory_tokens:-1, :]
    shift_labels = window_ids[:, 1:]
    shift_mask = window_mask[:, 1:]

    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction = 'none',
    ).view(shift_labels.shape)

    losses = losses * shift_mask
    num_tokens = shift_mask.sum()
    if num_tokens.item() == 0:
        return None, 0
    return losses.sum() / num_tokens, int(num_tokens.item())


def pack_examples(lengths_lists, pack_length):
    """Greedy first-fit packing of variable-length examples into fixed bins.

    `lengths_lists` is a list of token-id lists. Returns a list of packed
    sequences, each a flat list of token ids of exactly `pack_length` tokens.
    Packing ensures most training sequences exceed `max_window`, so the
    compression path receives gradient on (almost) every step.
    """
    bins, remaining = [], []
    for ids in lengths_lists:
        ids = ids[:pack_length]
        placed = False
        for i, rem in enumerate(remaining):
            if len(ids) <= rem:
                bins[i].extend(ids)
                remaining[i] -= len(ids)
                placed = True
                break
        if not placed:
            bins.append(list(ids))
            remaining.append(pack_length - len(ids))
    return bins


def make_collate_fn(pad_id):
    """Pads rows of token ids (packed or single examples) into a batch tensor."""
    def collate_fn(batch):
        rows = [b['input_ids'] for b in batch]
        max_len = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), max_len), pad_id, dtype = torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype = torch.long)
        for i, r in enumerate(rows):
            ids = torch.tensor(r, dtype = torch.long)
            input_ids[i, :len(r)] = ids
            attention_mask[i, :len(r)] = 1
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    return collate_fn


def build_windows_for_generate(tokenized_lists, max_window, min_prompt_tokens, max_prompt_tokens, num_prompts, seed):
    """Pick a few long examples and split them into (prompt, full) pairs."""
    rng = random.Random(seed)
    long_ids = [ids for ids in tokenized_lists if len(ids) > max_window + 8]
    rng.shuffle(long_ids)
    windows = []
    for ids in long_ids[:num_prompts]:
        lo = max(min_prompt_tokens, max_window // 2)
        hi = min(max_prompt_tokens, len(ids) - 1)
        if hi <= lo:
            continue
        cut = rng.randint(lo, hi)
        windows.append({'prompt_ids': ids[:cut], 'full_ids': ids})
    return windows


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

@click.command(context_settings = {'show_default': True})
# ---- models ----
@click.option('--llm-name', default = 'aisquared/bolt-instruct-1b', help = 'HF name/path of the causal LM.')
@click.option('--encoder-name', default = 'aisquared/bolt-embedding-small', help = 'HF name/path of the encoder.')
# ---- memory bridge architecture ----
@click.option('--max-window', type = int, default = 1024, help = 'Active (uncompressed) LLM window. Smaller values trigger compression more often.')
@click.option('--compression-window', type = int, default = 3072, help = 'Overflow tokens compressed per chunk.')
@click.option('--compression-slots', type = int, default = 128, help = 'Latent slots (super-tokens) produced per chunk.')
@click.option('--compression-n-heads', type = int, default = 8, help = 'Attention heads in the latent compressor.')
@click.option('--compression-n-layers', type = int, default = 2, help = 'Transformer decoder layers in the latent compressor.')
# ---- freezing ----
@click.option('--train-encoder', is_flag = True, default = False, help = 'Unfreeze the encoder (default: frozen).')
@click.option('--train-llm', is_flag = True, default = False, help = 'Unfreeze the LLM (default: frozen).')
# ---- data ----
@click.option('--dataset-name', default = 'aisquared/bolt-sft-final', help = 'HF dataset with a "messages" column.')
@click.option('--dataset-split', default = 'train', help = 'Dataset split to use.')
@click.option('--dataset-config', default = None, help = 'Optional dataset config name.')
@click.option('--max-examples', type = int, default = None, help = 'Cap on raw examples loaded before packing.')
@click.option('--pack-length', type = int, default = 4096, help = 'Token length of packed training sequences.')
@click.option('--no-packing', is_flag = True, default = False, help = 'Disable sequence packing (one example per row).')
@click.option('--eval-fraction', type = float, default = 0.02, help = 'Fraction of sequences held out for eval (0 disables).')
# ---- optimization ----
@click.option('--epochs', type = int, default = 1)
@click.option('--batch-size', type = int, default = 4)
@click.option('--grad-accum', type = int, default = 1, help = 'Gradient accumulation steps.')
@click.option('--lr', type = float, default = 1e-4)
@click.option('--weight-decay', type = float, default = 0.01)
@click.option('--warmup-ratio', type = float, default = 0.03, help = 'Fraction of total steps used for LR warmup.')
@click.option('--max-grad-norm', type = float, default = 1.0)
@click.option('--seed', type = int, default = 42)
# ---- precision / runtime ----
@click.option('--bf16/--no-bf16', default = True, help = 'bf16 autocast (CUDA only).')
@click.option('--fp16', is_flag = True, default = False, help = 'fp16 autocast + GradScaler (CUDA only, overrides bf16).')
@click.option('--gradient-checkpointing', is_flag = True, default = False, help = 'Enable gradient checkpointing on unfrozen LLM/encoder.')
@click.option('--num-workers', type = int, default = 0)
@click.option('--device', default = 'auto', help = 'auto | cpu | cuda | cuda:N | mps')
# ---- logging ----
@click.option('--log-every', type = int, default = 10, help = 'Console/W&B logging interval (optimizer steps).')
@click.option('--eval-every', type = int, default = 200, help = 'Eval interval in optimizer steps (0 = only at epoch end).')
@click.option('--max-steps', type = int, default = None, help = 'Stop after this many optimizer steps.')
# ---- generation sampling ----
@click.option('--gen-samples', type = int, default = 2, help = 'Number of eval prompts for generation samples (0 disables).')
@click.option('--gen-max-new-tokens', type = int, default = 32)
@click.option('--gen-every', type = int, default = 500, help = 'Generation sample interval in optimizer steps.')
@click.option('--gen-min-prompt-tokens', type = int, default = 256)
@click.option('--gen-max-prompt-tokens', type = int, default = 2048)
# ---- checkpointing / hub ----
@click.option('--output-dir', default = './memory-bridge-output')
@click.option('--save-every', type = int, default = 0, help = 'Save an intermediate checkpoint every N optimizer steps (0 disables).')
@click.option('--hf-repo-id', default = None, help = 'e.g. aisquared/bolt-memory-compressed-1b — pushes the final model + tokenizers.')
@click.option('--hf-private', is_flag = True, default = False, help = 'Create the Hub repo as private.')
# ---- weights & biases ----
@click.option('--wandb-project', default = None, help = 'W&B project (experiment) name. Disabled if omitted.')
@click.option('--wandb-run-name', default = None, help = 'W&B run name.')
@click.option('--wandb-entity', default = None, help = 'W&B entity (team/user).')
@click.option('--wandb-watch/--no-wandb-watch', default = True, help = 'Log gradients/parameters of the bridge + compressor.')
def main(**cfg):
    run(**cfg)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def run(**cfg):
    from datasets import Dataset, load_dataset
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoTokenizer,
        BatchEncoding,
        get_cosine_schedule_with_warmup,
    )

    set_seed(cfg['seed'])
    device = resolve_device(cfg['device'])
    click.echo(f'Using device: {device}')

    use_cuda = device.type == 'cuda'
    use_amp = use_cuda and (cfg['bf16'] or cfg['fp16'])
    amp_dtype = torch.float16 if cfg['fp16'] else torch.bfloat16
    if (cfg['bf16'] or cfg['fp16']) and not use_cuda:
        click.echo('bf16/fp16 requested but CUDA unavailable — running in fp32.')

    # ---------------- Load tokenizers & models ----------------
    click.echo(f"Loading LLM: {cfg['llm_name']}")
    llm_tokenizer = AutoTokenizer.from_pretrained(cfg['llm_name'])
    llm_model = AutoModelForCausalLM.from_pretrained(cfg['llm_name'])

    click.echo(f"Loading encoder: {cfg['encoder_name']}")
    enc_tokenizer = AutoTokenizer.from_pretrained(cfg['encoder_name'])
    enc_model = AutoModel.from_pretrained(cfg['encoder_name'])

    model = MemoryBridgeLLM(
        llm_model = llm_model,
        encoder_model = enc_model,
        max_window = cfg['max_window'],
        compression_window = cfg['compression_window'],
        compression_slots = cfg['compression_slots'],
        compression_n_heads = cfg['compression_n_heads'],
        compression_n_layers = cfg['compression_n_layers'],
    )

    # ---------------- Freeze / unfreeze ----------------
    model.llm.requires_grad_(cfg['train_llm'])
    model.encoder.requires_grad_(cfg['train_encoder'])
    # compressor + bridge always trainable

    if cfg['gradient_checkpointing']:
        if cfg['train_llm']:
            model.llm.gradient_checkpointing_enable()
        if cfg['train_encoder']:
            model.encoder.gradient_checkpointing_enable()

    model.to(device)

    param_report = {
        'llm': count_parameters(model.llm),
        'encoder': count_parameters(model.encoder),
        'compressor': count_parameters(model.compressor),
        'bridge': count_parameters(model.bridge),
    }
    total_params = sum(t for t, _ in param_report.values())
    trainable_params = sum(tr for _, tr in param_report.values())
    click.echo('\nParameters (total / trainable):')
    for name, (t, tr) in param_report.items():
        click.echo(f'  {name:10s} {t:>15,} / {tr:>15,}')
    click.echo(f'  {"TOTAL":10s} {total_params:>15,} / {trainable_params:>15,}')
    click.echo(f'\nModel max_window: {model.max_window}')

    # ---------------- Dataset ----------------
    click.echo(f"\nLoading dataset: {cfg['dataset_name']} ({cfg['dataset_split']})")
    ds = load_dataset(cfg['dataset_name'], cfg['dataset_config'], split = cfg['dataset_split'])
    if cfg['max_examples'] is not None:
        ds = ds.select(range(min(cfg['max_examples'], len(ds))))

    def tokenize_batch(examples):
        tokenized = llm_tokenizer.apply_chat_template(examples['messages'], tokenize = True)
        if isinstance(tokenized, BatchEncoding):
            tokenized = tokenized['input_ids']
        return {'input_ids': [list(ids) for ids in tokenized]}

    tokenized_ds = ds.map(
        tokenize_batch,
        batched = True,
        batch_size = 100,
        remove_columns = ds.column_names,
        desc = 'Tokenizing',
    )
    all_ids = tokenized_ds['input_ids']

    pad_id = llm_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = llm_tokenizer.eos_token_id

    if cfg['no_packing']:
        sequences = all_ids
    else:
        if cfg['pack_length'] <= model.max_window:
            click.echo(f"WARNING: pack_length ({cfg['pack_length']}) <= max_window ({model.max_window}); "
                       'compression will rarely trigger. Consider raising --pack-length or lowering --max-window.')
        click.echo(f"Packing {len(all_ids)} examples into {cfg['pack_length']}-token sequences...")
        sequences = pack_examples(all_ids, cfg['pack_length'])
        click.echo(f'  -> {len(sequences)} packed sequences')

    dataset = Dataset.from_dict({'input_ids': sequences})

    # Train/eval split
    eval_ds = None
    if cfg['eval_fraction'] > 0 and len(dataset) > 1:
        n_eval = max(1, int(len(dataset) * cfg['eval_fraction']))
        n_eval = min(n_eval, len(dataset) - 1)
        eval_ds = dataset.select(range(len(dataset) - n_eval, len(dataset)))
        dataset = dataset.select(range(len(dataset) - n_eval))

    collate_fn = make_collate_fn(pad_id)
    train_loader = DataLoader(
        dataset,
        batch_size = cfg['batch_size'],
        shuffle = True,
        collate_fn = collate_fn,
        num_workers = cfg['num_workers'],
        drop_last = False,
    )
    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(
            eval_ds,
            batch_size = cfg['batch_size'],
            shuffle = False,
            collate_fn = collate_fn,
            num_workers = cfg['num_workers'],
        )

    click.echo(f'\nDataset: {len(dataset)} train sequences, {len(eval_ds) if eval_ds is not None else 0} eval sequences')
    if not cfg['no_packing']:
        click.echo(f'Packed length: {cfg["pack_length"]} tokens (max_window={model.max_window} -> '
                   f'{model.num_memory_tokens(cfg["pack_length"])} super-tokens per packed sequence)')

    # ---------------- Optimizer & schedule ----------------
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {'params': decay, 'weight_decay': cfg['weight_decay']},
            {'params': no_decay, 'weight_decay': 0.0},
        ],
        lr = cfg['lr'],
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg['grad_accum'])
    total_steps = steps_per_epoch * cfg['epochs']
    if cfg['max_steps'] is not None:
        total_steps = min(total_steps, cfg['max_steps'])
    warmup_steps = max(1, int(total_steps * cfg['warmup_ratio']))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    click.echo(f'\nOptimization: {total_steps} optimizer steps ({steps_per_epoch}/epoch x {cfg["epochs"]} epochs), warmup {warmup_steps}')

    scaler = torch.amp.GradScaler('cuda') if (use_amp and cfg['fp16']) else None

    # ---------------- W&B ----------------
    wandb_run = None
    wandb_mod = None
    if cfg['wandb_project']:
        import wandb
        wandb_mod = wandb
        wandb_run = wandb.init(
            project = cfg['wandb_project'],
            name = cfg['wandb_run_name'],
            entity = cfg['wandb_entity'],
            config = {**cfg, 'total_steps': total_steps, 'trainable_params': trainable_params, 'total_params': total_params},
        )
        if cfg['wandb_watch']:
            wandb.watch(model.compressor, log = 'all', log_freq = max(1, cfg['log_every']))
            wandb.watch(model.bridge, log = 'all', log_freq = max(1, cfg['log_every']))
        click.echo(f'W&B run: {wandb_run.url}')

    def wandb_log(metrics, step):
        if wandb_run is not None:
            wandb_run.log(metrics, step = step)

    # ---------------- Generation prompt windows ----------------
    gen_windows = []
    if cfg['gen_samples'] > 0:
        gen_windows = build_windows_for_generate(
            all_ids,
            model.max_window,
            cfg['gen_min_prompt_tokens'],
            cfg['gen_max_prompt_tokens'],
            cfg['gen_samples'],
            cfg['seed'],
        )
        click.echo(f'Prepared {len(gen_windows)} long-context generation prompts')

    stop_token_ids = set()
    if llm_tokenizer.eos_token_id is not None:
        stop_token_ids.add(llm_tokenizer.eos_token_id)
    eot_ids = []
    if llm_tokenizer.eos_token_id is not None:
        eot_ids.append(llm_tokenizer.eos_token_id)

    # ---------------- Eval function ----------------
    def run_eval():
        model.eval()
        total_loss, total_tokens = 0.0, 0
        compressed_batches = 0
        t0 = time.time()
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                seq_len = input_ids.shape[1]
                num_mem = model.num_memory_tokens(seq_len)
                if num_mem > 0:
                    compressed_batches += 1
                with torch.autocast(device.type, dtype = amp_dtype, enabled = use_amp):
                    output = model(input_ids, llm_tokenizer, enc_tokenizer, attention_mask = attention_mask)
                    loss, ntok = compute_loss(output.logits, input_ids, attention_mask, num_mem)
                if loss is not None:
                    total_loss += loss.item() * ntok
                    total_tokens += ntok
        avg_loss = total_loss / total_tokens if total_tokens else float('nan')
        metrics = {
            'eval/loss': avg_loss,
            'eval/perplexity': math.exp(min(avg_loss, 20)) if total_tokens else float('nan'),
            'eval/tokens': total_tokens,
            'eval/compressed_batch_frac': compressed_batches / max(1, len(eval_loader)),
            'eval/time_sec': time.time() - t0,
        }
        model.train()
        return metrics

    # ---------------- Generation sample function ----------------
    def run_generation_samples(global_step):
        if not gen_windows:
            return
        rows = []
        for w in gen_windows:
            prompt_ids = torch.tensor([w['prompt_ids']], dtype = torch.long, device = device)
            prompt_mask = torch.ones_like(prompt_ids)
            with torch.autocast(device.type, dtype = amp_dtype, enabled = use_amp):
                gen_ids = model.generate(
                    prompt_ids,
                    llm_tokenizer,
                    enc_tokenizer,
                    attention_mask = prompt_mask,
                    max_new_tokens = cfg['gen_max_new_tokens'],
                    stop_token_ids = stop_token_ids,
                )[0]
            tail = w['full_ids'][len(w['prompt_ids']): len(w['prompt_ids']) + cfg['gen_max_new_tokens']]
            prompt_text = llm_tokenizer.decode(w['prompt_ids'], skip_special_tokens = False)
            gen_text = llm_tokenizer.decode(gen_ids, skip_special_tokens = False)
            ref_text = llm_tokenizer.decode(tail, skip_special_tokens = False)
            rows.append({'prompt_tail': prompt_text[-300:], 'generated': gen_text, 'reference': ref_text})
            click.echo('\n--- generation sample ---')
            click.echo(f'PROMPT (last 300 chars): {prompt_text[-300:]!r}')
            click.echo(f'GENERATED: {gen_text!r}')
            click.echo(f'REFERENCE: {ref_text!r}')
        if wandb_run is not None:
            table = wandb_mod.Table(columns = ['step', 'prompt_tail', 'generated', 'reference'])
            for r in rows:
                table.add_data(global_step, r['prompt_tail'], r['generated'], r['reference'])
            wandb_run.log({'generation/samples': table}, step = global_step)

    # ---------------- Training loop ----------------
    os.makedirs(cfg['output_dir'], exist_ok = True)
    click.echo('\n' + '=' * 70 + '\nStarting training\n' + '=' * 70)

    global_step = 0
    micro_step = 0
    accum_loss, accum_tokens = 0.0, 0
    accum_compressed, accum_super, accum_ratio = 0, 0.0, 0
    step_start = time.time()
    stop_training = False

    for epoch in range(cfg['epochs']):
        if stop_training:
            break
        click.echo(f'\nEpoch {epoch + 1}/{cfg["epochs"]}')
        model.train()
        optimizer.zero_grad(set_to_none = True)

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seq_len = input_ids.shape[1]
            num_chunks, num_super, ratio = batch_compression_stats(model, seq_len)

            with torch.autocast(device.type, dtype = amp_dtype, enabled = use_amp):
                output = model(input_ids, llm_tokenizer, enc_tokenizer, attention_mask = attention_mask)
                loss, ntok = compute_loss(output.logits, input_ids, attention_mask, num_super)

            if loss is None:
                continue

            micro_step += 1
            accum_loss += loss.item() * ntok
            accum_tokens += ntok
            if num_super > 0:
                accum_compressed += 1
                accum_super += num_super
                accum_ratio += ratio

            (scaler.scale(loss) if scaler else loss).div(cfg['grad_accum']).backward()

            if micro_step % cfg['grad_accum'] != 0:
                continue

            # ---- optimizer step ----
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg['max_grad_norm']
            )
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none = True)
            global_step += 1

            if global_step % cfg['log_every'] == 0:
                avg_loss = accum_loss / max(1, accum_tokens)
                elapsed = time.time() - step_start
                toks_per_sec = accum_tokens / elapsed if elapsed > 0 else 0.0
                metrics = {
                    'train/loss': avg_loss,
                    'train/perplexity': math.exp(min(avg_loss, 20)),
                    'train/lr': scheduler.get_last_lr()[0],
                    'train/grad_norm': float(grad_norm),
                    'train/epoch': epoch + 1,
                    'train/tokens_per_sec': toks_per_sec,
                    'compression/batches_with_compression_frac': accum_compressed / max(1, cfg['log_every'] * cfg['grad_accum']),
                    'compression/avg_super_tokens': accum_super / max(1, accum_compressed) if accum_compressed else 0.0,
                    'compression/avg_ratio': accum_ratio / max(1, accum_compressed) if accum_compressed else 0.0,
                }
                if use_cuda:
                    metrics['system/gpu_mem_allocated_mb'] = torch.cuda.memory_allocated() / 1024 ** 2
                    metrics['system/gpu_mem_reserved_mb'] = torch.cuda.memory_reserved() / 1024 ** 2
                wandb_log(metrics, global_step)
                click.echo(
                    f'step {global_step:>6}/{total_steps} | loss {avg_loss:.4f} | ppl {math.exp(min(avg_loss, 20)):8.2f} '
                    f'| lr {scheduler.get_last_lr()[0]:.2e} | gnorm {float(grad_norm):.2f} | {toks_per_sec:,.0f} tok/s'
                )
                accum_loss, accum_tokens = 0.0, 0
                accum_compressed, accum_super, accum_ratio = 0, 0.0, 0
                step_start = time.time()

            if eval_loader is not None and cfg['eval_every'] > 0 and global_step % cfg['eval_every'] == 0:
                eval_metrics = run_eval()
                wandb_log(eval_metrics, global_step)
                click.echo(f'  [eval] loss {eval_metrics["eval/loss"]:.4f} | ppl {eval_metrics["eval/perplexity"]:.2f}')

            if cfg['gen_every'] > 0 and global_step % cfg['gen_every'] == 0:
                run_generation_samples(global_step)

            if cfg['save_every'] > 0 and global_step % cfg['save_every'] == 0:
                ckpt = os.path.join(cfg['output_dir'], f'checkpoint-{global_step}')
                model.save_pretrained(ckpt)
                click.echo(f'  [checkpoint] saved to {ckpt}')

            if global_step >= total_steps:
                stop_training = True
                break

        # ---- end-of-epoch eval ----
        if eval_loader is not None and not stop_training:
            eval_metrics = run_eval()
            wandb_log(eval_metrics, global_step)
            click.echo(f'  [epoch {epoch + 1} eval] loss {eval_metrics["eval/loss"]:.4f} | ppl {eval_metrics["eval/perplexity"]:.2f}')

    # ---------------- Final eval + generation ----------------
    if eval_loader is not None:
        eval_metrics = run_eval()
        wandb_log({f'final/{k.split("/", 1)[1]}': v for k, v in eval_metrics.items()}, global_step)
        click.echo(f'\nFinal eval: loss {eval_metrics["eval/loss"]:.4f} | ppl {eval_metrics["eval/perplexity"]:.2f}')
    run_generation_samples(global_step)

    # ---------------- Save & push ----------------
    final_dir = os.path.join(cfg['output_dir'], 'final')
    click.echo(f'\nSaving model to {final_dir}')
    model.save_pretrained(final_dir)
    llm_tokenizer.save_pretrained(final_dir)
    enc_tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, 'training_config.json'), 'w') as f:
        json.dump({k: v for k, v in cfg.items()}, f, indent = 2, default = str)

    if cfg['hf_repo_id']:
        from huggingface_hub import HfApi
        click.echo(f'Pushing model to Hugging Face repo: {cfg["hf_repo_id"]}')
        api = HfApi()
        api.create_repo(cfg['hf_repo_id'], private = cfg['hf_private'], exist_ok = True)
        api.upload_folder(folder_path = final_dir, repo_id = cfg['hf_repo_id'])
        click.echo(f'  -> https://huggingface.co/{cfg["hf_repo_id"]}')

    if wandb_run is not None:
        wandb_run.finish()

    click.echo('\nTraining complete.')


if __name__ == '__main__':
    main()
