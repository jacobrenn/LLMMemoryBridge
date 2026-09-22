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
