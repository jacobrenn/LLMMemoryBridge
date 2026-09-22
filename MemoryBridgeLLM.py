from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn as nn
import torch
import math

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


class MemoryBridgeLLM(nn.Module):
    def __init__(
        self,
        llm_model,
        encoder_model,
        max_window = None,
        compression_window = 8192,
        compression_slots = 128,
        compression_n_heads = 8,
        compression_n_layers = 2
    ):
        super().__init__()

        # The LLM
        self.llm = llm_model

        # The Encoder
        self.encoder = encoder_model

        # Set the number of tokens to be compressed
        llm_max_window = self.llm.config.max_position_embeddings if self.llm.config.max_position_embeddings else 512
        encoder_max_window = self.encoder.config.max_position_embeddings if self.encoder.config.max_position_embeddings else 512
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

            # Run encoder (do not default to frozen in case we want to train encoder)
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

            # Project to LLM embedding space
            super_tokens = self.bridge(latents)
            super_token_sets.append(super_tokens)

        # Concatenate all chunk super tokens into one memory tensor
        memory_embeds = torch.cat(super_token_sets, dim = 1)

        return memory_embeds

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


# Example Usage
if __name__ == "__main__":
    # Use small models for the prototype to avoid OOM
    LLM_NAME = "aisquared/bolt-instruct-1b"  # Small footprint for testing
    ENC_NAME = "aisquared/bolt-embedding-small"
    DATASET_NAME = "aisquared/bolt-sft-final"

    # Load the tokenizers
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(LLM_NAME)

    # Load the models
    enc_tokenizer = AutoTokenizer.from_pretrained(ENC_NAME)
    enc_model = AutoModel.from_pretrained(ENC_NAME)
    
    # Load the dataset
    ds = load_dataset(DATASET_NAME)

    model = MemoryBridgeLLM(
        llm_model = llm_model,
        encoder_model = enc_model,
        max_window = None,
        compression_window = 3072,
        compression_slots = 128,
        compression_n_heads = 8,
        compression_n_layers = 2
    )

    tokenized_inputs = llm_tokenizer.apply_chat_template(
        list(ds['train']['messages'][:1000]),
        return_tensors = 'pt',
        padding = True
    )

    tokens_max_sequence_length = tokenized_inputs['input_ids'].shape[-1]
    llm_max_sequence_length = model.llm.config.max_position_embeddings

    print(f'Max sequence length seen: {tokens_max_sequence_length}\nLLM Max sequence length: {llm_max_sequence_length}')
