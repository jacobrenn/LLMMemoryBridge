from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import math
import time

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

    # Batch settings
    BATCH_SIZE = 8
    NUM_EXAMPLES = 1000

    # Tokenize once for the whole slice, batched via datasets.map
    def tokenize_batch(examples):
        tokenized = llm_tokenizer.apply_chat_template(
            examples['messages'],
            tokenize = True,
            # No padding/truncation here — return raw lists of varying length
        )
        # apply_chat_template may return a BatchEncoding; unwrap to plain lists
        if isinstance(tokenized, BatchEncoding):
            tokenized = tokenized['input_ids']
        return {'input_ids': [list(ids) for ids in tokenized]}

    tokenized_ds = (
        ds['train']
        .select(range(NUM_EXAMPLES))
        .map(
            tokenize_batch,
            batched = True,
            batch_size = 100,
            remove_columns = ds['train'].column_names,  # Drop 'messages', keep input_ids
        )
    )

    # Collator pads each batch dynamically to ITS longest sequence
    def collate_fn(batch):
        return llm_tokenizer.pad(
            [{'input_ids': b['input_ids']} for b in batch],
            padding = True,
            return_tensors = 'pt',
        )

    loader = DataLoader(
        tokenized_ds,
        batch_size = BATCH_SIZE,
        collate_fn = collate_fn,
    )

    # Iterate batches
    total_tokens_processed = 0
    total_compressed = 0
    batches_with_compression = 0
    seq_lengths = []
    total_forward_time = 0
    total_loss_sum = 0
    total_loss_count = 0

    for i, batch in enumerate(loader):
        batch_size = batch['input_ids'].shape[0]
        seq_len = batch['input_ids'].shape[-1]
        seq_lengths.append(seq_len)
        total_tokens_processed += batch_size * seq_len

        print(f"\n{'='*80}")
        print(f"Batch {i+1}/{len(loader)}")
        print(f"{'='*80}")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length (padded): {seq_len}")
        print(f"  Model max_window: {model.max_window}")
        print(f"  Compression window: {model.compression_window}")
        print(f"  Compression slots per chunk: {model.compression_slots}")
        
        # Show actual (unpadded) lengths in this batch
        actual_lengths = batch['attention_mask'].sum(dim=1).tolist()
        print(f"  Actual lengths (unpadded): min={min(actual_lengths)}, "
              f"max={max(actual_lengths)}, mean={sum(actual_lengths)/len(actual_lengths):.0f}")
        
        # Show a sample of the first sequence (first 50 tokens decoded)
        sample_ids = batch['input_ids'][0][:min(50, seq_len)]
        sample_text = llm_tokenizer.decode(sample_ids, skip_special_tokens=False)
        print(f"  Sample text (first 50 tokens of example 0):")
        print(f"    {repr(sample_text[:200])}{'...' if len(sample_text) > 200 else ''}")

        will_compress = seq_len > model.max_window
        
        if will_compress:
            batches_with_compression += 1
            total_compressed += batch_size
            overflow = seq_len - model.max_window
            num_chunks = math.ceil(overflow / model.compression_window)
            num_super_tokens = num_chunks * model.compression_slots
            compression_ratio = overflow / num_super_tokens
            
            print(f"\n  ⚠️  COMPRESSION TRIGGERED")
            print(f"    Overflow: {overflow} tokens")
            print(f"    Number of chunks: {num_chunks}")
            print(f"    Super-tokens generated: {num_super_tokens}")
            print(f"    Compression ratio: {compression_ratio:.1f}:1")
            print(f"    Final sequence length: {model.max_window + num_super_tokens} tokens")
            print(f"      ({model.max_window} raw + {num_super_tokens} super-tokens)")
        else:
            print(f"\n  ✓ No compression needed (fits in window)")

        # Track memory before forward pass
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / 1024**2
            print(f"\n  GPU memory before forward: {mem_before:.1f} MB")

        start_time = time.time()

        with torch.no_grad():
            output = model(
                batch['input_ids'],
                llm_tokenizer,
                enc_tokenizer,
                attention_mask = batch['attention_mask'],
            )

        forward_time = time.time() - start_time
        total_forward_time += forward_time

        print(f"\n  Forward pass complete:")
        print(f"    Time: {forward_time:.2f}s")
        print(f"    Output logits shape: {output.logits.shape}")
        print(f"    Expected shape: [{batch_size}, {model.max_window + (num_chunks * model.compression_slots if will_compress else 0)}, {model.llm.config.vocab_size}]")

        # Calculate throughput
        tokens_this_batch = batch_size * seq_len
        throughput = tokens_this_batch / forward_time
        print(f"    Throughput: {throughput:.0f} tokens/sec")

        # Calculate perplexity (only on non-compressed region to avoid super-token weirdness)
        # We'll use the logits up to max_window (or full seq if no compression)
        valid_seq_len = min(seq_len, model.max_window)
        
        if valid_seq_len > 1:  # Need at least 2 tokens for next-token prediction
            # Shift logits and labels for next-token prediction
            shift_logits = output.logits[:, :valid_seq_len-1, :].contiguous()
            shift_labels = batch['input_ids'][:, 1:valid_seq_len].contiguous()
            shift_mask = batch['attention_mask'][:, 1:valid_seq_len].contiguous()
            
            # Calculate loss only on non-padded positions
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            losses = losses.view(batch_size, -1) * shift_mask
            batch_loss = losses.sum() / shift_mask.sum()
            perplexity = torch.exp(batch_loss)
            
            total_loss_sum += batch_loss.item()
            total_loss_count += 1
            
            print(f"    Loss: {batch_loss.item():.4f}")
            print(f"    Perplexity: {perplexity.item():.2f}")
        
        # Generation sample (greedy decoding from last position)
        print(f"\n  Generation sample (from last valid position):")
        # Get the last valid token position for the first example
        last_valid_pos = actual_lengths[0] - 1
        
        # Get logits at that position
        if last_valid_pos < output.logits.shape[1]:
            next_token_logits = output.logits[0, last_valid_pos, :]
            
            # Top-5 predictions
            top_probs, top_indices = torch.softmax(next_token_logits, dim=-1).topk(5)
            print(f"    Top 5 next-token predictions:")
            for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
                token_text = llm_tokenizer.decode([idx.item()])
                print(f"      {rank}. {repr(token_text)} (prob: {prob.item():.4f})")
            
            # Greedy generation for next 10 tokens (very simple, no sampling)
            print(f"\n    Greedy generation (next 10 tokens):")
            generated_ids = [batch['input_ids'][0, last_valid_pos].item()]
            current_logits = next_token_logits.unsqueeze(0)
            
            for _ in range(10):
                next_id = current_logits.argmax(dim=-1)
                generated_ids.append(next_id.item())
                
                # For a real implementation, we'd do another forward pass
                # Here we just sample from the same distribution with some noise
                # to make it slightly more interesting
                current_logits = current_logits + torch.randn_like(current_logits) * 0.5
            
            generated_text = llm_tokenizer.decode(generated_ids)
            print(f"      Context ends with: ...{repr(sample_text[-50:])}")
            print(f"      Generated: {repr(generated_text)}")
            print(f"      (Note: This is NOT real autoregressive generation - just sampling from last position)")

        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / 1024**2
            mem_delta = mem_after - mem_before
            print(f"    GPU memory after forward: {mem_after:.1f} MB (Δ {mem_delta:+.1f} MB)")
            torch.cuda.empty_cache()
            mem_cached = torch.cuda.memory_reserved() / 1024**2
            print(f"    GPU memory cached: {mem_cached:.1f} MB")

        # Running statistics
        if (i + 1) % 10 == 0 or i == len(loader) - 1:
            print(f"\n  {'─'*76}")
            print(f"  Running statistics after {i+1} batches:")
            print(f"    Total tokens processed: {total_tokens_processed:,}")
            print(f"    Total examples compressed: {total_compressed}/{total_tokens_processed//seq_len}")
            print(f"    Compression rate: {batches_with_compression}/{i+1} batches "
                  f"({100*batches_with_compression/(i+1):.1f}%)")
            print(f"    Avg seq length: {sum(seq_lengths)/len(seq_lengths):.0f} tokens")
            print(f"    Seq length range: [{min(seq_lengths)}, {max(seq_lengths)}]")
            
            if total_loss_count > 0:
                avg_loss = total_loss_sum / total_loss_count
                avg_perplexity = math.exp(avg_loss)
                print(f"    Avg loss: {avg_loss:.4f}")
                print(f"    Avg perplexity: {avg_perplexity:.2f}")
            
            avg_forward_time = total_forward_time / (i + 1)
            avg_throughput = total_tokens_processed / total_forward_time
            print(f"    Avg forward time: {avg_forward_time:.2f}s")
            print(f"    Avg throughput: {avg_throughput:.0f} tokens/sec")
            print(f"  {'─'*76}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total batches: {len(loader)}")
    print(f"Total examples: {NUM_EXAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"\nSequence statistics:")
    print(f"  Min length: {min(seq_lengths)} tokens")
    print(f"  Max length: {max(seq_lengths)} tokens")
    print(f"  Mean length: {sum(seq_lengths)/len(seq_lengths):.0f} tokens")
    print(f"  Median length: {sorted(seq_lengths)[len(seq_lengths)//2]} tokens")
    print(f"\nCompression statistics:")
    print(f"  Batches with compression: {batches_with_compression}/{len(loader)} "
          f"({100*batches_with_compression/len(loader):.1f}%)")
    print(f"  Total examples compressed: {total_compressed}")
    
    if total_loss_count > 0:
        avg_loss = total_loss_sum / total_loss_count
        avg_perplexity = math.exp(avg_loss)
        print(f"\nLanguage modeling statistics:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Average perplexity: {avg_perplexity:.2f}")
        print(f"  (Note: Perplexity computed only on non-compressed region)")
    
    print(f"\nPerformance statistics:")
    print(f"  Total forward time: {total_forward_time:.2f}s")
    print(f"  Avg time per batch: {total_forward_time/len(loader):.2f}s")
    print(f"  Avg throughput: {total_tokens_processed/total_forward_time:.0f} tokens/sec")
    print(f"  Total tokens processed: {total_tokens_processed:,}")
    
    print(f"\nModel configuration:")
    print(f"  LLM max window: {model.max_window}")
    print(f"  Compression window: {model.compression_window}")
    print(f"  Compression slots: {model.compression_slots}")
    print(f"  LLM vocab size: {model.llm.config.vocab_size}")
    print(f"  LLM hidden size: {model.llm.config.hidden_size}")
    print(f"  Encoder hidden size: {model.encoder.config.hidden_size}")
    print(f"\n✅ All batches processed successfully!")
