#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh -- quick end-to-end smoke test for the MemoryBridgeLLM trainer
# =============================================================================
#
# PURPOSE
# -------
# Runs a *tiny* but complete training pipeline through
# `simplememorybridgeexperiment.py` on a small open model/dataset so you can
# verify, before pushing to the cluster, that:
#
#   1. Dependencies resolve (via `uv` inline script metadata).
#   2. The LLM + encoder load, the model builds, and parameters are wired up.
#   3. The compressor + bridge receive gradients and the loss goes DOWN.
#   4. Eval, generation sampling, and checkpoint saving all execute.
#   5. The saved artifact can be reloaded with `MemoryBridgeLLM.from_pretrained`.
#
# It intentionally uses SMALL models (SmolLM2-135M + MiniLM) and a handful of
# optimizer steps so the whole thing finishes in a few minutes on a laptop
# (CPU/MPS) or seconds on a GPU. Everything is meant to be inspected and
# edited -- see the CONFIGURATION block below.
#
# USAGE
# -----
#   bash smoke_test.sh                 # run everything
#   bash smoke_test.sh --keep-output   # don't delete the scratch output dir
#   bash smoke_test.sh --skip-train    # only run the reload sanity check
#                                      #   (expects a previous run's output)
#
# REQUIREMENTS
# ------------
#   * `uv` on PATH (https://docs.astral.sh/uv/) -- used to resolve and run the
#     training script's inline dependencies. To use your own python env
#     instead, set USE_UV=0 below (or `export USE_UV=0`).
#   * Internet access on first run (downloads models + dataset from the HF
#     Hub). Subsequent runs use the local HF cache.
#   * bash 3.2+ (this script avoids bash-4-only features).
#
# WHAT TO EDIT FOR THE CLUSTER
# ----------------------------
# The variables in the CONFIGURATION block map 1:1 onto the CLI flags of
# `simplememorybridgeexperiment.py`. For a cluster run you would typically:
#   * swap LLM_NAME / ENCODER_NAME / DATASET_NAME to your production models,
#   * raise MAX_WINDOW, PACK_LENGTH, MAX_STEPS (and drop MAX_EXAMPLES),
#   * enable W&B (WANDB_PROJECT) and/or Hub push (HF_REPO_ID),
#   * set DEVICE=cuda and BF16=1,
#   * point OUTPUT_ROOT at shared/scratch storage.
# For anything beyond a smoke test, prefer invoking the trainer directly (see
# the docstring at the top of simplememorybridgeexperiment.py).
#
# EXIT STATUS
# -----------
#   0  all checks passed
#   1  a prerequisite was missing or a check failed
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Pretty logging helpers
# -----------------------------------------------------------------------------
if [ -t 1 ]; then
    BOLD="$(printf '\033[1m')"
    GREEN="$(printf '\033[32m')"
    YELLOW="$(printf '\033[33m')"
    RED="$(printf '\033[31m')"
    RESET="$(printf '\033[0m')"
else
    BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

log()  { printf '%s==>%s %s\n'  "${GREEN}${BOLD}" "${RESET}" "$*"; }
warn() { printf '%sWARN:%s %s\n' "${YELLOW}${BOLD}" "${RESET}" "$*" >&2; }
die()  { printf '%sERROR:%s %s\n' "${RED}${BOLD}" "${RESET}" "$*" >&2; exit 1; }

# Run a command, echoing it first so the log shows exactly what was executed.
run() {
    printf '%s$%s' "${BOLD}" "${RESET}"
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

# =============================================================================
# CONFIGURATION  --  >>> EDIT THESE VALUES FOR YOUR LOCAL / CLUSTER RUN <<<
# =============================================================================

# ---- Models -----------------------------------------------------------------
# Small, public, fast-downloading models so the smoke test runs anywhere.
# Swap these for your production LLM / encoder for a real run.
LLM_NAME="${LLM_NAME:-aisquared/bolt-instruct-1b}"
ENCODER_NAME="${ENCODER_NAME:-aisquared/bolt-embedding-small}"

# ---- Dataset ----------------------------------------------------------------
# Needs a "messages" column (chat format). no_robots is small and public.
DATASET_NAME="${DATASET_NAME:-aisquared/bolt-sft-final}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_CONFIG="${DATASET_CONFIG:-}"          # leave empty for no config
MAX_EXAMPLES="${MAX_EXAMPLES:-64000}"            # cap raw examples (keeps it fast)

# ---- Memory bridge architecture ---------------------------------------------
# Deliberately tiny so compression triggers on short sequences:
#   pack_length (256) > max_window (64)  =>  the overflow is compressed into
#   ceil((256-64)/48) * 12 = 48 super-tokens per packed sequence.
MAX_WINDOW="${MAX_WINDOW:-64}"
COMPRESSION_WINDOW="${COMPRESSION_WINDOW:-48}"
COMPRESSION_SLOTS="${COMPRESSION_SLOTS:-12}"
COMPRESSION_N_HEADS="${COMPRESSION_N_HEADS:-8}"
COMPRESSION_N_LAYERS="${COMPRESSION_N_LAYERS:-2}"

# ---- Data packing / batching ------------------------------------------------
PACK_LENGTH="${PACK_LENGTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
EVAL_FRACTION="${EVAL_FRACTION:-0.1}"

# ---- Optimization ------------------------------------------------------------
# Just enough steps to prove gradients flow and loss decreases.
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-10000}"
LR="${LR:-1e-4}"
LOG_EVERY="${LOG_EVERY:-1}"
EVAL_EVERY="${EVAL_EVERY:-0}"                # 0 = only epoch-end/final eval

# ---- Generation sampling ------------------------------------------------------
GEN_SAMPLES="${GEN_SAMPLES:-1}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-4}"
GEN_MIN_PROMPT_TOKENS="${GEN_MIN_PROMPT_TOKENS:-24}"
GEN_MAX_PROMPT_TOKENS="${GEN_MAX_PROMPT_TOKENS:-96}"

# ---- Freezing ------------------------------------------------------------------
# 1 = train the submodule, 0 = keep frozen. Defaults match the intended setup:
# only the compressor + bridge learn.
TRAIN_LLM="${TRAIN_LLM:-1}"
TRAIN_ENCODER="${TRAIN_ENCODER:-1}"

# ---- Precision / runtime --------------------------------------------------------
# On CUDA you probably want BF16=1. On MPS/CPU the script auto-falls-back to
# fp32 with a warning, which is fine for a smoke test.
DEVICE="${DEVICE:-auto}"                      # auto | cpu | cuda | cuda:N | mps
BF16="${BF16:-1}"
FP16="${FP16:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

# ---- Output / experiment tracking ------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/memory-bridge-smoke}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/run-${RUN_ID}}"
SAVE_EVERY="${SAVE_EVERY:-0}"                # 0 = no intermediate checkpoints

# W&B / HF Hub push are OFF by default for a smoke test. Set these to enable.
WANDB_PROJECT="${WANDB_PROJECT:-}"
HF_REPO_ID="${HF_REPO_ID:-}"

# ---- Runner ----------------------------------------------------------------------
# 1 = run via `uv run` (resolves inline deps automatically). 0 = use the python
#     currently on PATH (you must have the deps installed yourself).
USE_UV="${USE_UV:-1}"

# ---- Script location ---------------------------------------------------------------
# Directory containing this script (and the trainer). Allows running from
# anywhere: `bash path/to/smoke_test.sh`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
TRAINER="${SCRIPT_DIR}/simplememorybridgeexperiment.py"

# =============================================================================
# END OF CONFIGURATION -- you normally shouldn't need to edit below this line.
# =============================================================================

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
KEEP_OUTPUT=0
SKIP_TRAIN=0
for arg in "$@"; do
    case "$arg" in
        --keep-output) KEEP_OUTPUT=1 ;;
        --skip-train)  SKIP_TRAIN=1 ;;
        -h|--help)
            sed -n '2,40p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) die "Unknown argument: $arg (try --help)" ;;
    esac
done

# -----------------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------------
log "Pre-flight checks"

[ -f "${TRAINER}" ] || die "Trainer not found at ${TRAINER}"
[ -f "${SCRIPT_DIR}/MemoryBridgeLLM.py" ] || die "MemoryBridgeLLM.py not found in ${SCRIPT_DIR}"

if [ "${USE_UV}" = "1" ]; then
    command -v uv >/dev/null 2>&1 || die "\`uv\` not found on PATH. Install it (https://docs.astral.sh/uv/) or set USE_UV=0."
    PY_RUN=(uv run)
    log "Using uv to resolve inline script dependencies"
else
    command -v python3 >/dev/null 2>&1 || die "python3 not found on PATH"
    PY_RUN=(python3)
    log "Using system python3 (USE_UV=0) -- assuming dependencies are installed"
fi

log "Configuration:"
cat <<EOF
    LLM_NAME .............. ${LLM_NAME}
    ENCODER_NAME .......... ${ENCODER_NAME}
    DATASET_NAME .......... ${DATASET_NAME} (${DATASET_SPLIT})
    MAX_EXAMPLES .......... ${MAX_EXAMPLES}
    MAX_WINDOW ............ ${MAX_WINDOW}
    COMPRESSION_WINDOW .... ${COMPRESSION_WINDOW}
    COMPRESSION_SLOTS ..... ${COMPRESSION_SLOTS}
    PACK_LENGTH ........... ${PACK_LENGTH}
    BATCH_SIZE ............ ${BATCH_SIZE} (grad_accum=${GRAD_ACCUM})
    MAX_STEPS ............. ${MAX_STEPS}   LR=${LR}   EPOCHS=${EPOCHS}
    DEVICE ................ ${DEVICE}   bf16=${BF16}   fp16=${FP16}
    TRAIN_LLM ............. ${TRAIN_LLM}   TRAIN_ENCODER=${TRAIN_ENCODER}
    OUTPUT_DIR ............ ${OUTPUT_DIR}
    WANDB_PROJECT ......... ${WANDB_PROJECT:-<disabled>}
    HF_REPO_ID ............ ${HF_REPO_ID:-<disabled>}
EOF

mkdir -p "${OUTPUT_DIR}"

# -----------------------------------------------------------------------------
# Assemble the trainer CLI from the configuration above.
# (Uses a plain indexed array so it works on bash 3.2 / macOS.)
# -----------------------------------------------------------------------------
TRAIN_ARGS=(
    --llm-name            "${LLM_NAME}"
    --encoder-name        "${ENCODER_NAME}"
    --dataset-name        "${DATASET_NAME}"
    --dataset-split       "${DATASET_SPLIT}"
    --max-examples        "${MAX_EXAMPLES}"
    --max-window          "${MAX_WINDOW}"
    --compression-window  "${COMPRESSION_WINDOW}"
    --compression-slots   "${COMPRESSION_SLOTS}"
    --compression-n-heads "${COMPRESSION_N_HEADS}"
    --compression-n-layers "${COMPRESSION_N_LAYERS}"
    --pack-length         "${PACK_LENGTH}"
    --batch-size          "${BATCH_SIZE}"
    --grad-accum          "${GRAD_ACCUM}"
    --eval-fraction       "${EVAL_FRACTION}"
    --epochs              "${EPOCHS}"
    --max-steps           "${MAX_STEPS}"
    --lr                  "${LR}"
    --log-every           "${LOG_EVERY}"
    --eval-every          "${EVAL_EVERY}"
    --gen-samples         "${GEN_SAMPLES}"
    --gen-max-new-tokens  "${GEN_MAX_NEW_TOKENS}"
    --gen-min-prompt-tokens "${GEN_MIN_PROMPT_TOKENS}"
    --gen-max-prompt-tokens "${GEN_MAX_PROMPT_TOKENS}"
    --save-every          "${SAVE_EVERY}"
    --num-workers         "${NUM_WORKERS}"
    --device              "${DEVICE}"
    --output-dir          "${OUTPUT_DIR}"
)

# Optional toggles (only appended when enabled, to keep the command readable).
[ -n "${DATASET_CONFIG}" ]          && TRAIN_ARGS+=(--dataset-config "${DATASET_CONFIG}")
[ "${TRAIN_LLM}" = "1" ]            && TRAIN_ARGS+=(--train-llm)
[ "${TRAIN_ENCODER}" = "1" ]        && TRAIN_ARGS+=(--train-encoder)
[ "${BF16}" = "1" ]                 && TRAIN_ARGS+=(--bf16) || TRAIN_ARGS+=(--no-bf16)
[ "${FP16}" = "1" ]                 && TRAIN_ARGS+=(--fp16)
[ "${GRADIENT_CHECKPOINTING}" = "1" ] && TRAIN_ARGS+=(--gradient-checkpointing)
[ -n "${WANDB_PROJECT}" ]           && TRAIN_ARGS+=(--wandb-project "${WANDB_PROJECT}")
[ -n "${HF_REPO_ID}" ]              && TRAIN_ARGS+=(--hf-repo-id "${HF_REPO_ID}")

# =============================================================================
# Step 1: training smoke run
# =============================================================================
TRAIN_LOG="${OUTPUT_DIR}/train.log"

if [ "${SKIP_TRAIN}" = "0" ]; then
    log "Step 1/3: running training smoke test (${MAX_STEPS} optimizer steps)"
    log "Full log: ${TRAIN_LOG}"

    # Tee the output so the user sees progress live AND we keep a log to grep.
    set +e
    run "${PY_RUN[@]}" "${TRAINER}" "${TRAIN_ARGS[@]}" 2>&1 | tee "${TRAIN_LOG}"
    TRAIN_RC=${PIPESTATUS[0]}
    set -e

    [ "${TRAIN_RC}" -eq 0 ] || die "Training exited with status ${TRAIN_RC} (see ${TRAIN_LOG})"
    log "Training completed successfully"
else
    log "Step 1/3: SKIPPED (--skip-train); expecting prior output in ${OUTPUT_DIR}"
    [ -f "${TRAIN_LOG}" ] || warn "No train.log found; loss checks will be skipped."
fi

# =============================================================================
# Step 2: verify the run actually trained
# =============================================================================
log "Step 2/3: verifying training signals"

if [ -f "${TRAIN_LOG}" ]; then
    # 2a: the trainer must have reached the final step.
    if grep -q "step *${MAX_STEPS}/${MAX_STEPS}" "${TRAIN_LOG}"; then
        log "  reached final optimizer step (${MAX_STEPS})"
    else
        die "  did not find final step ${MAX_STEPS}/${MAX_STEPS} in ${TRAIN_LOG}"
    fi

    # 2b: loss must be finite and (loosely) non-increasing across the run.
    #     We parse the `loss X.XXXX` fields from the step log lines.
    LOSSES="$(sed -n 's/.*step *[0-9]*\/[0-9]* | loss \([0-9.]*\).*/\1/p' "${TRAIN_LOG}")"
    if [ -n "${LOSSES}" ]; then
        FIRST_LOSS="$(printf '%s\n' "${LOSSES}" | head -1)"
        LAST_LOSS="$(printf '%s\n' "${LOSSES}" | tail -1)"
        log "  loss: first=${FIRST_LOSS}  last=${LAST_LOSS}"
        # Fail only if the loss EXPLODED (last > first * 1.5); a 3-step run is
        # too short to demand strict monotone decrease.
        EXPLODED="$(awk -v a="${FIRST_LOSS}" -v b="${LAST_LOSS}" 'BEGIN{print (b > a*1.5) ? 1 : 0}')"
        [ "${EXPLODED}" = "0" ] || die "  loss appears to have exploded (${FIRST_LOSS} -> ${LAST_LOSS})"
    else
        warn "  could not parse loss values from ${TRAIN_LOG}"
    fi

    # 2c: compression must actually have triggered (that's the point!).
    if grep -q "super-tokens per packed sequence" "${TRAIN_LOG}"; then
        SUPER="$(sed -n 's/.*-> \([0-9]*\) super-tokens per packed sequence.*/\1/p' "${TRAIN_LOG}" | head -1)"
        log "  compression path active (${SUPER} super-tokens per packed sequence)"
    else
        warn "  could not confirm compression triggered (check max_window < pack_length)"
    fi
else
    warn "Skipping log-based checks (no train.log)."
fi

# 2d: the final checkpoint directory must exist and contain model weights.
#     MemoryBridgeLLM.save_pretrained writes weights to 'model.safetensors.pt'
#     (a torch state_dict, despite the extension), so we check for that file
#     specifically but fall back to any *.safetensors / *.bin just in case.
FINAL_DIR="${OUTPUT_DIR}/final"
[ -d "${FINAL_DIR}" ] || die "Final checkpoint dir missing: ${FINAL_DIR}"
if [ ! -f "${FINAL_DIR}/model.safetensors.pt" ] \
    && ! ls "${FINAL_DIR}"/*.safetensors >/dev/null 2>&1 \
    && ! ls "${FINAL_DIR}"/*.bin >/dev/null 2>&1; then
    die "No model weights found in ${FINAL_DIR}"
fi
[ -f "${FINAL_DIR}/training_config.json" ] || warn "training_config.json not found in ${FINAL_DIR}"
log "  checkpoint present: ${FINAL_DIR}"

# =============================================================================
# Step 3: reload the saved model and run a forward + generate sanity check
# =============================================================================
log "Step 3/3: reloading checkpoint and running inference sanity check"

RELOAD_LOG="${OUTPUT_DIR}/reload.log"
set +e
run "${PY_RUN[@]}" - "${FINAL_DIR}" "${LLM_NAME}" "${ENCODER_NAME}" <<'PYEOF' 2>&1 | tee "${RELOAD_LOG}"
import sys
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

sys.path.insert(0, r"${SCRIPT_DIR}")
from MemoryBridgeLLM import MemoryBridgeLLM

final_dir, llm_name, enc_name = sys.argv[1], sys.argv[2], sys.argv[3]

llm = AutoModelForCausalLM.from_pretrained(llm_name)
enc = AutoModel.from_pretrained(enc_name)
llm_tok = AutoTokenizer.from_pretrained(llm_name)
enc_tok = AutoTokenizer.from_pretrained(enc_name)

model = MemoryBridgeLLM.from_pretrained(final_dir, llm_model=llm, encoder_model=enc)
counts = model.count_parameters()
print(f"params: total={counts['total']:,} trainable={counts['trainable']:,} frozen={counts['frozen']:,}")
assert counts["trainable"] > 0, "expected some trainable params (compressor+bridge)"

# Frozen submodules must stay in eval mode even under model.train().
model.train()
assert not model.llm.training and not model.encoder.training, "frozen submodules left eval mode"

# Long enough to overflow max_window and exercise the compression path.
text = "The quick brown fox jumps over the lazy dog. " * 40
ids = llm_tok(text, return_tensors="pt")["input_ids"]
mem = model.compress_context(ids, llm_tok, enc_tok)
print("memory_embeds:", None if mem is None else tuple(mem.shape))

out = model(ids, llm_tok, enc_tok)
print("logits:", tuple(out.logits.shape))

gen = model.generate(ids, llm_tok, enc_tok, max_new_tokens=3,
                     eos_token_id=llm_tok.eos_token_id)
# generate() may return a tensor or a nested list depending on version, and
# returns ONLY the newly generated tokens (not prompt + completion).
first = gen[0]
first = first.tolist() if hasattr(first, "tolist") else list(first)
print("generated ids:", first)
assert len(first) >= 1, "generate produced no new tokens"
print("RELOAD CHECK PASSED")
PYEOF
RELOAD_RC=${PIPESTATUS[0]}
set -e

[ "${RELOAD_RC}" -eq 0 ] || die "Reload/inference check failed (see ${RELOAD_LOG})"
grep -q "RELOAD CHECK PASSED" "${RELOAD_LOG}" || die "Reload check did not report success"

# =============================================================================
# Summary
# =============================================================================
printf '\n%s%s' "${GREEN}${BOLD}" "${RESET}"
cat <<EOF
============================================================
 SMOKE TEST PASSED
============================================================
  trainer log : ${TRAIN_LOG}
  reload log  : ${RELOAD_LOG}
  checkpoint  : ${FINAL_DIR}

Next steps for a real (cluster) run:
  * Edit the CONFIGURATION block in this script (models, dataset,
    window sizes, steps), or invoke simplememorybridgeexperiment.py
    directly -- see its module docstring for a full example.
  * Enable W&B (WANDB_PROJECT) and/or Hub push (HF_REPO_ID).
EOF
printf '%s' "${RESET}"

if [ "${KEEP_OUTPUT}" = "0" ]; then
    log "Cleaning up scratch output (use --keep-output to preserve): ${OUTPUT_ROOT}"
    rm -rf "${OUTPUT_ROOT}"
else
    log "Keeping output in ${OUTPUT_ROOT}"
fi

exit 0
