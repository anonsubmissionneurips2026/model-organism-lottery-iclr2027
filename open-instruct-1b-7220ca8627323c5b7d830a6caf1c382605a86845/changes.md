# Changes Log

**IMPORTANT**: This file tracks all changes made during debugging sessions. Update this file after each fix. This instruction persists across /compact commands.

---

## 2026-02-03: DPO Training Script Fixes

### 1. `--gradient_checkpointing` argument not recognized
- **Error**: `ValueError: Some specified arguments are not used by the HfArgumentParser: ['--gradient_checkpointing']`
- **Cause**: Argument missing from dataclass config
- **Fix**: Added `gradient_checkpointing: bool = False` to `ModelConfig` in `open_instruct/dpo_utils.py`

### 2. BOS token conflict
- **Error**: `ValueError: You specified add_bos=True, but the chat template already has a bos_token at the beginning.`
- **Cause**: Script had `--add_bos` but model's chat template already includes BOS
- **Fix**: Removed `--add_bos \` from `scripts/train/olmo2/dpo_1b_deepspeed.sh`

### 3. Logger `main_process_only` argument invalid
- **Error**: `TypeError: Logger._log() got an unexpected keyword argument 'main_process_only'`
- **Cause**: Code used Accelerate-specific logging argument with standard Python logger
- **Fix**: Removed `main_process_only=False` from `logger.info()` calls in:
  - `open_instruct/dpo_tune_cache.py:230`
  - `open_instruct/finetune.py:452`

### 4. `build_reference_logprobs_cache()` wrong arguments
- **Error**: `TypeError: build_reference_logprobs_cache() got an unexpected keyword argument 'accelerator'`
- **Cause**: Function signature changed but call site in `dpo_tune_cache.py` wasn't updated
- **Fix**: Updated call at `open_instruct/dpo_tune_cache.py:523` to match new signature:
  - Added `import pathlib`
  - Replaced `accelerator=` with `device=accelerator.device`
  - Replaced `reference_cache_hash=` with proper `cache_path=`
  - Added `is_main_process=`, `model_dims=`, `disable_adapter_context=`
