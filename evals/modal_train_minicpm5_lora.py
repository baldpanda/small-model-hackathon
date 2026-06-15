from __future__ import annotations

import os
from pathlib import PurePosixPath
from typing import Any

import modal


MODEL_ID = "openbmb/MiniCPM5-1B"
APP_NAME = "speech-feedback-minicpm5-finetune"
VOLUME_NAME = "speech-feedback-minicpm5-ft"
REMOTE_VOLUME_DIR = "/vol"
GPU_TYPE = os.environ.get("MODAL_GPU", "L4")


TRAIN_CHAT_TEMPLATE = (
    "{{- bos_token }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'user' %}"
    "{{- '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{%- elif message['role'] == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- generation %}"
    "{{- message['content'] + '<|im_end|>' }}"
    "{%- endgeneration %}"
    "{{- '\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- endif %}"
)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate>=1.12.0,<2.0",
        "datasets>=4.0.0,<5.0",
        "peft>=0.13.0,<1.0",
        "safetensors>=0.7.0,<1.0",
        "sentencepiece>=0.2.1,<0.3",
        "tensorboard>=2.20.0,<3.0",
        "torch>=2.11.0",
        "torchvision",
        "transformers>=5.6.0,<6.0",
        "trl>=0.21.0,<1.0",
    )
)


secrets = []
if os.environ.get("HF_TOKEN"):
    secrets.append(modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}))
elif os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    secrets.append(modal.Secret.from_dict({"HUGGINGFACEHUB_API_TOKEN": os.environ["HUGGINGFACEHUB_API_TOKEN"]}))


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    secrets=secrets,
    timeout=7200,
    volumes={REMOTE_VOLUME_DIR: volume},
)
def train_lora_remote(
    *,
    run_name: str,
    data_rel_path: str,
    val_rel_path: str | None = None,
    base_model: str = MODEL_ID,
    epochs: int = 4,
    limit: int | None = None,
    max_steps: int = 0,
    save_steps: int = 25,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
) -> dict[str, Any]:
    import json
    from pathlib import Path

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    validate_run_name(run_name)
    set_seed(42)

    output_dir = Path(REMOTE_VOLUME_DIR) / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter_final"
    train_rows = load_message_rows(Path(REMOTE_VOLUME_DIR) / data_rel_path, limit=limit)
    val_rows = load_message_rows(Path(REMOTE_VOLUME_DIR) / val_rel_path) if val_rel_path else None
    train_dataset = Dataset.from_list([{"messages": row["messages"]} for row in train_rows])
    eval_dataset = Dataset.from_list([{"messages": row["messages"]} for row in val_rows]) if val_rows else None

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, **hf_kwargs())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = TRAIN_CHAT_TEMPLATE
    mask_report = build_assistant_mask_report(tokenizer, train_rows)
    (output_dir / "assistant_mask_check.json").write_text(
        json.dumps(mask_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        **hf_kwargs(),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    config_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "bf16": True,
        "max_length": 2048,
        "packing": False,
        "assistant_only_loss": True,
        "logging_steps": 1 if max_steps else 10,
        "save_total_limit": 2,
        "report_to": ["tensorboard"],
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
        "seed": 42,
    }
    if max_steps:
        config_kwargs["max_steps"] = max_steps
        config_kwargs["save_strategy"] = "steps"
        config_kwargs["save_steps"] = 1
    elif eval_dataset is not None:
        config_kwargs["eval_strategy"] = "epoch"
        config_kwargs["save_strategy"] = "epoch"
        config_kwargs["load_best_model_at_end"] = True
        config_kwargs["metric_for_best_model"] = "eval_loss"
        config_kwargs["greater_is_better"] = False
    else:
        config_kwargs["save_strategy"] = "steps"
        config_kwargs["save_steps"] = save_steps

    trainer_kwargs = {
        "model": model,
        "args": SFTConfig(**config_kwargs),
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    trainer = SFTTrainer(**trainer_kwargs)
    train_result = trainer.train()
    trainer.save_state()
    train_metrics = make_json_safe(getattr(train_result, "metrics", {}) or {})
    (output_dir / "train_metrics.json").write_text(
        json.dumps(train_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "log_history.json").write_text(
        json.dumps(make_json_safe(trainer.state.log_history), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trainer.model.save_pretrained(adapter_dir)
    volume.commit()

    adapter_files = sorted(path.name for path in adapter_dir.iterdir())
    return {
        "run_name": run_name,
        "gpu": GPU_TYPE,
        "base_model": base_model,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows or []),
        "epochs": epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": 4 * gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "adapter_dir": str(adapter_dir.relative_to(REMOTE_VOLUME_DIR)),
        "assistant_mask_check": str((output_dir / "assistant_mask_check.json").relative_to(REMOTE_VOLUME_DIR)),
        "trainer_state": str((output_dir / "trainer_state.json").relative_to(REMOTE_VOLUME_DIR)),
        "train_metrics": str((output_dir / "train_metrics.json").relative_to(REMOTE_VOLUME_DIR)),
        "log_history": str((output_dir / "log_history.json").relative_to(REMOTE_VOLUME_DIR)),
        "adapter_files": adapter_files,
        "training_loss": train_metrics.get("train_loss"),
    }


@app.local_entrypoint()
def main(
    run_name: str = "smoke",
    data_rel_path: str = "data/sft_smoke_messages.jsonl",
    val_rel_path: str | None = None,
    base_model: str = MODEL_ID,
    epochs: int = 1,
    limit: int | None = None,
    max_steps: int = 2,
    save_steps: int = 25,
    gradient_accumulation_steps: int = 2,
    learning_rate: float = 2e-4,
) -> None:
    result = train_lora_remote.remote(
        run_name=run_name,
        data_rel_path=data_rel_path,
        val_rel_path=val_rel_path,
        base_model=base_model,
        epochs=epochs,
        limit=limit,
        max_steps=max_steps,
        save_steps=save_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
    )
    print(result)


def load_message_rows(path: Any, limit: int | None = None) -> list[dict[str, Any]]:
    import json
    from pathlib import Path

    if path is None:
        return []
    full_path = Path(path)
    rows = [json.loads(line) for line in full_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows found in {full_path}")
    return rows


def build_assistant_mask_report(tokenizer: Any, rows: list[dict[str, Any]], sample_size: int = 3) -> dict[str, Any]:
    reports = []
    for row in rows[:sample_size]:
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{row.get('id', '<unknown>')} has no messages for mask check")

        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = flatten_token_values(encoded.get("input_ids"))
        assistant_mask = encoded.get("assistant_masks")
        if assistant_mask is None:
            assistant_mask = encoded.get("assistant_tokens_mask")
        assistant_mask = flatten_token_values(assistant_mask)
        if not assistant_mask:
            raise RuntimeError(
                "Tokenizer did not return an assistant token mask. "
                "Check the training chat template generation block and TRL/transformers versions."
            )
        if len(input_ids) != len(assistant_mask):
            raise RuntimeError(
                f"Assistant mask length mismatch for {row.get('id', '<unknown>')}: "
                f"{len(assistant_mask)} mask values for {len(input_ids)} tokens"
            )

        assistant_token_ids = [token_id for token_id, keep in zip(input_ids, assistant_mask) if keep]
        decoded_assistant_tokens = tokenizer.decode(assistant_token_ids, skip_special_tokens=False)
        leaked_markers = [
            marker
            for marker in ("<|im_start|>system", "<|im_start|>user", "Stats:", "Transcript:")
            if marker in decoded_assistant_tokens
        ]
        if not assistant_token_ids:
            raise RuntimeError(f"Assistant mask is empty for {row.get('id', '<unknown>')}")
        if leaked_markers:
            raise RuntimeError(
                f"Assistant mask for {row.get('id', '<unknown>')} includes non-assistant markers: {leaked_markers}"
            )
        if "Strength:" not in decoded_assistant_tokens:
            raise RuntimeError(
                f"Assistant mask for {row.get('id', '<unknown>')} does not include the gold feedback labels"
            )

        assistant_text = str(messages[-1].get("content") or "")
        reports.append(
            {
                "id": row.get("id"),
                "total_tokens": len(input_ids),
                "assistant_loss_tokens": len(assistant_token_ids),
                "assistant_token_ratio": round(len(assistant_token_ids) / len(input_ids), 4),
                "assistant_words": len(assistant_text.split()),
                "masked_preview": decoded_assistant_tokens[:500],
            }
        )

    return {
        "sample_size": len(reports),
        "checked_ids": [report["id"] for report in reports],
        "reports": reports,
    }


def flatten_token_values(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return make_json_safe(value.item())
    return str(value)


def hf_kwargs() -> dict[str, str]:
    for name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(name)
        if value:
            return {"token": value}
    return {}


def validate_run_name(run_name: str) -> None:
    if not run_name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in run_name):
        raise ValueError("run_name may only contain letters, numbers, underscores, and hyphens")
