import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o classificador de tickets com LoRA.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=384)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/lora-final")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA nao esta disponivel. Execute scripts/check_gpu.py antes do treino.")

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("json", data_files=str(ROOT / "data" / "train.jsonl"), split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example: dict) -> dict:
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=args.max_seq_length,
            return_dict=True,
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = vars(args) | {"model": args.model, "gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda}
    (output_dir / "training_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Treino concluido em {output_dir}")


if __name__ == "__main__":
    main()
