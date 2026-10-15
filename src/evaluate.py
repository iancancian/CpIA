import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]


def generate(model, tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids=inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/lora-final")
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()
    adapter = ROOT / args.adapter
    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).cuda()
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    data = load_dataset("json", data_files=str(ROOT / "data" / "eval.jsonl"), split="train")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    categories = priorities = required = 0
    rows = []
    for item in data:
        raw = generate(model, tokenizer, item["prompt"])
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {}
        categories += parsed.get("categoria") == item["category"]
        priorities += parsed.get("prioridade") == item["priority"]
        required += all(key in parsed for key in ("categoria", "prioridade", "proximo_passo"))
        rows.append({"prompt": item["prompt"], "output": raw, "rouge_l": scorer.score(item["reference"], parsed.get("proximo_passo", ""))["rougeL"].fmeasure})
    result = {"examples": len(data), "category_accuracy": categories / len(data), "priority_accuracy": priorities / len(data), "required_fields_rate": required / len(data), "mean_rouge_l": sum(row["rouge_l"] for row in rows) / len(rows), "rows": rows}
    output = ROOT / args.output
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
