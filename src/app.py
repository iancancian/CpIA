import argparse
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parents[1]


def load_pipeline(adapter: str):
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / adapter))
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).cuda()
    return PeftModel.from_pretrained(base, str(ROOT / adapter)).eval(), tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/lora-10ep-lr5e-4")
    args = parser.parse_args()
    model, tokenizer = load_pipeline(args.adapter)

    def answer(ticket: str) -> str:
        if not ticket.strip():
            return "Informe um chamado para iniciar a análise."
        messages = [{"role": "user", "content": ticket}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=120, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    with gr.Blocks(title="Assistente de Suporte de TI") as demo:
        gr.Markdown("# Assistente de Suporte de TI\nClassifique um chamado e receba uma orientação objetiva.")
        gr.Markdown("**Modelo:** Qwen 1.5B + LoRA  |  **Execução:** GPU NVIDIA local")
        with gr.Row():
            with gr.Column(scale=1):
                ticket = gr.Textbox(
                    lines=8,
                    label="1. Descreva o problema",
                    placeholder="Ex.: A VPN conecta, mas não consigo abrir o sistema interno.",
                )
                with gr.Row():
                    analyze = gr.Button("Analisar chamado", variant="primary")
                    clear = gr.ClearButton([ticket], value="Limpar")
            with gr.Column(scale=1):
                result = gr.Code(
                    label="2. Resultado estruturado",
                    language="json",
                    lines=12,
                    interactive=False,
                )
                gr.Markdown("A resposta contém categoria, prioridade e próximo passo.")
        analyze.click(fn=answer, inputs=ticket, outputs=result)
        ticket.submit(fn=answer, inputs=ticket, outputs=result)
        clear.add(result)
    demo.launch()


if __name__ == "__main__":
    main()
