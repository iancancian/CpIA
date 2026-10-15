Grupo 4espy:
Açussena Macedo Mautone - 552568
Felipe Heilmann Marques - 551026
Felipe Voidela Toledo - 98595
Carlos Eduardo Caramante Ribeiro - 552159
Ian Cancian Nachtergaele - 98387

# Assistente Local de Suporte de TI

Projeto de fine-tuning de uma LLM local para classificar chamados de suporte em portugues.

## Modelo e tarefa

- Modelo: `Qwen/Qwen2.5-1.5B-Instruct`
- Tecnica: LoRA
- GPU: NVIDIA GeForce GTX 1060 6 GB
- Saida: JSON com `categoria`, `prioridade` e `proximo_passo`
- Categorias: rede, acesso, hardware, software e seguranca

## Instalar

PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/check_gpu.py
```

O diagnostico deve mostrar `cuda_disponivel: True`.

## Treinar e avaliar

```powershell
python src/train.py --epochs 10 --learning-rate 0.0005 --gradient-accumulation-steps 1 --output-dir outputs/lora-10ep-lr5e-4
python src/evaluate.py --adapter outputs/lora-10ep-lr5e-4 --output outputs/evaluation-10ep-lr5e-4.json
```

O treino usa [data/train.jsonl](data/train.jsonl). A avaliacao usa [data/eval.jsonl](data/eval.jsonl).

## Resultado

Melhor configuracao testada:

- 10 epocas
- Learning rate: `0.0005`
- Categoria: `100%`
- Prioridade: `60%`
- JSON completo: `100%`
- ROUGE-L medio: `0,2613`

O conjunto tem apenas 8 exemplos de treino. Para maior confiabilidade, aumente os dados.

## Frontend

```powershell
python src/app.py --adapter outputs/lora-10ep-lr5e-4
```

Acesse `http://127.0.0.1:7860` e insira um chamado.

## VS Code e GPU

As tarefas estao em [.vscode/tasks.json](.vscode/tasks.json). A documentacao esta em [docs/GPU.md](docs/GPU.md).

Extensoes recomendadas:

- GPU Environment
- GPU Monitor

Para o print, execute a tarefa **GPU: diagnostico** e capture o terminal com `nvidia-smi` e `cuda_disponivel: True`.

## Publicar no GitHub

```powershell
git init
git add .
git commit -m "feat: projeto de fine tuning local com GPU"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

