# Projeto de LLM Local com GPU

## Grupo

- Acussena Macedo Mautone - 552568
- Felipe Heilmann Marques - 551026
- Felipe Voidela Toledo - 98595
- Carlos Eduardo Caramante Ribeiro - 552159
- Ian Cancian Nachtergaele - 98387

## 1. Objetivo do projeto

Este projeto treina e avalia uma LLM local para resolver um problema especifico de suporte de TI.

Dado um chamado escrito em portugues, o modelo deve retornar uma resposta estruturada em JSON com:

- `categoria`: rede, acesso, hardware, software ou seguranca;
- `prioridade`: baixa, media, alta ou critica;
- `proximo_passo`: orientacao pratica para o usuario.

Exemplo de problema:

```text
Meu segundo monitor nao mostra imagem pelo HDMI.
```

Resposta esperada:

```json
{
  "categoria": "hardware",
  "prioridade": "media",
  "proximo_passo": "Teste outro cabo e outra porta HDMI."
}
```

## 2. Modelo escolhido

- Modelo base: `Qwen/Qwen2.5-1.5B-Instruct`;
- Tecnica de treinamento: LoRA (Low-Rank Adaptation);
- Frameworks: PyTorch, Transformers, PEFT e Datasets;
- Motivo da escolha: modelo pequeno o suficiente para uma GPU de 6 GB e adequado para gerar respostas em portugues;
- Pesos do modelo: baixados automaticamente do Hugging Face durante a primeira execucao.

O projeto treina somente os adaptadores LoRA. Os pesos originais do modelo nao sao modificados.

## 3. Hardware e CUDA

Hardware utilizado:

- GPU: NVIDIA GeForce GTX 1060 6 GB;
- Driver NVIDIA: 582.28;
- CUDA reportada pelo driver: 13.0;
- PyTorch instalado: `2.5.1+cu124`;
- CUDA usada pelo PyTorch: 12.4;
- Python: 3.12.

A versao CUDA do driver pode ser maior que a versao CUDA incluida na wheel do PyTorch. O driver 13.0 e retrocompativel com a wheel `cu124` usada neste projeto.

O treinamento e a avaliacao foram executados com GPU. Os scripts interrompem o processo se CUDA nao estiver disponivel.

## 4. Estrutura do repositorio

```text
data/train.jsonl                 Dados usados no treinamento
data/eval.jsonl                  Dados separados para avaliacao
src/train.py                     Fine-tuning LoRA
src/evaluate.py                  Metricas e respostas do modelo
src/app.py                       Frontend Gradio
scripts/check_gpu.py             Diagnostico da GPU
.vscode/tasks.json               Tarefas do VS Code
docs/GPU.md                      Evidencias e instrucoes da GPU
requirements.txt                 Dependencias fixadas
.gitignore                       Arquivos que nao devem ir para o GitHub
```

## 5. Instalar o projeto

Deve criar o seu proprio ambiente.

No PowerShell:

```powershell
cd "C:\caminho\do\repositorio"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativacao:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

A GPU deve ser verificada antes do treinamento:

```powershell
python scripts/check_gpu.py
```

A saida precisa conter:

```text
cuda_disponivel: True
gpu: NVIDIA GeForce GTX 1060 6GB
```

## 6. Dados do treinamento

Os dados estao em `data/train.jsonl`. Cada linha possui um chamado e uma resposta JSON esperada.

O conjunto atual possui 8 exemplos, cobrindo rede, seguranca, hardware, software e acesso.

Os dados de avaliacao estao separados em `data/eval.jsonl` e nao sao usados para atualizar os pesos durante o treinamento.

## 7. Teste de parametros

Foram testadas duas configuracoes, sempre utilizando a GPU:

| Configuracao | Epocas | Learning rate | Acumulacao | Resultado |
|---|---:|---:|---:|---|
| Experimento 1 | 3 | `0.0002` | 8 | Nao manteve o JSON no teste |
| Experimento 2 | 10 | `0.0005` | 1 | Melhor resultado |

Para executar o melhor experimento:

```powershell
python src/train.py `
  --epochs 10 `
  --learning-rate 0.0005 `
  --gradient-accumulation-steps 1 `
  --output-dir outputs/lora-10ep-lr5e-4
```

No PowerShell, o caractere `` ` `` continua o comando na linha seguinte. Tambem e possivel executar em uma linha:

```powershell
python src/train.py --epochs 10 --learning-rate 0.0005 --gradient-accumulation-steps 1 --output-dir outputs/lora-10ep-lr5e-4
```

O treinamento salva o adaptador LoRA em `outputs/lora-10ep-lr5e-4`.

## 8. Avaliacao do modelo

A avaliacao verifica as respostas produzidas para os 5 exemplos de `data/eval.jsonl`.

Metricas utilizadas:

- Acuracia de categoria;
- Acuracia de prioridade;
- Taxa de respostas com os tres campos obrigatorios;
- ROUGE-L entre a orientacao gerada e a referencia.

Executar avaliacao:

```powershell
python src/evaluate.py `
  --adapter outputs/lora-10ep-lr5e-4 `
  --output outputs/evaluation-10ep-lr5e-4.json
```

Resultado obtido:

- Categoria: `100%`;
- Prioridade: `60%`;
- Campos JSON obrigatorios: `100%`;
- ROUGE-L medio: `0,2613`.

O resultado mostra que o modelo aprendeu o formato e as categorias. A prioridade e o texto da orientacao ainda podem melhorar com mais exemplos de treinamento.

## 9. Frontend

O frontend permite que o usuario digite um novo chamado e receba a resposta do modelo.

Iniciar:

```powershell
python src/app.py --adapter outputs/lora-10ep-lr5e-4
```

Abrir no navegador:

```text
http://127.0.0.1:7860
```

Funcionalidades:

- Campo para inserir o chamado;
- Botao `Analisar chamado`;
- Botao `Limpar`;
- Envio pelo botao ou pela tecla Enter;
- Resposta exibida em formato JSON;
- Modelo carregado localmente na GPU NVIDIA.


## 11. Tarefas do VS Code

O arquivo [.vscode/tasks.json](.vscode/tasks.json) possui tarefas para:

- Diagnosticar a GPU;
- Treinar o modelo;
- Avaliar o modelo;
- Iniciar o frontend.

## 12. GitHub

O repositorio deve conter codigo, dados de exemplo, documentacao e configuracoes. O `.gitignore` impede o envio de:

- `.venv/`;
- credenciais e arquivos `.env`;
- tokens e certificados;
- pesos e checkpoints;
- caches e logs.

Criar o repositorio e publicar:

```powershell
git init
git add .
git status
git commit -m "feat: projeto de fine tuning local com GPU"
git branch -M main
git remote add origin https://github.com/iancancian/CpIA.git
git push -u origin main
```

Depois de publicado, o professor pode executar:

```powershell
git clone https://github.com/iancancian/CpIA.git
cd SEU_REPOSITORIO
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/check_gpu.py
```

Depois da verificacao da GPU, ele pode treinar, avaliar e abrir o frontend seguindo as secoes acima.

## 13. Limitacoes

- O conjunto de treino e pequeno e serve como demonstracao academica;
- O resultado nao representa um modelo pronto para uso em producao;
- O professor precisa de uma GPU NVIDIA compativel para repetir o treinamento;
- Os pesos treinados nao sao enviados ao GitHub por causa do tamanho;
- A primeira execucao precisa baixar aproximadamente 3 GB do modelo base.

