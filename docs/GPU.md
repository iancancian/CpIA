# Evidencia de GPU

## O que foi verificado

Comando executado no Windows:

```powershell
nvidia-smi
```

Resultado observado em 15/10/2026:

- GPU: NVIDIA GeForce GTX 1060 6GB
- Driver: 582.28
- CUDA reportado pelo driver: 13.0
- Memoria em uso no momento da verificacao: 1830 MiB de 6144 MiB

## Print no VS Code

1. Abra a paleta de comandos e execute `Tasks: Run Task`.
2. Escolha `GPU: diagnostico`.
3. Deixe o painel Terminal visivel com a saida de `nvidia-smi` e `cuda_disponivel: True`.
4. Capture esse painel para anexar ao relatorio.

O resultado e uma evidencia reproduzivel e nao apenas uma leitura visual do monitor.

## CUDA

O driver 582.28 aceita executaveis compilados para CUDA 12.x. O projeto instala o PyTorch com wheel `cu124`, que e a combinacao pratica para Windows e Python 3.12 neste hardware. A versao exibida pelo driver (13.0) nao precisa ser igual a versao embutida no wheel do PyTorch.
