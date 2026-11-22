# Auditoria — Integrating-AI-Models-for-Advanced-Image-Processing — 2026-11-22

## Estado anterior
- `ai-project-structure.py` (3 arquivos concatenados), `visualization-component.tsx`, `requirements.txt`, README. Sem testes, sem CI (`gh run list` vazio).
- O README descreve um sistema multimodal com CNN e visão computacional, a estrutura `src/...` e os comandos `--train/--predict`: nada disso existe. O código é classificação de texto com BERT.
- O componente TSX exibe métricas fixas no código (dados de exemplo apresentados como "Training Metrics").

## Problemas encontrados

| # | Sev. | Problema | Evidência |
|---|---|---|---|
| 1 | P0 | `main()` usa `train_loader`/`val_loader` indefinidos → `NameError` (depois de baixar o BERT) | `ruff`: `F821 Undefined name train_loader/val_loader`; o teste `test_main_sem_dataloaders...` falhou no código antigo |
| 2 | P0 | Pisos de versão vulneráveis (`torch>=1.8.0`, `transformers>=4.30.0`, `Pillow>=9.5.0`...) permitiam instalar versões com CVEs | `pip-audit --no-deps` nas versões mínimas: 31 avisos para torch, 49 para transformers, 71 para pillow |
| 3 | P1 | `get_attention_weights` quebra no transformers ≥ 5: a atenção `sdpa` não devolve pesos → `IndexError: tuple index out of range` | execução real com transformers 5.24.0 e `bert-base-uncased` |
| 4 | P1 | `plotly` é usado, mas falta no requirements | `import plotly` no código |
| 5 | P2 | Imports não usados; ferramentas de dev misturadas às dependências; dependências não usadas (opencv, pillow, pandas, sentencepiece, python-dotenv) | ruff F401 / grep |
| 6 | P2 | README e nome do repositório não correspondem ao código; TSX com dados fixos | comparação |

Segredos: nenhum (busca por regex em todo o histórico; gitleaks não está instalado). O README tinha o e-mail pessoal do autor (não é segredo); removido do README por precaução, mantido o LinkedIn.

## Mudanças
- `main(train_loader, val_loader)` lança `ValueError` com mensagem clara; a execução como script termina com essa mensagem, sem traceback.
- `attn_implementation='eager'` em `from_pretrained` (necessário para `output_attentions`).
- `requirements.txt` só com as dependências usadas e pisos sem CVEs conhecidas (`torch>=2.6`, `transformers>=4.53`, `tqdm>=4.66.3`); `requirements-dev.txt`; `ruff.toml`; `.gitignore`.
- 4 testes (`tests/test_ai_project.py`): `main` sem loaders; `train_model` com um modelo PyTorch pequeno **real** (treina 2 épocas em dados determinísticos); `plot_training_metrics` (só `fig.show` é substituído); atenção com BERT real (só roda com `RUN_HF_TESTS=1`).
- CI (ruff, compileall, pytest; PyTorch CPU; `contents: read`; actions fixadas por SHA). README reescrito.

## Testes executados (Windows, Python 3.12, torch 2.15.0+cpu, transformers 5.24.0)
- Antes: `ruff check` → 7 erros (2× F821); teste de `main` → 1 failed.
- Depois: `RUN_HF_TESTS=1 pytest -q tests` → **4 passed**; `ruff check .` OK; `pip-audit -l` → nenhuma vulnerabilidade conhecida (torch/torchvision CPU não puderam ser auditados porque a versão `+cpu` não existe no PyPI).
- `get_attention_weights("This is an example sentence.")` → matriz 8×8, 8 tokens.

## Não executado / riscos residuais
- Treino real do BERT: não há dataset no repositório. Treino em GPU não testado (a validação usou só CPU).
- `visualization-component.tsx`: não há `package.json` nem bundler; não foi compilado nem testado.
- Decisão do dono: renomear/redescrever o repositório (ele não trata imagens) ou implementar a parte de visão. Não há arquivo LICENSE.
