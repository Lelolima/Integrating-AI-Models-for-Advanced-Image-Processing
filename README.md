# Integrating AI Models for Advanced Image Processing

> ⚠️ **Estado real (auditoria de 2026-11-22):** apesar do nome, **não há processamento de imagens nem CNN**
> no código. O que existe é um classificador de **texto** baseado em BERT (Hugging Face) com loop de treino
> e visualizações de métricas e de pesos de atenção. Não há dataset, DataLoaders, modelos treinados nem
> resultados de performance no repositório.

## Conteúdo

| Arquivo | O que é |
|---|---|
| `ai-project-structure.py` | Três módulos concatenados em um arquivo: `VisualizationManager` (Plotly/Seaborn), `TransformerModel` (`AutoModelForSequenceClassification`, padrão `bert-base-uncased`), `train_model` (loop de treino/validação com acurácia, precisão e recall) e `main(train_loader, val_loader)` |
| `visualization-component.tsx` | Componente React avulso (Recharts + classes Tailwind) que plota curvas de loss/acurácia com **valores fixos de exemplo** escritos no código. Não há `package.json` nem build para ele. |
| `requirements.txt` | Dependências Python usadas pelo código |

A estrutura `src/data`, `src/models`, `src/utils` e os comandos `--train`/`--predict` descritos em versões
anteriores do README não existem.

## Instalação

Python 3.10+ (testado com 3.12, PyTorch CPU).

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
```

## Uso

O script não roda sozinho de forma útil: `main()` exige DataLoaders que produzam dicionários com
`input_ids`, `attention_mask` e `labels`. Sem eles, `python ai-project-structure.py` termina com uma mensagem
de erro explicando isso. Exemplo de uso como biblioteca:

```python
import importlib.util
spec = importlib.util.spec_from_file_location("ai", "ai-project-structure.py")
ai = importlib.util.module_from_spec(spec); spec.loader.exec_module(ai)
ai.main(train_loader, val_loader)   # seus DataLoaders
```

Na primeira execução, `TransformerModel` baixa `bert-base-uncased` (~440 MB) do Hugging Face (gratuito).
O treino usa GPU se houver (`cuda`); se não houver, usa CPU.

Não há variáveis de ambiente nem segredos.

## Testes e lint

```bash
ruff check .
pytest -q tests                    # sem rede: loop de treino com modelo pequeno real e visualização
RUN_HF_TESTS=1 pytest -q tests     # inclui teste com BERT real (baixa o modelo)
```

CI em `.github/workflows/ci.yml` (lint, compilação e testes, sem baixar o BERT).

## Contato

Wellington de Lima Catarina — LinkedIn: wellington-de-lima-catarina

## Licença

O README antigo citava `LICENSE.md`, mas o arquivo não existe no repositório.
