import importlib.util
import pathlib
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("ai_project", ROOT / "ai-project-structure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _ToyData(Dataset):
    """Dataset pequeno e determinístico no formato esperado por train_model."""

    def __init__(self, n=16):
        g = torch.Generator().manual_seed(0)
        self.ids = torch.randint(0, 20, (n, 6), generator=g)
        self.labels = (self.ids[:, 0] % 2).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": torch.ones(6, dtype=torch.long), "labels": self.labels[i]}


class _TinyClassifier(nn.Module):
    """Modelo real (treinável) com a mesma interface de TransformerModel.forward."""

    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(20, 8)
        self.fc = nn.Linear(8, 2)

    def forward(self, input_ids, attention_mask, labels=None):
        logits = self.fc(self.emb(input_ids))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss, "attention": None}


def test_main_sem_dataloaders_falha_com_mensagem_clara():
    # Antes: NameError (train_loader/val_loader indefinidos)
    with pytest.raises(ValueError, match="train_loader"):
        _load().main()


def test_train_model_executa_e_preenche_historico():
    mod = _load()
    torch.manual_seed(0)
    loader = DataLoader(_ToyData(), batch_size=4)
    _, hist = mod.train_model(_TinyClassifier(), loader, loader, num_epochs=2, learning_rate=0.05, device="cpu")
    for k in ("train_loss", "val_loss", "train_acc", "val_acc", "train_precision", "val_precision"):
        assert len(hist[k]) == 2
    assert all(0.0 <= a <= 1.0 for a in hist["val_acc"])


def test_plot_training_metrics_monta_4_paineis():
    mod = _load()
    hist = {k: [0.5, 0.4] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "train_precision",
                                     "val_precision", "train_recall", "val_recall")}
    with patch("plotly.graph_objects.Figure.show") as show:
        mod.VisualizationManager.plot_training_metrics(hist)
        show.assert_called_once()


@pytest.mark.skipif(
    __import__("os").getenv("RUN_HF_TESTS") != "1",
    reason="baixa bert-base-uncased (~440 MB) do Hugging Face; rode com RUN_HF_TESTS=1",
)
def test_pesos_de_atencao_bert():
    # Antes: IndexError (sdpa não devolve atenções no transformers >= 5)
    attn, tokens = _load().TransformerModel().get_attention_weights("This is an example sentence.")
    assert attn.shape == (len(tokens), len(tokens))
