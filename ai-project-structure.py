# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class VisualizationManager:
    @staticmethod
    def plot_training_metrics(history: dict):
        """
        Plot training and validation metrics over epochs
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Accuracy', 'Precision', 'Recall')
        )

        # Plot Loss
        fig.add_trace(
            go.Scatter(y=history['train_loss'], name='Train Loss'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_loss'], name='Val Loss'),
            row=1, col=1
        )

        # Plot Accuracy
        fig.add_trace(
            go.Scatter(y=history['train_acc'], name='Train Accuracy'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(y=history['val_acc'], name='Val Accuracy'),
            row=1, col=2
        )

        # Plot Precision
        fig.add_trace(
            go.Scatter(y=history['train_precision'], name='Train Precision'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(y=history['val_precision'], name='Val Precision'),
            row=2, col=1
        )

        # Plot Recall
        fig.add_trace(
            go.Scatter(y=history['train_recall'], name='Train Recall'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(y=history['val_recall'], name='Val Recall'),
            row=2, col=2
        )

        fig.update_layout(height=800, title_text="Training Metrics")
        fig.show()

    @staticmethod
    def plot_attention_weights(attention_weights, text_tokens):
        """
        Plot attention weights from transformer models
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_weights,
            xticklabels=text_tokens,
            yticklabels=text_tokens,
            cmap='YlOrRd'
        )
        plt.title('Transformer Attention Weights')
        plt.show()

# llm_model.py
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
from typing import Dict, List

class TransformerModel(nn.Module):
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_labels: int = 2,
        fine_tune: bool = True
    ):
        super(TransformerModel, self).__init__()
        self.transformer = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze transformer parameters if not fine-tuning
        if not fine_tune:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Dict:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True
        )
        
        return {
            'logits': outputs.logits,
            'loss': outputs.loss if labels is not None else None,
            'attention': outputs.attentions
        }
    
    def get_attention_weights(
        self,
        text: str,
        layer: int = -1
    ) -> tuple:
        """
        Get attention weights for visualization
        """
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        
        with torch.no_grad():
            outputs = self.transformer(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                output_attentions=True
            )
        
        # Get attention weights from specified layer
        attention = outputs.attentions[layer][0]
        
        # Average attention weights across heads
        attention = attention.mean(dim=0)
        
        # Get tokens for visualization
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        return attention.numpy(), tokens

# main.py
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score

def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 2e-5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs['logits'], dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, labels)
                val_loss += outputs['loss'].item()
                preds = torch.argmax(outputs['logits'], dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        
        train_precision = precision_score(train_labels, train_preds, average='weighted')
        val_precision = precision_score(val_labels, val_preds, average='weighted')
        
        train_recall = recall_score(train_labels, train_preds, average='weighted')
        val_recall = recall_score(val_labels, val_preds, average='weighted')
        
        # Update history
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    return model, history

# Example usage
def main():
    # Initialize model
    model = TransformerModel(
        model_name='bert-base-uncased',
        num_labels=2,
        fine_tune=True
    )
    
    # Create visualization manager
    viz_manager = VisualizationManager()
    
    # Train model (assuming you have your DataLoader setup)
    trained_model, history = train_model(model, train_loader, val_loader)
    
    # Plot training metrics
    viz_manager.plot_training_metrics(history)
    
    # Example of attention visualization
    sample_text = "This is an example sentence for attention visualization."
    attention_weights, tokens = trained_model.get_attention_weights(sample_text)
    viz_manager.plot_attention_weights(attention_weights, tokens)

if __name__ == "__main__":
    main()
