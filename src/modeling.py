"""DistilBERT model variants for the Day 3 ablation study."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from transformers import DistilBertModel, DistilBertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class DistilBertCustomClassifier(DistilBertPreTrainedModel):
    """DistilBERT encoder with a configurable MLP classification head."""

    def __init__(
        self,
        config,
        classifier_hidden_sizes: Sequence[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.classifier_hidden_sizes = list(classifier_hidden_sizes or [])
        self.dropout_rate = dropout

        hidden_size = config.dim
        layers: list[nn.Module] = []

        if self.classifier_hidden_sizes:
            input_size = hidden_size
            for output_size in self.classifier_hidden_sizes:
                layers.extend(
                    [
                        nn.Linear(input_size, output_size),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                input_size = output_size
            layers.append(nn.Linear(input_size, self.num_labels))
        else:
            layers.extend(
                [
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, self.num_labels),
                ]
            )

        self.classifier = nn.Sequential(*layers)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """Run the encoder and classifier, returning Trainer-compatible outputs."""
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def freeze_entire_transformer(model: nn.Module) -> None:
    """Freeze all DistilBERT transformer parameters."""
    for parameter in model.distilbert.parameters():
        parameter.requires_grad = False


def freeze_embeddings(model: nn.Module) -> None:
    """Freeze DistilBERT embedding parameters."""
    for parameter in model.distilbert.embeddings.parameters():
        parameter.requires_grad = False


def freeze_lower_transformer_layers(model: nn.Module, n_layers: int) -> None:
    """Freeze the first n DistilBERT transformer layers."""
    layers = model.distilbert.transformer.layer
    for layer in layers[:n_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def count_total_parameters(model: nn.Module) -> int:
    """Count all parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
