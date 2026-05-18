"""Transformer classifier variants for ablations and final comparison."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, BertModel, BertPreTrainedModel, DistilBertModel, DistilBertPreTrainedModel
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


class BertCustomClassifier(BertPreTrainedModel):
    """BERT encoder with a configurable MLP classification head."""

    def __init__(
        self,
        config,
        classifier_hidden_sizes: Sequence[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.classifier_hidden_sizes = list(classifier_hidden_sizes or [])
        self.dropout_rate = dropout

        hidden_size = config.hidden_size
        layers: list[nn.Module] = []
        if self.classifier_hidden_sizes:
            input_size = hidden_size
            for output_size in self.classifier_hidden_sizes:
                layers.extend([nn.Linear(input_size, output_size), nn.ReLU(), nn.Dropout(dropout)])
                input_size = output_size
            layers.append(nn.Linear(input_size, self.num_labels))
        else:
            layers.extend(
                [
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
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """Run the encoder and classifier, returning Trainer-compatible outputs."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        pooled_output = outputs.pooler_output
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
    """Freeze all encoder parameters for BERT or DistilBERT models."""
    encoder = get_encoder(model)
    for parameter in encoder.parameters():
        parameter.requires_grad = False


def freeze_embeddings(model: nn.Module) -> None:
    """Freeze embedding parameters for BERT or DistilBERT models."""
    encoder = get_encoder(model)
    for parameter in encoder.embeddings.parameters():
        parameter.requires_grad = False


def freeze_lower_transformer_layers(model: nn.Module, n_layers: int) -> None:
    """Freeze the first n transformer layers for BERT or DistilBERT models."""
    layers = get_transformer_layers(model)
    for layer in layers[:n_layers]:
        for parameter in layer.parameters():
            parameter.requires_grad = False


def get_encoder(model: nn.Module) -> nn.Module:
    """Return the encoder module for supported BERT-family models."""
    if hasattr(model, "distilbert"):
        return model.distilbert
    if hasattr(model, "bert"):
        return model.bert
    raise ValueError("Unsupported model: expected a .distilbert or .bert encoder.")


def get_transformer_layers(model: nn.Module):
    """Return transformer layers for supported BERT-family models."""
    encoder = get_encoder(model)
    if hasattr(encoder, "transformer"):
        return encoder.transformer.layer
    if hasattr(encoder, "encoder"):
        return encoder.encoder.layer
    raise ValueError("Unsupported encoder: no transformer layer stack found.")


def build_model(
    model_name: str,
    num_labels: int,
    classifier_hidden_sizes: Sequence[int] | None = None,
    dropout: float = 0.2,
    freeze_mode: str = "none",
    num_frozen_layers: int = 0,
    freeze_embeddings_flag: bool = False,
) -> nn.Module:
    """Build a BERT or DistilBERT classifier and apply requested freezing."""
    hidden_sizes = list(classifier_hidden_sizes or [])
    if "distilbert" in model_name:
        if hidden_sizes or freeze_mode != "none" or freeze_embeddings_flag:
            model = DistilBertCustomClassifier.from_pretrained(
                model_name,
                num_labels=num_labels,
                classifier_hidden_sizes=hidden_sizes,
                dropout=dropout,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    elif "bert" in model_name:
        if hidden_sizes or freeze_mode != "none" or freeze_embeddings_flag:
            model = BertCustomClassifier.from_pretrained(
                model_name,
                num_labels=num_labels,
                classifier_hidden_sizes=hidden_sizes,
                dropout=dropout,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if freeze_embeddings_flag:
        freeze_embeddings(model)
    if freeze_mode == "all":
        freeze_entire_transformer(model)
    elif freeze_mode == "lower_layers":
        freeze_lower_transformer_layers(model, num_frozen_layers)
    elif freeze_mode != "none":
        raise ValueError(f"Unsupported freeze_mode: {freeze_mode}")

    return model


def count_total_parameters(model: nn.Module) -> int:
    """Count all parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
