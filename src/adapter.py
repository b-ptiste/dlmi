####################################
# credit : adapted from TP DLMI 2024.
####################################

# standard libraries
from __future__ import annotations
from typing import Any

# third-party libraries
import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    def __init__(
        self, in_dim: int, activation: str = "ReLU", reduction_factor: int = 16
    ) -> None:
        super(BottleneckAdapter, self).__init__()
        hidden_dim = in_dim // reduction_factor
        self.adapter_downsample = nn.Linear(in_dim, hidden_dim)
        self.activation = getattr(nn, activation)()
        self.adapter_upsample = nn.Linear(hidden_dim, in_dim)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.adapter_downsample(x_in)
        x = self.activation(x)
        x = self.adapter_upsample(x)
        return x + x_in


def add_bottleneck_adapter(model: nn.Module) -> None:
    """Add a bottleneck adapter to the model

    Args:
        model (nn.Module): The model to which the bottleneck adapter will be added
    """
    for block in model.blocks:
        block.attn.proj = nn.Sequential(
            block.attn.proj, BottleneckAdapter(block.attn.proj.in_features)
        )
        block.mlp = nn.Sequential(
            block.mlp, BottleneckAdapter(block.mlp.fc1.in_features)
        )


def freeze_model_bottleneck(model: nn.Module) -> None:
    """Freeze the model's parameters except for the bottleneck adapter

    Args:
        model (nn.Module): The model to freeze
    """
    for name, param in model.named_parameters():
        if not ("adapter" in name or "norm1" in name or "norm2" in name):
            param.requires_grad = False


class AdaptFormer(nn.Module):
    def __init__(
        self,
        layer_norm: Any,
        linear: Any,
        activation: str = "ReLU",
        reduction_factor: int = 8,
    ) -> None:
        super(AdaptFormer, self).__init__()
        self.layer_norm = layer_norm
        self.linear = linear
        self.adapter_alpha = nn.Parameter(torch.ones(1))

        hidden_dim = self.linear.fc1.in_features // reduction_factor
        self.adapter_downsample = nn.Linear(self.linear.fc1.in_features, hidden_dim)
        self.activation = getattr(nn, activation)()
        self.adapter_upsample = nn.Linear(hidden_dim, self.linear.fc1.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main_x = self.linear(self.layer_norm(x))
        adapted_x = (
            self.adapter_upsample(self.activation(self.adapter_downsample(x)))
            * self.adapter_alpha
        )
        return main_x + adapted_x


def add_adaptformer(model: nn.Module) -> None:
    """Add an AdaptFormer to the model

    Args:
        model (nn.Module): The model to which the AdaptFormer will be added
    """
    for encoder_block in model.blocks:
        encoder_block.mlp = AdaptFormer(encoder_block.norm2, encoder_block.mlp)
        encoder_block.norm2 = nn.Identity()


def freeze_model_adaptformer(model: nn.Module) -> None:
    """Freeze the model's parameters except for the AdaptFormer

    Args:
        model (nn.Module): The model to freeze
    """
    for name, param in model.named_parameters():
        if not "adapter" in name:
            param.requires_grad = False


class LoRA(nn.Module):
    def __init__(
        self, linear_layer: Any, in_dim: int, rank: int = 32, alpha: int = 16
    ) -> None:
        super(LoRA, self).__init__()
        self.linear_layer = linear_layer
        std = 1 / torch.sqrt(torch.tensor(rank).float())
        self.adapter_Q_downsample = nn.Parameter(torch.randn(in_dim, rank) * std)
        self.adapter_Q_upsample = nn.Parameter(torch.zeros(rank, in_dim))
        self.adapter_V_downsample = nn.Parameter(torch.randn(in_dim, rank) * std)
        self.adapter_V_upsample = nn.Parameter(torch.zeros(rank, in_dim))
        self.adapter_alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.adapter_alpha * (
            x @ self.adapter_Q_downsample @ self.adapter_Q_upsample
        )
        x_v = self.adapter_alpha * (
            x @ self.adapter_V_downsample @ self.adapter_V_upsample
        )
        x_lora = torch.cat([x_q, torch.zeros_like(x_v), x_v], dim=-1)
        x = self.linear_layer(x) + x_lora
        return x


def add_lora(model: nn.Module) -> None:
    """Add a LoRA to the model

    Args:
        model (nn.Module): The model to which the LoRA will be added
    """
    for encoder_block in model.blocks:
        encoder_block.attn.qkv = LoRA(
            encoder_block.attn.qkv, encoder_block.attn.qkv.in_features
        )


def freeze_model_lora(model: nn.Module) -> None:
    """Freeze the model's parameters except for the LoRA

    Args:
        model (nn.Module): The model to freeze
    """
    for name, param in model.named_parameters():
        if not "adapter" in name:
            param.requires_grad = False


class PromptTuning(nn.Module):
    def __init__(self, num_tokens: int, token_dim: int = 384) -> None:
        super(PromptTuning, self).__init__()
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.adapter_prompts = nn.Parameter(torch.randn(num_tokens, token_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [self.adapter_prompts.unsqueeze(0).repeat(x.shape[0], 1, 1), x], dim=1
        )
        return x


def add_prompttuning(
    model: nn.Module, all_layers: bool = True, num_tokens: int = 128
) -> None:
    """Add a PromptTuning to the model

    Args:
        model (nn.Module): The model to which the PromptTuning will be added
        all_layers (bool, optional): condition to add PromptTuning to all layers. Defaults to True.
        num_tokens (int, optional): number of tokens. Defaults to 128.
    """
    num_layers = len(model.blocks) - 1 if all_layers else 1
    for block_id in range(num_layers):
        model.blocks[block_id] = nn.Sequential(
            PromptTuning(num_tokens), model.blocks[block_id]
        )


def freeze_model_prompttuning(model: nn.Module) -> None:
    """Freeze the model's parameters except for the PromptTuning

    Args:
        model (nn.Module): The model to freeze
    """
    for name, param in model.named_parameters():
        if not "adapter" in name:
            param.requires_grad = False
