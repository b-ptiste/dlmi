# third party libraries
import timm
import torch
import torch.nn as nn

# local libraries
from .utils import aggregation

from .adapter import (
    add_bottleneck_adapter,
    freeze_model_bottleneck,
    add_adaptformer,
    freeze_model_adaptformer,
    add_lora,
    freeze_model_lora,
    add_prompttuning,
    freeze_model_prompttuning,
)


class ModelFactory:
    """
    Class to create and get all the models
    """

    def __init__(self) -> None:
        pass

    def __call__(self, cfg: dict) -> nn.Module:
        model_name = cfg["model_name"]

        # print the configuration
        print("The configuration is:")
        for k, v in cfg.items():
            print(f"{k} : {v}")

        # load the model if registered
        if cfg["model_name"] == "PatientModel":
            print(f"Loading custom model {cfg['model_name']}")
            return PatientModel(cfg)

        elif cfg["model_name"] == "PatientModelAttention":
            print(f"Loading custom model {cfg['model_name']}")
            return PatientModelAttention(cfg)

        elif cfg["model_name"] == "PatientModelAttentionTab":
            print(f"Loading custom model {cfg['model_name']}")
            return PatientModelAttentionTab(cfg)

        elif cfg["timm"]:
            print(f"Loading timm model {cfg['timm_model']}")
            model = build_timm(cfg)
            add_adapter(model, cfg["adapter"])
            return model

        elif cfg["dino"]:
            print(f"Loading dino model {cfg['dino_size']}")
            model = build_dino(cfg["dino_size"])
            model.head = nn.Linear(cfg["feature_dim"], cfg["nb_class"])
            add_adapter(model, cfg["adapter"])
            return model
        else:
            raise NotImplemented(f"{model_name} don't register")


class PatientModel(nn.Module):
    """
    Custom model for the patient level
    """

    def __init__(self, cfg: dict) -> None:
        super(PatientModel, self).__init__()

        # variable definition
        self.aggregation = cfg["aggregation"]
        self.device_1 = cfg["device_1"]
        self.sub_batch_size = cfg["sub_batch_size"]

        # pick the right encoder model
        if cfg["timm"]:
            self.model = build_timm(cfg)
            
        elif cfg["dino"]:
            self.model = build_dino(cfg["dino_size"])
            self.model.head = nn.Linear(cfg["feature_dim"], cfg["nb_class"])

        # load a pretrained model
        if len(cfg["pretrained_path"]) > 0:
            self.model.load_state_dict(
                torch.load(cfg["pretrained_path"])["model_state_dict"]
            )

        # add adapters
        add_adapter(self.model, cfg["adapter"])

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        xout_sub_batch = torch.zeros((x.size(0), 2)).to(self.device_1)

        # loop over the data in sub-batches as the model is too big
        # it corresponds to gradient accumulation
        
        for i in range(0, x.size(0) // self.sub_batch_size + 1):
            start_idx = i * self.sub_batch_size
            end_idx = min(start_idx + self.sub_batch_size, x.size(0))
            
            if mode == "val":
                # we don't need to compute the gradient
                with torch.no_grad():
                    # drop edge case
                    if start_idx != end_idx:
                        x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                        xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            elif mode == "train":
                # drop edge case
                if start_idx != end_idx:
                    x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                    xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            else:
                raise f"Mode {mode} not implemented"

        xout_sub_batch = aggregation(xout_sub_batch, self.aggregation)

        return xout_sub_batch


class PatientModelAttention(nn.Module):
    """
    Custom model for the patient level with attention
    """

    def __init__(self, cfg: dict) -> None:
        super(PatientModelAttention, self).__init__()

        # variable definition
        self.device_1 = cfg["device_1"]
        self.latent_att = cfg["latent_att"]
        self.sub_batch_size = cfg["sub_batch_size"]
        self.aggregation = cfg["aggregation"]

        # pick the right encoder model
        if cfg["timm"]:
            self.model = build_timm(cfg)
            
        elif cfg["dino"]:
            self.model = build_dino(cfg["dino_size"])
            self.model.head = nn.Linear(cfg["feature_dim"], cfg["nb_class"])

        # load a pretrained model
        if len(cfg["pretrained_path"]) > 0:
            print(f"We load the weigths {cfg['pretrained_path']}")
            self.model.load_state_dict(
                torch.load(cfg["pretrained_path"])["model_state_dict"]
            )
        else:
            print("The training is from scatch")

        # add adapters
        add_adapter(self.model, cfg["adapter"])

        # we put it after loading the pretrained
        self.model.head = nn.Linear(cfg["feature_dim"], cfg["latent_att"])
        
        # multihead attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=cfg["latent_att"], num_heads=cfg["head"]
        )
        
        # classifier
        self.proj_1 = nn.Linear(cfg["latent_att"], cfg["latent_att"])
        self.proj_2 = nn.Linear(cfg["latent_att"], 2)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:

        # loop over the data in sub-batches as the model is too big
        # it corresponds to gradient accumulation
        
        xout_sub_batch = torch.zeros((x.size(0), self.latent_att)).to(self.device_1)
        for i in range(0, x.size(0) // self.sub_batch_size + 1):
            start_idx = i * self.sub_batch_size
            end_idx = min(start_idx + self.sub_batch_size, x.size(0))
            if mode == "val":
                # we don't need to compute the gradient
                with torch.no_grad():
                    # drop edge case
                    if start_idx != end_idx:
                        x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                        xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            elif mode == "train":
                # drop edge case
                if start_idx != end_idx:
                    x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                    xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            else:
                raise f"Mode {mode} not implemented"

        # perfom multihead attention
        xout_sub_batch = xout_sub_batch.unsqueeze(0)
        attn_output, _ = self.multihead_attn(
            xout_sub_batch, xout_sub_batch, xout_sub_batch
        )

        attn_output = attn_output.squeeze(0)
        proj_output = torch.nn.functional.relu(self.proj_1(attn_output))  # classifier
        proj_output = self.proj_2(proj_output)
        proj_output = aggregation(proj_output, self.aggregation)

        return proj_output


class PatientModelAttentionTab(nn.Module):
    """
    Custom model for the patient level with attention and tabular data
    """

    def __init__(self, cfg: dict) -> None:
        super(PatientModelAttentionTab, self).__init__()

        # variable definition
        self.device_1 = cfg["device_1"]
        self.latent_att = cfg["latent_att"]
        self.sub_batch_size = cfg["sub_batch_size"]
        self.aggregation = cfg["aggregation"]

        # pick the right encoder model
        if cfg["timm"]:
            self.model = build_timm(cfg)
            
        elif cfg["dino"]:
            self.model = build_dino(cfg["dino_size"])
            self.model.head = nn.Linear(cfg["feature_dim"], cfg["nb_class"])

        # load a pretrained model
        if len(cfg["pretrained_path"]) > 0:
            print(f"We load the weigths {cfg['pretrained_path']}")
            self.model.load_state_dict(
                torch.load(cfg["pretrained_path"])["model_state_dict"]
            )
        else:
            print("The training is from scatch")

        # add adapters
        add_adapter(self.model, cfg["adapter"])

        # we put it after loading the pretrained
        self.model.head = nn.Linear(cfg["feature_dim"], cfg["latent_att"])
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=cfg["latent_att"], num_heads=cfg["head"]
        )
        
        # classifier
        self.proj_1 = nn.Linear(cfg["latent_att"], 4)
        self.proj_2 = nn.Linear(4, 4)
        self.proj_3 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, x_tab: torch.Tensor, mode: str) -> torch.Tensor:
        xout_sub_batch = torch.zeros((x.size(0), self.latent_att)).to(self.device_1)
        
        # loop over the data in sub-batches as the model is too big
        # it corresponds to gradient accumulation
        
        for i in range(0, x.size(0) // self.sub_batch_size + 1):
            start_idx = i * self.sub_batch_size
            end_idx = min(start_idx + self.sub_batch_size, x.size(0))
            if mode == "val":
                # we don't need to compute the gradient
                with torch.no_grad():
                    # drop edge case
                    if start_idx != end_idx:
                        x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                        xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            elif mode == "train":
                # drop edge case
                if start_idx != end_idx:
                    x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                    xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            else:
                raise f"Mode {mode} not implemented"

        # perfom multihead attention
        xout_sub_batch = xout_sub_batch.unsqueeze(0)
        attn_output, _ = self.multihead_attn(
            xout_sub_batch, xout_sub_batch, xout_sub_batch
        )

        # classifier
        attn_output = attn_output.squeeze(0)
        proj_output = torch.nn.functional.gelu(self.proj_1(attn_output))  # classifier
        proj_output = torch.cat((proj_output, x_tab), dim=0)
        proj_output = torch.nn.functional.gelu(self.proj_2(proj_output))
        proj_output = self.proj_3(proj_output)
        proj_output = aggregation(proj_output, self.aggregation)

        return proj_output


def build_dino(model_type: str) -> nn.Module:
    """
    Load a trained DINOv2 model.
    arguments:
        model_type [str]: type of model to train (vits, vitb, vitl, vitg)
    returns:
        model [DinoVisionTransformer]: trained DINOv2 model
    """
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{model_type}14")

    return model


def build_timm(cfg: dict) -> nn.Module:
    """
    Load a trained DINOv2 model.
    arguments:
        model_type [str]: type of model to train (vits, vitb, vitl, vitg)
    returns:
        model [DinoVisionTransformer]: trained DINOv2 model
    """
    model = timm.create_model(
        cfg["timm_model"],
        pretrained=cfg["pretrained"],
        num_classes=cfg["nb_class"],
    )

    return model


def add_adapter(model, adapter):
    """Add an adapter to the model

    Args:
        model (_type_): model to which the adapter will be added
        adapter (_type_): type of adapter to add
    """
    if adapter == "bottleneck":
        print("Use bottleneck adapter")
        add_bottleneck_adapter(model)
        freeze_model_bottleneck(model)

    elif adapter == "adaptformer":
        print("Use adaptformer adapter")
        add_adaptformer(model)
        freeze_model_adaptformer(model)

    elif adapter == "lora":
        print("Use lora adapter")
        add_lora(model)
        freeze_model_lora(model)

    elif adapter == "prompttuning":
        print("Use prompttuning adapter")
        add_prompttuning(model)
        freeze_model_prompttuning(model)

    else:
        print("No adapter used")
