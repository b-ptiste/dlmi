import timm
import torch
import torch.nn as nn

# third party library
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
    def __init__(self):
        pass

    def __call__(self, cfg):
        model_name = cfg["model_name"]

        print("The configuration is:")
        for k, v in cfg.items():
            print(f"{k} : {v}")

        if cfg["model_name"] == "PatientModel":
            print(f"Loading custom model {cfg['model_name']}")
            return PatientModel(cfg)

        elif cfg["model_name"] == "PatientModelAttention":
            print(f"Loading custom model {cfg['model_name']}")
            return PatientModelAttention(cfg)

        elif cfg["timm"]:
            print(f"Loading timm model {cfg['timm_model']}")
            model = build_timm(cfg)
            add_adapter(model, cfg['adapter'])
            return model

        elif cfg["dino"]:
            print(f"Loading dino model {cfg['dino_size']}")
            model = build_dino(cfg["dino_size"])
            model.head = nn.Linear(cfg["feature_dim"], cfg["nb_class"])
            add_adapter(model, cfg['adapter'])
            return model
        else:
            raise NotImplemented(f"{model_name} don't register")


class PatientModel(nn.Module):
    def __init__(self, cfg):
        super(PatientModel, self).__init__()

        self.aggregation = cfg["aggregation"]
        self.device_1 = cfg["device_1"]
        self.sub_batch_size = cfg["sub_batch_size"]

        if cfg["timm"]:
            self.model = build_timm(cfg)
        elif cfg["dino"]:
            self.model = build_dino(cfg["dino_size"])
            self.model.head = nn.Linear(cfg["feature_dim"], cfg["nb_class"])

        if len(cfg["pretrained_path"]) > 0:
            self.model.load_state_dict(
                torch.load(cfg["pretrained_path"])["model_state_dict"]
            )
            add_adapter(self.model, cfg['adapter'])

    def forward(self, x, mode):
        #         x = x[torch.randperm(x.size(0)), ...]
        xout_sub_batch = torch.zeros((x.size(0), 2)).to(self.device_1)

        for i in range(0, x.size(0) // self.sub_batch_size + 1):
            start_idx = i * self.sub_batch_size
            end_idx = min(start_idx + self.sub_batch_size, x.size(0))
            if mode == "val":
                with torch.no_grad():
                    if start_idx != end_idx:
                        x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                        xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            elif mode == "train":
                if start_idx != end_idx:
                    x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                    xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            else:
                raise f"Mode {mode} not implemented"

        xout_sub_batch = aggregation(xout_sub_batch, self.aggregation)

        return xout_sub_batch


class PatientModelAttention(nn.Module):
    def __init__(self, cfg):
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

        # load a pretrained model
        if len(cfg["pretrained_path"]) > 0:
            print(f"We load the weigths {cfg['pretrained_path']}")
            self.model.load_state_dict(
                torch.load(cfg["pretrained_path"])["model_state_dict"]
            )
        else:
            print("The training is from scatch")
        
        # add adapters
        add_adapter(self.model, cfg['adapter'])
        
        # we put it after loading the pretrained
        self.model.head = nn.Linear(cfg["feature_dim"], cfg["latent_att"])
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=cfg["latent_att"], num_heads=cfg["head"]
        )
        self.proj_1 = nn.Linear(cfg["latent_att"], cfg["latent_att"])
        self.proj_2 = nn.Linear(cfg["latent_att"], 2)

    def forward(self, x, mode):
        #         x = x[torch.randperm(x.size(0)), ...]
        xout_sub_batch = torch.zeros((x.size(0), self.latent_att)).to(self.device_1)
        for i in range(0, x.size(0) // self.sub_batch_size + 1):
            start_idx = i * self.sub_batch_size
            end_idx = min(start_idx + self.sub_batch_size, x.size(0))
            if mode == "val":
                with torch.no_grad():
                    if start_idx != end_idx:
                        x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                        xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            elif mode == "train":
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


def build_dino(model_type):
    """
    Credit : DLMI TP

    Load a trained DINOv2 model.
    arguments:
        model_type [str]: type of model to train (vits, vitb, vitl, vitg)
    returns:
        model [DinoVisionTransformer]: trained DINOv2 model
    """
    model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{model_type}14")

    return model



def build_timm(cfg):
    """
    Credit : DLMI TP

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