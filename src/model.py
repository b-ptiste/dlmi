import timm
import torch
import torch.nn as nn

class ModelFactory:
    def __init__(self):
        pass

    def __call__(self, cfg):
        model_name = cfg["model_name"]

        print('The configuration is:')
        for k, v in cfg.items():
            print(f"{k} : {v}")
            
        if cfg["timm"]:
            print(f"Loading timm model {cfg['model_name']}")
            return timm.create_model(
                cfg["model_name"],
                pretrained=cfg["pretrained"],
                num_classes=cfg["nb_class"],
            )
        
        elif cfg["dino"]:
            print(f"Loading dino model {cfg['model_name']}")
            model = build_dino(cfg['dino_size'], cfg['adapter'])
            model.head = nn.Linear(cfg['feature_dim'], cfg['nb_class'])
            return model
        
        elif cfg["model_name"] == "PatientModel":
            print(f"Loading custom model {cfg['model_name']}")
            return PatientModel(cfg)
        
        elif cfg["model_name"] == "PatientModelAttention":
            print(f"Loading custom model {cfg['model_name']}")
            return PatientModelAttention(cfg)

        else:
            raise NotImplemented(f"{model_name} don't register")
        

class PatientModel(nn.Module):
    def __init__(self, cfg):
        super(PatientModel, self).__init__()
        
        self.aggregation = cfg['aggregation']
        self.device_1 = cfg['device_1']
        self.sub_batch_size = cfg['sub_batch_size']
        
        if cfg['timm']:
            self.model = timm.create_model(
                    cfg["model_name"],
                    pretrained=cfg["pretrained"],
                    num_classes=cfg["nb_class"],
                )
        elif cfg["dino"]:
            self.model = build_dino(cfg['dino_size'], cfg['adapter'])
            self.model.head = nn.Linear(cfg['feature_dim'], cfg['nb_class'])  
            
        
        if len(cfg['pretrained']) > 0:
            self.model.load_state_dict(torch.load(cfg['pretrained'])['model_state_dict'])


    def forward(self, x, mode):
#         x = x[torch.randperm(x.size(0)), ...]
        xout_sub_batch = torch.zeros((x.size(0), 2)).to(self.device_1)
        
        for i in range(0, x.size(0) // self.sub_batch_size + 1):
            start_idx = i * self.sub_batch_size
            end_idx = min(start_idx + self.sub_batch_size, x.size(0))
            if mode == 'val':
                with torch.no_grad():
                    if start_idx != end_idx:
                        x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                        xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            elif mode == 'train':
                if start_idx != end_idx:
                    x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                    xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            else:
                raise f'Mode {mode} not implemented'
        
        xout_sub_batch = aggregation(xout_sub_batch, self.aggregation)

        return xout_sub_batch
    
    
class PatientModelAttention(nn.Module):
    def __init__(self, cfg):
        super(PatientModelAttention, self).__init__()
    
        # variable definition
        self.device_1 = cfg['device_1']
        self.latent_att = cfg['latent_att']
        self.sub_batch_size = cfg['sub_batch_size']
        self.aggregation = cfg['aggregation']
        
        # pick the right encoder model
        if cfg['timm']:
            self.model = timm.create_model(
                    cfg["model_name"],
                    pretrained=cfg["pretrained"],
                    num_classes=cfg["nb_class"],
                )
        elif cfg["dino"]:
            self.model = build_dino(cfg['dino_size'], cfg['adapter'])
        
        # load a pretrained model
        if len(cfg['pretrained'])>0:
            print(f'We load the weigths {cfg['pretrained']}')
            self.model.load_state_dict(torch.load(cfg['pretrained'])['model_state_dict'])
        else:
            print('The training is from scatch')
        
        # we put it after loading the pretrained
        self.model.head = nn.Linear(cfg['feature_dim'], cfg['latent_att']) 
        self.multihead_attn = nn.MultiheadAttention(embed_dim=cfg['latent_att'], num_heads=cfg['head'])
        self.proj_1 = nn.Linear(cfg['latent_att'], cfg['latent_att'])
        self.proj_2 = nn.Linear(cfg['latent_att'], 2)


    def forward(self, x, mode):
#         x = x[torch.randperm(x.size(0)), ...]
        xout_sub_batch = torch.zeros((x.size(0), self.latent_att)).to(self.device_1)
        for i in range(0, x.size(0) // self.sub_batch_size + 1):
            start_idx = i * self.sub_batch_size
            end_idx = min(start_idx + self.sub_batch_size, x.size(0))
            if mode == 'val':
                with torch.no_grad():
                    if start_idx != end_idx:
                        x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                        xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            elif mode == 'train':
                if start_idx != end_idx:
                    x_sub_batch = x[start_idx:end_idx].to(self.device_1)
                    xout_sub_batch[start_idx:end_idx] = self.model(x_sub_batch)
            else:
                raise f'Mode {mode} not implemented'
        
        # perfom multihead attention
        xout_sub_batch = xout_sub_batch.unsqueeze(0)
        attn_output, _ = self.multihead_attn(xout_sub_batch, xout_sub_batch, xout_sub_batch)
        
        attn_output = attn_output.squeeze(0)
        proj_output = torch.nn.functional.relu(self.proj_1(attn_output)) # classifier
        proj_output = self.proj_2(proj_output)
        proj_output = aggregation(proj_output, self.aggregation)

        return proj_output    

def aggregation(x, mode):
    if mode == 'sum':
        x = x.sum(0)
    elif mode == 'avg':
        x = x.mean(0)
    elif mode == 'max':
        x = x.max(0)
    return x
        
    
def build_dino(model_type, adapter):
    """
        Credit : DLMI TP
        
        Load a trained DINOv2 model.
        arguments:
            model_type [str]: type of model to train (vits, vitb, vitl, vitg)
        returns:
            model [DinoVisionTransformer]: trained DINOv2 model
    """
    model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_type}14')
    
    if adapter == 'bottleneck':
        print('Use bottleneck adapter')
        add_bottleneck_adapter(model)
        freeze_model_bottleneck(model)

    return model


class BottleneckAdapter(nn.Module):
    """
    Credit : DLMI TP
    """
    def __init__(self, in_dim, activation='ReLU', reduction_factor=16):
        super(BottleneckAdapter, self).__init__()

        hidden_dim = in_dim // reduction_factor
        self.adapter_downsample = nn.Linear(in_dim, hidden_dim)
        self.activation =  getattr(torch.nn, activation)()
        self.adapter_upsample = nn.Linear(hidden_dim, in_dim)

    def forward(self, x_in):
        x_hid = self.activation(self.adapter_downsample(x_in))
        x_out = self.adapter_upsample(x_hid)
        return x_out + x_in

def add_bottleneck_adapter(model):
    """
    Credit : DLMI TP
    """
    for block in model.blocks:
        block.attn.proj = nn.Sequential(block.attn.proj, BottleneckAdapter(block.attn.proj.in_features))
        block.attn.mlp = nn.Sequential(block.mlp, BottleneckAdapter(block.mlp.fc1.in_features))
    
def freeze_model_bottleneck(model):
    """
    Credit : DLMI TP
    """
    
    for name, param in model.named_parameters():
       
        if not('adapter' in name or 'norm1' in name or 'norm2' in name):
            param.requires_grad = False
