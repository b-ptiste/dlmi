These dictionaries are exhaustive, but for certain configurations you may not need to use all the keys.
If you use wandb, it will also record all these values for each experiment!


For the `final-notebook-experiments.ipynb`, `training_using_MAE.ipynb notebooks`.

```
cfg = {
    'who': 'your_name', # who run the experiements
    'no_wandb': True, # True if you don't want wandb
    'name_exp': '', # choose the name of the exp. in wandb
    'lr': 1e-5, # learning rate
    'batch_size': 124, # batch size
    'nb_epochs': 10, # nb epochs
    'timm': True, # is the model from timm ? (it is incompatible with 'timm': True)
    'pretrained': True, # Do you want timm to load the pretrain ckpt ?
    'pretrained_path': '', # Do you want to load you own pretrained ?
    'timm_model': 'vit_tiny_patch16_224.augreg_in21k', # name from timm registery
    'dino': False, # do we want to load dino from torchvision (it is incompatible with 'timm': True)
    'dino_size': 'vits', # vits, vitb, vitl, vitg : dino torchvision size
    'adapter': '', # bottleneck, adaptformer, lora, prompttuning # pick the adapter you want
    'model_name': '', # name of the classifier from our model factory
    'nb_class': 2, # binary classification.
    'scheduler': None, # could be empty or linear, expo ...
    'dataset_name': 'DatasetPerImg', # DataLoader from our factory
    'device_1': 'cuda:0', 
    'device_2': 'cuda:1', # for double device
    'filename': f'{path_working}/submission_pretrain.csv', # save the pretrain model
    'filename_finetune': f'{path_working}/submission_finetune.csv', # save the finetune model
    'sub_batch_size': 16, # we performed sub-batches we grad acc. see report
    'latent_att': 512, # size of the patch embeddings after projection
    'head_1': 8, # multi-head attention
    'head_2': 2, # cross attention
    'feature_dim': 192, # DINOv2, VIT: 192 - 384 # model encoder embedding
    'aggregation': 'sum', # sum, avg, max # aggregation
    'beta_1': 0.5, # adam parameters
    'beta_2': 0.9, # adam parameters
    'weight_decay': 5e-2, # adam parameters
    'weight_class_0': 3.0, # Weighted BCE
    'weight_class_1': 1.0, # Weighted BCE
    'mae_pretrained': '', # path to your pretrain
    'with_tab': True, # For Classifier that use tabular set True otherwise False
    'degrees': (-5, 5), # rotation aug.
    'translate': (0.1, 0.1), # transl. aug.
    'scale': (1.0, 1.1), # scale aug.
    'fill': (255, 232, 201), # color to fill after rotation or transl.
    'p': 0.1, # flip proba.
}  
```
For the MAE notebook : `self_supervised_MAE.ipynb`

```
_cfg = {
    'seed': 42,
    'batch_size': 16,
    'max_device_batch_size': 512,
    'base_learning_rate': 1.5e-4,
    'weight_decay': 0.05,
    'mask_ratio': 0.75,
    'total_epoch': 800,
    'warmup_epoch': 200,
    'model_path': '/kaggle/working/vit-backbone-mae.pt', # output
    'image_size': 224,
    'patch_size': 14,
    'emb_dim': 384,
    "dataset_name": "DatasetPerImg"
}

```
