This project was done during my semester at the MVA in the Deep Learning for Imaging course teached by O. COLLIOT (CNRS), M. VAKALOPOULOU.

## Problem definiton
Lymphocytosis is characterized by an incerease in the number of lymphocytes, a type of white blood cell, in the bloodstream. It can indicate the body’s immune system is fighting off pathogens (reactive) or it may result from chronic patologies including blood cancers (tumoral). The diagnosis relies on two stages : a microscopic examination and clinical tests. We tackle the lack of reproducibility of the first stage to better filter patients which should be referred for flow cytometry analysis. We approached this problem through the prism of Multi-Instance Learning, given that we have several bags of images per patient. To do this, we coupled the ability of Vision Transformers to capture long-distance relationships with self-attention mechanism between images of a patient we have adapted to medical imaging using self-supervision methods. We incorporated clinical data to create a trainable end-to- end model with low computational resources using the distillation mechanism and ingenious finetuning methods.

## Experiments 

If you are intrested by some interactive we invite you to navigate in our Wandb projet [here]([https://wandb.ai/ii_timm/DLMI/reports/Some-Insight-of-our-experiments--Vmlldzo3MzAwMjcx?accessToken=s8ywd5gx8m7891ocsohpyqfysst0tjza8ury9b790p9v37jt3hcfo4nci1r9p8xi](https://api.wandb.ai/links/ii_timm/kpkpu224)) !!

## Where to start

Code is available within notebooks which is kind of main.py but as we run it on kaggle (to get acces to GPU) it is easier. You will need to fill cfg configuration dictionnaries, you give the explaination of the key in the `notebooks/Configuration.md` in `src`

Our project right now support some models from timm:
- VIT based models
- ResNet based models
- EfficientNet based models
### Code organisation

First, you will find the source code in `src`.
- We provide in `data.py` and `model.py` and zoo of DataLoader and Models that you can access by the Factory. If you want to add another one, you just need to register it.
- We also provide in `adapter.py` some generic code for finetuning VIT from timm.
- Finally, the code for the MAE self-supervised pre-training is availale in : `mae_pretraining.py`.
- The file `utils.py` contains some usefull functions.

One data split is provide if you want to launch our code quicly in `data/split.py`. However, we encourage you to create yours. We have created some usefull function to do it interactively.

### Self-supervised pre-traning (MAE), 
code : self_supervised_MAE.ipynb

It is compatible with VIT timm based models. you just need to fill the configuration `_cfg` and create your own dataloader and it will run !

### Train with MAE pretraining
code : training_using_MAE.ipynb

You will need to define a `cfg` dictionary.
You may need to create a wandb account and link it. If you don't want to, you can simply set the key `no_wandb`: True.

Pay attention to change download you weight from Self-supervised pre-traning (MAE) (otherwise, our models weight are available check the `MAE pretraining``section for some models).

### Train with image instances pretraining
code : final-notebook-experiments.ipynb

You will need to define two `cfg` dictionaries :
- the first one corresponds to the pretraining, it will just train the encoder
- the second one correponds to the finetuning using adapters and you can pick classifier available from `model.py`

## MAE pretraining

The weights for the MAE section are available [here](https://drive.google.com/drive/folders/13yrd36hwnCahIzXtedJdakCQZdADHxLd?usp=sharing) for the model `vit_tiny_patch16_224.augreg_in21k` and `vit_small_patch16_224.augreg_in21k` model from [timm](https://huggingface.co/timm)

The models were trained with limited resources from 250 to 800 epochs and different datasets, and convergence was not achieved (lack of computing time).

![MAE training](https://github.com/b-ptiste/dlmi/assets/75781257/be0b2723-9ea7-47dc-bc82-26bbad606202)

## MIL training

We proposed to train an end-to-end model using pre-trained models : MAE, DinoV2, ResNet, EfficientNet. For the VIT based models, we implemented adapter finetuning : LoRA, prompttuning, bottleneck adapter and adaptformer.

![MLI model](https://github.com/b-ptiste/dlmi/assets/75781257/87914a15-3e35-40a0-8878-5e929ce117e8)

Finally, we have proposed several classifiers that use the self-attention mechanism. 

![Classifier - MLP - Attention](https://github.com/b-ptiste/dlmi/assets/75781257/99cea953-508f-4b8a-9f7d-b2f650f37a48)
![legend](https://github.com/b-ptiste/dlmi/assets/75781257/47ca437b-8975-459f-bb3f-b4a21d589b71)

We have also proposed a model that uses clinical data in the form of cross attention, as a doctor might do.

![Classifier - Cross attention](https://github.com/b-ptiste/dlmi/assets/75781257/efe3bbfc-f406-4468-9ded-ba1a2fa02653)
![legend](https://github.com/b-ptiste/dlmi/assets/75781257/47ca437b-8975-459f-bb3f-b4a21d589b71)

# Credit

We adapted the really great code of : https://github.com/IcarusWizard/MAE/tree/main. We also re-use some code from DLMI course practical works. 
