This project was done during my semester at the MVA in the Deep Learning for Imaging course teached by O. COLLIOT (CNRS), M. VAKALOPOULOU.

## Problem definiton
Lymphocytosis is characterized by an incerease in the number of lymphocytes, a type of white blood cell, in the bloodstream. It can indicate the body’s immune system is fighting off pathogens (reactive) or it may result from chronic patologies including blood cancers (tumoral). The diagnosis relies on two stages : a microscopic examination and clinical tests. We tackle the lack of reproducibility of the first stage to better filter patients which should be referred for flow cytometry analysis. We approached this problem through the prism of Multi-Instance Learning, given that we have several bags of images per patient. To do this, we coupled the ability of Vision Transformers to capture long-distance relationships with self-attention mechanism between images of a patient we have adapted to medical imaging using self-supervision methods. We incorporated clinical data to create a trainable end-to- end model with low computational resources using the distillation mechanism and ingenious finetuning methods.

## MAE pretraining.

The weights for the MAE section are available [here](https://drive.google.com/drive/folders/13yrd36hwnCahIzXtedJdakCQZdADHxLd?usp=sharing) for the model `vit_tiny_patch16_224.augreg_in21k` and `vit_small_patch16_224.augreg_in21k` model from [timm](https://huggingface.co/timm)

The models were trained with limited resources from 250 to 800 epochs and different datasets, and convergence was not achieved (lack of computing time).

![MAE training](https://github.com/b-ptiste/dlmi/assets/75781257/be0b2723-9ea7-47dc-bc82-26bbad606202)


# Credit


