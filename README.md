# Problem Definition 🩺  
Lymphocytosis, an increase in lymphocytes in the bloodstream, can signal an immune response (reactive) or chronic conditions like blood cancers (tumoral). Diagnosis typically involves microscopic examination and clinical tests. To address reproducibility issues in microscopy and improve patient triage for flow cytometry, we applied Multi-Instance Learning, leveraging Vision Transformers for their ability to capture long-range relationships through self-attention. Adapting these models to medical imaging with self-supervision, we integrated clinical data to build an efficient, end-to-end trainable model using distillation and fine-tuning techniques.

---

# Mini-Blog Post with Some Insights ✨  
Check out the : 
- **WandB project** [here](https://api.wandb.ai/links/ii_timm/kpkpu224)
- **Report** : [here](https://drive.google.com/file/d/1Ewp0DFXEhgEjMmSIXJdOwpG5lwtnP4aQ/view)

Access the weights for the MAE pretraining [here](https://drive.google.com/drive/folders/13yrd36hwnCahIzXtedJdakCQZdADHxLd?usp=sharing) from [timm](https://huggingface.co/timm) models.

- `vit_tiny_patch16_224.augreg_in21k`
- `vit_small_patch16_224.augreg_in21k` 

These models were trained with limited resources (250 to 800 epochs) on different datasets, and convergence was not achieved due to lack of computing time. 🕒

![MAE Training](https://github.com/b-ptiste/dlmi/assets/75781257/be0b2723-9ea7-47dc-bc82-26bbad606202)

---

# Where to Start 🚀

The code is available within notebooks as it was run on Kaggle to access GPUs.  
### Key Steps:
- You will need to fill the `cfg` configuration dictionaries.
- Documentation is available at `notebooks/Configuration.md` in the `src` directory.

Our project currently supports the following image encoders from **timm**:
- **VIT-based models**  
- **ResNet-based models**  
- **EfficientNet-based models**  

---

## Code Organisation 📂

The source code is in the `src` directory:
- **`data.py` and `model.py`**: Contain a zoo of DataLoaders and Models accessible via a Factory. To add a new one, simply register it.
- **`adapter.py`**: Includes generic code for fine-tuning VIT models from **timm**.
- **`mae_pretraining.py`**: Contains code for MAE self-supervised pretraining.
- **`utils.py`**: Provides useful utility functions.

A sample data split is available in `data/split.py` to quickly launch the code. However, we encourage you to create your own split using the interactive functions provided. 🎨

---

## Self-Supervised Pretraining (MAE) 🔍

**Code**: `self_supervised_MAE.ipynb`  
- Compatible with VIT-based models from **timm**.  
- Simply fill the `_cfg` configuration dictionary and create a custom DataLoader to start training!

---

## Training with MAE Pretraining 🔍

**Code**: `training_using_MAE.ipynb`  
- You need to define a `cfg` dictionary.  
- Optionally, link a **WandB** account for experiment tracking. If not, set `no_wandb: True` in the configuration.

⚠️ Make sure to download weights from the Self-Supervised Pretraining (MAE) section (or use the provided model weights). Refer to the `MAE Pretraining` section for details. 🔗

---

## Training with Image Instances Pretraining 🔍

**Code**: `final-notebook-experiments.ipynb`  
- Define **two `cfg` dictionaries**:
  1. One for pretraining (training the encoder).  
  2. One for fine-tuning (using adapters and classifiers from `model.py`).  

---

# MIL Training 📝

We proposed an end-to-end model trained using pre-trained models like **MAE**, **DinoV2**, **ResNet**, and **EfficientNet**. For VIT-based models, we implemented adapter fine-tuning methods:  
- **LoRA**  
- **Prompt-tuning**  
- **Bottleneck Adapter**  
- **AdaptFormer**  

![MLI Model](https://github.com/b-ptiste/dlmi/assets/75781257/87914a15-3e35-40a0-8878-5e929ce117e8)

---

# Credit 🙌✨

This project was completed during my semester at the MVA in the *Deep Learning for Imaging* course, taught by O. COLLIOT (CNRS) and M. VAKALOPOULOU. 
This project adapted the excellent code from [IcarusWizard/MAE](https://github.com/IcarusWizard/MAE/tree/main). We also reused some practical code from the DLMI course. 👏
