import timm


class ModelFactory:
    def __init__(self):
        pass

    def __call__(self, cfg):
        model_name = cfg["model_name"]

        if cfg["timm"]:
            return timm.create_model(
                cfg["model_name"],
                pretrained=cfg["pretrain"],
                num_classes=cfg["nb_class"],
            )
        elif cfg["model_name"] == "nvx_model":
            pass

        else:
            raise NotImplemented(f"{model_name} don't register")
