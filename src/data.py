import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class DataloaderFactory:
    def __init__(self):
        pass

    def __call__(
        self,
        cfg,
        split_indexes,
        mode,
        path_root,
        shuffle,
        drop_last=False,
        transform=None,
    ):
        if cfg["dataset_name"] == "DatasetPerImg":
            dataset = DatasetPerImg(path_root, split_indexes, mode, transform)
            dataloader = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                shuffle=shuffle,
                drop_last=drop_last,
            )
            return dataloader
        elif cfg["dataset_name"] == "nvx_data":
            pass


class DatasetPerImg(Dataset):
    def __init__(self, path_root, indexes, mode, transform=None):
        self.indexes = indexes
        self.transform = transform
        self.list_im = []

        if mode == "train":
            path_csv = os.path.join(path_root, "trainset", "trainset_true.csv")
            path_im = os.path.join(path_root, "trainset")
        elif mode == "test":
            path_csv = os.path.join(path_root, "testset", "testset_data.csv")
            path_im = os.path.join(path_root, "testset")

        self.df = csv_processing(path_csv)

        for patient in indexes:
            patient_im = os.listdir(os.path.join(path_im, patient))
            for path in patient_im:
                self.list_im.append(os.path.join(path_im, patient, path))

    def __getitem__(self, idx):
        _path = self.list_im[idx]
        image = Image.open(_path)

        idx_patient = _path.split("/")[-2]

        annotation = (
            self.df[["ID", "LYMPH_COUNT", "AGE", "BIN_GENDER", "LABEL"]]
            .loc[idx_patient]
            .to_dict()
        )

        if self.transform is not None:
            image = self.transform(image) / 255.0

        return image, annotation

    def __len__(self):
        return len(self.list_im)


from datetime import datetime, timedelta


def calculate_age(dob_str):
    """
    Calculate age from a format date string.
    """

    if "/" in dob_str:
        month, day, year = map(int, dob_str.split("/"))
    elif "-" in dob_str:
        day, month, year = map(int, dob_str.split("-"))
    birth_date = datetime(year, month, day)
    today = datetime.now()
    age_timedelta = today - birth_date
    # Returning age in years as a float
    return age_timedelta.days / 365.25


def csv_processing(path):
    df = pd.read_csv(path)

    df["AGE"] = df["DOB"].apply(calculate_age)

    # fix the issue of annotation
    df.loc[df["GENDER"] == "f", "GENDER"] = "F"
    # TODO one hot encoding
    df["BIN_GENDER"] = df["GENDER"].apply(lambda row: (row == "M") * 1.0)

    # normalize
    df["LYMPH_COUNT"] /= np.max(df["LYMPH_COUNT"].values)
    df["AGE"] /= np.max(df["AGE"].values)

    df.index = df["ID"]

    return df
