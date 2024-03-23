# standard libraries
from __future__ import annotations

import os
from typing import Any

# third-party libraries
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class DataloaderFactory:
    """
    Class to create and get all the dataloaders
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self,
        cfg: dict,
        split_indexes: list,
        mode: str,
        path_root: str,
        shuffle: bool,
        drop_last: bool = False,
        transform: Any = None,
        oversampling: dict = {"0": 1, "1": 1},
    ) -> DataLoader:
        if cfg["dataset_name"] == "DatasetPerImg":
            dataset = DatasetPerImg(
                path_root, split_indexes, mode, oversampling, transform
            )
            dataloader = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                shuffle=shuffle,
                drop_last=drop_last,
            )

        elif cfg["dataset_name"] == "DatasetPerPatient":
            dataset = DatasetPerPatient(
                path_root, split_indexes, mode, oversampling, transform
            )
            dataloader = DataLoader(
                dataset,
                batch_size=cfg["batch_size"],
                shuffle=shuffle,
                drop_last=drop_last,
            )
        else:
            raise NotImplemented(f"{cfg['dataset_name']} don't register")
        return dataloader


class DatasetPerImg(Dataset):
    """
    DatasetPerImg: Dataset for image level
    """

    def __init__(
        self,
        path_root: str,
        indexes: list,
        mode: str,
        oversampling: dict,
        transform: Any = None,
    ):
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
                # oversampling: given the labels 0 or 1
                # oversampling is a dictionary with the number of oversampling for each label
                if mode == "train":
                    for _ in range(oversampling[str(self.df.loc[patient, "LABEL"])]):
                        self.list_im.append(os.path.join(path_im, patient, path))
                else:
                    self.list_im.append(os.path.join(path_im, patient, path))

    def __getitem__(self, idx: int) -> Any:
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

    def __len__(self) -> int:
        return len(self.list_im)


class DatasetPerPatient(Dataset):
    """
    DatasetPerPatient: Dataset for patient level
    """

    def __init__(
        self,
        path_root: str,
        indexes: list,
        mode: str,
        oversampling: dict,
        transform: Any = None,
    ):
        self.indexes = indexes
        self.transform = transform
        self.patients_data = {}

        if mode == "train":
            path_csv = os.path.join(path_root, "trainset", "trainset_true.csv")
            path_im = os.path.join(path_root, "trainset")
        elif mode == "test":
            path_csv = os.path.join(path_root, "testset", "testset_data.csv")
            path_im = os.path.join(path_root, "testset")

        self.df = csv_processing(path_csv)

        idx = 0
        self.map_idx_patient = {}
        for patient in indexes:
            patient_folder = os.path.join(path_im, patient)
            patient_im = [
                os.path.join(patient_folder, img)
                for img in os.listdir(patient_folder)
                if img.endswith(".jpg")
            ]
            if mode == "train":
                # oversampling: given the labels 0 or 1
                for _ in range(oversampling[str(self.df.loc[patient, "LABEL"])]):
                    self.patients_data[idx] = patient_im
                    self.map_idx_patient[idx] = patient
                    idx += 1
            else:
                self.patients_data[idx] = patient_im
                self.map_idx_patient[idx] = patient
                idx += 1

    def __getitem__(self, idx: int) -> Any:
        patient_id = self.map_idx_patient[idx]
        patient_images = self.patients_data[idx]
        images = []

        for img_path in patient_images:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img) / 255.0
            images.append(img)

        images = torch.stack(images)

        annotation = (
            self.df[["ID", "LYMPH_COUNT", "AGE", "BIN_GENDER", "LABEL"]]
            .loc[patient_id]
            .to_dict()
        )

        return images, annotation

    def __len__(self) -> int:
        return len(self.map_idx_patient)


def calculate_age(dob_str: str) -> float:
    """calculate_age

    Args:
        dob_str (str): string representing the date of birth

    Returns:
        float: age in years normalized
    """

    # Parsing the date of birth
    if "/" in dob_str:
        month, day, year = map(int, dob_str.split("/"))
    elif "-" in dob_str:
        day, month, year = map(int, dob_str.split("-"))
    birth_date = datetime(year, month, day)
    today = datetime.now()
    age_timedelta = today - birth_date
    # Returning age in years as a float
    return age_timedelta.days / 365.25


def csv_processing(path: str):
    """_summary_

    Args:
        path (str): path to the csv file

    Returns:
        _type_: processed dataframe
    """
    df = pd.read_csv(path)

    df["AGE"] = df["DOB"].apply(calculate_age)

    # fix the issue of annotation
    df.loc[df["GENDER"] == "f", "GENDER"] = "F"
    df["BIN_GENDER"] = df["GENDER"].apply(lambda row: (row == "M") * 1.0)

    # normalize
    df["LYMPH_COUNT"] /= np.max(df["LYMPH_COUNT"].values)
    df["AGE"] /= np.max(df["AGE"].values)

    df.index = df["ID"]

    return df
