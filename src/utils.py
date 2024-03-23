# standart libraries
import json
import random

# third-party libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def aggregation(x: torch.Tensor, mode: str) -> torch.Tensor:
    """implement aggregation function

    Args:
        x (torch.Tensor): input tensor
        mode (str): aggregation mode

    Returns:
        torch.Tensor: aggregated tensor
    """
    if mode == "sum":
        x = x.sum(0)
    elif mode == "avg":
        x = x.mean(0)
    elif mode == "max":
        x = x.max(0)
    return x


def get_stratified_split(
    df_annotation_train: pd.DataFrame,
    df_annotation_test: pd.DataFrame,
    save_path: str = None,
    ratio: float = 0.8,
    with_plot: bool = True,
):
    """Get a stratified split of the data and save it as a json file

    Args:
        df_annotation_train (pd.DataFrame): dataframe containing the training data
        df_annotation_test (pd.DataFrame): dataframe containing the test data
        save_path (str, optional): path to save the json file. Defaults to None.
        ratio (float, optional): ratio of the training data. Defaults to 0.8.
        with_plot (bool, optional): whether to plot the distribution. Defaults to True.
    Returns:
        _type_: dictionary containing the indices for each split
    """
    n_train = int(ratio * len(df_annotation_train))

    train_indices = random.sample(range(len(df_annotation_train)), n_train)
    df_train = df_annotation_train.iloc[train_indices]
    df_val = df_annotation_train.drop(df_annotation_train.index[train_indices])

    df_train["Dataset"] = "Train"
    df_val["Dataset"] = "Validation"

    df_combined = pd.concat([df_train, df_val])

    # First, calculate the counts for LABEL and GENDER by Dataset
    label_counts = (
        df_combined.groupby(["LABEL", "Dataset"]).size().reset_index(name="Counts")
    )
    gender_counts = (
        df_combined.groupby(["GENDER", "Dataset"]).size().reset_index(name="Counts")
    )

    # Then, calculate the total counts for each Dataset to find ratios
    total_counts_label = label_counts.groupby("Dataset")["Counts"].transform("sum")
    total_counts_gender = gender_counts.groupby("Dataset")["Counts"].transform("sum")

    # Add a new column for ratios
    label_counts["Ratio"] = label_counts["Counts"] / total_counts_label
    gender_counts["Ratio"] = gender_counts["Counts"] / total_counts_gender

    if with_plot:
        # Set up the figure for the plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.tight_layout(pad=5)

        # Plotting the Label Distribution as ratios
        sns.barplot(
            x="LABEL",
            y="Ratio",
            hue="Dataset",
            data=label_counts,
            ax=axes[0],
            palette="viridis",
        )
        axes[0].set_title("Label Distribution Ratio")
        axes[0].set_ylabel("Ratio")

        # Plotting the Gender Distribution as ratios
        sns.barplot(
            x="GENDER",
            y="Ratio",
            hue="Dataset",
            data=gender_counts,
            ax=axes[1],
            palette="coolwarm",
        )
        axes[1].set_title("Gender Distribution Ratio")
        axes[1].set_ylabel("Ratio")

        # Overlayed histogram for LYMPH_COUNT for both DataFrames on the same axes
        # This part remains unchanged
        sns.histplot(
            df_train["LYMPH_COUNT"],
            kde=True,
            ax=axes[2],
            color="skyblue",
            label="Train",
        )
        sns.histplot(
            df_val["LYMPH_COUNT"],
            kde=True,
            ax=axes[2],
            color="orange",
            label="Validation",
        )
        axes[2].set_title("Lymph Count Distribution")
        axes[2].legend(title="Dataset")

        plt.show()

    train_index = df_train.index.to_list()
    val_index = df_val.index.to_list()
    test_index = df_annotation_test.index.to_list()

    map_mode_index = {
        "train": train_index,
        "val": val_index,
        "test": test_index,
    }
    if save_path is not None:
        # save as json
        with open(save_path, "w") as f:
            json.dump(map_mode_index, f)
    return map_mode_index


def get_index_from_json(path: str) -> dict:
    """Load a json file and return the content as a dictionary

    Args:
        path (str): The path to the json file

    Returns:
        dict: The content of the json file
    """
    with open(path, "r") as f:
        map_mode_index = json.load(f)
    return map_mode_index
