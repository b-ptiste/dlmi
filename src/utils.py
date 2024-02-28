import json
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_stratified_split(
    df_annotation_train, df_annotation_test, save_path=None, ratio=0.8, with_plot=True
):
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


def get_index_from_json(path):
    with open(path, "r") as f:
        map_mode_index = json.load(f)
    return map_mode_index
