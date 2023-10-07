# -*- coding: utf-8 -*-
"""Plotting module.

Created on: 6/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike


def partial_dependency_plot(
    data: pd.DataFrame, feature_name: str
) -> pd.DataFrame:
    """Plot the target partial dependence.

    Args:
        data: pd.DataFrame.
            Data Frame containing the data to analyze.
        feature_name: str
            feature name.

    Returns:
        pd.DataFrame
            Aggregated data.
    """
    agg_data = (
        data.groupby(by=feature_name, as_index=False)
        .agg(tx_fraud=("tx_fraud", "mean"), trx=("tx_datetime", "count"))
        .assign(tx_fraud_delta=lambda df: df.tx_fraud / data.tx_fraud.mean())
    )

    fig, ax1 = plt.subplots(figsize=(10, 4))

    # Creating a twin Axes sharing the xaxis
    ax2 = ax1.twinx()

    # Bar plot (trx) plotted first
    bar = sns.barplot(
        x=agg_data[feature_name],
        y=agg_data["trx"],
        ax=ax2,
        color="yellow",
        alpha=0.5,
        label="Transaction Count",
    )
    ax2.set_ylabel("Transaction Count")
    ax2.set_ylim(
        0, max(agg_data["trx"]) * 1.1
    )  # Adjust to ensure bars don't touch the top

    # Get bar positions
    bar_positions = bar.get_xticks()

    # Line plot (tx_fraud_delta) plotted secondly, so it's on top
    sns.lineplot(
        x=bar_positions,
        y=agg_data["tx_fraud_delta"],
        ax=ax1,
        label="E[fraud | hour] / E[fraud]",
        marker="o",
    )
    ax1.axhline(y=1, color="r", linestyle="--", label="Constant (1)")
    ax1.set_title(f"Partial dependence on {feature_name}")
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel("Delta")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Combining legends from both ax1 and ax2
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    plt.tight_layout()
    plt.show()
    return agg_data


def scatter_plot(
    data: pd.DataFrame,
    feature_name: str,
    n_samples: int = 30000,
) -> None:
    """Plot a joint distribution with scatter plot in the center.

    Args:
        data: pd.DataFrame.
            Data Frame containing the data to analyze.
        feature_name: str
            feature name.
        n_samples: int = 30000
            Number of samples.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.scatterplot(
        data=data.sample(n_samples, random_state=0),
        x=feature_name,
        y="tx_fraud",
    )
    ax.set_title(f"Partial dependence on {feature_name}")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


def plot_precision_recall_curve(
    precision: ArrayLike,
    recall: ArrayLike,
    pr_auc: float,
    pr_auc_random: float,
) -> None:
    """Plot the Precision-Recall Area Under the Curve.

    Args:
        precision: ArrayLike
            Precision values.
        recall: ArrayLike
            Recall values.
        pr_auc: float
            Model's Precision-Recall Area Under the Curve
        pr_auc_random: float
            Random Precision-Recall Area Under the Curve

    Returns:
        None
    """
    pr_curve, ax = plt.subplots(figsize=(5, 5))
    ax.step(recall, precision, label=f"AP-AUC Model = {round(pr_auc, 3)}")
    ax.set_title("Precision-Recall Curve Test Data", fontsize=15)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    ax.set_xlabel("Recall: Detected Frauds / Total Frauds", fontsize=10)
    ax.set_ylabel(
        "Precision:  Detected Frauds / Predicted Frauds", fontsize=10
    )
    ax.plot(
        [0, 1],
        [pr_auc_random, pr_auc_random],
        "r--",
        label=f"AP-AUC Random = {round(pr_auc_random, 3)}",
    )
    ax.legend(loc="upper right")
    plt.show()
