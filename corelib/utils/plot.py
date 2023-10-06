# -*- coding: utf-8 -*-
"""Plotting module.

Created on: 6/10/23
@author: Heber Trujillo <heber.trj.urt@gmail.com>
Licence,
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Creating a twin Axes sharing the xaxis
    ax2 = ax1.twinx()

    # Bar plot (trx) plotted first
    sns.barplot(
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

    # Line plot (tx_fraud_delta) plotted secondly, so it's on top
    sns.lineplot(
        x=agg_data[feature_name],
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
