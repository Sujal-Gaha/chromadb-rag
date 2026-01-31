from haystack.logging import getLogger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from typing import Optional, cast
from pathlib import Path
from matplotlib.figure import Figure
from pandas import Series

log = getLogger(__name__)


class EvaluationVisualizer:
    def __init__(
        self,
        dpi: int = 100,
    ):
        self.dpi = dpi
        self._setup_style()

    def _setup_style(self):
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")

    def plot_metric_distribution(
        self,
        df: pd.DataFrame,
        metric_column: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.hist(df[metric_column], bins=20, alpha=0.7, edgecolor="black")
        ax1.set_xlabel(metric_column)
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"Histogram of {metric_column}")
        ax1.grid(True, alpha=0.3)

        ax2.boxplot(df[metric_column], vert=True)
        ax2.set_ylabel(metric_column)
        ax2.set_title(f"Box Plot of {metric_column}")
        ax2.grid(True, alpha=0.3)

        mean_val = df[metric_column].mean()
        median_val = df[metric_column].median()
        std_val = df[metric_column].std()

        stats_text = (
            f"Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nStd: {std_val:.3f}"
        )
        fig.text(
            0.02,
            0.98,
            stats_text,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        if title:
            fig.suptitle(title, fontsize=14)

        fig.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_difficulty_comparison(
        self,
        df: pd.DataFrame,
        metric_column: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Figure:
        """Compare metrics across difficulty levels"""
        if "difficulty" not in df.columns:
            log.warning("No difficulty column found in data")
            raise ValueError("No difficulty column found in data")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        difficulties = df["difficulty"].unique()
        data_by_difficulty = [
            df[df["difficulty"] == d][metric_column] for d in difficulties
        ]

        ax1.boxplot(data_by_difficulty, labels=difficulties)
        ax1.set_xlabel("Difficulty")
        ax1.set_ylabel(metric_column)
        ax1.set_title(f"{metric_column} by Difficulty Level")
        ax1.grid(True, alpha=0.3)

        difficulty_means = cast(Series, df.groupby("difficulty")[metric_column].mean())
        difficulty_counts = cast(Series, df.groupby("difficulty").size())

        bars = ax2.bar(
            list(difficulty_means.index), list(difficulty_means.values), alpha=0.7
        )
        ax2.set_xlabel("Difficulty")
        ax2.set_ylabel(f"Average {metric_column}")
        ax2.set_title(f"Average {metric_column} by Difficulty")
        ax2.grid(True, alpha=0.3)

        for bar, count in zip(bars, difficulty_counts.values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}\n(n={count})",
                ha="center",
                va="bottom",
            )

        if title:
            fig.suptitle(title, fontsize=14)

        fig.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        metric_columns: list[str],
        title: str = "Metric Correlations",
        save_path: Optional[str] = None,
    ) -> Figure:
        numeric_cols = [
            col
            for col in metric_columns
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

        if len(numeric_cols) < 2:
            log.warning("Not enough numeric columns for correlation matrix")
            raise ValueError("Not enough numeric columns for correlation matrix")

        df_numeric = cast(pd.DataFrame, df[numeric_cols])
        correlation_matrix = df_numeric.corr()

        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)

        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                ax.text(
                    x=j,
                    y=i,
                    s=f"{correlation_matrix.iloc[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        ax.set_xticks(np.arange(len(numeric_cols)))
        ax.set_yticks(np.arange(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax.set_yticklabels(numeric_cols)

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom")

        ax.set_title(title)
        fig.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def plot_response_time_analysis(
        self,
        df: pd.DataFrame,
        title: str = "Response Time Analysis",
        save_path: Optional[str] = None,
    ) -> Figure:
        if "response_time" not in df.columns:
            log.warning("No response_time column found")
            raise ValueError("No response_time column found")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        ax1.hist(df["response_time"], bins=20, alpha=0.7, edgecolor="black")
        ax1.set_xlabel("Response Time (seconds)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Response Time Distribution")
        ax1.grid(True, alpha=0.3)

        if "AnswerEvaluator_semantic_similarity" in df.columns:
            ax2.scatter(
                df["response_time"],
                df["AnswerEvaluator_semantic_similarity"],
                alpha=0.6,
            )
            ax2.set_xlabel("Response Time (seconds)")
            ax2.set_ylabel("Semantic Similarity")
            ax2.set_title("Response Time vs Quality")
            ax2.grid(True, alpha=0.3)

        if "difficulty" in df.columns:
            difficulties = df["difficulty"].unique()
            time_by_difficulty = [
                df[df["difficulty"] == d]["response_time"] for d in difficulties
            ]
            ax3.boxplot(time_by_difficulty, labels=difficulties)
            ax3.set_xlabel("Difficulty")
            ax3.set_ylabel("Response Time (seconds)")
            ax3.set_title("Response Time by Difficulty")
            ax3.grid(True, alpha=0.3)

        sorted_times = np.sort(df["response_time"])
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax4.plot(sorted_times, cdf, linewidth=2)
        ax4.set_xlabel("Response Time (seconds)")
        ax4.set_ylabel("Cumulative Probability")
        ax4.set_title("Cumulative Distribution of Response Times")
        ax4.grid(True, alpha=0.3)

        for percentile in [50, 75, 90, 95]:
            time_at_percentile = np.percentile(df["response_time"], percentile)
            ax4.axvline(x=time_at_percentile, color="r", linestyle="--", alpha=0.5)
            ax4.text(
                time_at_percentile, 0.5, f"{percentile}%", rotation=90, va="center"
            )

        if title:
            fig.suptitle(title, fontsize=16)

        fig.tight_layout()

        if save_path:
            self._save_figure(fig, save_path)

        return fig

    def create_comprehensive_report(
        self,
        df: pd.DataFrame,
        output_dir: str = "evaluation_reports",
        batch_id: Optional[str] = None,
    ) -> dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_files: dict[str, str] = {}

        metric_columns = [
            col
            for col in df.columns
            if any(
                x in col
                for x in [
                    "similarity",
                    "match",
                    "precision",
                    "recall",
                    "mrr",
                    "time_score",
                ]
            )
        ]

        if "AnswerEvaluator_semantic_similarity" in df.columns:
            try:
                fig1 = self.plot_metric_distribution(
                    df,
                    "AnswerEvaluator_semantic_similarity",
                    title="Semantic Similarity Distribution",
                )
                similarity_path = (
                    output_path / f"similarity_distribution_{batch_id or ''}.png"
                )
                self._save_figure(fig1, str(similarity_path))
                report_files["similarity_distribution"] = str(similarity_path)
            except Exception as e:
                log.error(f"Failed to create similarity distribution: {e}")

        if (
            "difficulty" in df.columns
            and "AnswerEvaluator_semantic_similarity" in df.columns
        ):
            try:
                fig2 = self.plot_difficulty_comparison(
                    df,
                    "AnswerEvaluator_semantic_similarity",
                    title="Similarity by Difficulty Level",
                )
                difficulty_path = (
                    output_path / f"difficulty_comparison_{batch_id or ''}.png"
                )
                self._save_figure(fig2, str(difficulty_path))
                report_files["difficulty_comparison"] = str(difficulty_path)
            except Exception as e:
                log.error(f"Failed to create difficulty comparison: {e}")

        if "response_time" in df.columns:
            try:
                fig3 = self.plot_response_time_analysis(
                    df, title="Response Time Analysis"
                )
                time_path = output_path / f"response_time_analysis_{batch_id or ''}.png"
                self._save_figure(fig3, str(time_path))
                report_files["response_time_analysis"] = str(time_path)
            except Exception as e:
                log.error(f"Failed to create response time analysis: {e}")

        if len(metric_columns) >= 2:
            try:
                fig4 = self.plot_correlation_matrix(
                    df, metric_columns, title="Metric Correlations"
                )
                if fig4:
                    correlation_path = (
                        output_path / f"correlation_matrix_{batch_id or ''}.png"
                    )
                    self._save_figure(fig4, str(correlation_path))
                    report_files["correlation_matrix"] = str(correlation_path)
            except Exception as e:
                log.error(f"Failed to create correlation matrix: {e}")

        return report_files

    def _save_figure(self, fig: Figure, filepath: str):
        """Save figure with proper formatting"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        log.info(f"Saved visualization: {filepath}")
