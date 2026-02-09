import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from typing import Optional, cast
from pathlib import Path
from matplotlib.figure import Figure
from pandas import Series

from utils.logger import get_logger

log = get_logger(__name__)


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

    def _clean_numeric_columns(self, df: pd.DataFrame, cols: list[str]) -> list[str]:
        valid_cols = []
        for col in cols:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if df[col].nunique(dropna=True) <= 1:
                continue
            if bool(df[col].isna().all()):
                continue
            valid_cols.append(col)
        return valid_cols

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        metric_columns: list[str],
        title: str = "Metric Correlations",
        save_path: Optional[str] = None,
    ) -> Figure:

        numeric_cols = self._clean_numeric_columns(df, metric_columns)

        if len(numeric_cols) < 2:
            raise ValueError("Not enough valid numeric columns for correlation")

        df_numeric: pd.DataFrame = df.loc[:, numeric_cols]
        corr: pd.DataFrame = df_numeric.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={"label": "Correlation"},
            mask=corr.isna(),
            ax=ax,
        )

        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
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
            raise ValueError("No response_time column found")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Histogram
        ax1.hist(df["response_time"].dropna(), bins="auto", edgecolor="black")
        ax1.set_title("Response Time Distribution")
        ax1.set_xlabel("Seconds")
        ax1.set_ylabel("Frequency")

        # 2. Quality vs Time OR fallback
        if "AnswerEvaluator_semantic_similarity" in df.columns:
            ax2.scatter(
                df["response_time"],
                df["AnswerEvaluator_semantic_similarity"],
                alpha=0.6,
            )
            ax2.set_ylabel("Semantic Similarity")
            ax2.set_title("Response Time vs Quality")
        else:
            ax2.text(
                0.5,
                0.5,
                "No quality metric available",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=11,
                alpha=0.7,
            )
            ax2.set_title("Response Time vs Quality")

        ax2.set_xlabel("Seconds")

        # 3. Difficulty boxplot
        if "difficulty" in df.columns:
            order = ["easy", "medium", "hard"]
            data = []
            labels = []

            for d in order:
                if d not in df["difficulty"].unique():
                    continue

                subset: pd.Series = df.loc[df["difficulty"] == d, "response_time"]
                subset = subset.dropna()

                if not subset.empty:
                    data.append(subset)
                    labels.append(d)

            ax3.boxplot(data, labels=labels, showfliers=True)
            ax3.set_title("Response Time by Difficulty")
            ax3.set_ylabel("Seconds")
        else:
            ax3.text(
                0.5,
                0.5,
                "No difficulty data",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )

        # 4. CDF
        times = np.sort(df["response_time"].dropna())
        cdf = np.arange(1, len(times) + 1) / len(times)

        ax4.plot(times, cdf, linewidth=2)
        ax4.set_title("Cumulative Distribution")
        ax4.set_xlabel("Seconds")
        ax4.set_ylabel("Probability")

        for p in [50, 75, 90, 95]:
            t = np.percentile(times, p)
            ax4.axvline(t, linestyle="--", alpha=0.6)
            ax4.text(t, 0.02, f"{p}%", rotation=90, va="bottom")

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
