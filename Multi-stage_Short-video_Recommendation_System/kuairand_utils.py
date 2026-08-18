"""Reusable helpers for the KuaiRand recommender notebooks.

The notebooks in this project are intentionally lightweight, but a few pieces of
logic recur across stages: path handling, objective labels, temporal splitting,
retrieval, and evaluation. Keeping those pieces here makes each notebook easier
to read while avoiding a full Python package structure.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def find_project_root() -> Path:
    """Return the project root whether a notebook runs from root or notebooks/."""
    cwd = Path.cwd()
    if (cwd / "data").exists() or (cwd / "notebooks").exists():
        return cwd
    if (cwd.parent / "data").exists() or (cwd.parent / "notebooks").exists():
        return cwd.parent
    return cwd


def format_bytes(num_bytes: int) -> str:
    """Return a human-readable file size."""
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def count_lines(path: Path) -> int:
    """Count newline-delimited rows without parsing a CSV."""
    total = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            total += block.count(b"\n")
    return total


@dataclass
class KuaiRandConfig:
    """Central configuration shared by the KuaiRand notebooks."""

    project_root: Path = field(default_factory = find_project_root)
    dataset_relative_path: str = "data/kuairand/KuaiRand Dataset"
    user_column: str = "user_id"
    item_column: str = "video_id"
    timestamp_column: str = "event_ts"
    random_seed: int = 42
    nrows_per_standard_log: int | None = 600_000
    validation_days: int = 3
    recent_lookback_days: int = 5
    top_k: int = 50

    @property
    def dataset_root(self) -> Path:
        return self.project_root / self.dataset_relative_path

    @property
    def data_dir(self) -> Path:
        return self.dataset_root / "data"

    @property
    def standard_log_files(self) -> list[Path]:
        return [
            self.data_dir / "log_standard_4_08_to_4_21_1k.csv",
            self.data_dir / "log_standard_4_22_to_5_08_1k.csv",
        ]

    @property
    def random_log_file(self) -> Path:
        return self.data_dir / "log_random_4_22_to_5_08_1k.csv"

    @property
    def user_feature_file(self) -> Path:
        return self.data_dir / "user_features_1k.csv"

    @property
    def video_basic_file(self) -> Path:
        return self.data_dir / "video_features_basic_1k.csv"

    @property
    def video_statistic_file(self) -> Path:
        return self.data_dir / "video_features_statistic_1k.csv"

    @property
    def all_csv_files(self) -> list[Path]:
        return sorted(self.data_dir.glob("*.csv"))

    def validate_paths(self) -> None:
        """Raise a clear error if expected dataset files are missing."""
        expected = [
            *self.standard_log_files,
            self.random_log_file,
            self.user_feature_file,
            self.video_basic_file,
            self.video_statistic_file,
        ]
        missing = [path for path in expected if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing expected KuaiRand files: {missing}")


class KuaiRandDataCatalog:
    """Inspect KuaiRand CSV files and load interaction samples."""
    log_usecols = [
        "user_id", "video_id", # ID level
        "date", "hourmin", "time_ms", # timestamp
        "is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate", # user-video interaction (binary)
        "long_view", "play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time", # user-video interaction (continuous)
        "is_profile_enter", "is_rand", "tab", # other logging features
    ]

    def __init__(self, config: KuaiRandConfig):
        self.config = config
        self.config.validate_paths()

    def inventory(self) -> pd.DataFrame:
        """Return file size, row count, and column count for all CSVs."""
        rows = []
        for path in self.config.all_csv_files:
            columns = self.read_header(path)
            rows.append(
                {
                    "file": path.name,
                    "path": str(path),
                    "size": format_bytes(path.stat().st_size),
                    "data_rows": max(count_lines(path) - 1, 0),
                    "columns": len(columns),
                }
            )
        return pd.DataFrame(rows).sort_values("data_rows", ascending=False).reset_index(drop=True)

    @staticmethod
    def read_header(path: Path) -> list[str]:
        """Read only the CSV header."""
        return list(pd.read_csv(path, nrows=0).columns)

    def sample_rows(self, path: Path, nrows: int = 5) -> pd.DataFrame:
        """Read a small sample from one CSV."""
        return pd.read_csv(path, nrows=nrows)

    def schema_profile(self, nrows: int = 2_000) -> pd.DataFrame:
        """Profile sample dtypes, missingness, distinct counts, and examples."""
        rows = []
        for path in self.config.all_csv_files:
            sample = pd.read_csv(path, nrows=nrows)
            for column in sample.columns:
                rows.append(
                    {
                        "file": path.name,
                        "column": column,
                        "sample_dtype": str(sample[column].dtype),
                        "sample_missing": int(sample[column].isna().sum()),
                        "sample_unique": int(sample[column].nunique(dropna=True)),
                        "example_values": sample[column].dropna().head(3).tolist(),
                    }
                )
        return pd.DataFrame(rows)

    def shared_columns(self) -> pd.DataFrame:
        """Return shared-column overlaps for every CSV pair."""
        columns_by_file = {path.name: set(self.read_header(path)) for path in self.config.all_csv_files}
        file_names = list(columns_by_file)
        rows = []
        for i, left in enumerate(file_names):
            for right in file_names[i + 1 :]:
                shared = sorted(columns_by_file[left] & columns_by_file[right])
                rows.append(
                    {
                        "left_file": left,
                        "right_file": right,
                        "shared_column_count": len(shared),
                        "shared_columns": shared,
                    }
                )
        return pd.DataFrame(rows).sort_values("shared_column_count", ascending=False)

    @staticmethod
    def collect_unique_values(path: Path, column: str, chunksize: int = 500_000) -> tuple[int, set]:
        """Return row count and unique non-null values for one CSV column."""
        row_count = 0
        unique_values = set()
        for chunk in pd.read_csv(path, usecols=[column], chunksize=chunksize):
            row_count += len(chunk)
            unique_values.update(chunk[column].dropna().unique().tolist())
        return row_count, unique_values

    def key_profiles(self, chunksize: int = 500_000) -> tuple[pd.DataFrame, dict[tuple[str, str], set]]:
        """Profile user_id/video_id uniqueness and return the collected key sets."""
        key_sets: dict[tuple[str, str], set] = {}
        rows = []
        for path in self.config.all_csv_files:
            columns = set(self.read_header(path))
            for key_col in [self.config.user_column, self.config.item_column]:
                if key_col not in columns:
                    continue
                row_count, values = self.collect_unique_values(path, key_col, chunksize=chunksize)
                key_sets[(path.name, key_col)] = values
                rows.append(
                    {
                        "file": path.name,
                        "key_column": key_col,
                        "rows": row_count,
                        "unique_values": len(values),
                        "duplicate_key_rows": row_count - len(values),
                        "is_unique_key": row_count == len(values),
                    }
                )
        return pd.DataFrame(rows).sort_values(["key_column", "file"]), key_sets

    @staticmethod
    def relationship_row(left: str, right: str, key_col: str, key_sets: dict[tuple[str, str], set]) -> dict[str, object]:
        """Summarize overlap between two files on one key column."""
        left_values = key_sets.get((left, key_col), set())
        right_values = key_sets.get((right, key_col), set())
        overlap = left_values & right_values
        return {
            "left_file": left,
            "right_file": right,
            "key": key_col,
            "left_unique": len(left_values),
            "right_unique": len(right_values),
            "overlap": len(overlap),
            "left_coverage_in_right": len(overlap) / len(left_values) if left_values else np.nan,
            "right_coverage_in_left": len(overlap) / len(right_values) if right_values else np.nan,
        }

    def relationship_summary(self, key_sets: dict[tuple[str, str], set]) -> pd.DataFrame:
        """Summarize important KuaiRand table relationships."""
        standard_early = "log_standard_4_08_to_4_21_1k.csv"
        standard_late = "log_standard_4_22_to_5_08_1k.csv"
        random_log = "log_random_4_22_to_5_08_1k.csv"
        user_features = "user_features_1k.csv"
        video_basic = "video_features_basic_1k.csv"
        video_stats = "video_features_statistic_1k.csv"
        specs = [
            (standard_early, user_features, "user_id"),
            (standard_late, user_features, "user_id"),
            (random_log, user_features, "user_id"),
            (standard_early, video_basic, "video_id"),
            (standard_late, video_basic, "video_id"),
            (random_log, video_basic, "video_id"),
            (standard_early, video_stats, "video_id"),
            (standard_late, video_stats, "video_id"),
            (random_log, video_stats, "video_id"),
            (video_basic, video_stats, "video_id"),
            (standard_early, standard_late, "user_id"),
            (standard_early, standard_late, "video_id"),
            (standard_late, random_log, "user_id"),
            (standard_late, random_log, "video_id"),
        ]
        return pd.DataFrame([self.relationship_row(*spec, key_sets=key_sets) for spec in specs])

    def load_standard_logs(self, nrows_per_file: int | None = None) -> pd.DataFrame:
        """Load standard recommendation logs and add event_ts."""
        frames = []
        for path in self.config.standard_log_files:
            frame = pd.read_csv(path, usecols=self.log_usecols, nrows=nrows_per_file)
            frame["source_file"] = path.name
            frames.append(frame)
        logs = pd.concat(frames, ignore_index=True)
        logs["event_ts"] = pd.to_datetime(logs["time_ms"], unit="ms", errors="coerce")
        return logs.sort_values("event_ts").reset_index(drop=True)


@dataclass
class LabelConfig:
    """Configuration for observed objective-label construction."""

    watch_ratio_relevance_threshold: float = 0.7
    min_play_time_ms_for_relevance: int = 5_000
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "long_watch": 1.00,
            "watch_ratio": 0.75,
            "like": 0.35,
            "follow": 0.50,
            "comment": 0.20,
            "share": 0.20,
            "negative_feedback": 1.00,
        }
    )


class ObjectiveLabeler:
    """Create primary and secondary recommendation labels from interaction logs."""

    def __init__(self, config: LabelConfig | None = None):
        self.config = config or LabelConfig()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add objective labels and observed utility to an interaction DataFrame."""
        labeled = df.copy()
        labeled["duration_ms_safe"] = labeled["duration_ms"].where(labeled["duration_ms"] > 0, np.nan)
        labeled["watch_ratio_raw"] = labeled["play_time_ms"] / labeled["duration_ms_safe"]
        labeled["watch_ratio"] = labeled["watch_ratio_raw"].replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 5)
        labeled["completion_rate"] = labeled["watch_ratio"].clip(0, 1)
        labeled["long_watch_label"] = labeled["long_view"].fillna(0).astype(int)
        labeled["watch_ratio_relevant"] = (
            (labeled["watch_ratio"] >= self.config.watch_ratio_relevance_threshold)
            & (labeled["play_time_ms"] >= self.config.min_play_time_ms_for_relevance)
        ).astype(int)
        labeled["primary_relevance_label"] = (
            (labeled["long_watch_label"] == 1) | (labeled["watch_ratio_relevant"] == 1)
        ).astype(int)
        labeled["like_label"] = labeled["is_like"].fillna(0).astype(int)
        labeled["follow_label"] = labeled["is_follow"].fillna(0).astype(int)
        labeled["comment_label"] = labeled["is_comment"].fillna(0).astype(int)
        labeled["share_label"] = labeled["is_forward"].fillna(0).astype(int)
        labeled["negative_feedback_label"] = labeled["is_hate"].fillna(0).astype(int)
        weights = self.config.weights
        labeled["observed_utility"] = (
            weights["long_watch"] * labeled["long_watch_label"]
            + weights["watch_ratio"] * labeled["completion_rate"]
            + weights["like"] * labeled["like_label"]
            + weights["follow"] * labeled["follow_label"]
            + weights["comment"] * labeled["comment_label"]
            + weights["share"] * labeled["share_label"]
            - weights["negative_feedback"] * labeled["negative_feedback_label"]
        )
        return labeled

    @staticmethod
    def summary(df: pd.DataFrame) -> pd.DataFrame:
        """Summarize the main objective labels."""
        label_cols = [
            "long_watch_label",
            "watch_ratio",
            "completion_rate",
            "watch_ratio_relevant",
            "primary_relevance_label",
            "like_label",
            "follow_label",
            "comment_label",
            "share_label",
            "negative_feedback_label",
            "observed_utility",
        ]
        return pd.DataFrame({"mean": df[label_cols].mean(), "missing": df[label_cols].isna().sum()})


class TemporalSplitter:
    """Create leakage-aware chronological train/validation splits."""

    def __init__(self, timestamp_column: str = "event_ts", validation_days: int = 3):
        self.timestamp_column = timestamp_column
        self.validation_days = validation_days

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
        """Split by the most recent validation window."""
        max_ts = df[self.timestamp_column].max()
        cutoff = max_ts - pd.Timedelta(days=self.validation_days)
        train = df[df[self.timestamp_column] < cutoff].copy()
        validation = df[df[self.timestamp_column] >= cutoff].copy()
        if train.empty or validation.empty:
            raise ValueError("Empty train or validation split. Increase sample size or reduce validation_days.")
        assert train[self.timestamp_column].max() < validation[self.timestamp_column].min()
        return train.reset_index(drop=True), validation.reset_index(drop=True), cutoff

    def summary(self, train: pd.DataFrame, validation: pd.DataFrame, user_col: str, item_col: str) -> pd.DataFrame:
        """Return a compact split summary."""
        return pd.DataFrame(
            [
                {
                    "split": "train",
                    "rows": len(train),
                    "users": train[user_col].nunique(),
                    "items": train[item_col].nunique(),
                    "min_ts": train[self.timestamp_column].min(),
                    "max_ts": train[self.timestamp_column].max(),
                    "primary_relevance_rate": train["primary_relevance_label"].mean(),
                },
                {
                    "split": "validation",
                    "rows": len(validation),
                    "users": validation[user_col].nunique(),
                    "items": validation[item_col].nunique(),
                    "min_ts": validation[self.timestamp_column].min(),
                    "max_ts": validation[self.timestamp_column].max(),
                    "primary_relevance_rate": validation["primary_relevance_label"].mean(),
                },
            ]
        )


class PopularityRecommender:
    """Training-only popularity/trending candidate generator."""
    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "video_id",
        timestamp_col: str = "event_ts",
        score_col: str = "observed_utility",
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col
        self.score_col = score_col
        self.item_scores_: pd.Series | None = None
        self.source_name_: str | None = None

    def fit(self, train: pd.DataFrame, source_name: str = "global_utility_popularity") -> "PopularityRecommender":
        """Fit item scores using training interactions only."""
        grouped = train.groupby(self.item_col).agg(
            utility_sum = (self.score_col, "sum"),
            utility_mean = (self.score_col, "mean"),
            interaction_count = (self.score_col, "size"),
        )
        grouped["retrieval_score"] = grouped["utility_mean"] * np.log1p(grouped["interaction_count"])
        self.item_scores_ = grouped["retrieval_score"].sort_values(ascending=False)
        self.source_name_ = source_name
        return self

    def fit_recent(
        self,
        train: pd.DataFrame,
        lookback_days: int,
        source_name: str = "recent_utility_popularity",
    ) -> "PopularityRecommender":
        """Fit popularity scores on the most recent part of training only."""
        cutoff = train[self.timestamp_col].max() - pd.Timedelta(days=lookback_days)
        recent_train = train[train[self.timestamp_col] >= cutoff]
        return self.fit(recent_train if not recent_train.empty else train, source_name=source_name)

    @staticmethod
    def build_seen_items(train: pd.DataFrame, user_col: str, item_col: str) -> dict[object, set]:
        """Map each user to items seen in training."""
        return train.groupby(user_col)[item_col].agg(lambda values: set(values)).to_dict()

    def recommend(
        self,
        users: Iterable[object],
        top_k: int,
        seen_items_by_user: dict[object, set] | None = None,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Recommend top-k items for each user from the fitted item score table."""
        if self.item_scores_ is None or self.source_name_ is None:
            raise ValueError("Call fit or fit_recent before recommend.")
        ranked_items = list(self.item_scores_.items())
        rows = []
        for user_id in users:
            seen = seen_items_by_user.get(user_id, set()) if (seen_items_by_user and exclude_seen) else set()
            rank = 1
            for item_id, score in ranked_items:
                if item_id in seen:
                    continue
                rows.append(
                    {
                        self.user_col: user_id,
                        self.item_col: item_id,
                        "rank": rank,
                        "retrieval_score": float(score),
                        "retrieval_source": self.source_name_,
                    }
                )
                rank += 1
                if rank > top_k:
                    break
        return pd.DataFrame(rows)


class ItemToItemRecommender:
    """Co-occurrence-based item-to-item retrieval."""
    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "video_id",
        timestamp_col: str = "event_ts",
        positive_label_col: str = "primary_relevance_label",
        max_history_items_per_user: int = 80,
        max_neighbors_per_item: int = 100,
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col
        self.positive_label_col = positive_label_col
        self.max_history_items_per_user = max_history_items_per_user
        self.max_neighbors_per_item = max_neighbors_per_item
        self.histories_: dict[object, list] = {}
        self.neighbors_: dict[object, list[tuple[object, float]]] = {}

    def fit(self, train: pd.DataFrame) -> "ItemToItemRecommender":
        """Build positive histories and item co-occurrence neighbors from training data."""
        positive = train[train[self.positive_label_col] == 1].sort_values(self.timestamp_col)
        histories = {}
        for user_id, group in positive.groupby(self.user_col):
            items = list(dict.fromkeys(group[self.item_col].tolist()))[-self.max_history_items_per_user :]
            if items:
                histories[user_id] = items
        self.histories_ = histories
        self.neighbors_ = self._build_neighbors(histories)
        return self

    def _build_neighbors(self, histories: dict[object, list]) -> dict[object, list[tuple[object, float]]]:
        """Build cosine-normalized item neighbors from user-history co-occurrence."""
        co_counts: dict[object, Counter] = defaultdict(Counter)
        item_user_counts = Counter()
        for items in histories.values():
            unique_items = list(dict.fromkeys(items))
            for item in unique_items:
                item_user_counts[item] += 1
            for left_idx, left_item in enumerate(unique_items):
                for right_item in unique_items[left_idx + 1 :]:
                    co_counts[left_item][right_item] += 1
                    co_counts[right_item][left_item] += 1
        neighbors = {}
        for item, counter in co_counts.items():
            scored = []
            for neighbor, count in counter.items():
                denom = np.sqrt(item_user_counts[item] * item_user_counts[neighbor])
                scored.append((neighbor, count / denom if denom else 0.0))
            neighbors[item] = sorted(scored, key=lambda pair: pair[1], reverse=True)[: self.max_neighbors_per_item]
        return neighbors

    def recommend(
        self,
        users: Iterable[object],
        top_k: int,
        seen_items_by_user: dict[object, set] | None = None,
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """Recommend items by aggregating neighbors of a user's positive history items."""
        rows = []
        for user_id in users:
            seen = seen_items_by_user.get(user_id, set()) if (seen_items_by_user and exclude_seen) else set()
            candidate_scores = Counter()
            anchors = self.histories_.get(user_id, [])[-self.max_history_items_per_user :]
            for recency_position, anchor_item in enumerate(reversed(anchors), start=1):
                recency_weight = 1.0 / np.sqrt(recency_position)
                for neighbor_item, neighbor_score in self.neighbors_.get(anchor_item, []):
                    if neighbor_item in seen:
                        continue
                    candidate_scores[neighbor_item] += recency_weight * neighbor_score
            for rank, (item_id, score) in enumerate(candidate_scores.most_common(top_k), start=1):
                rows.append(
                    {
                        self.user_col: user_id,
                        self.item_col: item_id,
                        "rank": rank,
                        "retrieval_score": float(score),
                        "retrieval_source": "item_to_item_cooccurrence",
                    }
                )
        return pd.DataFrame(rows)

    def neighbors_for_item(self, item_id: object, top_n: int = 20) -> pd.DataFrame:
        """Return nearest co-occurrence neighbors for one item."""
        rows = [
            {"anchor_item": item_id, "neighbor_item": neighbor, "similarity_score": score, "rank": rank}
            for rank, (neighbor, score) in enumerate(self.neighbors_.get(item_id, [])[:top_n], start=1)
        ]
        return pd.DataFrame(rows)


class RetrievalEvaluator:
    """Evaluate candidate retrieval against validation relevance sets."""

    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "video_id",
        relevance_col: str = "primary_relevance_label",
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.relevance_col = relevance_col

    def ground_truth(self, validation: pd.DataFrame) -> dict[object, set]:
        """Return relevant validation items per user."""
        relevant = validation[validation[self.relevance_col] == 1]
        return relevant.groupby(self.user_col)[self.item_col].agg(lambda values: set(values)).to_dict()

    def recommendations_by_user(self, recs: pd.DataFrame, k: int) -> dict[object, list]:
        """Return ordered top-k recommendations per user."""
        if recs.empty:
            return {}
        top_recs = recs[recs["rank"] <= k].sort_values([self.user_col, "rank"])
        return top_recs.groupby(self.user_col)[self.item_col].agg(list).to_dict()

    def recall_at_k(self, recs: pd.DataFrame, ground_truth: dict[object, set], k: int) -> float:
        """Mean user-level Recall@K for users with relevant validation items."""
        if not ground_truth:
            return float("nan")
        rec_map = self.recommendations_by_user(recs, k)
        values = []
        for user_id, relevant_items in ground_truth.items():
            recommended_items = set(rec_map.get(user_id, []))
            values.append(len(recommended_items & relevant_items) / len(relevant_items))
        return float(np.mean(values)) if values else float("nan")

    def hit_rate_at_k(self, recs: pd.DataFrame, ground_truth: dict[object, set], k: int) -> float:
        """Mean user-level HitRate@K for users with relevant validation items."""
        if not ground_truth:
            return float("nan")
        rec_map = self.recommendations_by_user(recs, k)
        values = []
        for user_id, relevant_items in ground_truth.items():
            recommended_items = set(rec_map.get(user_id, []))
            values.append(float(bool(recommended_items & relevant_items)))
        return float(np.mean(values)) if values else float("nan")

    def catalog_coverage(self, recs: pd.DataFrame, catalog_items: set) -> float:
        """Return the share of training catalog represented in recommendation lists."""
        if not catalog_items:
            return float("nan")
        if recs.empty:
            return 0.0
        return recs[self.item_col].nunique() / len(catalog_items)

    def evaluate(self, named_recs: dict[str, pd.DataFrame], validation: pd.DataFrame, train_catalog: set, k: int) -> pd.DataFrame:
        """Evaluate several retrieval outputs in one comparison table."""
        gt = self.ground_truth(validation)
        rows = []
        for source, recs in named_recs.items():
            rows.append(
                {
                    "retrieval_source": source,
                    "users_with_recs": recs[self.user_col].nunique() if not recs.empty else 0,
                    "recommended_rows": len(recs),
                    f"recall@{k}": self.recall_at_k(recs, gt, k),
                    f"hit_rate@{k}": self.hit_rate_at_k(recs, gt, k),
                    "catalog_coverage": self.catalog_coverage(recs, train_catalog),
                }
            )
        return pd.DataFrame(rows).sort_values(f"recall@{k}", ascending=False)


@dataclass
class TwoTowerConfig:
    """Training and architecture settings for the two-tower retrieval baseline."""

    embedding_dim: int = 32
    hidden_dim: int = 64
    batch_size: int = 2048
    epochs: int = 3
    learning_rate: float = 1e-3
    negatives_per_positive: int = 3
    max_positive_interactions: int = 50_000
    random_seed: int = 42
    device: str | None = None

    def resolved_device(self) -> str:
        """Return the requested device, falling back to available local hardware."""
        if self.device:
            return self.device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


class IDMap:
    """Map raw KuaiRand IDs to contiguous integer indices for embeddings."""

    def __init__(self, values: Iterable[object]):
        unique_values = list(dict.fromkeys(values))
        self.id_to_index = {value: idx for idx, value in enumerate(unique_values)}
        self.index_to_id = {idx: value for value, idx in self.id_to_index.items()}

    def __len__(self) -> int:
        return len(self.id_to_index)

    def transform(self, values: Iterable[object]) -> np.ndarray:
        """Convert raw IDs to embedding indices. Unknown IDs become -1."""
        return np.array([self.id_to_index.get(value, -1) for value in values], dtype=np.int64)

    def inverse_transform(self, indices: Iterable[int]) -> list[object]:
        """Convert embedding indices back to raw IDs."""
        return [self.index_to_id[int(index)] for index in indices]


class TwoTowerInteractionDataset(Dataset):
    """PyTorch dataset of user indices, item indices, and binary labels."""

    def __init__(self, user_indices: np.ndarray, item_indices: np.ndarray, labels: np.ndarray):
        self.user_indices = torch.as_tensor(user_indices, dtype=torch.long)
        self.item_indices = torch.as_tensor(item_indices, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.user_indices[idx], self.item_indices[idx], self.labels[idx]


class TwoTowerRetrievalModel(nn.Module):
    """Small two-tower retrieval model using user and item embedding towers."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def encode_users(self, user_indices: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized user embeddings."""
        user_vectors = self.user_tower(self.user_embedding(user_indices))
        return nn.functional.normalize(user_vectors, dim=-1)

    def encode_items(self, item_indices: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized item embeddings."""
        item_vectors = self.item_tower(self.item_embedding(item_indices))
        return nn.functional.normalize(item_vectors, dim=-1)

    def forward(self, user_indices: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        """Return dot-product logits for user-item pairs."""
        user_vectors = self.encode_users(user_indices)
        item_vectors = self.encode_items(item_indices)
        return (user_vectors * item_vectors).sum(dim=-1)


class TwoTowerRecommender:
    """End-to-end helper for training and retrieving from a two-tower model."""

    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "video_id",
        timestamp_col: str = "event_ts",
        positive_label_col: str = "primary_relevance_label",
        config: TwoTowerConfig | None = None,
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col
        self.positive_label_col = positive_label_col
        self.config = config or TwoTowerConfig()
        self.user_map_: IDMap | None = None
        self.item_map_: IDMap | None = None
        self.model_: TwoTowerRetrievalModel | None = None
        self.training_history_: pd.DataFrame | None = None
        self.seen_items_by_user_: dict[object, set] = {}

    def build_training_examples(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build positive and sampled-negative training pairs from training data only."""
        rng = np.random.default_rng(self.config.random_seed)
        positive = (
            train[train[self.positive_label_col] == 1][[self.user_col, self.item_col, self.timestamp_col]]
            .drop_duplicates([self.user_col, self.item_col])
            .sort_values(self.timestamp_col)
        )
        if len(positive) > self.config.max_positive_interactions:
            positive = positive.sample(self.config.max_positive_interactions, random_state=self.config.random_seed)

        self.seen_items_by_user_ = PopularityRecommender.build_seen_items(train, self.user_col, self.item_col)
        item_pool = np.array(sorted(train[self.item_col].unique()))
        positive_users = positive[self.user_col].to_numpy()
        positive_items = positive[self.item_col].to_numpy()

        users = positive_users.tolist()
        items = positive_items.tolist()
        labels = [1.0] * len(positive)

        for user_id, positive_item in zip(positive_users, positive_items):
            seen = self.seen_items_by_user_.get(user_id, set())
            for _ in range(self.config.negatives_per_positive):
                negative_item = rng.choice(item_pool)
                retry_count = 0
                while negative_item in seen and retry_count < 50:
                    negative_item = rng.choice(item_pool)
                    retry_count += 1
                if negative_item == positive_item:
                    continue
                users.append(user_id)
                items.append(negative_item)
                labels.append(0.0)

        self.user_map_ = IDMap(sorted(train[self.user_col].unique()))
        self.item_map_ = IDMap(sorted(train[self.item_col].unique()))
        user_indices = self.user_map_.transform(users)
        item_indices = self.item_map_.transform(items)
        label_array = np.array(labels, dtype=np.float32)
        valid = (user_indices >= 0) & (item_indices >= 0)
        return user_indices[valid], item_indices[valid], label_array[valid]

    def fit(self, train: pd.DataFrame) -> "TwoTowerRecommender":
        """Train the two-tower model on positive and sampled-negative pairs."""
        torch.manual_seed(self.config.random_seed)
        user_indices, item_indices, labels = self.build_training_examples(train)
        if self.user_map_ is None or self.item_map_ is None:
            raise ValueError("ID maps were not initialized.")

        dataset = TwoTowerInteractionDataset(user_indices, item_indices, labels)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        device = self.config.resolved_device()
        model = TwoTowerRetrievalModel(
            n_users=len(self.user_map_),
            n_items=len(self.item_map_),
            embedding_dim=self.config.embedding_dim,
            hidden_dim=self.config.hidden_dim,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        history = []

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            total_loss = 0.0
            total_examples = 0
            for batch_users, batch_items, batch_labels in loader:
                batch_users = batch_users.to(device)
                batch_items = batch_items.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                logits = model(batch_users, batch_items)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()
                batch_size = len(batch_labels)
                total_loss += float(loss.detach().cpu()) * batch_size
                total_examples += batch_size
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / total_examples if total_examples else np.nan,
                    "examples": total_examples,
                    "device": device,
                }
            )

        self.model_ = model
        self.training_history_ = pd.DataFrame(history)
        return self

    def item_embedding_table(self, batch_size: int = 65_536) -> tuple[np.ndarray, np.ndarray]:
        """Return raw item IDs and normalized item embeddings for all training-catalog items."""
        if self.model_ is None or self.item_map_ is None:
            raise ValueError("Call fit before extracting item embeddings.")
        device = self.config.resolved_device()
        self.model_.eval()
        all_item_indices = np.arange(len(self.item_map_), dtype=np.int64)
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(all_item_indices), batch_size):
                batch = torch.as_tensor(all_item_indices[start : start + batch_size], dtype=torch.long, device=device)
                vectors = self.model_.encode_items(batch).cpu().numpy()
                embeddings.append(vectors)
        raw_item_ids = np.array(self.item_map_.inverse_transform(all_item_indices))
        return raw_item_ids, np.vstack(embeddings)

    def recommend(
        self,
        users: Iterable[object],
        top_k: int,
        seen_items_by_user: dict[object, set] | None = None,
        exclude_seen: bool = True,
        source_name: str = "two_tower_exact_dot_product",
    ) -> pd.DataFrame:
        """Retrieve top-k items by exact dot product over learned item embeddings."""
        if self.model_ is None or self.user_map_ is None:
            raise ValueError("Call fit before recommend.")
        raw_item_ids, item_embeddings = self.item_embedding_table()
        device = self.config.resolved_device()
        rows = []
        self.model_.eval()
        with torch.no_grad():
            for user_id in users:
                user_index = self.user_map_.id_to_index.get(user_id)
                if user_index is None:
                    continue
                user_tensor = torch.as_tensor([user_index], dtype=torch.long, device=device)
                user_vector = self.model_.encode_users(user_tensor).cpu().numpy()[0]
                scores = item_embeddings @ user_vector
                if exclude_seen:
                    seen = seen_items_by_user.get(user_id, set()) if seen_items_by_user else self.seen_items_by_user_.get(user_id, set())
                    if seen:
                        seen_mask = np.isin(raw_item_ids, list(seen))
                        scores[seen_mask] = -np.inf
                candidate_count = min(top_k, len(scores))
                if candidate_count == 0:
                    continue
                top_indices = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
                top_indices = top_indices[np.argsort(-scores[top_indices])]
                for rank, idx in enumerate(top_indices, start=1):
                    if not np.isfinite(scores[idx]):
                        continue
                    rows.append(
                        {
                            self.user_col: user_id,
                            self.item_col: raw_item_ids[idx],
                            "rank": rank,
                            "retrieval_score": float(scores[idx]),
                            "retrieval_source": source_name,
                        }
                    )
        return pd.DataFrame(rows)
