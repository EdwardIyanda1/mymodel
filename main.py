"""Explore FUNAAB ICTREC helpdesk data and train a PyTorch regression model.

Target: Resolution_Time_Hrs
Only features plausibly known when a ticket is opened are used to avoid leakage.

IMPORTANT DATA CAVEAT (fixed here)
-----------------------------------
`Resolution_Time_Hrs` is NOT comparable across all rows. For tickets that are
already Closed/Resolved, it is the true time-to-resolution. For tickets that
are still Pending or In Progress, the same column is ~4x higher on average
(28.4 hrs vs 7.4 hrs) -- because for an open ticket it can only be "hours
elapsed so far", not a finished duration. Training on the raw column mixes
two different quantities and biases a model toward predicting artificially
long resolution times.

Fix: the regression model is trained/evaluated only on tickets where
Resolved == "Yes". Unresolved tickets are still reported on in the
exploration summary (e.g. resolution rate) but excluded from modeling.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 42
TARGET = "Resolution_Time_Hrs"
CAT_FEATURES = ["User_Type", "Unit_Dept", "Issue_Category", "Priority"]
NUM_FEATURES = ["month", "day_of_week"]
PRIORITY_ORDER = ["Low", "Medium", "High", "Critical"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_and_prepare(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the CSV and split it into a full frame (for exploration) and a
    resolved-only frame (for modeling), which avoids the censoring issue
    described in the module docstring.
    """
    df = pd.read_csv(csv_path)
    required = set(CAT_FEATURES + ["Date", TARGET, "Resolved"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek

    resolved_df = df[df["Resolved"] == "Yes"].copy()
    X = resolved_df[CAT_FEATURES + NUM_FEATURES].copy()
    y = resolved_df[TARGET].astype("float32")
    return df, resolved_df, X, y


def save_exploration(df: pd.DataFrame, resolved_df: pd.DataFrame, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(df)
    n_resolved = len(resolved_df)
    response_resolution_corr = resolved_df[["Response_Time_Hrs", TARGET]].corr().iloc[0, 1]
    monthly_volume = df.groupby("month").size().to_dict()
    summary = {
        "rows_total": int(n_total),
        "rows_resolved_used_for_modeling": int(n_resolved),
        "rows_excluded_unresolved": int(n_total - n_resolved),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_ticket_ids": int(df["Ticket_ID"].duplicated().sum()),
        "date_start": str(df["Date"].min().date()),
        "date_end": str(df["Date"].max().date()),
        "mean_response_hours": round(float(df["Response_Time_Hrs"].mean()), 2),
        "median_response_hours": round(float(df["Response_Time_Hrs"].median()), 2),
        # Resolution-time stats are reported for resolved tickets only --
        # see module docstring for why mixing in open tickets is misleading.
        "mean_resolution_hours_resolved_only": round(float(resolved_df[TARGET].mean()), 2),
        "median_resolution_hours_resolved_only": round(float(resolved_df[TARGET].median()), 2),
        "mean_elapsed_hours_unresolved_tickets": round(
            float(df.loc[df["Resolved"] == "No", TARGET].mean()), 2
        ),
        "resolved_rate_percent": round(float(df["Resolved"].eq("Yes").mean() * 100), 1),
        "response_sla_rate_percent": round(float(df["Response_SLA"].eq("Within 8 Hours").mean() * 100), 1),
        # Weak/negative -> a fast first response does not predict a fast resolution.
        "response_vs_resolution_correlation_resolved_only": round(float(response_resolution_corr), 3),
        "monthly_ticket_volume": {int(k): int(v) for k, v in monthly_volume.items()},
        "note": (
            "Resolution-time figures use Resolved=='Yes' tickets only. For the "
            "remaining tickets, Resolution_Time_Hrs reflects hours elapsed so far "
            "on a still-open ticket, not a finished duration, so it is reported "
            "separately rather than averaged in."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sns.set_theme(style="whitegrid")
    status_order = ["Closed", "In Progress", "Pending"]
    monthly_counts = df.groupby("month").size().reindex(range(1, 13)).dropna()
    dept_order = resolved_df.groupby("Unit_Dept")[TARGET].mean().sort_values(ascending=False).index

    # Each entry is (filename, title, draw_fn). draw_fn(ax) renders one chart
    # onto whichever Axes it's given -- reused for both the individual PNGs
    # and the combined overview figure so the two can never drift apart.
    charts: list[tuple[str, str, "callable"]] = [
        (
            "01_tickets_by_issue_category.png",
            "Tickets by issue category (all tickets)",
            lambda ax: sns.countplot(
                data=df, y="Issue_Category", order=df["Issue_Category"].value_counts().index,
                ax=ax, color="#2563eb",
            ),
        ),
        (
            "02_tickets_by_priority.png",
            "Tickets by priority (all tickets)",
            lambda ax: sns.countplot(data=df, x="Priority", order=PRIORITY_ORDER, ax=ax, color="#0f766e"),
        ),
        (
            "03_resolution_time_by_status.png",
            "Resolution_Time_Hrs by status\n(Pending/In Progress = time elapsed so far, not resolved)",
            lambda ax: sns.boxplot(data=df, x="Status", y=TARGET, order=status_order, ax=ax, color="#f97316"),
        ),
        (
            "04_ticket_volume_by_month.png",
            "Ticket volume by month",
            lambda ax: (
                ax.bar(monthly_counts.index, monthly_counts.values, color="#0891b2"),
                ax.set_xlabel("Month"),
                ax.set_ylabel("Tickets opened"),
                ax.set_xticks(list(monthly_counts.index)),
            ),
        ),
        (
            "05_resolution_time_distribution.png",
            "Resolution-time distribution (resolved tickets only)",
            lambda ax: sns.histplot(data=resolved_df, x=TARGET, bins=25, kde=True, ax=ax, color="#d97706"),
        ),
        (
            "06_resolution_time_by_priority.png",
            "Resolution time by priority (resolved tickets only)",
            lambda ax: sns.boxplot(
                data=resolved_df, x="Priority", y=TARGET, order=PRIORITY_ORDER, ax=ax, color="#93c5fd",
            ),
        ),
        (
            "07_resolution_time_by_department.png",
            "Mean resolution time by department (resolved tickets only)",
            lambda ax: sns.barplot(
                data=resolved_df, y="Unit_Dept", x=TARGET, order=dept_order, ax=ax, color="#7c3aed",
                errorbar=None,
            ),
        ),
        (
            "08_response_vs_resolution_time.png",
            f"Response vs. resolution time (resolved only)\ncorrelation = {response_resolution_corr:.2f} "
            "-- fast response doesn't imply fast resolution",
            lambda ax: sns.regplot(
                data=resolved_df, x="Response_Time_Hrs", y=TARGET, ax=ax,
                scatter_kws={"color": "#be185d", "alpha": 0.5, "s": 20}, line_kws={"color": "#1f2937"},
            ),
        ),
    ]

    # Individual images -- one file per chart.
    for filename, title, draw_fn in charts:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        draw_fn(ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Combined overview -- all charts together in one image.
    fig, axes = plt.subplots(2, 4, figsize=(24, 9))
    for (filename, title, draw_fn), ax in zip(charts, axes.flat):
        draw_fn(ax)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "helpdesk_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


class ResolutionMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(X).cpu().numpy()
    actual = y.cpu().numpy()
    return {
        "MAE_hours": float(mean_absolute_error(actual, pred)),
        "RMSE_hours": float(mean_squared_error(actual, pred) ** 0.5),
        "R2": float(r2_score(actual, pred)),
    }


def train_model(X: pd.DataFrame, y: pd.Series, output_dir: Path, epochs: int) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.20, random_state=SEED
    )
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("numeric", StandardScaler(), NUM_FEATURES),
    ])
    train_np = np.ascontiguousarray(preprocessor.fit_transform(X_train).astype("float32"))
    val_np = np.ascontiguousarray(preprocessor.transform(X_val).astype("float32"))
    test_np = np.ascontiguousarray(preprocessor.transform(X_test).astype("float32"))
    train_x, val_x, test_x = map(torch.from_numpy, [train_np, val_np, test_np])
    train_y = torch.from_numpy(y_train.to_numpy(dtype="float32").copy())
    val_y = torch.from_numpy(y_val.to_numpy(dtype="float32").copy())
    test_y = torch.from_numpy(y_test.to_numpy(dtype="float32").copy())

    model = ResolutionMLP(train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    best_state, best_val, patience_left = None, float("inf"), 30
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_x), val_y).item()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-4:
            best_val, best_state, patience_left = val_loss, copy.deepcopy(model.state_dict()), 30
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    model.load_state_dict(best_state)

    metrics = evaluate(model, test_x, test_y)
    baseline = np.full(len(y_test), y_train.median(), dtype="float32")
    metrics["baseline_MAE_hours"] = float(mean_absolute_error(y_test, baseline))
    metrics["epochs_trained"] = len(history)
    metrics["n_train"] = int(len(y_train))
    metrics["n_val"] = int(len(y_val))
    metrics["n_test"] = int(len(y_test))
    metrics["note"] = (
        "Trained on Resolved=='Yes' tickets only. Sample size is small, so treat "
        "R2/MAE as indicative rather than a precise estimate -- consider k-fold "
        "cross-validation before reporting a single number."
    )
    torch.save({"model_state": model.state_dict(), "input_dim": train_x.shape[1]}, output_dir / "resolution_model.pt")
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    hist = pd.DataFrame(history)
    ax = hist.plot(x="epoch", y=["train_loss", "val_loss"], figsize=(8, 5), color=["#2563eb", "#dc2626"])
    ax.set(title="Training and validation loss", ylabel="Huber loss")
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "training_history.png", dpi=180)
    plt.close(ax.figure)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("FUNAAB_ICTREC_Helpdesk_500_Instances.csv"))
    parser.add_argument("--output", type=Path, default=Path("analysis_output"))
    parser.add_argument("--epochs", type=int, default=400)
    args = parser.parse_args()
    seed_everything()
    df, resolved_df, X, y = load_and_prepare(args.csv)
    print(
        f"Using {len(resolved_df)} resolved tickets for modeling "
        f"(excluded {len(df) - len(resolved_df)} still-open tickets)."
    )
    print("Observed summary:", json.dumps(save_exploration(df, resolved_df, args.output), indent=2))
    print("Model metrics:", json.dumps(train_model(X, y, args.output, args.epochs), indent=2))


if __name__ == "__main__":
    main()