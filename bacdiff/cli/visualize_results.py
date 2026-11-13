#!/usr/bin/env python3
"""可视化反事实预测结果，展示模型预测效果。"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from bacdiff.utils.config import load_config, resolve_run_dir


def plot_dose_response_curves(dose_df: pd.DataFrame, save_dir: Path, top_n: int = 10):
    """绘制剂量响应曲线：不同细菌剂量下的平均预测表达量。"""
    # 计算每个剂量的平均表达量
    dose_summary = dose_df.groupby("dose_scale")["mean_pred_expression"].agg(["mean", "std"]).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 平均表达量 vs 剂量
    ax = axes[0]
    ax.plot(dose_summary["dose_scale"], dose_summary["mean"], "o-", color="black", linewidth=2, markersize=8)
    ax.fill_between(
        dose_summary["dose_scale"],
        dose_summary["mean"] - dose_summary["std"],
        dose_summary["mean"] + dose_summary["std"],
        alpha=0.3, color="black"
    )
    ax.set_xlabel("Bacteria Dose Scale", fontsize=12)
    ax.set_ylabel("Mean Predicted Expression", fontsize=12)
    ax.set_title("Dose-Response: Global Average", fontsize=14)
    ax.grid(True, alpha=0.3)

    # 2. Delta norm vs 剂量
    ax = axes[1]
    delta_summary = dose_df.groupby("dose_scale")["delta_norm"].agg(["mean", "std"]).reset_index()
    ax.plot(delta_summary["dose_scale"], delta_summary["mean"], "s-", color="blue", linewidth=2, markersize=8)
    ax.fill_between(
        delta_summary["dose_scale"],
        delta_summary["mean"] - delta_summary["std"],
        delta_summary["mean"] + delta_summary["std"],
        alpha=0.3, color="blue"
    )
    ax.set_xlabel("Bacteria Dose Scale", fontsize=12)
    ax.set_ylabel("Delta Norm (Latent Shift)", fontsize=12)
    ax.set_title("Latent Space Shift vs Dose", fontsize=14)
    ax.grid(True, alpha=0.3)

    # 3. Effect norm vs 剂量
    ax = axes[2]
    effect_summary = dose_df.groupby("dose_scale")["effect_norm"].agg(["mean", "std"]).reset_index()
    ax.plot(effect_summary["dose_scale"], effect_summary["mean"], "^-", color="red", linewidth=2, markersize=8)
    ax.fill_between(
        effect_summary["dose_scale"],
        effect_summary["mean"] - effect_summary["std"],
        effect_summary["mean"] + effect_summary["std"],
        alpha=0.3, color="red"
    )
    ax.set_xlabel("Bacteria Dose Scale", fontsize=12)
    ax.set_ylabel("Effect Norm (Expression Change)", fontsize=12)
    ax.set_title("Expression Change vs Dose", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "dose_response_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_dir / 'dose_response_curves.png'}")


def plot_cell_delta_norm(dose_df: pd.DataFrame, save_dir: Path, max_cells: int = 20):
    """绘制每个细胞的delta norm随剂量的变化。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 细胞级 delta norm vs 剂量
    ax = axes[0]
    cell_ids = dose_df["cell_id"].unique()[:max_cells]

    for cell_id in cell_ids:
        cell_data = dose_df[dose_df["cell_id"] == cell_id]
        ax.plot(cell_data["dose_scale"], cell_data["delta_norm"], alpha=0.3, linewidth=1)

    # 平均线
    mean_line = dose_df.groupby("dose_scale")["delta_norm"].mean()
    ax.plot(mean_line.index, mean_line.values, color="black", linewidth=2, label="Mean")
    ax.set_xlabel("Bacteria Dose Scale", fontsize=12)
    ax.set_ylabel("Delta Norm", fontsize=12)
    ax.set_title("Cell-Level Latent Shift", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 细胞级 effect norm vs 剂量
    ax = axes[1]
    for cell_id in cell_ids:
        cell_data = dose_df[dose_df["cell_id"] == cell_id]
        ax.plot(cell_data["dose_scale"], cell_data["effect_norm"], alpha=0.3, linewidth=1)

    mean_effect = dose_df.groupby("dose_scale")["effect_norm"].mean()
    ax.plot(mean_effect.index, mean_effect.values, color="black", linewidth=2, label="Mean")
    ax.set_xlabel("Bacteria Dose Scale", fontsize=12)
    ax.set_ylabel("Effect Norm", fontsize=12)
    ax.set_title("Cell-Level Expression Change", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "cell_delta_norm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_dir / 'cell_delta_norm.png'}")


def plot_gene_effects(gene_effects_df: pd.DataFrame, save_dir: Path, top_n: int = 20):
    """绘制基因级别的细菌效应。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Top affected genes (by absolute mean effect)
    ax = axes[0]
    top_genes = gene_effects_df.nlargest(top_n, "abs_mean_effect")
    y_pos = np.arange(len(top_genes))
    ax.barh(y_pos, top_genes["mean_effect"], color=["red" if x > 0 else "blue" for x in top_genes["mean_effect"]])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_genes["gene"], fontsize=10)
    ax.set_xlabel("Mean Effect (log2 fold change)", fontsize=12)
    ax.set_title(f"Top {top_n} Bacteria-Responsive Genes", fontsize=14)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3, axis="x")

    # 2. Dose slope (sensitivity to bacteria dose)
    ax = axes[1]
    top_slope = gene_effects_df.nlargest(top_n, "dose_slope")
    y_pos = np.arange(len(top_slope))
    ax.barh(y_pos, top_slope["dose_slope"], color=["orange" if x > 0 else "cyan" for x in top_slope["dose_slope"]])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_slope["gene"], fontsize=10)
    ax.set_xlabel("Dose Slope (Sensitivity)", fontsize=12)
    ax.set_title(f"Top {top_n} Dose-Sensitive Genes", fontsize=14)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(save_dir / "gene_effects.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_dir / 'gene_effects.png'}")


def plot_cell_summary(cell_df: pd.DataFrame, save_dir: Path):
    """绘制细胞级别的反事实分析总结。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Delta norm vs 细菌负荷
    ax = axes[0]
    scatter = ax.scatter(
        cell_df["observed_bacteria_total_load"],
        cell_df["delta_norm_observed"],
        c=cell_df["effect_norm"],
        cmap="viridis",
        alpha=0.6,
        s=30
    )
    plt.colorbar(scatter, ax=ax, label="Effect Norm")
    ax.set_xlabel("Observed Bacteria Load (log1p)", fontsize=12)
    ax.set_ylabel("Delta Norm (Latent Shift)", fontsize=12)
    ax.set_title("Latent Shift vs Bacteria Load", fontsize=14)
    ax.grid(True, alpha=0.3)

    # 2. Effect norm vs 细菌负荷
    ax = axes[1]
    ax.scatter(
        cell_df["observed_bacteria_total_load"],
        cell_df["effect_norm"],
        alpha=0.6, s=30, color="red"
    )
    ax.set_xlabel("Observed Bacteria Load (log1p)", fontsize=12)
    ax.set_ylabel("Effect Norm (Expression Change)", fontsize=12)
    ax.set_title("Expression Change vs Bacteria Load", fontsize=14)
    ax.grid(True, alpha=0.3)

    # 3. Delta norm: zero vs observed
    ax = axes[2]
    ax.scatter(
        cell_df["delta_norm_zero"],
        cell_df["delta_norm_observed"],
        alpha=0.6, s=30, color="green"
    )
    max_val = max(cell_df["delta_norm_zero"].max(), cell_df["delta_norm_observed"].max())
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Delta Norm (Zero Bacteria)", fontsize=12)
    ax.set_ylabel("Delta Norm (Observed Bacteria)", fontsize=12)
    ax.set_title("Latent Shift: Zero vs Observed", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "cell_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_dir / 'cell_summary.png'}")


def plot_top_gene_dose_curves(dose_df: pd.DataFrame, gene_effects_df: pd.DataFrame,
                               save_dir: Path, top_n: int = 5):
    """绘制Top基因的剂量响应曲线（使用平均表达量）。"""
    # 获取top基因
    top_genes = gene_effects_df.nlargest(top_n, "abs_mean_effect")["gene"].tolist()

    fig, axes = plt.subplots(1, top_n, figsize=(4 * top_n, 4))
    if top_n == 1:
        axes = [axes]

    for idx, gene in enumerate(top_genes):
        ax = axes[idx]
        # 使用 mean_pred_expression 作为全局平均
        mean_by_dose = dose_df.groupby("dose_scale")["mean_pred_expression"].agg(["mean", "std"]).reset_index()

        ax.plot(mean_by_dose["dose_scale"], mean_by_dose["mean"], "o-", color="black", linewidth=2, markersize=6)
        ax.fill_between(
            mean_by_dose["dose_scale"],
            mean_by_dose["mean"] - mean_by_dose["std"],
            mean_by_dose["mean"] + mean_by_dose["std"],
            alpha=0.3, color="gray"
        )

        ax.set_xlabel("Dose", fontsize=10)
        ax.set_ylabel("Mean Expression", fontsize=10)
        ax.set_title(f"{gene}\n(effect={gene_effects_df.loc[gene_effects_df['gene']==gene, 'mean_effect'].values[0]:.3f})", fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "top_gene_dose_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_dir / 'top_gene_dose_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize counterfactual prediction results")
    parser.add_argument("--config", required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = resolve_run_dir(config)
    save_dir = run_dir / "counterfactual"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {save_dir}")

    # 加载数据
    dose_df = pd.read_csv(save_dir / "dose_scan_long.csv")
    gene_effects_df = pd.read_csv(save_dir / "gene_level_effects.csv")
    cell_df = pd.read_csv(save_dir / "cell_level_counterfactual.csv")

    print(f"Dose scan: {len(dose_df)} rows, {dose_df['cell_id'].nunique()} cells")
    print(f"Gene effects: {len(gene_effects_df)} genes")
    print(f"Cell summary: {len(cell_df)} cells")

    # 生成可视化
    print("\nGenerating visualizations...")
    plot_dose_response_curves(dose_df, save_dir)
    plot_cell_delta_norm(dose_df, save_dir)
    plot_gene_effects(gene_effects_df, save_dir)
    plot_cell_summary(cell_df, save_dir)
    plot_top_gene_dose_curves(dose_df, gene_effects_df, save_dir)

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Results saved to: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
