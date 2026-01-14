import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']


def plot_pic50_distributions(df, save_dir="save/", title="Neuraminidase pIC50 Distribution"):
    COLOR_HIST = "#17BB17"
    COLOR_BOX = "#AC0808"
    COLOR_VIOLIN = "#17BB17"
    DOT_COLOR = "#AC0808"
    FONT_TITLE = 24
    FONT_LABELS = 20
    FONT_TICKS = 16
    FONT_LEGEND = 16
    DOT_SIZE = 40
    ALPHA = 0.8
    FIG_SIZE = (18,6)
    DPI = 400

    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{title.replace(' ', '_')}.svg")

    sns.set_style("white")
    fig,axes = plt.subplots(1,3, figsize=FIG_SIZE, dpi=DPI)

    # Histogram + KDE
    sns.histplot(df["pIC50"], bins=30, kde=True, color=COLOR_HIST, alpha=0.8, ax=axes[0])
    axes[0].set_title("Histogram + KDE", fontsize=FONT_TITLE, fontweight='bold', pad=15)
    axes[0].set_xlabel("pIC50", fontsize=FONT_LABELS, labelpad=10)
    axes[0].set_ylabel("Count", fontsize=FONT_LABELS, labelpad=10)
    axes[0].tick_params(axis='both', labelsize=FONT_TICKS)

    # Boxplot
    sns.boxplot(x=df["pIC50"], color=COLOR_BOX, ax=axes[1])
    axes[1].set_title("Boxplot", fontsize=FONT_TITLE, fontweight='bold', pad=15)
    axes[1].set_xlabel("pIC50", fontsize=FONT_LABELS, labelpad=10)
    axes[1].tick_params(axis='x', labelsize=FONT_TICKS)
    axes[1].tick_params(axis='y', labelsize=0)

    # Violin + Scatter
    sns.violinplot(x=df["pIC50"], color=COLOR_VIOLIN, inner="quartile", ax=axes[2])
    sns.stripplot(x=df["pIC50"], color=DOT_COLOR, size=DOT_SIZE/10, alpha=0.6, ax=axes[2])
    axes[2].set_title("Violin + Scatter", fontsize=FONT_TITLE, fontweight='bold', pad=15)
    axes[2].set_xlabel("pIC50", fontsize=FONT_LABELS, labelpad=10)
    axes[2].tick_params(axis='x', labelsize=FONT_TICKS)
    axes[2].tick_params(axis='y', labelsize=0)

    plt.suptitle(title, fontsize=FONT_TITLE+2, fontweight='bold', y=1.02)
    sns.despine(fig=fig)
    plt.tight_layout()

    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    print(f"Figure saved as: {filename}")

def plot_chemspace_pca_umap(df, pca_coords, umap_coords, save_dir="save/",
                            title="Chemical Space Comparison", color_col='pIC50'):

    FONT_TITLE = 24
    FONT_LABELS = 18
    FONT_TICKS = 16
    DOT_SIZE = 40
    ALPHA = 0.8
    FIG_SIZE = (16,7)
    DPI = 400
    COLOR_MAP = "viridis_r"

    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{title.replace(' ', '_')}.svg")

    sns.set_style("white")
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, dpi=DPI)

    # PCA Scatter
    x_pca, y_pca = pca_coords[:,0], pca_coords[:,1]
    sc1 = axes[0].scatter(x_pca, y_pca, c=df[color_col], cmap=COLOR_MAP, s=DOT_SIZE, alpha=ALPHA)
    axes[0].set_title("PCA", fontsize=FONT_TITLE, fontweight='bold')
    axes[0].set_xlabel("PC 1", fontsize=FONT_LABELS)
    axes[0].set_ylabel("PC 2", fontsize=FONT_LABELS)
    axes[0].tick_params(axis='both', labelsize=FONT_TICKS)
    cbar1 = plt.colorbar(sc1, ax=axes[0], shrink=0.8)
    cbar1.set_label(color_col, fontsize=FONT_LABELS)

    # UMAP Scatter
    x_umap, y_umap = umap_coords[:,0], umap_coords[:,1]
    sc2 = axes[1].scatter(x_umap, y_umap, c=df[color_col], cmap=COLOR_MAP, s=DOT_SIZE, alpha=ALPHA)
    axes[1].set_title("UMAP", fontsize=FONT_TITLE, fontweight='bold')
    axes[1].set_xlabel("UMAP 1", fontsize=FONT_LABELS)
    axes[1].set_ylabel("UMAP 2", fontsize=FONT_LABELS)
    axes[1].tick_params(axis='both', labelsize=FONT_TICKS)
    cbar2 = plt.colorbar(sc2, ax=axes[1], shrink=0.8)
    cbar2.set_label(color_col, fontsize=FONT_LABELS)

    plt.suptitle(title, fontsize=FONT_TITLE+4, fontweight='bold', y=1.02)
    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"Figure saved as: {filename}")





