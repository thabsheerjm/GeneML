import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np, os
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']

def volcano_plot(df, title="Parkinson's Disease Volcano Plot", save="results/"):
    ## Plot Parameters
    # Colors
    COLOR_SIG = '#D35400'
    COLOR_NONSIG = '#AED6F1'
    COLOR_LINES = '#2C3E50'
    COLOR_TOP  = '#2C3E50'
    #Fonts & Sizes
    FONT_TITLE = 24
    FONT_LABELS = 20
    FONT_TICKS = 16
    FONT_LEGEND = 16
    DOT_SIZE = 40
    ALPHA = 0.8
    FIG_SIZE = (12,8)
    DPI = 400
    #Threshold
    P_CUTOFF = 0.01
    FC_CUTOFF = 0.25
    filename = os.path.join(save,str(title)+".svg")

    sns.set_style("white")
    plt.figure(figsize=FIG_SIZE, dpi=DPI)

    sns.scatterplot(data=df[df['significant']],x='log_fc',
        y=-np.log10(df['p_value']),color=COLOR_SIG,s=DOT_SIZE + 10,
        alpha=ALPHA, edgecolor='white',linewidth=0.8,label='Significant Marker')

    sns.scatterplot(data=df[~df['significant']],x='log_fc',
        y=-np.log10(df['p_value']),color=COLOR_NONSIG,s=DOT_SIZE,
        alpha=0.6,edgecolor='white',linewidth=0.5,label='Not Significant')

    # Add threshold lines
    plt.axhline(-np.log10(P_CUTOFF), color=COLOR_LINES, linestyle='--',linewidth=1.5, alpha=0.7 )
    plt.axvline(FC_CUTOFF, color=COLOR_LINES, linestyle='--',linewidth=1.5, alpha=0.7 )
    plt.axvline(-FC_CUTOFF, color=COLOR_LINES, linestyle='--',linewidth=1.5, alpha=0.7 )

    top_genes = df.sort_values('p_value').head(5)
    for gene in top_genes.index:
        x = df.loc[gene, 'log_fc']
        y = -np.log10(df.loc[gene, 'p_value'])
        plt.text(x + 0.04, y, gene, fontsize=14, fontweight='bold', color=COLOR_TOP)

    plt.title(title, fontsize=FONT_TITLE, fontweight='bold', pad=20)
    plt.xlabel("Effect Size ($Log_2$ Fold Change)", fontsize=FONT_LABELS, labelpad=10)
    plt.ylabel("Significance ($-Log_{10}$ P-value)", fontsize=FONT_LABELS, labelpad=10)

    plt.xticks(fontsize=FONT_TICKS)
    plt.yticks(fontsize=FONT_TICKS)

    plt.legend(fontsize=FONT_LEGEND, loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    sns.despine()
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as: {filename}")
    return True


def plot_lc_sensitivity(X, y, save="results/"):
    # Parameters
    COLOR_70_30 = '#1F77B4'
    COLOR_80_20 = '#E74C3C'
    COLOR_90_10 = '#27AE60'
    MARKER_TRAIN = 's'
    MARKER_VALID = '^'
    LINE_TRAIN = '-'
    LINE_VALID = '--'
    FONT_TITLE = 24
    FONT_LABELS = 18
    FONT_TICKS = 16
    MARKER_SIZE = 10
    LINE_WIDTH = 2
    FIG_SIZE = (20, 7)
    DPI = 400

    # Estimators
    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42),
        'SVM': SVC(kernel='linear', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    splits = [0.3, 0.2, 0.1]
    split_colors = [COLOR_70_30, COLOR_80_20, COLOR_90_10]
    split_labels = ['70/30 Split', '80/20 Split', '90/10 Split']

    sns.set_style("white")
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE, dpi=DPI, sharey=True)

    for i, (model_name, model) in enumerate(models.items()):
        ax = axes[i]
        for split, color, label in zip(splits, split_colors, split_labels):
            X_train_outer, _, y_train_outer, _ = train_test_split(
                X, y, test_size=split, random_state=42
            )

            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train_outer, y_train_outer,
                cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.2, 1.0, 5),
                scoring='accuracy'
            )
            train_mean = np.mean(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)

            ax.plot(train_sizes, train_mean, color=color, linestyle=LINE_TRAIN,
                    linewidth=LINE_WIDTH, marker=MARKER_TRAIN, markersize=MARKER_SIZE,
                    label=f"Train {label}")
            ax.plot(train_sizes, test_mean, color=color, linestyle=LINE_VALID,
                    linewidth=LINE_WIDTH, marker=MARKER_VALID, markersize=MARKER_SIZE,
                    label=f"Valid {label}")

        ax.set_title(model_name, fontsize=FONT_TITLE, fontweight='bold', pad=15)
        ax.set_xlabel("Number of Samples", fontsize=FONT_LABELS)
        ax.tick_params(axis='both', labelsize=FONT_TICKS)

        ax.set_ylim(0.5, 1.05)

        if i == 0:
            ax.set_ylabel("Mean Accuracy (5-Fold CV)", fontsize=FONT_LABELS, labelpad=10)

        if i == 2:
            legend_elements = [
                Line2D([0], [0], color=COLOR_70_30, lw=2, label='70/30 Split'),
                Line2D([0], [0], color=COLOR_80_20, lw=2, label='80/20 Split'),
                Line2D([0], [0], color=COLOR_90_10, lw=2, label='90/10 Split'),
                Line2D([0], [0], color='white', label=''),
                Line2D([0], [0], marker=MARKER_TRAIN, color='black', label='Training (Solid)',
                       linestyle=LINE_TRAIN, markersize=8),
                Line2D([0], [0], marker=MARKER_VALID, color='black', label='Validation (Dashed)',
                       linestyle=LINE_VALID, markersize=8),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
                      frameon=True, framealpha=0.9, title="Legend")

        sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(f"{save}trajectory_sensitivity.svg", dpi=DPI, bbox_inches='tight')
    return True

def plot_enrichment_results(enr_results, title="Functional Enrichment Analysis", save="results/", top_n=5):
    """
    Horizontal bar plot for Enrichment Results.
    """
    COLOR_BARS = '#000080'
    COLOR_TEXT = '#2C3E50'
    FONT_TITLE = 24
    FONT_LABELS = 18
    FONT_TICKS = 16
    FIG_SIZE = (10, 8)
    DPI = 400

    sorted_df = enr_results.sort_values('P-value', ascending=True)

    df_unique = sorted_df.drop_duplicates(subset='Term', keep='first')
    df_plot = df_unique.head(top_n).copy()
    df_plot['log_p'] = -np.log10(df_plot['P-value'])

    sns.set_style("white")
    plt.figure(figsize=FIG_SIZE, dpi=DPI)

    ax = sns.barplot(
        data=df_plot,
        x='log_p',
        y='Term',
        color=COLOR_BARS,
        edgecolor='black',
        linewidth=0.5
    )

    for i, (index, row) in enumerate(df_plot.iterrows()):
        p_val = row['P-value']
        log_p = row['log_p']
        x_pos = max(log_p, 0.1) + 0.1
        ax.text(x_pos, i, f"p={p_val:.3f}", va='center', fontsize=12, color=COLOR_TEXT)

    plt.title(title, fontsize=FONT_TITLE, fontweight='bold', pad=20)
    plt.xlabel("Significance ($-Log_{10}$ P-value)", fontsize=FONT_LABELS, labelpad=10)
    plt.ylabel("")

    plt.xticks(fontsize=FONT_TICKS)
    plt.yticks(fontsize=FONT_TICKS)

    threshold = -np.log10(0.05)
    plt.axvline(threshold, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    plt.text(threshold, -0.5, 'p=0.05 threshold', color='red', fontsize=11, va='bottom', ha='left')

    sns.despine()
    plt.tight_layout()

    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, "enrichment_barplot.svg")
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    plt.show()
    print(f"Plot saved as: {filename}")
    return True








