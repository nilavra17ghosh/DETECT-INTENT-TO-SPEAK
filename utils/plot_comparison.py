import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_comparison_plot():
    # Set premium aesthetic
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Roboto", "Arial"],
        "axes.edgecolor": "#333333",
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.6
    })

    # Data based on project benchmarks
    models = ['EEGNet', 'ShallowConvNet', 'Riemannian+LR', 'SVM+Features']
    within_subj = [0.82, 0.78, 0.81, 0.72]
    cross_subj = [0.64, 0.61, 0.63, 0.58]
    fine_tuned = [0.72, 0.69, 0.71, 0.65]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)

    # Gradient colors
    colors = ['#2563eb', '#ef4444', '#10b981'] # Blue, Red, Green
    
    rects1 = ax.bar(x - width, within_subj, width, label='Within-Subject (K-Fold)', 
                    color=colors[0], alpha=0.85, edgecolor='white', linewidth=1,
                    zorder=3)
    rects2 = ax.bar(x, cross_subj, width, label='Cross-Subject (LOSO)', 
                    color=colors[1], alpha=0.85, edgecolor='white', linewidth=1,
                    zorder=3)
    rects3 = ax.bar(x + width, fine_tuned, width, label='Cross-Subject + Fine-Tuning', 
                    color=colors[2], alpha=0.85, edgecolor='white', linewidth=1,
                    zorder=3)

    # Add labels and title
    ax.set_ylabel('Balanced Accuracy', fontsize=12, fontweight='600', labelpad=10)
    ax.set_title('Generalization Performance: Within-Subject vs. Cross-Subject', 
                 fontsize=16, fontweight='bold', pad=20, color='#1f2937')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='500')
    ax.set_ylim(0, 1.0)
    
    # Add horizontal line for chance level
    ax.axhline(0.5, color='#9ca3af', linestyle=':', linewidth=1.5, alpha=0.8, zorder=2)
    ax.text(-0.45, 0.51, 'Chance (50%)', color='#6b7280', fontsize=10, fontweight='500')

    # Legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.9, facecolor='white', 
              edgecolor='#e5e7eb', fontsize=10)

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='600',
                        color='#374151')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Background styling
    ax.set_facecolor('#f9fafb')
    fig.patch.set_facecolor('white')

    # Despine
    sns.despine(left=True, bottom=False)

    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/within_vs_cross_comparison.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    generate_comparison_plot()
