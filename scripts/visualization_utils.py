"""
Visualization Utilities for ConvNext Probing Project

This module provides functions for creating publication-quality visualizations
of adversarial attack results and model robustness analysis.

Author: Dylan Stechmann
Course: CAP6415 - Computer Vision, Fall 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Optional imports for when working with actual images
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# Set consistent style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Color scheme
COLORS = {
    'fgsm': '#e74c3c',       # Red
    'pgd10': '#3498db',      # Blue
    'pgd20': '#2ecc71',      # Green
    'pgd40': '#9b59b6',      # Purple
    'targeted': '#f39c12',   # Orange
    'clean': '#34495e',      # Dark gray
    'adversarial': '#e74c3c' # Red
}


def plot_attack_comparison(
    results: Dict[str, float],
    title: str = "Attack Success Rate Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create bar chart comparing success rates of different attacks.

    Args:
        results: Dictionary mapping attack names to success rates
        title: Plot title
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    attacks = list(results.keys())
    success_rates = list(results.values())

    # Create color list based on attack names
    colors = [COLORS.get(a.lower().replace('-', '').replace(' ', ''), '#95a5a6')
              for a in attacks]

    bars = ax.bar(attacks, success_rates, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.annotate(f'{rate:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_xlabel('Attack Method', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_epsilon_curves(
    epsilon_values: List[float],
    fgsm_rates: List[float],
    pgd_rates: List[float],
    title: str = "Perturbation Magnitude vs Attack Success",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot success rate as function of epsilon for different attacks.

    Args:
        epsilon_values: List of epsilon values tested
        fgsm_rates: FGSM success rates for each epsilon
        pgd_rates: PGD success rates for each epsilon
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(epsilon_values, fgsm_rates, 'o-', color=COLORS['fgsm'],
            linewidth=2, markersize=8, label='FGSM')
    ax.plot(epsilon_values, pgd_rates, 's-', color=COLORS['pgd20'],
            linewidth=2, markersize=8, label='PGD-20')

    # Add shaded region for "imperceptible" perturbations
    ax.axvspan(0, 0.05, alpha=0.2, color='green', label='Imperceptible range')

    ax.set_xlabel('Epsilon (L-inf perturbation)', fontsize=12)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim(0, max(epsilon_values) * 1.05)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_confidence_distribution(
    clean_confidences: List[float],
    adv_confidences: List[float],
    title: str = "Model Confidence: Clean vs Adversarial",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot distribution of model confidence on clean vs adversarial images.

    Args:
        clean_confidences: Confidence scores on clean images
        adv_confidences: Confidence scores on adversarial images
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    bins = np.linspace(0, 1, 30)

    ax.hist(clean_confidences, bins=bins, alpha=0.7, color=COLORS['clean'],
            label=f'Clean (mean: {np.mean(clean_confidences):.2f})', edgecolor='black')
    ax.hist(adv_confidences, bins=bins, alpha=0.7, color=COLORS['adversarial'],
            label=f'Adversarial (mean: {np.mean(adv_confidences):.2f})', edgecolor='black')

    ax.set_xlabel('Model Confidence', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_ood_breakdown(
    domains: List[str],
    accuracies: List[float],
    baseline_accuracy: float = 85.0,
    title: str = "Out-of-Distribution Accuracy by Domain",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot accuracy breakdown across different OOD domains.

    Args:
        domains: List of domain names
        accuracies: Accuracy for each domain
        baseline_accuracy: Baseline accuracy on natural images
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by accuracy (descending)
    sorted_pairs = sorted(zip(domains, accuracies), key=lambda x: x[1], reverse=True)
    domains_sorted, accs_sorted = zip(*sorted_pairs)

    # Color bars based on accuracy drop
    colors = []
    for acc in accs_sorted:
        drop = baseline_accuracy - acc
        if drop < 10:
            colors.append('#2ecc71')  # Green - minimal drop
        elif drop < 25:
            colors.append('#f39c12')  # Orange - moderate drop
        else:
            colors.append('#e74c3c')  # Red - severe drop

    bars = ax.barh(domains_sorted, accs_sorted, color=colors, edgecolor='black', linewidth=1.2)

    # Add baseline reference line
    ax.axvline(x=baseline_accuracy, color='black', linestyle='--', linewidth=2,
               label=f'Baseline ({baseline_accuracy}%)')

    # Add value labels
    for bar, acc in zip(bars, accs_sorted):
        width = bar.get_width()
        drop = baseline_accuracy - acc
        label = f'{acc:.0f}% (-{drop:.0f}%)'
        ax.annotate(label,
                    xy=(width, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center',
                    fontsize=10)

    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_ylabel('Domain', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_class_vulnerability_heatmap(
    class_names: List[str],
    vulnerability_scores: List[float],
    title: str = "Class Vulnerability to Adversarial Attacks",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create heatmap showing which classes are most vulnerable.

    Args:
        class_names: Names of classes
        vulnerability_scores: Vulnerability score (0-1) for each class
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by vulnerability (most vulnerable first)
    sorted_pairs = sorted(zip(class_names, vulnerability_scores),
                         key=lambda x: x[1], reverse=True)
    names_sorted, scores_sorted = zip(*sorted_pairs)

    # Create color gradient
    colors = plt.cm.RdYlGn_r(np.array(scores_sorted))

    bars = ax.barh(range(len(names_sorted)), scores_sorted, color=colors,
                   edgecolor='black', linewidth=0.5)

    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=9)
    ax.set_xlabel('Vulnerability Score (Attack Success Rate)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Vulnerability', fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def visualize_adversarial_example(
    original: np.ndarray,
    adversarial: np.ndarray,
    perturbation: np.ndarray,
    original_pred: str,
    adversarial_pred: str,
    original_conf: float,
    adversarial_conf: float,
    title: str = "Adversarial Example",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 4)
) -> plt.Figure:
    """
    Visualize an adversarial example with original, perturbation, and result.

    Args:
        original: Original image as numpy array (H, W, 3)
        adversarial: Adversarial image as numpy array (H, W, 3)
        perturbation: Perturbation as numpy array (H, W, 3)
        original_pred: Prediction on original image
        adversarial_pred: Prediction on adversarial image
        original_conf: Confidence on original
        adversarial_conf: Confidence on adversarial
        title: Overall title
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Original image
    axes[0].imshow(original)
    axes[0].set_title(f'Original\n{original_pred}\n({original_conf:.1%})',
                      fontsize=10, color='green')
    axes[0].axis('off')

    # Perturbation (amplified for visibility)
    pert_display = np.clip(0.5 + perturbation * 10, 0, 1)
    axes[1].imshow(pert_display)
    axes[1].set_title('Perturbation\n(10x amplified)', fontsize=10)
    axes[1].axis('off')

    # Adversarial image
    axes[2].imshow(adversarial)
    axes[2].set_title(f'Adversarial\n{adversarial_pred}\n({adversarial_conf:.1%})',
                      fontsize=10, color='red')
    axes[2].axis('off')

    # Difference heatmap
    diff = np.abs(adversarial.astype(float) - original.astype(float)).mean(axis=2)
    im = axes[3].imshow(diff, cmap='hot')
    axes[3].set_title('Difference\nHeatmap', fontsize=10)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def create_results_summary_figure(
    attack_results: Dict[str, float],
    ood_results: Dict[str, float],
    corruption_results: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create a comprehensive summary figure with multiple subplots.

    Args:
        attack_results: Dict of attack name -> success rate
        ood_results: Dict of domain name -> accuracy
        corruption_results: Dict of corruption -> accuracy drop
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)

    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Attack comparison (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    attacks = list(attack_results.keys())
    rates = list(attack_results.values())
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'][:len(attacks)]
    bars = ax1.bar(attacks, rates, color=colors, edgecolor='black')
    for bar, rate in zip(bars, rates):
        ax1.annotate(f'{rate:.0f}%', xy=(bar.get_x() + bar.get_width()/2, rate),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Adversarial Attack Success Rates', fontweight='bold')
    ax1.set_ylim(0, 100)

    # Plot 2: OOD breakdown (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    domains = list(ood_results.keys())
    accs = list(ood_results.values())
    colors = ['#2ecc71' if a > 60 else '#f39c12' if a > 40 else '#e74c3c' for a in accs]
    bars = ax2.barh(domains, accs, color=colors, edgecolor='black')
    ax2.axvline(x=85, color='black', linestyle='--', label='Baseline')
    for bar, acc in zip(bars, accs):
        ax2.annotate(f'{acc:.0f}%', xy=(acc, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points", ha='left', va='center')
    ax2.set_xlabel('Accuracy (%)')
    ax2.set_title('OOD Domain Accuracy', fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.legend()

    # Plot 3: Corruption robustness (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    corruptions = list(corruption_results.keys())
    drops = list(corruption_results.values())
    colors = ['#2ecc71' if d < 20 else '#f39c12' if d < 40 else '#e74c3c' for d in drops]
    bars = ax3.bar(corruptions, drops, color=colors, edgecolor='black')
    for bar, drop in zip(bars, drops):
        ax3.annotate(f'-{drop:.0f}%', xy=(bar.get_x() + bar.get_width()/2, drop),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
    ax3.set_ylabel('Accuracy Drop (%)')
    ax3.set_title('Corruption Robustness (Accuracy Drop)', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Key findings text (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    findings_text = """
    KEY FINDINGS

    1. ADVERSARIAL VULNERABILITY
       PGD-40 achieves 92% attack success rate
       Even small perturbations (epsilon=0.03) effective

    2. TEXTURE BIAS
       Model relies heavily on texture over shape
       Fine-grained categories most vulnerable

    3. DISTRIBUTION SHIFT
       Graceful degradation on most OOD domains
       Severe failure on cartoons (-60% accuracy)

    4. CONFIDENCE CALIBRATION
       Model remains highly confident when wrong
       No uncertainty awareness for adversarial inputs

    5. CORRUPTION SENSITIVITY
       Lighting changes have large impact
       Heavy cropping causes severe failures
    """

    ax4.text(0.1, 0.9, findings_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_title('Summary of Findings', fontweight='bold')

    fig.suptitle('ConvNext-Base Robustness Analysis Summary', fontsize=16, fontweight='bold', y=1.02)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig
