#!/usr/bin/env python3
"""
Run Adversarial Attacks on ConvNext

This script runs adversarial attacks on ConvNext-Base and saves results.
Supports FGSM, PGD, and targeted attacks.

Usage:
    python run_attacks.py --attack fgsm --epsilon 0.03
    python run_attacks.py --attack pgd --epsilon 0.03 --steps 20
    python run_attacks.py --attack targeted --epsilon 0.05 --target 281

Author: Dylan Stechmann
Course: CAP6415 - Computer Vision, Fall 2025
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import timm
from torchvision import transforms
from tqdm import tqdm

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))
from attack_utils import fgsm_attack, pgd_attack, evaluate_attack_success


# Configuration
MODEL_NAME = 'convnext_base'
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(device):
    """Load pretrained ConvNext-Base model."""
    print(f"Loading {MODEL_NAME}...")
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1000)
    model = model.to(device)
    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def get_transforms():
    """Get preprocessing transforms."""
    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return preprocess


def denormalize(tensor):
    """Denormalize image tensor for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


def create_sample_batch(batch_size, device):
    """
    Create a sample batch of images for demonstration.
    In practice, you would load real images here.
    """
    # Generate random images (for demonstration)
    # In actual use, replace this with real ImageNet images
    images = torch.rand(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Normalize
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    images = (images - mean) / std

    # Random labels
    labels = torch.randint(0, 1000, (batch_size,))

    return images.to(device), labels.to(device)


def run_fgsm_experiment(model, images, labels, epsilon, device):
    """Run FGSM attack experiment."""
    print(f"\nRunning FGSM attack with epsilon={epsilon}...")

    adv_images, perturbations = fgsm_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon
    )

    results = evaluate_attack_success(model, images, adv_images, labels)
    return adv_images, perturbations, results


def run_pgd_experiment(model, images, labels, epsilon, steps, device):
    """Run PGD attack experiment."""
    print(f"\nRunning PGD-{steps} attack with epsilon={epsilon}...")

    alpha = epsilon / steps * 2  # Step size

    adv_images, perturbations = pgd_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        alpha=alpha,
        num_steps=steps,
        random_start=True
    )

    results = evaluate_attack_success(model, images, adv_images, labels)
    return adv_images, perturbations, results


def run_targeted_experiment(model, images, labels, epsilon, target_class, device):
    """Run targeted attack experiment."""
    print(f"\nRunning targeted attack toward class {target_class}...")

    target_labels = torch.full_like(labels, target_class)

    adv_images, perturbations = pgd_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        alpha=epsilon / 20,
        num_steps=40,
        random_start=True,
        targeted=True,
        target_labels=target_labels
    )

    results = evaluate_attack_success(
        model, images, adv_images, labels,
        targeted=True, target_labels=target_labels
    )
    return adv_images, perturbations, results


def save_results(results, attack_type, epsilon, output_dir):
    """Save experiment results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{attack_type}_eps{epsilon}_{timestamp}.json"

    results_dict = {
        'attack_type': attack_type,
        'epsilon': epsilon,
        'timestamp': timestamp,
        **results
    }

    with open(output_path / filename, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"Results saved to {output_path / filename}")
    return output_path / filename


def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial attacks on ConvNext-Base"
    )
    parser.add_argument(
        '--attack', '-a',
        type=str,
        choices=['fgsm', 'pgd', 'targeted'],
        default='fgsm',
        help='Attack type (default: fgsm)'
    )
    parser.add_argument(
        '--epsilon', '-e',
        type=float,
        default=0.03,
        help='Perturbation magnitude (default: 0.03)'
    )
    parser.add_argument(
        '--steps', '-s',
        type=int,
        default=20,
        help='Number of PGD steps (default: 20)'
    )
    parser.add_argument(
        '--target', '-t',
        type=int,
        default=281,
        help='Target class for targeted attack (default: 281 = tabby cat)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='../results/attack_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_model(device)

    # Create sample batch (replace with real data loading in practice)
    print(f"\nCreating sample batch of {args.batch_size} images...")
    images, labels = create_sample_batch(args.batch_size, device)

    # Run attack
    print("\n" + "=" * 60)
    print(f"RUNNING {args.attack.upper()} ATTACK")
    print("=" * 60)

    if args.attack == 'fgsm':
        adv_images, perturbations, results = run_fgsm_experiment(
            model, images, labels, args.epsilon, device
        )
    elif args.attack == 'pgd':
        adv_images, perturbations, results = run_pgd_experiment(
            model, images, labels, args.epsilon, args.steps, device
        )
    elif args.attack == 'targeted':
        adv_images, perturbations, results = run_targeted_experiment(
            model, images, labels, args.epsilon, args.target, device
        )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Attack Type:           {args.attack.upper()}")
    print(f"Epsilon:               {args.epsilon}")
    print(f"Clean Accuracy:        {results['clean_accuracy']:.2f}%")
    print(f"Adversarial Accuracy:  {results['adversarial_accuracy']:.2f}%")
    print(f"Attack Success Rate:   {results['attack_success_rate']:.2f}%")
    print(f"Accuracy Drop:         {results['accuracy_drop']:.2f}%")
    print(f"Clean Confidence:      {results['clean_confidence']:.4f}")
    print(f"Adversarial Confidence:{results['adversarial_confidence']:.4f}")
    print("=" * 60)

    # Save results
    save_results(results, args.attack, args.epsilon, args.output)

    print("\nDone!")


if __name__ == '__main__':
    main()
