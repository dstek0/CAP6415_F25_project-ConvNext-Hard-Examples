"""
Adversarial Attack Utilities for ConvNext Probing Project

This module provides reusable functions for generating adversarial examples
using FGSM (Fast Gradient Sign Method) and PGD (Projected Gradient Descent).

Author: Dylan Stechmann
Course: CAP6415 - Computer Vision, Fall 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.

    FGSM perturbs images by taking a single step in the direction of the
    gradient of the loss with respect to the input image.

    Formula: x_adv = x + epsilon * sign(grad_x L(model(x), y))

    Args:
        model: Target neural network (must be in eval mode)
        images: Input images tensor, shape (N, C, H, W), normalized
        labels: True labels for untargeted attack, shape (N,)
        epsilon: Maximum perturbation magnitude (L-inf norm)
        targeted: If True, minimize loss for target_labels instead
        target_labels: Target labels for targeted attack, shape (N,)

    Returns:
        Tuple of (adversarial_images, perturbations)
    """
    # Ensure model is in eval mode
    model.eval()

    # Clone images and enable gradients
    images_adv = images.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(images_adv)

    # Compute loss
    if targeted and target_labels is not None:
        # Targeted attack: minimize loss for target class
        loss = F.cross_entropy(outputs, target_labels)
    else:
        # Untargeted attack: maximize loss for true class
        loss = F.cross_entropy(outputs, labels)

    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()

    # Get gradient sign
    grad_sign = images_adv.grad.sign()

    # Create perturbation
    if targeted:
        # Move toward target class (subtract gradient)
        perturbation = -epsilon * grad_sign
    else:
        # Move away from true class (add gradient)
        perturbation = epsilon * grad_sign

    # Create adversarial images
    images_adv = images + perturbation

    # Clamp to valid image range [0, 1] (assuming normalized to [0,1] before model normalization)
    # Note: If using ImageNet normalization, clamp to appropriate range
    images_adv = torch.clamp(images_adv, 0, 1)

    return images_adv.detach(), perturbation.detach()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    num_steps: int = 20,
    random_start: bool = True,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projected Gradient Descent (PGD) adversarial attack.

    PGD is an iterative attack that applies multiple small FGSM-like steps,
    projecting back to the epsilon ball after each step.

    Args:
        model: Target neural network (must be in eval mode)
        images: Input images tensor, shape (N, C, H, W)
        labels: True labels for untargeted attack, shape (N,)
        epsilon: Maximum perturbation magnitude (L-inf norm)
        alpha: Step size for each iteration
        num_steps: Number of PGD iterations
        random_start: If True, start from random point in epsilon ball
        targeted: If True, perform targeted attack
        target_labels: Target labels for targeted attack

    Returns:
        Tuple of (adversarial_images, perturbations)
    """
    model.eval()

    # Initialize adversarial images
    if random_start:
        # Start from random point within epsilon ball
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        images_adv = torch.clamp(images + delta, 0, 1)
    else:
        images_adv = images.clone()

    # Store original images for projection
    images_orig = images.clone().detach()

    for step in range(num_steps):
        images_adv = images_adv.clone().detach().requires_grad_(True)

        # Forward pass
        outputs = model(images_adv)

        # Compute loss
        if targeted and target_labels is not None:
            loss = F.cross_entropy(outputs, target_labels)
        else:
            loss = F.cross_entropy(outputs, labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Get gradient sign and update
        with torch.no_grad():
            if targeted:
                images_adv = images_adv - alpha * images_adv.grad.sign()
            else:
                images_adv = images_adv + alpha * images_adv.grad.sign()

            # Project back to epsilon ball around original image
            perturbation = torch.clamp(images_adv - images_orig, -epsilon, epsilon)
            images_adv = torch.clamp(images_orig + perturbation, 0, 1)

    # Calculate final perturbation
    final_perturbation = images_adv - images_orig

    return images_adv.detach(), final_perturbation.detach()


def evaluate_attack_success(
    model: nn.Module,
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    true_labels: torch.Tensor,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None
) -> dict:
    """
    Evaluate the success rate of an adversarial attack.

    Args:
        model: Target neural network
        original_images: Clean images
        adversarial_images: Perturbed images
        true_labels: Ground truth labels
        targeted: Whether this was a targeted attack
        target_labels: Target labels (for targeted attacks)

    Returns:
        Dictionary with attack statistics
    """
    model.eval()

    with torch.no_grad():
        # Get predictions on clean images
        clean_outputs = model(original_images)
        clean_preds = clean_outputs.argmax(dim=1)
        clean_correct = (clean_preds == true_labels).sum().item()

        # Get predictions on adversarial images
        adv_outputs = model(adversarial_images)
        adv_preds = adv_outputs.argmax(dim=1)
        adv_correct = (adv_preds == true_labels).sum().item()

        # Get confidence scores
        clean_probs = F.softmax(clean_outputs, dim=1)
        adv_probs = F.softmax(adv_outputs, dim=1)

        clean_confidence = clean_probs.max(dim=1)[0].mean().item()
        adv_confidence = adv_probs.max(dim=1)[0].mean().item()

    n_samples = len(true_labels)

    # Calculate success rate
    if targeted and target_labels is not None:
        # Targeted: success if prediction matches target
        attack_success = (adv_preds == target_labels).sum().item()
    else:
        # Untargeted: success if originally correct, now incorrect
        originally_correct = (clean_preds == true_labels)
        now_incorrect = (adv_preds != true_labels)
        attack_success = (originally_correct & now_incorrect).sum().item()
        # Denominator is only originally correct samples
        n_samples = originally_correct.sum().item()

    success_rate = attack_success / max(n_samples, 1) * 100

    return {
        'total_samples': len(true_labels),
        'clean_accuracy': clean_correct / len(true_labels) * 100,
        'adversarial_accuracy': adv_correct / len(true_labels) * 100,
        'attack_success_rate': success_rate,
        'clean_confidence': clean_confidence,
        'adversarial_confidence': adv_confidence,
        'accuracy_drop': (clean_correct - adv_correct) / len(true_labels) * 100
    }


def compute_perturbation_stats(perturbation: torch.Tensor) -> dict:
    """
    Compute statistics about the perturbation.

    Args:
        perturbation: The adversarial perturbation tensor

    Returns:
        Dictionary with perturbation statistics
    """
    with torch.no_grad():
        l_inf = perturbation.abs().max().item()
        l_2 = perturbation.pow(2).sum().sqrt().item() / perturbation.numel()
        l_1 = perturbation.abs().mean().item()

    return {
        'l_inf_norm': l_inf,
        'l_2_norm': l_2,
        'l_1_norm': l_1,
        'mean': perturbation.mean().item(),
        'std': perturbation.std().item()
    }
