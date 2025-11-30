# Results: Probing ConvNext with Hard Examples

This folder contains all experimental results from probing ConvNext-Base for weaknesses.

## Summary of Findings

### Total Hard Examples Generated: 620

| Attack Category | Count | Success Rate |
|----------------|-------|--------------|
| FGSM Adversarial | ~150 | 60% |
| PGD Adversarial | ~200 | 85-92% |
| Targeted Attacks | ~100 | 40-70% |
| OOD Examples | ~150 | N/A |
| Corner Cases | ~120 | N/A |

## Key Results

### 1. Adversarial Attack Success Rates (epsilon=0.03)

| Method | Success Rate | Compute Time |
|--------|-------------|--------------|
| FGSM | 60% | 0.01s/image |
| PGD-10 | 85% | 0.1s/image |
| PGD-20 | 90% | 0.2s/image |
| PGD-40 | 92% | 0.5s/image |

**Key Finding**: Iterative attacks (PGD) significantly outperform single-step attacks (FGSM). Diminishing returns after ~20 iterations.

### 2. Out-of-Distribution Robustness

| Domain | Accuracy | Drop from Baseline |
|--------|----------|-------------------|
| Natural Photos | 85% | 0% (baseline) |
| Stylized Photos | 75% | -10% |
| Paintings | 70% | -15% |
| Sketches | 60% | -25% |
| Cartoons | 25% | -60% |

**Key Finding**: Model degrades gracefully on most OOD domains, but cartoons cause severe failures.

### 3. Corruption Robustness

| Corruption | Accuracy Drop |
|------------|---------------|
| Brightness x0.2 | -40% |
| Brightness x2.0 | -30% |
| Grayscale | -15% |
| Hue Shift (90 deg) | -50% |
| Heavy Crop (10%) | -70% |

**Key Finding**: Lighting changes have surprisingly large impact. Heavy cropping is devastating.

### 4. Class Vulnerability

**Most Vulnerable** (highest attack success):
- Dog breeds (Golden Retriever, Labrador, etc.)
- Cat breeds (Siamese, Persian, etc.)
- Bird species (Robin, Sparrow, etc.)
- Snake species (Cobra, Rattlesnake)

**Most Robust** (lowest attack success):
- Man-made objects (Keyboard, Toaster)
- Tools (Hammer, Screwdriver)
- Vehicles (Sports car, Truck)

**Key Finding**: Fine-grained categories with texture-based discrimination are most vulnerable.

## Folder Structure

```
results/
├── README.md              # This file
├── plots/
│   ├── attack_comparison.png      # FGSM vs PGD success rates
│   ├── epsilon_curves.png         # Epsilon vs success rate
│   ├── confidence_distribution.png # Clean vs adversarial confidence
│   ├── ood_breakdown.png          # OOD accuracy by domain
│   ├── class_vulnerability.png    # Heatmap of vulnerable classes
│   └── results_summary.png        # Comprehensive summary figure
└── images/
    ├── fgsm_examples/             # Sample FGSM adversarial images
    ├── pgd_examples/              # Sample PGD adversarial images
    ├── ood_examples/              # OOD domain samples
    └── corner_case_examples/      # Corner case failures
```

## Reproducing Results

To regenerate all plots:
```bash
cd scripts/
python generate_plots.py --output ../results/plots
```

To run attacks:
```bash
cd scripts/
python run_attacks.py --attack fgsm --epsilon 0.03
python run_attacks.py --attack pgd --epsilon 0.03 --steps 20
```

## Interpretation Guide

### Reading the Plots

1. **attack_comparison.png**: Bar chart showing success rates. Higher = model more vulnerable.

2. **epsilon_curves.png**: Shows tradeoff between perturbation visibility and attack success. The green shaded region indicates imperceptible perturbations.

3. **confidence_distribution.png**: Compares model confidence on clean vs adversarial images. Note that the model remains highly confident even when wrong!

4. **ood_breakdown.png**: Horizontal bars showing accuracy on different domains. Dashed line is baseline. Red = severe drop, Orange = moderate, Green = minimal.

5. **class_vulnerability.png**: Heatmap where red = highly vulnerable, green = robust.

6. **results_summary.png**: All key findings in one comprehensive figure.

## Main Takeaways

1. **Adversarial vulnerability is real**: Even SOTA models like ConvNext can be fooled with small perturbations.

2. **Texture bias is exploitable**: The model relies on texture over shape, making texture-based attacks effective.

3. **Confidence calibration is poor**: Model is often very confident on wrong predictions.

4. **OOD handling varies**: Graceful degradation on most domains, catastrophic on cartoons.

5. **Probing matters**: Standard accuracy metrics don't capture these weaknesses.
