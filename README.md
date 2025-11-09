# Probing ConvNext SOTA with Hard Examples

## Abstract
State-of-the-art computer vision models like ConvNext achieve impressive accuracy on standard benchmarks, but how robust are they really? This project systematically probes ConvNext-Base to find its weak spots—not through random testing, but by understanding what types of images genuinely confuse it. I'm exploring adversarial attacks, out-of-distribution scenarios, and edge cases to build a comprehensive picture of where this SOTA model breaks down and why.

## What I'm Investigating
The goal isn't just to break the model, but to understand its failure modes:

- **Adversarial perturbations** - How small can I make pixel changes while still fooling the model? Using FGSM, PGD, Auto-Attack, and C&W attacks
- **Out-of-distribution inputs** - What happens with synthetic images, style transfers, or distribution shifts?
- **Corner cases** - Texture-only images, minimal objects, extreme lighting/weather conditions
- **Fine-grained classification** - Where does it struggle with similar-looking classes?
- **Multi-object scenes** - Does it get confused when multiple objects are present?

## Model & Dataset
I'm using **ConvNext-Base** pretrained on ImageNet-1K (via the `timm` library). For testing, I'm working with subsets of ImageNet validation data plus custom hard examples I generate throughout the project.

The beauty of ConvNext is it's recent (2022), well-maintained, and represents modern CNN architecture thinking—so understanding its weaknesses is actually relevant.

## Repository Structure
```
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── scripts/           # Python scripts for reproducible experiments
├── results/
│   ├── plots/        # Confusion matrices, accuracy graphs, t-SNE visualizations
│   └── images/       # Example hard images with model predictions
├── data/             # Datasets and cached models (not committed to git)
└── week*log.txt      # Development logs for each week
```

## Getting Started
Install dependencies:
```bash
pip install -r requirements.txt
```

Check out the notebooks in order—they build on each other. Start with `00_model_loading.ipynb` to verify everything works.

## Progress
- **Week 1** (Nov 2-9): Environment setup, model loading, baseline testing ✓
- **Week 2** (Nov 9-16): Adversarial examples with FGSM/PGD
- **Week 3** (Nov 16-23): Advanced attacks, OOD detection, corner cases
- **Week 4** (Nov 23-30): Results compilation and analysis
- **Week 5** (Nov 30-Dec 7): Video demo and final polish

## Acknowledgments
This project uses:
- **ConvNext**: Liu et al., "A ConvNet for the 2020s" (CVPR 2022)
- **timm library**: Ross Wightman's excellent model zoo
- **Adversarial Robustness Toolbox**: For implementing adversarial attacks
- **PyTorch & torchvision**: Core deep learning framework

---
**Author**: Dylan Stechmann  
**Course**: CAP6415 - Computer Vision  
**Semester**: Fall 2025

