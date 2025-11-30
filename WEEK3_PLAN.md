# Week 3 Action Plan (Nov 16-23, 2025)

## 🎯 Main Goal
Expand hard example generation beyond FGSM to include iterative attacks, targeted attacks, and non-adversarial corner cases. Aim for **500-1000 total hard examples** by end of week.

---

## 📋 Tasks Breakdown

### 1. Iterative Adversarial Attacks (Priority: HIGH)
**Time estimate: 4-6 hours**

- [ ] Implement PGD (Projected Gradient Descent) attack
  - PGD is basically FGSM repeated N times with projection back to epsilon ball
  - Parameters: epsilon=0.03, alpha=0.01, num_steps=10-40
  - Compare success rate vs FGSM

- [ ] Test different PGD configurations
  - Random start vs zero start
  - Different step sizes
  - Different iteration counts

- [ ] Generate ~200 PGD adversarial examples

**Notebook**: `03_pgd_attacks.ipynb`

**Key implementation detail**: 
```python
for step in range(num_steps):
    x_adv.requires_grad = True
    output = model(x_adv)
    loss = criterion(output, target)
    loss.backward()
    
    # Take a step
    x_adv = x_adv + alpha * x_adv.grad.sign()
    
    # Project back to epsilon ball around original image
    perturbation = torch.clamp(x_adv - x_original, -epsilon, epsilon)
    x_adv = torch.clamp(x_original + perturbation, 0, 1).detach()
```

---

### 2. Targeted Adversarial Attacks (Priority: MEDIUM)
**Time estimate: 3-4 hours**

- [ ] Modify FGSM/PGD for targeted attacks
  - Instead of maximizing loss, MINIMIZE loss for target class
  - Pick interesting target classes (e.g., make dogs→cats)

- [ ] Create targeted examples for:
  - Cross-category (dog→cat, car→airplane)
  - Similar classes (golden retriever→labrador)
  - Semantic opposites (day→night objects)

- [ ] Generate ~100 targeted adversarial examples

**Implementation note**: Change loss from maximizing to minimizing:
```python
# Untargeted: x_adv = x + epsilon * sign(∇loss)
# Targeted: x_adv = x - epsilon * sign(∇loss_target)
```

---

### 3. Out-of-Distribution (OOD) Hard Examples (Priority: HIGH)
**Time estimate: 4-5 hours**

- [ ] Test model on:
  - Sketch images (find sketch datasets online)
  - Cartoon/animated images
  - Paintings/artistic renderings
  - Heavily stylized images

- [ ] Simple style transfer experiment
  - Use pretrained style transfer model or AdaIN
  - Apply artistic styles to ImageNet images
  - Test if model still recognizes content

- [ ] Generate ~150 OOD examples

**Datasets to consider**:
- Sketchy dataset (sketch images)
- DomainNet (clipart, painting, sketch domains)
- Random internet images with unusual styles

---

### 4. Corner Cases & Edge Scenarios (Priority: MEDIUM)
**Time estimate: 3-4 hours**

- [ ] Texture vs Shape
  - Create texture-only images (random shapes with object textures)
  - Test if model relies more on texture than shape
  - Reference: Geirhos et al. texture bias paper

- [ ] Extreme lighting/color manipulations
  - Very dark images (simulate low light)
  - Very bright/overexposed
  - Color shifts (change hue/saturation dramatically)
  - Grayscale versions

- [ ] Minimal objects
  - Crop images to show only small portion of object
  - Test if model can still classify

- [ ] Multi-object confusion
  - Images with multiple objects from different classes
  - See which object model "focuses on"

- [ ] Generate ~100-150 corner case examples

**Notebook**: `04_corner_cases.ipynb`

---

### 5. Analysis & Visualization (Priority: HIGH)
**Time estimate: 4-5 hours**

- [ ] Create comprehensive visualizations:
  - **Attack success rate comparison chart** (FGSM vs PGD vs others)
  - **Confusion matrix** - which classes are most vulnerable?
  - **Confidence distribution plots** - clean vs adversarial
  - **Epsilon vs success rate curve**
  - **Examples grid** - show best/worst attacks for each method

- [ ] Start populating `results/` folder:
  - `results/plots/` - all graphs and charts
  - `results/images/` - example images with predictions
  - Create `results/README.md` explaining all results

- [ ] Statistical analysis:
  - Which categories are most vulnerable to attacks?
  - Is there correlation between baseline confidence and attack success?
  - Compare different attack methods quantitatively

**Notebook**: `05_analysis.ipynb`

---

### 6. Code Organization & Documentation (Priority: MEDIUM)
**Time estimate: 2-3 hours**

- [ ] Add detailed docstrings to all functions
- [ ] Create utility module: `utils.py` or `attack_utils.py`
  - Move reusable functions from notebooks to scripts
  - Makes code more maintainable

- [ ] Update README.md with:
  - Week 3 progress
  - New results/findings
  - Updated repository structure

- [ ] Ensure all notebooks run end-to-end
- [ ] Update requirements.txt if added new libraries

---

## 📊 Expected Outputs by End of Week 3

### Hard Examples Generated:
- ✅ ~150 FGSM examples (from Week 2)
- 🆕 ~200 PGD examples
- 🆕 ~100 Targeted attack examples
- 🆕 ~150 OOD examples
- 🆕 ~100-150 Corner case examples
- **Total: 700-850 hard examples**

### Visualizations:
- Attack comparison charts
- Confusion matrices
- Success rate plots
- Example galleries
- t-SNE embeddings (optional, if time permits)

### Code:
- 3-4 new notebooks (PGD, corner cases, analysis)
- Well-documented, reproducible code
- Organized results folder

---

## 🚧 Potential Challenges & Solutions

### Challenge 1: Too many experiments, not enough time
**Solution**: Prioritize breadth over depth. Better to have all attack types working at basic level than perfect implementation of one method.

### Challenge 2: Data management (hundreds of images + metadata)
**Solution**: 
- Use structured JSON to store metadata
- Organize images in subfolders by attack type
- Consider using pickle/HDF5 for bulk storage

### Challenge 3: Computing resources (GPU time)
**Solution**:
- Work with smaller dataset subsets (100-200 images)
- Cache model outputs to avoid recomputation
- PGD is expensive - limit to reasonable iteration counts (20-40 steps)

### Challenge 4: Some attacks might not work well
**Solution**: That's actually interesting! Document which attacks fail and hypothesize why. This is valuable insight.

---

## 📈 Progress Tracking

Use this checklist throughout the week:

```
Day-by-day plan:
□ Sunday (Nov 16):    PGD implementation + initial testing (3hrs)
□ Monday (Nov 17):    Generate PGD examples, start targeted attacks (3hrs)
□ Tuesday (Nov 18):   OOD dataset collection & testing (3hrs)
□ Wednesday (Nov 19): Corner cases experiments (3hrs)
□ Thursday (Nov 20):  Analysis & visualization (3hrs)
□ Friday (Nov 21):    More visualizations, organize results (3hrs)
□ Saturday (Nov 22):  Code cleanup, documentation (2hrs)
□ Sunday (Nov 23):    Create week3log.txt, prepare for Week 4 (1hr)
```

Total time: ~20 hours spread across 8 days

---

## 🎓 Papers to Reference

1. **Madry et al. (2018)** - "Towards Deep Learning Models Resistant to Adversarial Attacks"
   - PGD attack original paper

2. **Carlini & Wagner (2017)** - "Towards Evaluating the Robustness of Neural Networks"
   - C&W attack (implement if time permits)

3. **Geirhos et al. (2019)** - "ImageNet-trained CNNs are biased towards texture"
   - Texture vs shape experiments

4. **Hendrycks et al. (2019)** - "Benchmarking Neural Network Robustness to Common Corruptions"
   - Ideas for corner cases

---

## 💡 Bonus Ideas (If You Have Extra Time)

- Feature visualization: What does the model "see" in adversarial examples?
- Transfer attacks: Do adversarial examples from ConvNext fool other models?
- Defense testing: Apply simple defenses (JPEG compression, smoothing) and see if attacks still work
- Gradient masking check: Verify the model isn't gradient-masked
- Auto-Attack: Industry-standard adversarial benchmark (very strong attack)

---

## ✅ Definition of "Done" for Week 3

- [ ] 500+ hard examples generated and saved
- [ ] At least 3 different attack strategies implemented
- [ ] Results folder populated with images and plots
- [ ] All notebooks documented and runnable
- [ ] Week3log.txt completed
- [ ] README.md updated with progress

---

**Remember**: Week 4 is for compilation and analysis, so Week 3 is your last chance to generate examples. Focus on getting diverse examples rather than perfecting any single attack!

Good luck! 🚀
