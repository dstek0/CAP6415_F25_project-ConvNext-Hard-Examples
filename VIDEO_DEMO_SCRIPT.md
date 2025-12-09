# Video Demo Script: Probing ConvNext SOTA with Hard Examples

**Target Duration:** 15-18 minutes (within 10-20 minute requirement)  
**Student:** Dylan Stechmann  
**Course:** CAP6415 - Computer Vision, Fall 2025

---

## Pre-Recording Checklist

Before you start recording, make sure:

- [ ] Close all unnecessary browser tabs and applications
- [ ] Have GitHub repository open in browser: https://github.com/dstek0/CAP6415_F25_project-ConvNext-Hard-Examples
- [ ] Have your project folder open in file explorer
- [ ] Have terminal/command prompt ready (cd to project directory)
- [ ] Have Jupyter Lab or notebook viewer ready
- [ ] Test your microphone levels (speak at normal volume)
- [ ] Turn off notifications (Slack, email, etc.)
- [ ] Have a glass of water nearby

---

## Part 1: Introduction & Repository Overview (3-5 minutes)

### Opening [0:00-0:30]

**[Start screen recording - show your desktop briefly then open browser]**

> "Hi, I'm Dylan Stechmann, and this is my final project demo for CAP6415 Computer Vision. Today I'm going to show you my project on probing state-of-the-art models with hard examples."

> "The goal of this project was to systematically find weaknesses in ConvNext, which is a modern SOTA vision model, by understanding what types of images genuinely confuse it—not through random testing, but by analyzing specific failure modes."

**[Navigate to GitHub repository]**

### Repository Overview [0:30-2:00]

> "Let me start by showing you the repository structure on GitHub."

**[Scroll through the main README]**

> "In the README, I provide an abstract explaining the project motivation, key findings summarized in a table, and details about what I investigated."

**[Point out the key findings table]**

> "Some of the main findings: PGD attacks achieve 100% success rate with very small perturbations - meaning they fool the model on every single image. Even simple FGSM attacks achieve 92.6% success. The model was tested on an NVIDIA A100 GPU with 1,000 real test images."

**[Scroll down to repository structure section]**

> "The repository is organized into several folders:
> - **notebooks** - for Jupyter notebooks where I did exploratory work
> - **scripts** - for reproducible Python scripts  
> - **results** - with plots and images showing all the findings
> - **data** - for datasets, though these aren't committed to git since they're large
> - And I have weekly logs tracking my progress over the 5 weeks."

### Code Structure [2:00-3:30]

**[Click into the scripts folder]**

> "In the scripts folder, I have four main Python files:"
> - "**attack_utils.py** implements FGSM and PGD adversarial attacks"
> - "**visualization_utils.py** has plotting functions"  
> - "**run_attacks.py** is the main script to generate adversarial examples"
> - "**generate_plots.py** creates all the result visualizations"

**[Click into the notebooks folder]**

> "The notebooks folder has two notebooks: one for loading the model and verifying the setup, and another for baseline testing."

**[Click into results folder]**

> "And the results folder contains all the plots I generated plus detailed documentation of the findings."

**[Brief pause]**

> "Alright, now let me show you the code actually running."

---

## Part 2: Running the Code (8-12 minutes)

### Notebook 1: Model Loading [3:30-6:00]

**[Open terminal or Jupyter Lab]**

> "First, I'll start Jupyter Lab to run through the notebooks."

**[Type command]**
```bash
jupyter lab
```

> "While that's loading... [wait for Jupyter to open]... okay, here we go."

**[Navigate to 00_model_loading.ipynb]**

> "This first notebook loads the ConvNext-Base model and verifies everything is set up correctly."

**[Run the first few cells]**

> "Let me run these cells. First cell is just markdown explaining the notebook..."

**[Run cell 2 - environment setup]**

> "This cell imports all the necessary libraries and checks the versions. You can see PyTorch is installed, CUDA is available—or if not, it falls back to CPU which is fine for this demo."

**[Run cell 4 - model loading]**

> "Now loading the ConvNext-Base model from the timm library. This model has about 89 million parameters and was pretrained on ImageNet-1K."

**[Run cell 6 - data transforms]**

> "Setting up the standard ImageNet preprocessing pipeline with normalization."

**[Run cell 8 - test inference]**

> "And here's a quick test to make sure inference works. You can see the output shape is 1 by 1000, which makes sense—1000 ImageNet classes."

**[Run cell 10 - load labels]**

> "Loading the ImageNet class labels so we can interpret the predictions."

> "So this notebook just verifies that the model loads correctly and we can run inference. Pretty straightforward."

### Notebook 2: Baseline Testing [6:00-8:00]

**[Navigate to 01_baseline_testing.ipynb]**

> "The second notebook does baseline accuracy testing on clean ImageNet images."

**[Run a few key cells showing baseline results]**

> "This establishes the baseline performance before we start attacking the model. On natural photos from ImageNet validation set, the model gets around 85% top-1 accuracy, which is pretty good for ConvNext-Base."

**[Show any visualization cells if they exist]**

> "You can see some example predictions here—the model is quite confident and usually correct on clean, natural images."

### Running Attack Scripts [8:00-10:30]

**[Switch to terminal, navigate to scripts folder]**

```bash
cd scripts/
```

> "Now let me show you the attack scripts. These are Python scripts you can run from the command line to reproduce all the results."

**[Run FGSM attack]**

```bash
python run_attacks.py --attack fgsm --epsilon 0.03
```

> "This runs the FGSM attack with epsilon of 0.03. FGSM is the Fast Gradient Sign Method—it's a single-step attack that adds a small perturbation in the direction of the gradient."

**[While it runs, explain]**

> "You can see it's processing images and showing the attack success rate. FGSM typically gets around 60% success rate, meaning 60% of the images are misclassified after adding the adversarial perturbation."

**[If it finishes quickly, run PGD too, otherwise skip to next]**

> "I could also run PGD attacks with:"

```bash
python run_attacks.py --attack pgd --epsilon 0.03 --steps 20
```

> "But since these take longer—PGD does 20 iterative steps—I'll skip running this live. The results are already captured in the plots."

### Generating Visualizations [10:30-12:00]

**[Run plot generation script]**

```bash
python generate_plots.py --output ../results/plots
```

> "Now let me regenerate all the result plots using the generate_plots script."

**[While it runs]**

> "This creates six key visualizations:
> - Attack comparison showing FGSM vs PGD
> - Epsilon curves showing the tradeoff between perturbation size and attack success
> - Confidence distributions
> - Out-of-distribution accuracy breakdown
> - Class vulnerability heatmap
> - And a comprehensive results summary"

**[When finished]**

> "Great, all plots generated successfully in the results/plots folder."

---

## Part 3: Results Discussion (5-8 minutes)

### Opening Results [12:00-12:30]

**[Navigate to results/plots folder in file browser or show in GitHub]**

> "Now let's look at the actual findings. This is the interesting part—what did we learn about ConvNext's weaknesses?"

### Plot 1: Attack Comparison [12:30-13:30]

**[Open attack_comparison.png]**

> "First, the attack comparison. This shows the success rates of different attacks at epsilon 0.03."

> "FGSM, the simple one-step attack, achieves 92.6% success rate - that's already devastating. But PGD, the iterative attack, is even more powerful—PGD with just 5 steps achieves 100% success rate. PGD-10, PGD-20, and PGD-40 all maintain 100% success."

> "This is a striking finding: even with minimal iterations, PGD completely breaks ConvNext-Base. Every single image can be successfully attacked with imperceptible perturbations."

### Plot 2: Epsilon Curves [13:30-14:30]

**[Open epsilon_curves.png]**

> "This plot shows how attack success rate changes with epsilon—the size of the perturbation. Epsilon of 0.03 means we change each pixel by at most 3 out of 255, which is generally imperceptible to humans."

> "You can see at very small epsilons, attacks barely work. But there's a sharp transition around epsilon 0.02-0.03 where success rate jumps significantly. This is the 'adversarial regime' where perturbations are small enough to be nearly invisible but large enough to fool the model."

### Plot 3: OOD Breakdown [14:30-15:30]

**[Open ood_breakdown.png]**

> "Now for out-of-distribution robustness. I tested the model on different visual domains—not adversarial perturbations, just different styles of images."

> "Natural photos are the baseline at 85%. Stylized photos drop to 75%—only a 10% decrease, not bad. Paintings drop to 70%, sketches to 60%. But look at cartoons—they drop all the way to 25% accuracy. That's a 60% decrease!"

> "This tells us the model is extremely brittle to cartoon-style images. It probably relies on photorealistic textures and lighting that aren't present in cartoons."

### Plot 4: Class Vulnerability [15:30-16:30]

**[Open class_vulnerability.png]**

> "This heatmap shows which ImageNet classes are most vulnerable to attacks. Red means highly vulnerable, green means robust."

> "The most vulnerable classes are fine-grained categories: dog breeds, cat breeds, bird species, snake species. These classes look very similar and are often distinguished by texture patterns, making them easy to fool."

> "The most robust classes are man-made objects: keyboards, toasters, hammers, vehicles. These have strong shape cues and distinct structures, so texture-based attacks don't work as well."

> "This confirms the texture bias hypothesis—ConvNext, like many CNNs, relies heavily on texture over shape for classification."

### Summary Plot [16:30-17:00]

**[Open results_summary.png]**

> "Finally, here's the comprehensive results summary showing all key findings in one figure."

> "The main takeaways are:
> 1. ConvNext is vulnerable to adversarial attacks even at small epsilon
> 2. The model has a strong texture bias
> 3. Confidence calibration is poor—it stays confident even when wrong
> 4. OOD robustness varies dramatically by domain
> 5. Fine-grained categories are the weakest point"

---

## Part 4: Conclusions & Future Work (2-3 minutes)

### Limitations [17:00-17:45]

> "Now, let me discuss some limitations of this work."

> "First, I only tested ConvNext-Base. It would be interesting to test other architectures like Vision Transformers or newer ConvNext variants to see if the vulnerabilities are architecture-specific or more general."

> "Second, I was limited to ImageNet validation data and subsets of it due to computational constraints. Testing on larger datasets or domain-specific datasets could reveal other weaknesses."

> "Third, I focused primarily on FGSM and PGD attacks. There are more sophisticated attacks like Carlini & Wagner or Auto-Attack that might reveal additional vulnerabilities, but they're much more computationally expensive."

### Future Work [17:45-18:30]

> "For future work, I'd be interested in:
> - Testing adversarial training—can we make ConvNext more robust?
> - Exploring why cartoons specifically break the model so badly
> - Investigating the texture bias more deeply with controlled experiments
> - Trying transfer attacks—do adversarial examples generated for ConvNext also fool other models?"

### Closing [18:30-19:00]

> "Overall, this project taught me that standard accuracy benchmarks don't tell the whole story. ConvNext achieves 85% on ImageNet, which sounds great, but it has serious weaknesses that only show up when you probe it systematically."

> "Understanding these failure modes is important for deploying vision models in real-world applications where robustness and reliability matter."

**[Show GitHub repository one final time]**

> "All the code, results, and documentation are available in the GitHub repository. Everything is reproducible—you can clone the repo, install the requirements, and run all the scripts and notebooks yourself."

> "Thanks for watching, and I hope you found the project interesting!"

**[End recording]**

---

## Post-Recording Checklist

After recording:

- [ ] Review the video to make sure audio is clear
- [ ] Check that all important parts are visible (no overlapping windows)
- [ ] Verify video is between 10-20 minutes
- [ ] Export/render the video in MP4 format (H.264 codec recommended)
- [ ] Test that the video plays correctly
- [ ] Upload to the required submission platform

---

## Troubleshooting Tips

**If notebooks don't run:**
- Restart kernel and run all cells before recording
- Or, record the notebook having already been run (showing outputs)

**If scripts take too long:**
- Use smaller dataset subsets for demo purposes
- Or, show the command and explain what it does without waiting for completion
- Or, cut the video and resume after it finishes (mention "this took X minutes")

**If you make a mistake while recording:**
- Pause briefly, then continue from where you made the mistake
- You can edit out the pause later, or just leave it—minor mistakes make it feel more human!

**If you go over 20 minutes:**
- Cut down the results discussion section
- Skip running one of the scripts live (just describe it)
- Speak slightly faster (but still clearly)

---

## Key Things to Remember

1. **Sound natural!** Don't read the script word-for-word like a robot. Use it as a guide.

2. **Speak clearly and at a moderate pace.** Professors will appreciate not having to rewind.

3. **Show enthusiasm!** You did good work—let that come through.

4. **Point out your key contributions:** The attacks you implemented, the analysis you did, the insights you discovered.

5. **Be honest about limitations.** It shows maturity and understanding.

6. **End strong with the GitHub repo link** so they know where to find everything.

Good luck! You've got this! 🎬

