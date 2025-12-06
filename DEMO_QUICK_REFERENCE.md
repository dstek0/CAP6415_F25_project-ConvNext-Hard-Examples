# Quick Reference for Video Demo

## Commands You'll Need

### Start Jupyter Lab
```bash
cd /path/to/CAP6415_F25_project-ConvNext-Hard-Examples
jupyter lab
```

### Run Attack Scripts
```bash
cd scripts/

# FGSM attack
python run_attacks.py --attack fgsm --epsilon 0.03

# PGD attack  
python run_attacks.py --attack pgd --epsilon 0.03 --steps 20

# Targeted attack
python run_attacks.py --attack targeted --epsilon 0.05 --target 281
```

### Generate All Plots
```bash
cd scripts/
python generate_plots.py --output ../results/plots
```

---

## Before Recording - 5 Minute Setup

1. **Close Everything**
   - Close all browser tabs except GitHub
   - Close Slack, Discord, email
   - Close any other applications
   - Turn off notifications

2. **Open What You Need**
   - Browser: https://github.com/dstek0/CAP6415_F25_project-ConvNext-Hard-Examples
   - Terminal (in project directory)
   - File explorer (showing project folder)

3. **Test**
   - Test microphone (record 10 seconds and play back)
   - Make sure screen resolution is readable
   - Have water nearby

---

## Timing Guide (Aim for 15-18 minutes total)

| Section | Time | What to Show |
|---------|------|--------------|
| Intro & Repo | 3-5 min | GitHub README, folder structure |
| Running Code | 8-12 min | Notebooks, scripts, generate plots |
| Results | 5-8 min | Walk through each plot |
| Conclusion | 2-3 min | Limitations, future work, wrap up |

---

## What to Say (Very Short Version)

**Intro:** "Hi, I'm Dylan Stechmann. This project probes ConvNext to find weaknesses."

**Repo Tour:** "Here's the structure: notebooks for exploration, scripts for reproducibility, results with plots."

**Running Code:** "Let me show it working. First, load the model... now run attacks... generate plots."

**Results:** "Key findings: PGD gets 92% success, cartoons break the model, texture bias confirmed."

**Conclusion:** "Limitations: only ConvNext-Base, limited data. Future work: test other models, adversarial training."

---

## If Something Goes Wrong

**Script takes too long:**
- Say "This typically takes about X minutes, so I'll cut ahead to when it finishes"
- Or use smaller test set for demo

**Notebook errors:**
- Have notebooks pre-run with outputs saved
- Say "I've already run this to save time"

**You forget what to say:**
- Pause briefly (can be edited out)
- Refer to the plots and describe what you see
- Natural pauses are fine!

---

## Key Points to Hit

✓ Mention it's a SOTA model (ConvNext-Base)  
✓ Explain you're probing for weaknesses, not just attacking randomly  
✓ Show the repository is well-organized  
✓ Demonstrate code actually runs  
✓ Explain your key findings clearly  
✓ Be honest about limitations  
✓ End with GitHub link

---

## After Recording

1. Watch it once through to check quality
2. Make sure it's 10-20 minutes
3. Export as MP4 (H.264 codec)
4. Test that the file plays
5. Upload to submission platform
6. Submit before December 8, 11:59 PM!

---

**You've got this! The hard work is done—now just explain it clearly.** 🎬

