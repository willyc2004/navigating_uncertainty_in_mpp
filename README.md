# Navigating Demand Uncertainty in Container Shipping: Deep Reinforcement Learning for Enabling Adaptive and Feasible Master Stowage Planning

## Overview
This repository contains supplementary code for our double-blind submission(s).

## Submission Guidelines
To ensure compliance with double-blind review requirements, we have taken the following measures:
- **Author identities are not disclosed** anywhere in the code or associated files.
- **Metadata in scripts and documentation** has been scrubbed of personal information.
- **Code comments and version control history** do not contain author-identifying information.

## Repository Structure with reference to the paper
```
📂 project_root
├── 📂 environment/          # Environment scripts (Section 4, Appendix A, Appendix C)
├── 📂 models/               # Model scripts (Section 5)
├── 📂 rl_algorithms/        # Reinforcement learning algorithms (Appendix B, Appendix E - DRL Algorithms)
├── 📂 results/              # Results of experiments (Section 6, Appendix F)
├── 📄 main.py               # Main script (Appendix E - Main Execution Script)
├── 📄 sweep.py              # Sweep script (Appendix G)
├── 📄 scenario_tree_mip.py  # SMIP script (Appendix D)
├── 📄 requirements.txt      # Required dependencies (Software Requirements)
├── 📄 config.yaml           # Configuration file (Appendix E - Hyperparameter Configuration)
├── 📄 sweep_config.yaml     # Configuration file for hyperparameter sweeps (Appendix E - Hyperparameter Ranges)
├── 📄 README.md             # This document
├── 📄 .gitignore            # Git ignore files
└── 📄 LICENSE               # License information
```

## Running the Code
To execute the supplementary code, ensure the config files are correctly set up. Then, use the following instructions:
```sh
python main.py  # Modify based on actual usage
```
Ensure that any required dependencies are installed using:
```sh
pip install -r requirements.txt
```

## Reproducibility
To facilitate reproducibility while maintaining anonymity, we:
- Provide necessary scripts with minimal setup.
- Avoid hard-coded paths linked to personal systems.

## Contact
For inquiries, please use the conference/journal’s anonymous submission system. Do not include identifying contact information in this repository.

---
> **Note:** This repository will be updated with author details after the review process is complete.
