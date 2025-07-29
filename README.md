# GAN-Based Synthetic Data Generation

## Problem Statement
Generate synthetic tabular data using a Copula-based Wasserstein GAN (CopulaGAN), preserving both feature-wise distributions and inter-feature correlations.

## How to Run
1. Place `data.xlsx` in the project folder.  
2. Run `w_cop_GAN_training.py` to train and save models.  
3. Open `analysis_WCopulaGAN.ipynb` to evaluate saved models and generate synthetic data using `G_epoch706.pth`. 
4. Models upto 1000 epochs are given in `models3_subset`
4. Final output is saved as `synthetic_data.csv`.
