# GAN-Based Synthetic Data Generation

## Problem Statement
Generate synthetic tabular data using a Copula-based Wasserstein GAN (CopulaGAN), preserving both feature-wise distributions and inter-feature correlations.

Here’s the **resume-style project entry** for your **CopulaGAN (Wasserstein GAN for synthetic data)**, keeping it short and consistent with the others:

---

**Synthetic Data Generation using CopulaGAN**

* Implemented a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** combined with **copula transformations** to generate high-quality synthetic tabular data.
* Preserved both **feature distributions** and **inter-feature correlations**; evaluated realism using **Earth Mover’s Distance, KS-test, and correlation MSE**.
* Achieved strong alignment between real and synthetic data across most features, validating CopulaGAN as a robust framework for **privacy-preserving data augmentation**.


## How to Run
1. Place `data.xlsx` in the project folder.  
2. Run `w_cop_GAN_training.py` to train and save models.  
3. Open `analysis_WCopulaGAN.ipynb` to evaluate saved models and generate synthetic data using `G_epoch706.pth`. 
4. Models upto 1000 epochs are given in `models3_subset`
4. Final output is saved as `synthetic_data.csv`.

