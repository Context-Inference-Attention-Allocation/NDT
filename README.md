# **Contextual Inference and Attention Allocation**  

This repository contains code and data for the paper **Attention Allocation is Adapted to Context:
Evidence from a Naturalistic Discrimination Task**. The repository is divided into two main sections:  
1. **Illustrative Example** (synthetic data generation and parameter estimation)  
2. **Naturalistic Umpire Dataset** (contextual inference and policy estimation for real umpires)  

## **Repository Structure**  

### **1. Illustrative Example** (`illustrative_example/`)  
This folder contains synthetic data and scripts for generating and estimating model parameters.  

- `synthetic_data_observations.npy` – Generated synthetic observations  
- `synthetic_data_outcomes.npy` – Generated synthetic outcomes  
- `synthetic_data_generator.ipynb` – Code to generate synthetic data  
- `synthetic_data_estimation.ipynb` – Code to estimate model parameters from synthetic data (results referenced in the paper)  

### **2. Naturalistic Umpire Dataset** (`naturalistic_umpire_dataset/`)  
This section contains real-world data for six umpires, structured as follows:  

- **Each umpire has a dedicated folder** (`umpire_1/`, `umpire_2/`, ..., `umpire_6/`)  
- Each umpire's folder contains:  
  - `context_inference_umpire_#.csv` – Dataset for contextual inference (Phase 1)  
  - `utility_estimation_umpire_#.csv` – Dataset for estimating attention and task policy (Phase 2, includes pre-trained info from Phase 1)  

The main script for processing umpire data is:  
- `overall_estimation.ipynb` – Code to reproduce umpire results  

## **Usage Instructions**  

### **1. Synthetic Data Example**  
Run the following notebook in `illustrative_example/`:  
- Estimate model parameters: `synthetic_data_estimation.ipynb`  

### **2. Umpire Dataset Estimation**  
To reproduce results for umpires, navigate to `naturalistic_umpire_dataset/` and run:  
- `overall_estimation.ipynb`  

## **Dependencies**  
Ensure you have the following Python libraries installed:  
```bash
pip install numpy pandas matplotlib scipy jupyter
