# PeptideFunction: ML-Driven Prediction of Bioactive Peptides

## 🔬 Project Overview
A machine learning pipeline designed to predict the biological function of peptides (e.g., Cell-Penetrating Peptides) using physicochemical properties. This project bridges **Bioinformatics** (feature extraction via AAindex) and **Ensemble Learning** (stacking multiple classifiers).

## 📊 Methodology
1.  **Data Curation**: 
    - Aggregated peptide sequences from published benchmarks (**C2PRED**, **MLCPP**, **KELM-CPPpred**).
    - Removed duplicates to create a high-quality dataset of ~1,800 sequences.
2.  **Feature Engineering**: 
    - Mapped amino acid sequences to a **566-dimensional vector space** using the **AAindex1** database (representing hydrophobicity, steric hindrance, isoelectric point, etc.).
3.  **Modeling Strategy**: 
    - Implemented a **Voting Ensemble Classifier** combining five distinct algorithms to maximize robustness:
        - Random Forest (Bagging)
        - Support Vector Machine (Kernel-based)
        - K-Nearest Neighbors (Instance-based)
        - Naive Bayes (Probabilistic)
        - Decision Tree
    
## 🚀 Results
The Ensemble model achieved an accuracy of **>84%** on the independent test set, demonstrating that combining structural/physicochemical features with voting logic outperforms simple baselines.

## 🛠 Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python model_training.py`
