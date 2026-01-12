# Coating Motifs ML  
**Motif-aware machine learning for hardness prediction in multilayer YbSi–Mullite–Si environmental barrier coatings**

This repository provides the full **machine learning pipeline**, **feature engineering framework**, and **analysis scripts** used in the following study:

> **Data-Driven Prediction of Hardness and Layer Behavior in YbSi–Mullite–Si Environmental Barriers**  
> Emre Bal, Muhammet Karabas, Sadettin Y. Ugurlu  
> *Submitted in an ACS journal*

The codebase enables **reproducible motif-aware hardness prediction**, **through-thickness profiling**, and **layer-resolved analysis** for multilayer and functionally graded environmental barrier coatings (EBCs).

---

## 🔬 Scope of This Repository

This repository includes:

- Physics-aware **synthetic feature engineering** from nanoindentation data  
- **Layer-aware composition decoding** (YbSi / Mullite / Si fractions)  
- **ANOVA-based feature selection**  
- Training and evaluation of:
  - XGBoost
  - LightGBM
  - CatBoost
  - Equal-weight ensembles
- **Cross-motif generalization protocol** (train on motif E → test on A–D)
- **Noise-mitigation strategies**:
  - Within-layer neighborhood smoothing
  - Layer-mean reformulation
- **Model interpretability**:
  - SHAP analysis
  - Hierarchical clustering
  - Graph-theoretic (NetworkX) feature community analysis

---

## 📂 What Is *Not* Included (Important)

⚠️ **Experimental nanoindentation data are NOT included in this repository.**

Due to **copyright and data ownership restrictions**, the raw indentation datasets **cannot be publicly redistributed**.

All data usage in the associated manuscript complies with institutional and publisher policies.

---

## 📊 Data Access

The experimental nanoindentation data used in this study were obtained from prior experimental work and are **available upon reasonable request**.

To request access to the data, please contact:

- **Sadettin Y. Ugurlu**  
  📧 syavuzugurlu@akdeniz.edu.tr  
  ORCID: https://orcid.org/0000-0001-9589-0269

or

- **Emre Bal**  
  📧 emrebal@akdeniz.edu.tr

When requesting data, please briefly describe:
- Your affiliation  
- Intended academic use  
- Whether the data will be used for replication or extension

---

## ⚙️ Installation

### 1️⃣ Create a Python environment (recommended)

```bash
conda create -n coating_ml python=3.10 -y
conda activate coating_ml
