Aşağıda, **ACS makalesiyle birebir uyumlu**, akademik olarak temiz, dergi/etik kurallara uygun ve GitHub için **doğrudan kullanılabilir** kapsamlı bir **`README.md`** taslağı veriyorum.
Metin; **kod–makale ilişkisini**, **veri erişim kısıtlarını (copyright)**, **kurulum**, **çalıştırma**, **tekrarlanabilirlik** ve **atıf** konularını net biçimde kapsar.

Bunu **aynen kopyalayıp** repoya `README.md` olarak koyabilirsin.

---

````markdown
# Coating Motifs ML  
**Motif-aware machine learning for hardness prediction in multilayer YbSi–Mullite–Si environmental barrier coatings**

This repository provides the full **machine learning pipeline**, **feature engineering framework**, and **analysis scripts** used in the following study:

> **Data-Driven Prediction of Hardness and Layer Behavior in YbSi–Mullite–Si Environmental Barriers**  
> Emre Bal, Muhammet Karabas, Sadettin Y. Ugurlu  
> *Submitted / under review / accepted in an ACS journal*

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
````

### 2️⃣ Install required packages

```bash
pip install numpy pandas scipy scikit-learn
pip install xgboost lightgbm catboost
pip install shap matplotlib seaborn
pip install networkx tqdm joblib
```

> Tested with Python ≥ 3.9.
> GPU is **not required**.

---

## 🧪 Pipeline Overview

The typical workflow is:

1. **Raw indentation tables**
2. Layer-wise concatenation → **continuous through-thickness coordinate**
3. Parsing of categorical layer labels into:

   * `ybsi_pct`
   * `mullite_pct`
   * `si_pct`
4. Synthetic feature construction:

   * Device-signal transforms
   * Depth coupling
   * Composition–mechanics interactions
   * Hybrid template deviations
5. Feature selection (ANOVA F-test, training-only)
6. Model training (Motif E)
7. Evaluation on unseen motifs (A–D)
8. Post-processing:

   * Neighborhood smoothing
   * Layer-mean aggregation
9. Interpretation:

   * SHAP
   * Clustering
   * Feature networks

---

## ▶️ Running the Code

> **Note:** You must first place the experimental CSV files (obtained via request) into the expected data directory.

Example execution order:

```bash
python 1_preprocess_and_concat.py
python 2_feature_engineering.py
python 3_feature_selection_anova.py
python 4_train_models.py
python 5_evaluate_cross_motif.py
python 6_shap_analysis.py
python 7_feature_clustering.py
```

Script names may vary depending on the exact analysis stage; see inline documentation in each file.

---

## 📈 Outputs

The pipeline produces:

* Depth-wise hardness predictions
* Smoothed hardness profiles
* Layer-mean hardness predictions
* Performance metrics:

  * R²
  * MAE
* SHAP importance plots
* Feature dendrograms
* Feature correlation networks
* CSV summaries for publication figures

---

## 📌 Reproducibility Notes

* Feature selection is **fit only on the training motif** to prevent leakage
* Identical feature sets are applied to all test motifs
* Ensemble models use **fixed equal weights**
* No hyperparameter tuning is performed on test motifs

---

## 🧠 Intended Use

This repository is intended for:

* Academic research
* Methodological comparison
* Extension to other multilayer / graded ceramic systems
* Educational purposes in materials informatics

❌ Not intended for proprietary or commercial use without permission.

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{Bal2025CoatingMotifsML,
  title   = {Data-Driven Prediction of Hardness and Layer Behavior in YbSi–Mullite–Si Environmental Barriers},
  author  = {Bal, Emre and Karabas, Muhammet and Ugurlu, Sadettin Y.},
  journal = {ACS Journal (under review)},
  year    = {2025}
}
```

---

## 📄 License

This repository is released for **academic use only**.

All experimental data remain the property of the original authors and institutions.

---

## 📬 Contact

For questions, collaboration, or extensions:

**Sadettin Y. Ugurlu**
Akdeniz University – Materials Science & Engineering
📧 [syavuzugurlu@akdeniz.edu.tr](mailto:syavuzugurlu@akdeniz.edu.tr)

```

