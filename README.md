# Perovskite Stability ML Pipeline

> **Status: Work in Progress** 🚧
> This project is under active development. Core modules (data ingestion, filtering, 
> Digital Twin screening, T80 tracking, RUL forecasting, and XAI) are functional, 
> but refactoring, documentation, and additional validation are still ongoing.

End-to-end machine learning pipeline for outdoor perovskite solar cell stability analysis: data engineering, Digital Twin early screening, T80 survival tracking, RUL forecasting, and Explainable AI (SHAP).

---

## Overview

This project builds a complete ML stack on top of **28M+ telemetry points** from outdoor perovskite solar cells. The pipeline covers:

- **Data Engineering:** modular 9-stage architecture processing 28M telemetry points into optimized Parquet, with disk-to-disk streaming and empirical validation of every threshold.
- **Physical QC & Filtering:** three-layer J-V curve filtering (physical invariants → per-curve scores → traceable decision) with automated noise-floor estimation.
- **Digital Twin Early Screening:** dual XGBoost model (PCE + pFF) trained on the mature phase of healthy cells, with OOF-residual thresholds for anomaly detection during the burn-in window.
- **T80 Survival Tracking:** physical lifecycle tracking of the 80% degradation threshold over consecutive days.
- **RUL Forecasting:** hybrid kinematic engine (XGBoost velocity + structural floor + soft countdown) with walk-forward API calibration.
- **Multivariate Trajectory Forecasting:** per-parameter autoregressive engines (XGBoost for PCE, RandomForest for FF/Jsc/Voc) with LOOCV validation and persistence blending.
- **Explainable AI (XAI):** SHAP values, surrogate decision trees, and forensic analysis of physical vs. environmental degradation.

---

## Repository Structure

```
src/
├── 01_ingest_raw.py
├── 02_jv_filtering.py
├── 03_jv_clustering.py
├── 04_mppt_aggregation.py
├── 05_merge_jv_mppt.py
├── 06_jv_mppt_t80_tracker.py
├── 07_jv_mppt_early_screening.py
├── 08_mppt_rul_forecasting.py
├── 09_jv_mppt_trajectory_forecasting.py
├── config.py
├── audits/          # Diagnostic and calibration audit scripts
├── calibration/     # Optimizers for RUL and trajectory coefficients
├── viz/             # Plotting utilities
├── xai/             # SHAP-based explainability modules
└── archive/         # Deprecated code (kept for reference)

streamlit_app.py     # Interactive dashboard
main.py              # Pipeline orchestrator
requirements.txt     # Python dependencies
```

---

## Stack

Python, pandas, numpy, scikit-learn, XGBoost, PyArrow, Plotly, Streamlit, SHAP, joblib.

---

## Roadmap

- [ ] Complete documentation and docstrings for all modules
- [ ] Add unit tests for critical stages (filtering, physics extraction, T80 tracking)
- [ ] Finalize calibration of kinematic coefficients across the full cohort
- [ ] Improve the Streamlit dashboard's interaction flow
- [ ] Package the pipeline for reproducibility (Dockerfile + CI)

---

## Data Source & Acknowledgments

The telemetry data analyzed in this project was provided by the **ParaSol platform** at the **Open Solar Stability (OSS) Lab**, University of Zaragoza (Spain), led by Dr. Emilio J. Juarez-Perez. Data was shared with the **University of Seville** for collaborative research.

**ParaSol platform details:**

- Outdoor testing facility for perovskite solar cell stability under real environmental conditions
- MPPT tracking: Perovskino galvanostatic tracker (open-source, high-hysteresis capable)
- IV sweeps: Reverse + Forward scan directions
- Sensors: POA reference cell, ambient/module thermistors, capacitive humidity sensor
- Platform: [www.emiliojuarez.es](https://www.emiliojuarez.es)
- OSS Lab GitHub: [github.com/ej-jp/perovskino](https://github.com/ej-jp/perovskino)

**My contribution** focuses on the machine learning layer: data engineering (28M+ telemetry points), Digital Twin early screening, T80 survival tracking, RUL forecasting, and Explainable AI (SHAP). The platform hardware, data collection, and experimental design are the work of the OSS Lab team.

---

## Author

**Cristian Carretero Fernández**
Data Scientist & ML Researcher | Physicist & Materials Engineer | MSc Data Science (UOC)

- GitHub: [@cristian-carretero](https://github.com/cristian-carretero)
- LinkedIn: [cristian-carretero-fernandez](https://www.linkedin.com/in/cristian-carretero-fernandez)