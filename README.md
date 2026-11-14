# Multi-Level Driver Drowsiness Detection Using Hybrid Facial Features and Machine Learning

A computer vision and machine learning project that classifies driver drowsiness into three levels — **Alert**, **Low Vigilant**, and **Drowsy** — using facial-behaviour features and machine learning models.

## Project Overview

Driver fatigue can reduce attention and reaction time, increasing the risk of road accidents. This project investigates a camera-based approach to detecting drowsiness from facial cues without requiring wearable sensors.

The approach extracts facial-behaviour features with MediaPipe FaceMesh, applies subject-specific calibration, and evaluates classical machine learning models and a soft-voting ensemble on the UTA-RLDD dataset. Experiments also investigate how performance changes across different validation protocols, calibration modes and feature groups.

> **Research note:** This repository reports offline model evaluations on recorded videos. They do not demonstrate a deployed, real-time in-vehicle warning system, and the project is an academic research prototype that must not be used as the sole basis for deciding whether a driver is safe to drive.

> **Calibration caveat (read before the results):** The headline results use `CALIBRATION = subject_alert`, where each subject's baseline is computed from that subject's entire alert recording. This requires a known-alert reference at test time, so it is **not label-free**, and because each state is a separate recording in UTA-RLDD it may also let recording-session differences help the classifier. These results should be read as an **oracle (upper-bound) setting**, not as expected performance of a deployed system. Label-free calibration modes (`per_video_head`, `subject_alert_head`) are implemented in the code but are not part of the reported results. Details are in [Calibration modes](#calibration-modes) and [Limitations and Future Work](#limitations-and-future-work).

## Objectives

- Detect three levels of driver vigilance: Alert, Low Vigilant, and Drowsy.
- Represent driver behaviour using multiple facial regions rather than relying on a single cue.
- Evaluate Random Forest, XGBoost, and Multilayer Perceptron (MLP) classifiers.
- Test a soft-voting ensemble of the classifiers.
- Compare performance on seen-subject and unseen-subject evaluation protocols.
- Measure the contribution of individual facial feature groups through ablation experiments.

## Dataset

The project uses the **UTA-RLDD (UTA Real-Life Drowsiness Dataset)** as the evaluation dataset. It contains labelled recordings across alert (code 0), low-vigilant (code 5), and drowsy (code 10) states and includes 60 subjects.

The dataset is not included in this repository. Obtain it from its official distribution source and follow its access and usage conditions before running experiments.

## Methodology

The processing pipeline is:

1. **Video preparation:** Organize recordings; split videos (for example `10_1`, `10_2`) are processed as separate parts.
2. **Facial landmark extraction:** `extract_features.py` runs MediaPipe FaceMesh on every video and caches per-frame signals (eye aspect ratio, mouth aspect ratio, head pose, eyebrow-to-lid distance) as `.npz` files.
3. **Feature engineering:** Notebook `02` builds 30-second windows (15-second step) and derives 28 window features from the per-frame signals.
4. **Subject-specific calibration:** Features are normalized against a baseline; the mode is selected by `CALIBRATION` (see below). The reported results use an alert-state reference for the subject.
5. **Model training and evaluation:** Notebooks `03` and `04` tune and train the individual classifiers and evaluate the ensemble.
6. **Prediction:** Each evaluated window is assigned to Alert, Low Vigilant, or Drowsy; window predictions can also be aggregated per video.

### Per-frame signals

The extractor writes 10 columns per frame: `ear_l, ear_r, mar, pitch, yaw, roll, brow_l, brow_r, interocular, face_found`. Head pose is computed with `cv2.solvePnP` on six landmarks using a generic 3D face model and an approximate camera matrix, so the angles are relative indicators rather than calibrated measurements. Eyebrow distance is the brow-to-upper-lid distance divided by the inter-ocular distance. Frames are down-scaled to at most 640 px width and every second frame is kept (`--stride 2`); timestamps use the real frame index and fps.

### Facial feature groups

The 28 window features fall into four groups (exact column names are in `feature_groups.json`):

| Feature group | Number of features | Examples / role |
|---|---:|---|
| Eye | 14 | Eye Aspect Ratio (EAR), blink behaviour, eye-closure measures and PERCLOS |
| Mouth | 5 | Mouth Aspect Ratio (MAR) and mouth movement |
| Head | 7 | Head-pose-related measurements |
| Eyebrow | 2 | Eyebrow movement measurements |

### Windowing and calibration configuration

The `run_info.json` file records the configuration used for the reported subject-alert run:

| Setting | Value |
|---|---:|
| Calibration | `subject_alert` |
| Window length | 30 seconds |
| Window step | 15 seconds |
| Blink threshold | 0.8 |
| Closed-eye thresholds | 0.5, 0.7, 0.8, 0.9 |

The windowing configuration uses overlapping 30-second windows with a 15-second step.

### Calibration modes

Calibration normalizes window features by a baseline (open-eye EAR level, MAR / eyebrow / head-pose medians). The mode is set by `CALIBRATION` in notebook `02`:

| Mode | Baseline computed from | Label-free? | Status in this report |
|---|---|---|---|
| `none` | No baseline (raw features) | Yes | Diagnostic run, macro-F1 reported below |
| `per_video` | The whole of each video | Yes, but uses the full video | Diagnostic run, macro-F1 reported below |
| `subject_all` | All of a subject's videos | No state labels needed, but pools all three states | Diagnostic run, macro-F1 reported below |
| `per_video_head` | First `CAL_S` seconds (default 120) of each video; those windows are excluded | Yes | Implemented in code, not evaluated in the reported results |
| `subject_alert_head` | First `CAL_S` seconds of the subject's alert video; those windows are excluded | No, needs a known-alert recording | Implemented in code, not evaluated in the reported results |
| `subject_alert` | Entire alert video(s) of the subject | No, needs a known-alert recording | **Headline results (oracle upper bound)** |

Unseen-subject macro-F1 of the ensemble under each evaluated calibration mode:

| Calibration | Macro-F1 (unseen-subject, ensemble) |
|---|---:|
| `none` | 0.436 |
| `per_video` | 0.451 |
| `subject_all` | 0.451 |
| `subject_alert` | 0.597 |

The large gap between `subject_alert` and the other modes shows how strongly the result depends on the calibration choice. It is consistent with `subject_alert` benefiting from its use of the alert reference, but it is not proof of a specific cause.

## Machine Learning Models

The tuned model configuration covers:

- **Random Forest (RF)**
- **XGBoost (XGB)**
- **Multilayer Perceptron (MLP)**
- **Soft-voting ensemble** combining the classical classifiers

### Best recorded hyperparameters

These values are taken from `best_params.json`. They were tuned on folds 0–3 under `subject_alert` calibration and were not re-tuned for the other calibration modes.

**Random Forest**
```json
{
  "max_depth": 6,
  "max_features": "log2",
  "min_samples_leaf": 6,
  "n_estimators": 222
}
```

**XGBoost**
```json
{
  "colsample_bytree": 0.8902645881432277,
  "learning_rate": 0.027550628717863315,
  "max_depth": 2,
  "n_estimators": 365,
  "subsample": 0.8149492917796043
}
```

**MLP**
```json
{
  "alpha": 1.685978023170582e-05,
  "hidden_layer_sizes": [32],
  "learning_rate_init": 0.0005846326121643416
}
```

## Experimental Results

All results in this section use **calibration = `subject_alert` (oracle upper bound)**. The following results are transcribed from `subject_alert_w30/main_results.csv`. Values are percentages. The primary cross-subject result is the **unseen-subject** (leave-one-subject-out) evaluation.

### Unseen-subject evaluation

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Majority-class baseline | 33.75% | 11.25% | 33.33% | 16.82% |
| PERCLOS-only Logistic Regression | 47.36% | 49.19% | 47.50% | 45.60% |
| Random Forest | 58.95% | 57.87% | 59.06% | 58.13% |
| XGBoost | 60.14% | 58.95% | 60.26% | 59.32% |
| MLP | 59.23% | 58.03% | 59.34% | 58.42% |
| **Soft-voting ensemble** | **60.65%** | **59.31%** | **60.76%** | **59.72%** |

The ensemble's reported 95% accuracy confidence interval is **56.7%–64.7%**. Class-wise, the ensemble recognises Alert best (recall about 0.86) and Low Vigilant worst (recall about 0.41).

### Other validation protocols

| Validation protocol | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Seen-subject (time-blocked) | Soft-voting ensemble | 74.30% | 73.98% |
| Seen-subject (video held out) | Soft-voting ensemble | 54.94% | 53.91% |
| Unseen-subject | Soft-voting ensemble | 60.65% | 59.72% |

These protocols measure different generalization settings and should not be treated as interchangeable. In particular, the higher time-blocked seen-subject score does not mean the model achieves the same performance on previously unseen drivers.

### Video-level accuracy

For the unseen-subject ensemble, the results report:

- Majority-vote video accuracy: **64.44%**
- Mean-probability video accuracy: **63.89%**

These are separate video-level aggregation metrics; they are not the same as window-level classification accuracy.

### Paired comparisons

The paired-comparison results report that, in the unseen-subject evaluation:

- The ensemble exceeded the PERCLOS-only Logistic Regression baseline by **13.3 percentage points** (95% interval for the difference: +9.0 to +17.5 points; supported in the comparison).
- The ensemble's differences versus XGBoost, Random Forest, and MLP were not supported by the reported paired comparisons.

This means the ensemble performed better than the simple PERCLOS-only baseline in this evaluation, but the statistical comparisons do not establish that it is definitively better than each individual machine learning model.

## Ablation Study

The ablation results are from `subject_alert_w30/ablation.csv`. They use **GroupKFold(5)** with `subject_alert` calibration, a different protocol from the leave-one-subject-out main table.

| Feature configuration | Accuracy | Macro F1 |
|---|---:|---:|
| Only mouth | 47.46% | 45.20% |
| Only eyebrow | 50.17% | 47.46% |
| Only head | 52.35% | 51.01% |
| Only eye | 53.61% | 52.87% |
| Without head | 57.40% | 56.16% |
| Without eyebrow | 60.03% | 58.94% |
| Without mouth | 60.14% | 59.28% |
| All features | 62.31% | 61.39% |
| Without eye | **63.67%** | **63.00%** |

The ablation results show that the eye-only feature group is the strongest individual group among those tested. However, the highest accuracy in this particular ablation table was obtained by the **without-eye** configuration, not the all-feature configuration. Therefore, the results do not support claiming that every feature group improves performance when combined.

The proposal expected the full feature set to outperform every subset. Here, removing the eye group gave a slightly higher score than using all features (63.67% vs 62.31%, about 1.4 points). The difference is small, so we do not claim that the eye features hurt performance; we report it as an observed result that contradicts the original expectation. Possible explanations, none of which was tested directly, are redundancy between feature groups and calibration effects: under `subject_alert` calibration, head, mouth and eyebrow offsets may carry information that is not purely drowsiness-related.

The ablation table is a separate experiment output. Its accuracy should not be substituted for the unseen-subject ensemble accuracy in the main results table.

## Additional Sequence-Model Experiment

The archive also contains `seq_subject_alert_w30/seq_results.csv` (sequence experiment, notebook `05_sequence_lstm.ipynb`, `subject_alert` calibration), reporting a five-fold subject-grouped comparison of four approaches:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| A — Ensemble (handcrafted) | 59.19% | 58.16% |
| B — FrameNet (no time) | **60.09%** | **59.39%** |
| C — SeqNet (conv + LSTM) | 56.28% | 55.61% |
| D — SeqNet + handcrafted | 57.46% | 57.00% |

In this experiment, FrameNet (no time) recorded the highest accuracy. The paired comparisons report that SeqNet (conv + LSTM) performed significantly below FrameNet in the comparison. The differences between the other listed model pairs were not supported by those paired tests.

This five-fold subject-grouped experiment is reported separately from the `subject_alert_w30` main evaluation and should not be merged into its validation table.

## Outcome Versus the Proposal's Expectations

| Proposal expectation | Observed outcome (calibration `subject_alert`) |
|---|---|
| Ensemble outperforms single models | The ensemble had the highest accuracy (60.65%), but the difference from XGBoost (60.14%), MLP (59.23%) and Random Forest (58.95%) was not statistically supported |
| Ensemble outperforms a simple baseline | Supported: +13.3 points over PERCLOS-only logistic regression (95% interval +9.0 to +17.5) |
| Full feature set outperforms single-region feature sets | Supported: all features (62.31%) beat every single-group configuration (47.46% to 53.61%) |
| Full feature set is the best configuration | Not supported: the without-eye configuration scored higher (63.67%) |
| Seen-subject versus unseen-subject generalization | Seen-subject time-blocked accuracy (74.30%) is much higher than unseen-subject accuracy (60.65%) |

## Evaluation Metrics

The result files report:

- **Accuracy:** Fraction of correctly classified windows.
- **Macro precision:** Precision calculated per class and averaged equally across classes.
- **Macro recall:** Recall calculated per class and averaged equally across classes.
- **Macro F1-score:** F1-score calculated per class and averaged equally across classes.
- **Video-level accuracy:** Accuracy after aggregating window predictions at video level.
- **95% confidence intervals:** Reported for accuracy in the main and sequence result tables (paired subject-level bootstrap).

## Results Archive

The `report_results.zip` file contains these experiment folders:

| Folder | Contents |
|---|---|
| `subject_alert_w30` | Main subject-alert results, ablation results, paired comparisons, confusion matrices, per-subject plot |
| `none_w30` | Results from the no-calibration configuration |
| `per_video_w30` | Results from the per-video configuration |
| `subject_all_w30` | Results from the subject-all configuration |
| `seq_subject_alert_w30` | Additional sequence-model comparison |

The experiment folders include CSV result tables and PNG visualizations. The main subject-alert folder includes:

- `main_results.csv`
- `ablation.csv`
- `paired_comparisons.csv`
- `confusion_matrices.png`
- `per_subject.png`
- `ablation.png`

The other experiment folders contain corresponding result files where applicable. The sequence folder contains `seq_results.csv`, `seq_paired.csv`, and `seq_confusion.png`.

## Repository Files and Reproduction

### Repository structure

```
UTA-RLDD videos
  |  extract_features.py              MediaPipe FaceMesh -> per-frame signals
  v
features/*.npz                        t, X (T x 10), cols, fps
  |  01_extract_landmarks.ipynb       landmark extraction
  |  02_features_calibration.ipynb    windows, calibration, 28 window features
  v
data/windows.csv, feature_groups.json, subject_folds.csv, run_info.json
  |  03_models_ensemble.ipynb         tuning -> best_params.json
  |  04_evaluation_ablation.ipynb     unseen/seen-subject evaluation, bootstrap CIs, ablation, error analysis
  v
results/
     05_sequence_lstm.ipynb           additional sequence-model experiment
```

| File / folder | Purpose |
|---|---|
| `extract_features.py` | Stage 1, run once: videos to `features/<subject>_<label>[_p<part>].npz` (resumable). |
| `01_extract_landmarks.ipynb` to `04_evaluation_ablation.ipynb` | Pipeline stages as shown above. |
| `05_sequence_lstm.ipynb` | Additional experiment: frame and sequence neural models compared with the handcrafted ensemble. |
| `drowsy_utils.py` | Shared helpers: `make_models`, `cv_proba`, `summarize`, bootstrap confidence intervals. |
| `best_params.json` | Tuned hyperparameters for RF, XGBoost, and MLP. |
| `run_info.json` | Configuration of the reported run. |
| `check_features.py`, `peek_video.py`, `ear_check.png` | Development and sanity-check utilities. |
| `data/`, `data_subject_all/`, `features/`, `results/` | Generated outputs of the pipeline. |
| `report_results.zip`, `features.zip` | Archived result tables and plots, and cached features. |
| `requirements.txt` | Python dependencies. |

### Reproduction steps

1. **Environment** (Python 3.10–3.12): `pip install -r requirements.txt`. The extractor requires `mediapipe==0.10.14`; other dependencies are `opencv-python`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `matplotlib` (plus `torch` or `tensorflow` for notebook `05` if needed).
2. **Dataset:** Obtain UTA-RLDD from its authorized source. The extractor expects `<videos>/**/<subject_id>/{0,5,10}.mov|mp4` (0 alert, 5 low vigilant, 10 drowsy) and treats split videos such as `10_1.mp4` as separate parts.
3. **Extract per-frame signals (once):**
   ```bash
   python extract_features.py --videos <path-to-RLDD> --out features --stride 2 --workers 3
   ```
   Add `--limit 2` first to benchmark on two videos.
4. **Run the notebooks in order:** `01_extract_landmarks.ipynb`, `02_features_calibration.ipynb`, `03_models_ensemble.ipynb`, `04_evaluation_ablation.ipynb`.
   - In `02`, set `CALIBRATION` (and `CAL_S` for the `*_head` modes). The reported results use `subject_alert`.
   - In `04`, set `QUICK=False`, `LOSO=True`, `RUN_SEEN=True`, `RUN_ABLATION=True` for full runs.
5. Compare the regenerated metrics and plots with the CSV and PNG files in `report_results.zip`.

Exact reproduction also depends on the dataset, the split definitions in `subject_folds.csv`, and random seeds.

## Limitations and Future Work

- **Calibration is the main caveat.** The headline results rely on the held-out subject's alert recording (oracle setting). The label-free alternatives (`per_video_head`, `subject_alert_head`) are implemented but not evaluated in the reported results, so label-free performance is unknown and is expected to be lower. Future work: evaluate them and re-tune hyperparameters under the chosen mode.
- **Session confound.** In UTA-RLDD each state is a separate recording, so calibration offsets can reflect camera or posture differences between recordings rather than drowsiness. The without-eye ablation result is consistent with this, but does not prove it.
- Unseen-subject accuracy is lower than seen-subject time-blocked accuracy, indicating that generalization to new drivers remains challenging.
- The reported unseen-subject ensemble accuracy is 60.65%; this is a research result, not evidence of production-ready safety performance.
- The ensemble is not shown to be significantly better than its individual models.
- Low Vigilant is the hardest class to recognise (recall about 0.41 for the ensemble).
- The ablation results show that removing the eye feature group improved the reported score in that particular experiment. Feature interactions and robustness should be investigated further. The ablation uses GroupKFold(5), not the leave-one-subject-out protocol of the main table.
- The sequence-model comparison did not outperform the FrameNet (no time) result in the five-fold subject-grouped experiment.
- Hyperparameters were tuned on folds 0–3; a separate check on the untouched fold 4 is not reported. No comparison with HM-LSTM on its official folds is reported, so no like-for-like claim against that baseline is made.
- Head pose uses a generic 3D face model and an approximate camera matrix.
- Further work can investigate feature robustness across subjects, lighting conditions, camera positions, occlusions, and real driving environments.
- Real-time performance, alert timing, false-alarm rates, and deployment reliability require dedicated evaluation before any safety-critical use.

## References

1. Xu, C., Huang, W., Liu, J., & Li, L. (2025). *Detecting Driver Drowsiness Using Hybrid Facial Features and Ensemble Learning*. Information, 16, 294. https://doi.org/10.3390/info16040294
2. Choi, Y., Yang, S., Park, Y., Choi, C., & Lee, E. C. (2025). *Feasibility Study on Contactless Feature Analysis for Early Drowsiness Detection in Driving Scenarios*. Electronics, 14, 662. https://doi.org/10.3390/electronics14040662
3. Ghoddoosian, R., Galib, M., & Athitsos, V. (2019). *A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection*. arXiv:1904.07312. https://arxiv.org/abs/1904.07312

## Team Members

| Name | Enrollment Number | Contribution in project proposal |
|---|---|---|
| Geethika | IIB2024024 | Dataset collection and preprocessing: dataset organisation, segmentation into fixed-length clips, frame-rate standardisation, landmark extraction |
| Rithwika | IIB2024021 | Feature engineering and subject calibration: EAR, MAR, blink, PERCLOS, head pose and eyebrow features, per-subject calibration, final feature vectors |
| Archee | IIB2024022 | Model development and training: Random Forest, XGBoost and MLP, soft-voting ensemble, hyperparameter tuning |
| Sumayya | IIB2024032 | Experimentation and evaluation: seen-subject and unseen-subject validation, metrics, feature-group ablation, results and report |
