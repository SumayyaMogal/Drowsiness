# Multi-Level Driver Drowsiness Detection Using Hybrid Facial Features and Machine Learning

A computer vision and machine learning project that classifies driver drowsiness into three levels — **Alert**, **Low Vigilant**, and **Drowsy** — using facial-behaviour features and machine learning models.

## Project Overview

Driver fatigue can reduce attention and reaction time, increasing the risk of road accidents. This project investigates a camera-based approach to detecting drowsiness from facial cues without requiring wearable sensors.

The proposed approach extracts facial-behaviour features, applies subject-specific calibration, and evaluates classical machine learning models and an ensemble. Experiments also investigate how performance changes across different validation protocols and feature groups.

> **Research note:** This repository's supplied result files report offline model evaluations. They do not, by themselves, demonstrate a deployed, real-time in-vehicle warning system.

## Objectives

- Detect three levels of driver vigilance: Alert, Low Vigilant, and Drowsy.
- Represent driver behaviour using multiple facial regions rather than relying on a single cue.
- Evaluate Random Forest, XGBoost, and Multilayer Perceptron (MLP) classifiers.
- Test a soft-voting ensemble of the classifiers.
- Compare performance on seen-subject and unseen-subject evaluation protocols.
- Measure the contribution of individual facial feature groups through ablation experiments.

## Dataset

The project proposal identifies the **UTA-RLDD (UTA Real-Life Drowsiness Dataset)** as the evaluation dataset. It contains labelled recordings across alert, low-vigilant, and drowsy states and includes 60 subjects.

The dataset is not included in the supplied results archive. Obtain it from its official distribution source and follow its access and usage conditions before running experiments.

## Methodology

The proposed processing pipeline is:

1. **Video preparation:** Organize recordings and prepare fixed-length video segments.
2. **Facial landmark extraction:** Detect facial landmarks around the eyes, mouth, eyebrows, and head pose using the landmark-extraction approach described in the project proposal (MediaPipe/dlib-based).
3. **Feature engineering:** Derive numerical features from facial landmarks and temporal behaviour.
4. **Subject-specific calibration:** Normalize features using an alert-state reference for the subject.
5. **Model training and evaluation:** Train individual classifiers and evaluate the ensemble.
6. **Prediction:** Assign each evaluated segment to Alert, Low Vigilant, or Drowsy.

### Facial feature groups

The proposal and experiment outputs refer to the following feature groups:

| Feature group | Examples / role |
|---|---|
| Eye | Eye Aspect Ratio (EAR), blink behaviour, eye-closure measures and PERCLOS |
| Mouth | Mouth Aspect Ratio (MAR) and mouth movement |
| Head | Head-pose-related measurements |
| Eyebrow | Eyebrow movement measurements |

The supplied ablation output records 28 features for the full feature set. The exact feature-column definitions should be taken from the feature extraction code or dataset files.

### Windowing and calibration configuration

The supplied `run_info.json` records the configuration used for the reported subject-alert run:

| Setting | Value |
|---|---:|
| Calibration | `subject_alert` |
| Window length | 30 seconds |
| Window step | 15 seconds |
| Blink threshold | 0.8 |
| Closed-eye thresholds | 0.5, 0.7, 0.8, 0.9 |

The windowing configuration uses overlapping 30-second windows with a 15-second step.

## Machine Learning Models

The supplied tuned model configuration covers:

- **Random Forest (RF)**
- **XGBoost (XGB)**
- **Multilayer Perceptron (MLP)**
- **Soft-voting ensemble** combining the classical classifiers

### Best recorded hyperparameters

These values are taken from `best_params.json`.

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

The following results are transcribed from `subject_alert_w30/main_results.csv`. Values are percentages. The primary cross-subject result is the **unseen-subject** evaluation.

### Unseen-subject evaluation

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Majority-class baseline | 33.75% | 11.25% | 33.33% | 16.82% |
| PERCLOS-only Logistic Regression | 47.36% | 49.19% | 47.50% | 45.60% |
| Random Forest | 58.95% | 57.87% | 59.06% | 58.13% |
| XGBoost | 60.14% | 58.95% | 60.26% | 59.32% |
| MLP | 59.23% | 58.03% | 59.34% | 58.42% |
| **Soft-voting ensemble** | **60.65%** | **59.31%** | **60.76%** | **59.72%** |

The ensemble's reported 95% accuracy confidence interval is **56.7%–64.7%**.

### Other validation protocols

| Validation protocol | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Seen-subject (time-blocked) | Soft-voting ensemble | 74.30% | 73.98% |
| Seen-subject (video held out) | Soft-voting ensemble | 54.94% | 53.91% |
| Unseen-subject | Soft-voting ensemble | 60.65% | 59.72% |

These protocols measure different generalization settings and should not be treated as interchangeable. In particular, the higher time-blocked seen-subject score does not mean the model achieves the same performance on previously unseen drivers.

### Video-level accuracy

For the unseen-subject ensemble, the supplied results report:

- Majority-vote video accuracy: **64.44%**
- Mean-probability video accuracy: **63.89%**

These are separate video-level aggregation metrics; they are not the same as segment-level classification accuracy.

### Paired comparisons

The supplied paired-comparison results report that, in the unseen-subject evaluation:

- The ensemble exceeded the PERCLOS-only Logistic Regression baseline by **13.3 percentage points** (95% interval for the difference: +9.0 to +17.5 points; supported in the supplied comparison).
- The ensemble's differences versus XGBoost, Random Forest, and MLP were not supported by the reported paired comparisons.

This means the ensemble performed better than the simple PERCLOS-only baseline in this evaluation, but the supplied statistical comparisons do not establish that it is definitively better than each individual machine learning model.

## Ablation Study

The subject-alert ablation results are from `subject_alert_w30/ablation.csv`.

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

The ablation table is a separate experiment output. Its accuracy should not be substituted for the unseen-subject ensemble accuracy in the main results table.

## Additional Sequence-Model Experiment

The archive also contains `seq_subject_alert_w30/seq_results.csv`, reporting a five-fold subject-grouped comparison of four approaches:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| A — Ensemble (handcrafted) | 59.19% | 58.16% |
| B — FrameNet (no time) | **60.09%** | **59.39%** |
| C — SeqNet (conv + LSTM) | 56.28% | 55.61% |
| D — SeqNet + handcrafted | 57.46% | 57.00% |

In this experiment, FrameNet (no time) recorded the highest accuracy. The paired comparisons report that SeqNet (conv + LSTM) performed significantly below FrameNet in the supplied comparison. The differences between the other listed model pairs were not supported by those paired tests.

This five-fold subject-grouped experiment is reported separately from the `subject_alert_w30` main evaluation and should not be merged into its validation table.

## Evaluation Metrics

The result files report:

- **Accuracy:** Fraction of correctly classified segments.
- **Macro precision:** Precision calculated per class and averaged equally across classes.
- **Macro recall:** Recall calculated per class and averaged equally across classes.
- **Macro F1-score:** F1-score calculated per class and averaged equally across classes.
- **Video-level accuracy:** Accuracy after aggregating segment predictions at video level.
- **95% confidence intervals:** Reported for accuracy in the main and sequence result tables.

## Results Archive

The supplied `report_results.zip` contains these experiment folders:

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

The supplied project materials for this README include:

- `IML_FINAL_projectproposal.pdf` — project proposal and team contribution plan.
- `best_params.json` — tuned hyperparameters for RF, XGBoost, and MLP.
- `run_info.json` — subject-alert run configuration.
- `report_results.zip` — experiment result tables and visualizations.

**Important:** The supplied materials do not include the implementation source code, a dependency lock file, or a confirmed training/inference entry-point. Consequently, exact installation commands and execution commands cannot be specified reliably from these files alone.

To reproduce the experiments:

1. Obtain the UTA-RLDD dataset from its authorized source.
2. Place the dataset in the location expected by the project implementation.
3. Set up the Python environment using the dependencies specified by the actual source code.
4. Run the project's preprocessing, landmark extraction, feature engineering, calibration, training, and evaluation scripts.
5. Use the recorded hyperparameters and run configuration where the corresponding experiment is being reproduced.
6. Compare newly generated metrics and plots with the CSV and PNG files in `report_results.zip`.

Do not assume that reproducing a result is possible from the CSV files alone; the dataset, preprocessing implementation, split definitions, random seeds, and model code are also required.

## Limitations and Future Work

- Unseen-subject accuracy is lower than seen-subject time-blocked accuracy, indicating that generalization to new drivers remains challenging.
- The reported unseen-subject ensemble accuracy is 60.65%; this is a research result, not evidence of production-ready safety performance.
- The ablation results show that removing the eye feature group improved the reported score in that particular experiment. Feature interactions and robustness should be investigated further.
- The sequence-model comparison did not outperform the FrameNet (no time) result in the supplied five-fold subject-grouped experiment.
- Further work can investigate feature robustness across subjects, lighting conditions, camera positions, occlusions, and real driving environments.
- Real-time performance, alert timing, false-alarm rates, and deployment reliability require dedicated evaluation before any safety-critical use.

## Team Members

| Name | Enrollment Number | Contribution in project proposal |
|---|---|---|
| Geethika | IIB2024024 | Dataset collection and preprocessing |
| Rithwika | IIB2024021 | Feature engineering and subject calibration |
| Archee | IIB2024022 | Model development and training |
| Summaya | IIB2024032 | Experimentation and evaluation |

## References

1. Xu, C., Huang, W., Liu, J., & Li, L. (2025). *Detecting Driver Drowsiness Using Hybrid Facial Features and Ensemble Learning*. Information, 16, 294. https://doi.org/10.3390/info16040294
2. Choi, Y., Yang, S., Park, Y., Choi, C., & Lee, E. C. (2025). *Feasibility Study on Contactless Feature Analysis for Early Drowsiness Detection in Driving Scenarios*. Electronics, 14, 662. https://doi.org/10.3390/electronics14040662
3. Ghoddoosian, R., Galib, M., & Athitsos, V. (2019). *A Realistic Dataset and Baseline Temporal Model for Early Drowsiness Detection*. arXiv:1904.07312. https://arxiv.org/abs/1904.07312

---

**Disclaimer:** This project is an academic research prototype. It must not be used as the sole basis for deciding whether a driver is safe to drive.
