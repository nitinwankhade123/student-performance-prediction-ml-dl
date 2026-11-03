# Methodology

## Pipeline overview

The project uses a fixed synthetic tabular dataset to demonstrate binary
classification. The implemented workflow is:

1. Generate and inspect `datasets/student_data.csv`.
2. Encode the result label and create stratified train, validation, and test
   partitions.
3. Fit and save a `StandardScaler` on training features only; transform all
   three partitions with that same fitted object.
4. Train the ANN and fit the classical models on the training partition.
5. Compare validation probabilities and select a threshold per experiment using
   validation balanced accuracy.
6. Evaluate the frozen Logistic Regression candidate once on the held-out test
   partition.
7. Serve the saved Logistic Regression estimator and scaler in Streamlit.

The testing split is not loaded by the ANN training code or by the controlled
model-comparison script. The independent test scripts use it only for inference
and metric calculation.

## Synthetic data generation

`datasets/generate_synthetic_data.py` uses NumPy's `default_rng` with seed 42 to
generate 1,000 records. Feature values are drawn uniformly and rounded to one
decimal place:

- `study_hours`: 0.0–10.0 hours per day
- `attendance`: 40.0–100.0 percent
- `previous_marks`: 20.0–100.0

For each row, the generator calculates:

```text
score = 0.40 * previous_marks
      + 0.30 * attendance
      + 0.30 * min(study_hours * 10, 100)
      + Normal(0, 8)
```

`result` is `pass` when `score >= 50`; otherwise it is `fail`. The normal noise
has mean 0 and standard deviation 8. This rule is an educational simulation,
not a grading policy or a model of observed student outcomes. The generated
CSV contains 733 pass and 267 fail records.

`datasets/load_dataset.py` reads the CSV with Pandas, checks required columns,
and reports shape, missing values, duplicates, and a small preview.

## Preprocessing

`preprocessing/preprocess.py` selects `study_hours`, `attendance`, and
`previous_marks` as `X`, and `result` as `y`. It normalizes label strings and
encodes `fail = 0`, `pass = 1`.

The split is stratified in two steps using random seed 42:

| Partition | Records | Fail (0) | Pass (1) |
|---|---:|---:|---:|
| Training | 700 | 187 | 513 |
| Validation | 150 | 40 | 110 |
| Testing | 150 | 40 | 110 |

`StandardScaler` is fitted on the 700 training feature rows only. That scaler
transforms training, validation, and test features. It is persisted as
`artifacts/standard_scaler.joblib`; scaled partitions are saved under
`artifacts/processed/`. Both the ANN and the scikit-learn experiment models use
these same processed feature columns in the same order.

## ANN architecture and training

`models/ann_model.py` defines a feed-forward PyTorch network:

```text
3 standardized input features
        ↓
Linear(3, 8)
        ↓
ReLU
        ↓
Linear(8, 1 raw logit)
```

There is no sigmoid in the model's forward method. The model contains 41
trainable parameters. `training/train.py` uses `BCEWithLogitsLoss`, Adam with
learning rate 0.001, full-batch updates, a maximum of 100 epochs, and a
validation-loss early-stopping patience of 10. Random seeds are set to 42 and
deterministic PyTorch algorithms are requested. The saved run completed 100
epochs without triggering early stopping; minimum validation loss was
0.570101 at epoch 100. The best state dictionary is
`artifacts/best_model.pth`.

`validation/validate.py` loads the saved ANN and the validation split without
training. It calculates validation probabilities, ROC-AUC, and threshold
diagnostics. The saved validation ROC-AUC was 0.699545. At threshold 0.5, every
validation probability was above the threshold and all records were classified
as pass. The threshold that maximized validation balanced accuracy was
0.6012358665466309, with balanced accuracy 0.678409 and fail recall 0.675.

## Classical models

The controlled comparison in `experiments/compare_models.py` uses the same
processed training and validation CSVs for all candidates.

### Logistic Regression

Scikit-learn `LogisticRegression` uses `C=1.0`, the `lbfgs` solver, maximum
2,000 iterations, no class weighting, and random state 42. It is fitted on the
training partition and saved to
`experiments/logistic_regression/model.joblib`.

### Random Forest

Scikit-learn `RandomForestClassifier` uses 300 trees, `max_depth=None`,
`min_samples_split=2`, `min_samples_leaf=1`, no class weighting, random state
42, and `n_jobs=-1`.

### Class-weighted ANN

The class-weighted candidate uses the same 3–8–1 architecture and trains for
100 full-batch Adam epochs at learning rate 0.001. It applies sample-weighted
`BCEWithLogitsLoss`: fail examples (label 0) receive weight
`513 / 187 = 2.743315508...`; pass examples (label 1) receive weight 1.0. The
weights and training history are saved separately under
`experiments/weighted_ann/`. This experiment did not change the original ANN
checkpoint.

## Model comparison and threshold selection

Each model produces validation probabilities for class 1 (pass). Candidate
thresholds include 0, 1, 0.5, and the midpoints between adjacent distinct
validation probabilities. A probability at or above a candidate threshold is
classified as pass; a lower probability is classified as fail.

The selected threshold maximizes validation balanced accuracy. Ties are
resolved by higher fail precision, then by closeness to 0.5. This process is
run independently for each model using validation predictions only. The
candidate-level results are in
`experiments/results/threshold_comparison.csv` (549 rows total); summary
metrics are in `experiments/results/model_comparison.csv`.

The comparison favors Logistic Regression on validation ROC-AUC (0.914545),
balanced accuracy (0.842045), and accuracy (0.873333). Its selected validation
threshold is 0.45458775758743286. The result is plausible for this dataset
because its synthetic labels arise from an additive score of the same three
features. It does not prove that Logistic Regression will generalize to real
student outcomes.

## Independent testing

`experiments/final_test_logistic_regression.py` verifies the saved fit-split
and threshold provenance, loads the fitted Logistic Regression estimator and
validation-selected threshold, and only then reads the test partition. It
does not call `fit()` or tune a threshold. The frozen threshold is
0.45458775758743286.

On the 150-record synthetic test split (40 fail, 110 pass), Logistic Regression
achieved accuracy 0.866667, fail recall 0.700000, pass recall 0.927273, balanced
accuracy 0.813636, and ROC-AUC 0.904545. Its confusion matrix, with actual rows
and predicted columns ordered `[fail, pass]`, is `[[28, 12], [8, 102]]`.

The original ANN test script used the fixed threshold 0.5 and predicted every
test row as pass. Its accuracy was 0.733333 and ROC-AUC was 0.699773; its
confusion matrix was `[[0, 40], [0, 110]]`. These are results on synthetic
records, not real-world evidence.

## Streamlit inference

`deployment/app.py` loads the saved Logistic Regression estimator and the
training-fitted `StandardScaler`. It accepts inputs in the generated ranges,
preserves feature order, transforms the input with the saved scaler, calls
`predict_proba`, and applies the frozen threshold to the pass probability. It
does not train or change the model. It displays both class probabilities and a
synthetic-data disclaimer. See the root README for local run commands.
