# Model Comparison and Independent Test Results

All validation figures below are copied from
`experiments/results/model_comparison.csv`. All candidates use the same 700-row
training and 150-row validation partitions. Thresholds shown were selected from
validation probabilities by maximizing balanced accuracy; no test labels or
probabilities were used to select them.

## Validation results

Precision, recall, and F1 are separated by class because the target uses
`fail = 0` and `pass = 1`. Confusion matrices have rows = actual `[fail, pass]`
and columns = predicted `[fail, pass]`.

| Model | Selected threshold | Accuracy | Balanced accuracy | ROC-AUC | Fail precision | Fail recall | Fail F1 | Pass precision | Pass recall | Pass F1 | Confusion matrix |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Existing ANN | 0.6012358665466309 | 0.680000 | 0.678409 | 0.699545 | 0.435484 | 0.675000 | 0.529412 | 0.852273 | 0.681818 | 0.757576 | `[[27, 13], [35, 75]]` |
| Logistic Regression | 0.45458775758743286 | 0.873333 | 0.842045 | 0.914545 | 0.756098 | 0.775000 | 0.765432 | 0.917431 | 0.909091 | 0.913242 | `[[31, 9], [10, 100]]` |
| Random Forest | 0.8266666666666667 | 0.740000 | 0.806818 | 0.872386 | 0.506667 | 0.950000 | 0.660870 | 0.973333 | 0.663636 | 0.789189 | `[[38, 2], [37, 73]]` |
| Class-weighted ANN | 0.5576108694076538 | 0.626667 | 0.729545 | 0.810227 | 0.413043 | 0.950000 | 0.575758 | 0.965517 | 0.509091 | 0.666667 | `[[38, 2], [54, 56]]` |

Logistic Regression has the strongest validation ROC-AUC, balanced accuracy,
and overall accuracy. Its validation threshold is exactly
**0.45458775758743286** in the threshold candidate CSV. The summary CSV stores
the threshold rounded to six decimal places; the candidate-level CSV retains
the full stored value.

## Threshold selection

For each model, candidate thresholds were derived from the validation
probabilities: 0, 1, 0.5, and midpoints between adjacent distinct probabilities.
The rule is `predict pass if P(pass) >= threshold`, otherwise predict fail.
The selected threshold maximizes validation balanced accuracy. Ties use higher
fail precision, then closeness to 0.5. The exhaustive candidate results are in
[`experiments/results/threshold_comparison.csv`](../experiments/results/threshold_comparison.csv).

| Model | Threshold at 0.5: fail recall | Threshold at 0.5: balanced accuracy | Selected threshold: fail recall | Selected threshold: balanced accuracy |
|---|---:|---:|---:|---:|
| Existing ANN | 0.000 | 0.500000 | 0.675 | 0.678409 |
| Logistic Regression | 0.775 | 0.828409 | 0.775 | 0.842045 |
| Random Forest | 0.675 | 0.778409 | 0.950 | 0.806818 |
| Class-weighted ANN | 0.600 | 0.709091 | 0.950 | 0.729545 |

The weighted ANN and Random Forest reach higher fail recall at their selected
thresholds, but they also make substantially more pass-to-fail errors. For
example, the weighted ANN correctly detects 38 of 40 validation fails, while
also classifying 54 of 110 validation passes as fail.

## Why Logistic Regression was selected as the demonstration model

The synthetic label is generated from a weighted additive score over the same
three inputs, plus Gaussian noise, with a fixed cutoff. In the generated
study-hours range of 0–10, the capped study-hours term is effectively linear.
Logistic Regression is therefore a suitable low-complexity fit for this
synthetic boundary. On validation it also led the four experiments in ROC-AUC,
balanced accuracy, and accuracy, while providing a more balanced error
profile than the baseline ANN.

The class-weighted ANN increased fail recall from 0.675 to 0.950 compared with
the ANN baseline at their respective validation-selected thresholds, but fail
precision decreased from 0.435484 to 0.413043 and accuracy decreased from
0.680000 to 0.626667. Random Forest also achieved 0.950 fail recall, with 37
actual pass rows predicted fail. These trade-offs are why balanced accuracy
and class-specific metrics are reported alongside accuracy.

This is a choice of the current **demonstration inference model** based on
validation results; it is not evidence of a real-world academic assessment
system.

## Independent test comparison

The final Logistic Regression test used the already selected validation
threshold **0.45458775758743286**. The original ANN test run used threshold
**0.5**. Both were evaluated on the same 150-row held-out test partition (40
fail, 110 pass). No threshold was tuned using test results.

| Test metric | Original ANN (threshold 0.5) | Logistic Regression (validation threshold) |
|---|---:|---:|
| Accuracy | 0.733333 | 0.866667 |
| Fail precision | 0.000000 | 0.777778 |
| Fail recall | 0.000000 | 0.700000 |
| Fail F1 | 0.000000 | 0.736842 |
| Pass precision | 0.733333 | 0.894737 |
| Pass recall | 1.000000 | 0.927273 |
| Pass F1 | 0.846154 | 0.910714 |
| Balanced accuracy | 0.500000* | 0.813636 |
| ROC-AUC | 0.699773 | 0.904545 |
| Confusion matrix | `[[0, 40], [0, 110]]` | `[[28, 12], [8, 102]]` |

`*` The original ANN metrics CSV does not contain balanced accuracy. The value
0.500000 is calculated from the saved confusion matrix: average of fail recall
0.0 and pass recall 1.0.

For Logistic Regression, the matrix is given with pass as the conventional
positive class (`pass = 1`): TP=102, TN=28, FP=12, FN=8. Its fail recall is
28/40 = 0.700000. The original ANN predicted every row as pass, matching the
test set's 73.33% pass share in accuracy while missing all 40 fail cases.

The test result is an independent evaluation of one synthetic split. It does
not establish performance on real students or other institutions.

## Saved comparison artifacts

- Validation summary: [`experiments/results/model_comparison.csv`](../experiments/results/model_comparison.csv)
- All validation thresholds: [`experiments/results/threshold_comparison.csv`](../experiments/results/threshold_comparison.csv)
- Validation ROC comparison: [`experiments/results/validation_roc_comparison.png`](../experiments/results/validation_roc_comparison.png)
- Final Logistic Regression test metrics: [`results/final_test_metrics.csv`](../results/final_test_metrics.csv)
- Final Logistic Regression predictions: [`results/final_test_predictions.csv`](../results/final_test_predictions.csv)
- Original ANN test metrics: [`results/test_metrics.csv`](../results/test_metrics.csv)
