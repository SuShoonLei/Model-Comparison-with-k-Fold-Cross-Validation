# Homework Assignment 3: Model Comparison with k-Fold Cross-Validation

CSC 461: Machine Learning

This project compares two regression models on the scikit-learn diabetes dataset using k-fold cross-validation written by hand (no `KFold`, `cross_val_score`, or `cross_validate`).

- **Model A:** linear regression, `target ~ bmi + sex_dummy`
- **Model B:** an ensemble of two decision trees (one on `bmi`, one on `sex_dummy`), combined with inverse-MSE weights

## Files

| File | Description |
|---|---|
| `cross_validation_by_hand.py` | Full implementation of Parts 1 and 2 |
| `README.md` | This report (instructions, implementation, results, Part 3 essay) |

## How to Run

Requires Python 3. From the project folder:

```bash
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn pandas numpy
python cross_validation_by_hand.py
```

On Windows, activate the environment with `venv\Scripts\activate` instead of the `source` line.

The script prints the Part 1 results (fit on the full dataset) followed by the Part 2 cross-validation table.

## Implementation Description

The code is organized into sections so the models, the fold-splitting procedure, and the cross-validation loop are easy to tell apart.

**Data loading (`load_data`).** Loads the diabetes dataset and builds `sex_category` and `sex_dummy` exactly as specified (`Moonbeam` = 0, `Thunderpaws` = 1, based on whether the standardized `sex` value is below zero).

**Model functions.**
- `fit_linear(train_df)` fits `LinearRegression` on `bmi` and `sex_dummy`.
- `fit_ensemble(train_df)` fits `tree_bmi` (`max_depth=2`) on `bmi` only and `tree_sex` (`max_depth=1`) on `sex_dummy` only. It computes each tree's MSE on the rows it was given, then the weights `w_i = (1/MSE_i) / (1/MSE_bmi + 1/MSE_sex)`. All split points are learned by the trees; none are hard-coded.
- `predict_ensemble(model_tuple, test_df)` returns `w_bmi * pred_bmi + w_sex * pred_sex`.
- `mse(y_true, y_pred)` computes `((y_true - y_pred) ** 2).mean()`.

Part 1 and the CV loop both call the same `fit_ensemble`, so there is one code path for fitting the ensemble.

**Fold construction (`make_folds`).** Shuffles the row indices with `np.random.default_rng(42).permutation(n)` and cuts them into `k` consecutive blocks. Each block has `n // k` rows, and the first `n % k` blocks get one extra row. For n = 442 and k = 5 the fold sizes are 89, 89, 88, 88, 88. An `assert` checks that the folds are disjoint and cover every row.

**Cross-validation loop (`k_fold_cv`).** For each fold, the test set is that block and the training set is every other row. Everything is refit from scratch in every fold: the linear model, both trees, and both ensemble weights. The weights are computed from the current fold's training rows only, so the test rows never influence the coefficients, the split points, or the weights. The loop records the test MSE for the linear model, `tree_bmi`, `tree_sex`, and the ensemble. The summary uses the mean and the sample standard deviation (`np.std(values, ddof=1)`) across the 5 folds.

## Part 1 Results (fit on the full dataset, n = 442)

### Model A: Linear regression

| Quantity | Value |
|---|---|
| Intercept | 152.7628 |
| BMI coefficient | 950.6781 |
| `sex_dummy` coefficient | -1.3438 |
| SSE | 1719384.6087 |
| MSE (SSE / n) | 3890.0104 |

### Model B: Ensemble of two trees

**`tree_bmi` (`max_depth=2`)**

- Learned thresholds: 0.009422 (root), -0.021834 (left child), 0.073013 (right child)
- Leaf values, left to right: 105.0424, 143.9554, 191.5630, 264.2333
- Training MSE: 3757.3440

```
|--- bmi <= 0.009422
|   |--- bmi <= -0.021834
|   |   |--- value: [105.042424]
|   |--- bmi >  -0.021834
|   |   |--- value: [143.955357]
|--- bmi >  0.009422
|   |--- bmi <= 0.073013
|   |   |--- value: [191.562963]
|   |--- bmi >  0.073013
|   |   |--- value: [264.233333]
```

**`tree_sex` (`max_depth=1`)**

- Learned threshold: 0.500000
- Leaf values, left to right: 149.0213 (Moonbeam), 155.6667 (Thunderpaws)
- Training MSE: 5918.8889

```
|--- sex_dummy <= 0.500000
|   |--- value: [149.021277]
|--- sex_dummy >  0.500000
|   |--- value: [155.666667]
```

**Ensemble weights and training MSE**

| Quantity | Value |
|---|---|
| `w_bmi` | 0.612 |
| `w_sex` | 0.388 |
| Sum of weights | 1.000 |
| Ensemble MSE (training data) | 4081.8168 |
| Baseline MSE (always predict mean of y) | 5929.8849 |

## Part 2 Results: 5-Fold Cross-Validation

Fold sizes: 89, 89, 88, 88, 88 (seed = 42). All values are test MSE.

### Per-fold table

| Fold | Linear | tree_bmi | tree_sex | Ensemble |
|---:|---:|---:|---:|---:|
| 1 | 4270.71 | 4418.20 | 5933.92 | 4457.26 |
| 2 | 4167.62 | 3987.05 | 5342.80 | 4113.97 |
| 3 | 4278.40 | 4360.14 | 6011.84 | 4459.22 |
| 4 | 3590.69 | 3470.65 | 6196.12 | 3988.56 |
| 5 | 3590.74 | 4363.07 | 6427.87 | 4778.81 |

### Summary table

| | Linear | tree_bmi | tree_sex | Ensemble |
|---|---:|---:|---:|---:|
| Mean test MSE | 3979.63 | 4119.82 | 5982.51 | 4359.56 |
| SD (sample, ddof=1) | 357.72 | 401.56 | 405.10 | 313.53 |

Baseline MSE (always predict mean of y): 5929.88

**Sanity check:** the linear model and the ensemble both average below the baseline, and the sex tree's mean (5982.51) is slightly above it, as expected.

### In-sample vs. test MSE

| Model | Part 1 (in-sample) | Part 2 (CV mean) | Gap |
|---|---:|---:|---:|
| Linear | 3890.01 | 3979.63 | +89.6 |
| tree_bmi | 3757.34 | 4119.82 | +362.5 |
| tree_sex | 5918.89 | 5982.51 | +63.6 |
| Ensemble | 4081.82 | 4359.56 | +277.7 |

## Part 3: Comparing the Two Models

Based on my results, I would choose the linear model for predicting a new patient's target. Its mean test MSE across the five folds was 3979.63, compared with 4119.82 for the BMI tree and 4359.56 for the ensemble. All three beat the mean-prediction baseline of 5929.88, so BMI clearly carries real signal. The sex tree did not: its mean test MSE of 5982.51 was slightly above the baseline, which makes sense because `sex_dummy` has almost no relationship with target. The linear model's `sex_dummy` coefficient is only -1.34, so the linear model is effectively a line through BMI with a slope of about 951.

The ensemble did worse than the BMI tree it contains. It lost to `tree_bmi` in all five folds, by as little as about 39 in fold 1 and as much as about 518 in fold 4, with an average gap of roughly 240. The reason is the weighting. The weights come from training MSE (3757 for the BMI tree versus 5919 for the sex tree), and those two numbers are not far enough apart to push the sex tree's weight toward zero. It ended up with `w_sex` = 0.388, so almost 40% of every ensemble prediction came from a model that is barely better than guessing the mean, which dragged the good BMI predictions toward the center. The ensemble did have a lower standard deviation (313.53 versus 401.56 for `tree_bmi`), but a slightly steadier error that is also consistently higher is not much of a win.

Comparing the in-sample results from Part 1 with the test results from Part 2 shows how well each model generalizes. In Part 1, `tree_bmi` looked like the best model, with a training MSE of 3757.34 against 3890.01 for the linear model. On unseen data the order flipped. The linear model's MSE rose by only about 90 (to 3979.63), while `tree_bmi`'s rose by about 362 (to 4119.82) and the ensemble's rose by about 278 (to 4359.56). This is the problem from Assignment 2: a model always looks best on data it has already seen, and the more flexible model gets the bigger flattering bonus. The tree picks its own cutoffs from the training data, so it can fit quirks of those particular patients, while the line has only three parameters and much less room to chase noise. The sex tree's gap was small (about 64), but only because it never learned much to begin with.

The fold-by-fold results back up this conclusion, but not as strongly as the means suggest. Fold 5 supports the linear model most clearly: it scored 3590.74 there, while `tree_bmi` scored 4363.07, a gap of about 772 that is far bigger than in any other fold. Folds 2 and 4 complicate the story, because `tree_bmi` beat linear in both (3987.05 versus 4167.62 in fold 2, and 3470.65 versus 3590.69 in fold 4). Fold 2 also complicates the comparison with the ensemble, since it is the one fold where the ensemble (4113.97) edged out linear (4167.62). Overall, linear beat `tree_bmi` in only three of five folds, and its average advantage of about 140 is small next to fold-to-fold standard deviations of about 358 to 402. If I drop fold 5, the tree is actually ahead by about 18 on average across the other four folds. So a single fold is doing most of the work in separating linear from `tree_bmi`.

Several things could explain this variation. Each test fold has only 88 or 89 patients, so a few extreme target values or unusual BMI values landing in one fold can move that fold's MSE a lot. Each fold also trains on a different set of rows, so the tree's learned thresholds move around, and a threshold in the wrong place hurts every patient near it. That may be what happened to `tree_bmi` in fold 5, though I did not print the per-fold thresholds to confirm it. Because of this, I am fairly confident that the linear model is better than the ensemble, since it won four of five folds, and that the ensemble is worse than its own BMI tree. I am not confident that five folds settle linear versus `tree_bmi`. I would want repeated CV with different seeds, or more folds, before calling that one.

If I had used the test outcomes to choose the ensemble weights, the comparison would be unfair. The ensemble could shift weight toward the BMI tree (or set `w_sex` close to zero) after seeing the answers, and its test MSE would drop for a reason that has nothing to do with real predictive ability. That is data leakage: the test fold would no longer be unseen data, and the reported error would be too optimistic. A new patient doesn't come with an answer key, so the ensemble would do worse in practice than its tuned test score suggested.

Finally, the results connect to how each model represents the relationship between BMI and target. The line assumes a smooth, constant increase in target with BMI. The tree approximates the relationship with a four-step staircase, and its leaf values (105, 144, 192, 264) rise fairly steadily, which is close to a straight line anyway. When the true relationship is roughly linear, a line captures it with fewer parameters and less chance of overfitting, while the tree's flexibility mostly adds variance, because the choice of cutoffs depends on which rows it saw. This is likely why the line generalized a bit better here. If the true relationship had sharp jumps or plateaus, the tree could come out ahead.

## AI Usage

_To be added._