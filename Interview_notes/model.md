| Random Forest (Bagging) | XGBoost (Boosting)                                                                |
| ----------------------- | --------------------------------------------------------------------------------- |
| Trees are independent   | Trees are sequential                                                              |
| Majority voting         | Correct previous mistakes                                                         |
| Reduces variance        | Reduces both bias and variance (through iterative improvement and regularization) |
| Easier to parallelize   | More sequential training                                                          |
| Simpler to tune         | More hyperparameters                                                              |
| Very strong baseline    | Often state-of-the-art for tabular data                                           |


Why is XGBoost so good?

Because it combines several strengths:

Builds trees sequentially to correct previous mistakes.
Optimizes a well-defined loss function.
Includes regularization to reduce overfitting.
Handles missing values internally.
Can model complex nonlinear relationships.
Is highly optimized for speed and memory.

| Parameter          | Purpose                                                                        |
| ------------------ | ------------------------------------------------------------------------------ |
| `n_estimators`     | Number of boosting rounds (trees).                                             |
| `learning_rate`    | How much each new tree contributes. Smaller values usually require more trees. |
| `max_depth`        | Maximum depth of each tree. Controls model complexity.                         |
| `subsample`        | Fraction of training rows used for each tree. Helps reduce overfitting.        |
| `colsample_bytree` | Fraction of features considered when building each tree.                       |
| `scale_pos_weight` | Gives extra importance to the minority class in imbalanced datasets.           |
