## Measured results and observations

These observations are generated from the saved execution results, with seed 42.

### Q1

AND, OR and multilayer XOR classify all four inputs correctly. Single-layer XOR fails, as required by its lack of linear separability.

### Q2

Test accuracy is 97.37%, malignant-class precision 1.0000, recall 0.9286, and F1 0.9630. The validation-loss curve flattens while training loss continues to fall, indicating mild late overfitting rather than underfitting. Early stopping limits it; L2 would be a useful next experiment.

### Q3

Baseline test accuracy is 50.33%; dropout test accuracy is 44.70%, a change of -5.63 percentage points. The inference-time training-validation gap changes from 0.0573 to 0.0170. Dropout therefore needs to be assessed from both the gap and held-out accuracy, rather than assumed to improve performance.
Here, dropout reduces both training and test accuracy, consistent with too much regularization or insufficient training for this capacity. Prefer the baseline by its lower validation loss; consider weaker dropout or a longer validation-controlled budget in a future experiment.
The most frequent directed confusion for baseline is **airplane → ship**, with 230 test images.
The most frequent directed confusion for dropout is **bird → deer**, with 245 test images.

### Q4

Choose **SGD** for further testing because it has the lower validation MSE (0.2988). Its test MAE is 0.3786, MSE 0.2985, RMSE 0.5464 and R² 0.7722. RMSE corresponds to approximately $54,637. The actual-versus-predicted plot reveals residual spread and the dataset ceiling; the notebook discusses fitting behavior using the training/validation curves.

The alternative, Adam, has test RMSE 0.5396. The validation difference is small; repeated splits would be needed to establish a reliable optimizer ranking. Model selection remains based on validation performance even when the test ranking differs.

### Q5

- **neurons:** validation-selected setting 128; validation accuracy 97.47%, test accuracy 97.56%.
- **learning_rate:** validation-selected setting 0.01; validation accuracy 97.14%, test accuracy 97.07%.
- **optimizer:** validation-selected setting RMSprop; validation accuracy 97.05%, test accuracy 97.30%.
- **regularization:** validation-selected setting l2; validation accuracy 97.09%, test accuracy 97.19%.

Increasing hidden width from 16 to 128 changes validation accuracy from 94.06% to 97.47%. Wider layers add parameters and can model more digit variation, at greater computation cost.

Increasing the rate from 0.001 to 0.1 changes validation accuracy from 96.94% to 90.78%. The larger steps hurt the final solution in this run.

- **none:** training-validation accuracy gap 0.0163; test accuracy 97.18%.
- **dropout:** training-validation accuracy gap 0.0112; test accuracy 96.82%.
- **l2:** training-validation accuracy gap 0.0134; test accuracy 97.19%.
- **dropout+l2:** training-validation accuracy gap 0.0097; test accuracy 96.78%.

Compared with no regularization, **dropout** reduces the accuracy gap by 0.0051 and changes validation accuracy by -0.17 percentage points. The smaller capacity or penalty does not improve validation accuracy within this budget.

Compared with no regularization, **l2** reduces the accuracy gap by 0.0029 and changes validation accuracy by +0.15 percentage points. This improves validation performance in this run.

Compared with no regularization, **dropout+l2** reduces the accuracy gap by 0.0066 and changes validation accuracy by -0.21 percentage points. The smaller capacity or penalty does not improve validation accuracy within this budget.

Early stopping triggered in 2 of 11 unique runs; 9 reached the 15-epoch cap. The per-run stopping table records the epoch count and best epoch. Test scores are descriptive and were not used to choose settings.

These results use a single seed and a finite budget. Repeated seeds and further validation are needed before treating close differences as meaningful.
