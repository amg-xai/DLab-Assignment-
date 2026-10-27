# Lab Assignment 1 — Artificial Neural Networks

TensorFlow/Keras solutions for all five questions in the Foundations of Neural Networks and Deep Learning ANN lab.

Open [`LAB1_ANN_Assignment.ipynb`](LAB1_ANN_Assignment.ipynb) to read the code, explanations, outputs and conclusions. Tables and standalone plots are saved in `results/` during execution.
[`RESULTS.md`](RESULTS.md) provides a concise discussion generated from the measured results after a successful run.

| Question | Experiment |
| --- | --- |
| Q1 | AND/OR perceptrons, single-layer XOR failure, hidden-layer XOR success, loss plots and truth-table predictions |
| Q2 | Wisconsin breast cancer EDA, stratified 80/20 split, training-only scaling, binary ANN, accuracy/precision/recall/F1 and confusion matrix |
| Q3 | Full CIFAR-10 dataset, labeled image samples, dense ANN with two hidden layers, dropout comparison and confused classes |
| Q4 | California Housing exploration and preprocessing, two-hidden-layer regression ANN, Adam vs SGD, MAE/MSE/RMSE/R² and actual-vs-predicted plot |
| Q5 | Full MNIST dataset; widths 16/32/64/128; rates 0.001/0.01/0.1; SGD/RMSprop/Adam; four regularization settings; early stopping |

## Run locally (Python 3.11)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe execute_notebook.py
```

The runner executes and saves the notebook using the current interpreter, with checkpoints after each cell. Network access is needed on the first run to download public datasets. No paid service or API key is required. Training uses CPU by default and can take several minutes.
`requirements-lock.txt` records the complete Windows execution environment; `requirements.txt` lists the main reproducible dependencies for local or Colab setup.

For Google Colab, upload the notebook, install the requirements if needed, and choose **Runtime → Run all**. Hardware and package differences may change numerical results slightly.

`build_notebook.py` regenerates the notebook source and **clears saved outputs**. Use it only when editing the cell definitions, then run `execute_notebook.py` again.
If a run is interrupted during Q5 after Q1-Q4 finish, `.\.venv\Scripts\python.exe execute_notebook.py --resume-q5` preserves those outputs and re-executes Q5 plus the final summary using the saved earlier CSV files.

## Experimental controls

- Fixed random seed 42, stratified validation splits for classification, training-only preprocessing.
- Complete official CIFAR-10/MNIST test sets; no small-data shortcut.
- Test results are reported for the required comparisons; validation data selects models.
- Repeated baseline configurations in Q5 share one identical run: 14 comparison rows correspond to 11 unique configurations.
- Best validation-loss weights are restored after training. The early-stopping table distinguishes patience stops from epoch-limit termination.
- Breast cancer label 1 means malignant. Housing errors are in $100,000 units (MSE in squared units).
- One-seed comparisons and finite training budgets are educational experiments, not statistically significant rankings.

Dataset references are included in the notebook. Dataset downloads, virtual environments and temporary files are excluded from Git.
