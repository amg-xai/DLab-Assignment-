"""Add a concise, measured results discussion after a successful notebook run."""
from pathlib import Path
import json
import pandas as pd
import nbformat

def finalize():
    root=Path(__file__).resolve().parent
    result=root/'results'
    summary=json.loads((result/'summary.json').read_text())
    cifar=pd.read_csv(result/'q3_comparison.csv').set_index('model')
    housing=pd.read_csv(result/'q4_optimizer_comparison.csv')
    mnist=pd.read_csv(result/'q5_all_experiments.csv')
    selected=housing.loc[housing.validation_mse.idxmin()]
    bc=summary['q2']
    lines=['## Measured results and observations',
        '', 'These observations are generated from the saved execution results, with seed 42.', '',
        '### Q1', '',
        'AND, OR and multilayer XOR classify all four inputs correctly. Single-layer XOR fails, '
        'as required by its lack of linear separability.', '',
        '### Q2', '',
        f'Test accuracy is {bc["accuracy"]:.2%}, malignant-class precision {bc["precision"]:.4f}, '
        f'recall {bc["recall"]:.4f}, and F1 {bc["f1"]:.4f}. '
        'The validation-loss curve flattens while training loss continues to fall, indicating mild late '
        'overfitting rather than underfitting. Early stopping limits it; L2 would be a useful next experiment.', '',
        '### Q3', '']
    baseline=cifar.loc['baseline']; dropout=cifar.loc['dropout']
    delta=dropout.test_accuracy-baseline.test_accuracy
    lines.append(f'Baseline test accuracy is {baseline.test_accuracy:.2%}; dropout test accuracy is '
        f'{dropout.test_accuracy:.2%}, a change of {delta*100:+.2f} percentage points. '
        f'The inference-time training-validation gap changes from '
        f'{baseline.train_accuracy-baseline.validation_accuracy:.4f} to '
        f'{dropout.train_accuracy-dropout.validation_accuracy:.4f}. '
        'Dropout therefore needs to be assessed from both the gap and held-out accuracy, rather than assumed to improve performance.')
    if delta < 0 and dropout.train_accuracy < baseline.train_accuracy:
        lines.append('Here, dropout reduces both training and test accuracy, consistent with too much '
            'regularization or insufficient training for this capacity. Prefer the baseline by its lower '
            'validation loss; consider weaker dropout or a longer validation-controlled budget in a future experiment.')
    for name in ['baseline','dropout']:
        pair=pd.read_csv(result/f'q3_{name}_confused_pairs.csv').iloc[0]
        lines.append(f'The most frequent directed confusion for {name} is **{pair.actual} → '
                     f'{pair.predicted}**, with {pair["count"]} test images.')
    lines += ['', '### Q4', '',
        f'Choose **{selected.optimizer}** for further testing because it has the lower validation MSE '
        f'({selected.validation_mse:.4f}). Its test MAE is {selected.MAE:.4f}, MSE {selected.MSE:.4f}, '
        f'RMSE {selected.RMSE:.4f} and R² {selected.R2:.4f}. '
        f'RMSE corresponds to approximately ${selected.RMSE*100000:,.0f}. '
        'The actual-versus-predicted plot reveals residual spread and the dataset ceiling; '
        'the notebook discusses fitting behavior using the training/validation curves.', '', '### Q5', '']
    alternative=housing.loc[housing.optimizer != selected.optimizer].iloc[0]
    housing_note=(f'The alternative, {alternative.optimizer}, has test RMSE {alternative.RMSE:.4f}. '
        'The validation difference is small; repeated splits would be needed to establish a reliable '
        'optimizer ranking. Model selection remains based on validation performance even when the '
        'test ranking differs.')
    q5_heading=lines.index('### Q5')
    lines[q5_heading:q5_heading]=[housing_note,'']
    for group in ['neurons','learning_rate','optimizer','regularization']:
        group_rows=mnist[mnist.group==group]
        best=group_rows.loc[group_rows.validation_accuracy.idxmax()]
        lines.append(f'- **{group}:** validation-selected setting {best.value}; '
            f'validation accuracy {best.validation_accuracy:.2%}, test accuracy {best.test_accuracy:.2%}.')
    widths=mnist[mnist.group=='neurons'].set_index('width')
    lines += ['',f'Increasing hidden width from 16 to 128 changes validation accuracy from '
        f'{widths.loc[16].validation_accuracy:.2%} to {widths.loc[128].validation_accuracy:.2%}. '
        'Wider layers add parameters and can model more digit variation, at greater computation cost.']
    rates=mnist[mnist.group=='learning_rate'].set_index('value')
    rate_low=rates.loc['0.001']; rate_high=rates.loc['0.1']
    lines += ['', f'Increasing the rate from 0.001 to 0.1 changes validation accuracy from '
        f'{rate_low.validation_accuracy:.2%} to {rate_high.validation_accuracy:.2%}. '
        + ('The larger steps hurt the final solution in this run.' if rate_high.validation_accuracy < rate_low.validation_accuracy
           else 'The larger steps did not hurt validation accuracy in this run; inspect the curves for stability.'), '']
    reg=mnist[mnist.group=='regularization'].set_index('value')
    for name in ['none','dropout','l2','dropout+l2']:
        row=reg.loc[name]
        lines.append(f'- **{name}:** training-validation accuracy gap '
            f'{row.train_accuracy-row.validation_accuracy:.4f}; test accuracy {row.test_accuracy:.2%}.')
    base=reg.loc['none']; base_gap=base.train_accuracy-base.validation_accuracy
    for name in ['dropout','l2','dropout+l2']:
        row=reg.loc[name]; gap=row.train_accuracy-row.validation_accuracy
        lines += ['',f'Compared with no regularization, **{name}** '
            f'{"reduces" if gap < base_gap else "increases"} the accuracy gap by '
            f'{abs(gap-base_gap):.4f} and changes validation accuracy by '
            f'{(row.validation_accuracy-base.validation_accuracy)*100:+.2f} percentage points. '
            + ('This improves validation performance in this run.' if row.validation_accuracy > base.validation_accuracy
               else 'The smaller capacity or penalty does not improve validation accuracy within this budget.')]
    unique=mnist.drop_duplicates(['width','learning_rate','optimizer','regularization'])
    stopped=int(unique.stopped_early.sum())
    lines += ['', f'Early stopping triggered in {stopped} of {len(unique)} unique runs; '
        f'{len(unique)-stopped} reached the 15-epoch cap. The per-run stopping table records the '
        'epoch count and best epoch. Test scores are descriptive and were not used to choose settings.', '',
        'These results use a single seed and a finite budget. Repeated seeds and further validation are '
        'needed before treating close differences as meaningful.']
    text='\n'.join(lines)+'\n'
    (root/'RESULTS.md').write_text(text,encoding='utf-8')
    path=root/'LAB1_ANN_Assignment.ipynb'
    notebook=nbformat.read(path,as_version=4)
    notebook.cells=[cell for cell in notebook.cells if cell.get('id')!='measured-results']
    cell=nbformat.v4.new_markdown_cell(text); cell.id='measured-results'
    notebook.cells.append(cell)
    nbformat.validate(notebook); nbformat.write(notebook,path)
    print('Generated RESULTS.md and notebook measured-results discussion.')

if __name__=='__main__': finalize()
