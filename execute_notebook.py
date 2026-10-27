"""Execute with this interpreter and checkpoint after each cell for recoverability."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client import AsyncKernelManager
import json

root = Path(__file__).resolve().parent
kernel_dir = root / '.cache' / 'kernels' / 'lab1'
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / 'kernel.json').write_text(json.dumps({
    'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'Lab 1 Python', 'language':'python'}), encoding='utf-8')
manager = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
path = root / 'LAB1_ANN_Assignment.ipynb'
notebook = nbformat.read(path,as_version=4)
def checkpoint(cell, cell_index, **kwargs):
    nbformat.write(notebook,path)
    print(f'Completed cell {cell_index}: {cell.source.splitlines()[0][:90]}',flush=True)
client=NotebookClient(notebook,timeout=-1,kernel_name='lab1',
    km=AsyncKernelManager(kernel_name='lab1',kernel_spec_manager=manager),resources={'metadata':{'path':str(root)}},
    on_cell_executed=checkpoint)
try:
    if '--resume-q5' in sys.argv:
        with client.setup_kernel():
            client.execute_cell(notebook.cells[1],1,execution_count=1)
            bootstrap=nbformat.v4.new_code_cell("""
bc_metrics=pd.read_csv(OUT/'q2_test_metrics.csv').iloc[0].to_dict()
cifar_rows=pd.read_csv(OUT/'q3_comparison.csv').to_dict('records')
housing_rows=pd.read_csv(OUT/'q4_optimizer_comparison.csv').to_dict('records')
logic_table=pd.read_csv(OUT/'q1_predictions.csv')
""")
            notebook.cells.append(bootstrap)
            try:
                client.execute_cell(bootstrap,len(notebook.cells)-1)
            finally:
                notebook.cells.pop()
            client.execute_cell(notebook.cells[11],11,execution_count=6)
            client.execute_cell(notebook.cells[13],13,execution_count=7)
    else:
        client.execute()
finally:
    nbformat.write(notebook,path)
print('Notebook execution finished.',flush=True)
from finalize_results import finalize
finalize()
