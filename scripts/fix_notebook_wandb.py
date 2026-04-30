import nbformat
import sys
import os

def fix_notebook(file_path):
    print(f"Processing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    modified = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            content = cell.source
            if 'def train_model(' in content and 'use_wandb=False' in content:
                print(f"  Found train_model definition in {file_path}. Updating...")
                cell.source = content.replace('use_wandb=False', 'use_wandb=True')
                modified = True
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"  Successfully updated {file_path}")
    else:
        print(f"  No changes needed for {file_path}")

if __name__ == "__main__":
    notebooks = [
        'eeg_intent_to_speak_real.ipynb',
        'eeg_intent_to_speak_physionet.ipynb',
        'eeg_intent_to_speak_synthetic.ipynb'
    ]
    for nb_file in notebooks:
        if os.path.exists(nb_file):
            fix_notebook(nb_file)
        else:
            print(f"File not found: {nb_file}")
