import nbformat
from nbformat.v4 import new_code_cell, new_notebook
import codecs


def parse_py(fn):
    with open(fn, "r") as f:
        lines = []
        for line_i in f:
            if line_i.startswith('# %%'):
                lines[-1] = lines[-1].strip('\n')
                if len(lines[-1]) == 0:
                    lines = lines[:(len(lines)-1)]
                lines[-1] = lines[-1].strip('\n')
                yield "".join(lines)
                lines = []
            lines.append(line_i)
        if lines:
            lines[-1] = lines[-1].strip('\n')
            yield "".join(lines)


def py_to_ipynb(source, dest):
    # Create the code cells by parsing the file in input
    cells = []
    for c in parse_py(source):
        cells.append(new_code_cell(source=c))

    # This creates a V4 Notebook with the code cells extracted above
    nb0 = new_notebook(cells=cells,
                       metadata={'language': 'python'})

    with codecs.open(dest, encoding='utf-8', mode='w') as f:
        nbformat.write(nb0, f, 4)


if __name__ == '__main__':
    import os
    root = '../python'
    files = [file for file in os.listdir(root) if '.py' in file]
    for file in files:
        py_to_ipynb(os.path.join(root, file), file.strip('.py') + '.ipynb')
