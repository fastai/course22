#!/usr/bin/env python

import nbformat
from nbdev.export import *
from nbdev.clean import *
from fastcore.all import *

_re_header = re.compile(r'^#+\s+\S+')
_re_clean  = re.compile(r'^\s*#\s*clean\s*')

def is_header_cell(cell): return _re_header.search(cell['source']) is not None
def is_clean_cell(cell): return _re_clean.search(cell['source']) is not None

def clean_tags(cell):
    if is_header_cell(cell): return cell
    for attr in ["id", "caption", "alt", "width", "hide_input", "hide_output", "clean"]:
        cell["source"] = re.sub(r'#\s*' + attr + r'.*?($|\n)', '', cell["source"])
    return cell

def proc_nb(fname, dest):
    nb = read_nb(fname)
    nb['cells'] = [clean_tags(c) for j,c in enumerate(nb['cells']) if
                   c['cell_type']=='code' or is_header_cell(c) or is_clean_cell(c)]
    clean_nb(nb, clear_all=True)
    with open(dest/fname.name, 'w') as f: nbformat.write(nb, f, version=4)

@call_parse
def proc_all(
        path:str='.', # Path for source NBs
        dest_path:str='clean'): # Path for dest NBs
    path,dest_path = Path(path),Path(dest_path)
    fns = [f for f in path.iterdir() if f.suffix == '.ipynb' and not f.name.startswith('_')]
    for fn in fns: proc_nb(fn, dest=dest_path)

