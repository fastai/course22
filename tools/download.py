#!/usr/bin/env python

from fastcore.utils import *
from kaggle import api

nbs = Path('tools/kaggles.txt').read_text().strip().splitlines()

for i,nb in enumerate(nbs):
    d = api.kernel_pull('jhoward', nb)
    src = d['blob']['sourceNullable']
    Path(f'{i:02d}-{nb}.ipynb').write_text(src)

