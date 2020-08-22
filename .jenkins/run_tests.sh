#!/bin/bash
set -e
set -x
pip3 install --upgrade wheel
pip3 install pytest
git clone https://github.com/gridtools/gt4py.git gt4py
pip3 install --no-cache-dir -e gt4py
python3 -m gt4py.gt_src_manager install
python3 -m pytest gt4py
