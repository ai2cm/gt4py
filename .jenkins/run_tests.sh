#!/bin/bash
set -e
set -x
pip3 install --upgrade wheel
python3 -m pip install --user virtualenv
python3 -m venv gt4pyenv
source gt4pyenv/bin/activate
pip3 install pytest==5.2
git clone https://github.com/gridtools/gt4py.git gt4py
pip3 install --no-cache-dir -e gt4py
python3 -m gt4py.gt_src_manager install
python3 -m pytest gt4py
