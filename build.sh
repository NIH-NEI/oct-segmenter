#!/bin/bash
rm -rf dist build oct_segmenter.egg-info
python3 setup.py bdist_wheel --universal