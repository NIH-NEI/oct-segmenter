#!/bin/bash
rm -rf dist build oct_segmenter.egg-info
python setup.py bdist_wheel --universal
