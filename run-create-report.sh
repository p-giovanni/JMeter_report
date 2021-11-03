#!/bin/bash

PROG_PATH="./src/CreateReport.py"
source /home/giovanni/add-on/venv/bin/activate

#export PYTHONPATH="/home/giovanni/code-personal/python/SchemaTools:${PYTHONPATH}"

python ${PROG_PATH} --report $1 $2 $3 --chart_title "VIDEO_API listing (TEST)"
