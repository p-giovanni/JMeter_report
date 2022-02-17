#!/bin/bash

source /Users/ERIZZAG5J/venv/bin/activate

PROJECT_HOME=$(cd $(dirname "$BASH_SOURCE[0]")&& pwd)
PROG_PATH="./src/report/CreateReport.py"

export PYTHONPATH="${PROJECT_HOME}/src:${PYTHONPATH}"

python ${PROJECT_HOME}/${PROG_PATH} --report $1 $2 $3 $4 --chart_title "VAS"
exit $?

