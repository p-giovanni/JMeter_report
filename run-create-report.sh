#!/bin/bash

PROJECT_HOME="/Users/ERIZZAG5J/repos/JMeter_report"
PROG_PATH="./src/report/CreateReport.py"
source /Users/ERIZZAG5J/venv/bin/activate

export PYTHONPATH="${PROJECT_HOME}/src:${PYTHONPATH}"

python ${PROG_PATH} --report $1 $2 $3 --chart_title "VIDEO_API_Playback_STAGE"


