#!/bin/bash

HOME="${HOME}"
PROJECT_HOME="${HOME}/repos/JMeter_report"

SOURCE_PATH="${PROJECT_HOME}/src"
CONFIG_FILE="${PROJECT_HOME}/config/JMeterMergeFileConfig.json"
JMETER_CONF_FILE="${HOME}/repos/ita-videoplatform-public-api-performance-tests/cdk-jmeter/jmeter/out_csv.properties"
JMETER_REPORT_BASE="${HOME}/Work/jmeter/report"
NOW=$(date +"%Y%m%d_%H%M%S")

export PYTHONPATH="${SOURCE_PATH}:${SOURCE_PATH}/common:${SOURCE_PATH}/tools:${PYTHONPATH}"

source ${HOME}/venv/bin/activate

python  ${SOURCE_PATH}/tools/JMeterFileMerge.py ${CONFIG_FILE}
if [ $? != "0" ]; then
    echo "Sample file merge failded - exit in error."
    exit 1
fi
exit 0