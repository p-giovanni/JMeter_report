# JMeter report
This is a project to explore a time serie - a JMeter performance csv file - using [Pandas](https://pandas.pydata.org/) and [Matplotlib](https://matplotlib.org/).

To see the code open the notebook here [JMeter_Report.ipynb](./notebook/JMeter_Report.ipynb).
The chars are in the [images](./images/) directory. 
The real output is the picture containing all the charts, that is this one: [Test-All-CHARTS.png](./images/Test-All-CHARTS.png)

The same chart can be exported in PDF format, see: [Test-All-CHARTS.pdf](./images/Test-All-CHARTS.pdf). 

All the settings are in a note cell - the one called **settings section**.

## Disclaimer
I do know very well Pandas, Matplotlib but, as all the programmers, I do bugs.
So beware, I have checked the results as carefully as I can but nevertheless do not take for granted my results, check by yourself my code and decide if it is correct or not.

## Project status
```diff
! Doing
```
I am not yet satisfied by the charts, there are a few improvements I'd like to do.

**TODO**: 
- [o] a batch Python script to create the charts; 

## Data file
The data file is the standard JMeter log file and have the following columns structure:

|timeStamp| elapsed | responseMessage | responseCode | responseCode | grpThreads | allThreads |
|:-------:|:---------------:| ------------:| ------------:| ----------:| ----------:| ----------:|
|1526484723967|125|AAA P1 - HTTP Request|200|OK|1|1|
|1526484723100|20|AAA P1 - HTTP Request|200|OK|1|1|

## Installation and build
Clone this repository and than:

```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```
To run **Jupyter** use this command:
```bash
jupyter lab --notebook-dir <path to your installation>/JMeter_report/notebook --port=9191
```
## Report chart

![JMeter report chart](./images/Test-All-CHARTS.pdf)
