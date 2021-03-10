import os
import re
import sys
import csv
import json
import codecs
import locale
import requests
import datetime as dt
import argparse

import logging

from typing import Union, Optional, Tuple, List, cast

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import matplotlib as mp                 # type: ignore
from matplotlib import pyplot as plt    # type: ignore
from matplotlib import colors           # type: ignore
import matplotlib.dates as mdates       # type: ignore
import matplotlib.gridspec as gridspec  # type: ignore
import matplotlib.ticker as mticker     # type: ignore

from typing import Any, Tuple, Dict, Union

from logger_init import init_logger
from result_value import ResultKo, ResultOk, ResultValue
from ChartTools import remove_tick_lines
from ChartTools import every_nth_tick
from ChartTools import autolabel
from ChartTools import set_axes_common_properties
from ChartTools import text_box

def picture_title_text(title:str, global_statistics:dict)-> ResultValue:
    log = logging.getLogger('picture_title_text')
    log.info(" >>")

    txt = None
    rv: ResultValue = ResultKo(Exception("Error"))
    try:
        dt_format = "%d/%m/%Y %H:%M"                                                           
        txt = title + " - " + global_statistics['test begin'].strftime(dt_format)
    except Exception as ex:
        msg = "Failed - {ex}".format(ex=ex)
        log.error(msg)
        rv = ResultKo(Exception(msg))
    else:
        rv = ResultOk(txt)
        
    return rv

def statistics_box_text(global_statistics:dict, labels:dict)-> ResultValue:
    log = logging.getLogger('statistics_box_text')
    log.info(" >>")

    dt_format = "%d/%m/%Y %H:%M"
    txt = None
    rv: ResultValue = ResultKo(Exception("Error"))
    try:
        title = "{title}\n\n".format(title=labels["statistics txt title"][0])
        txt_duration = labels["stats duration"].format(be=global_statistics['test begin'].strftime(dt_format)
                                                      ,en=global_statistics['test end'].strftime(dt_format)
                                                      ,d=global_statistics['duration']) 

        txt_sample_num = labels["stats sample num"].format(n=f"{global_statistics['sample num']:,}"
                                                          ,er=f"{global_statistics['samples in error']:,}")
                                                           
        txt_elapsed = labels["stats elapsed"].format(min=global_statistics['min elapsed']
                                                    ,max=global_statistics['max elapsed']
                                                    ,avg=global_statistics['mean elapsed']
                                                    ,stde=global_statistics['std error elapsed']
                                                    ,med=global_statistics['median elapsed'])

        txt_threads = "\n" + labels["stats max threads"].format(th=global_statistics["max threads"])
                                                           
        txt = title + txt_duration + txt_sample_num + txt_elapsed + txt_threads
    except Exception as ex:
        msg = "Failed - {ex}".format(ex=ex)
        log.error(msg)
        rv = ResultKo(Exception(msg))
    else:
        rv = ResultOk(txt)
        
    return rv

def create_label_dict(language:str="it")-> ResultValue :
  labels = {}
  if language.lower() == "it":
    labels = {
       "picture_title": ("Test di carico per XYZ", 24)
               
      ,"thread title": ("Numero thread", 16)
      ,"thread x":     ("Tempo", 12)
      ,"thread y":     ("Numero", 12)
               
      ,"elapsed title":      ("Tempo di servizio", 16)
      ,"elapsed title - log":("Tempo di servizio - scala logaritmica", 16)
      ,"elapsed x":          ("Tempo", 12)
      ,"elapsed y":          ("Tempo di servizio (ms)", 12)
               
      ,"elapsed-binned title":      ("Tempo di servizio medio", 14)
      ,"elapsed-binned title - log":("Tempo di servizio medio - scala logaritmica", 16)
      ,"elapsed-binned x":          ("Tempo", 12)
      ,"elapsed-binned y":          ("Tempo di servizio medio (ms)", 12)
      ,"elapsed-binned mean txt":   ("Tempo servizio medio", 12)
      ,"elapsed-binned snum y":     ("Numero campioni", 12)
               
      ,"elapsed-frequency title":   ("Analisi frequenza dei tempi di servizio", 16)
      ,"elapsed-frequency label":   ("Frequenza", 12)
      ,"elapsed-frequency y %":     ("Percentuale", 12)
      ,"elapsed-frequency y #":     ("Numero", 12)
      ,"elapsed-frequency y":       ("", 12)
      ,"elapsed-frequency x":       ("Intervallo tempo di servizio (ms)", 12)
               
      ,"tps title":                 ("Transazioni per secondo", 16)
      ,"tps label":                 ("Frequenza", 12)
      ,"tps y":                     ("Transazioni per secondo", 12)
      ,"tps x":                     ("Tempo", 12)
               
      ,"quantiles txt title":       ("Tabella percentile / valore", 12)
      ,"statistics txt title":      ("Dati statistici:", 12)
      ,"stats duration":            ("Inizio test: {be}\nFine test: {en}\nDurata: {d} (minuti)\n")
      ,"stats sample num":          ("Numero campioni: {n}\nCampioni in errore: {er}\n")
               
      ,"stats elapsed":             ("Tempo di servizio minimo: {min} (ms)\nTempo di servizio massimo: {max} (ms)\nTempo di servizio medio: {avg} (ms)\nVarianza: {stde} (ms)\nTempo di servizio - mediana: {med} (ms)\n")
      ,"stats max threads":         ("Numero massimo di thread: {th} (ms)\n")
               
      ,"quantiles":                 ("", 18)
      ,"statistics":                ("", 18)
      ,"notes":                     ("", 18)
    }
  elif language.lower() == "en":
    labels = {
       "picture_title": ("Load test of XYZ", 24)
      ,"thread title": ("Threads number", 16)
      ,"thread x":     ("Time", 12)
      ,"thread y":     ("Number", 12)
              
      ,"elapsed title":      ("Service time", 16)
      ,"elapsed title - log":("Service time - logarithmic scale", 16)
      ,"elapsed x":          ("Time", 12)
      ,"elapsed y":          ("Service time (ms)", 12)
              
      ,"elapsed-binned title":      ("Average service time", 16)
      ,"elapsed-binned title - log":("Average service time - logarithmic scale", 12)
      ,"elapsed-binned x":          ("Ttime", 12)
      ,"elapsed-binned y":          ("Average service time (ms)", 12)
      ,"elapsed-binned mean txt":   ("Elapsed mean value", 12)
      ,"elapsed-binned snum y":     ("Sample numerosity", 12)
              
      ,"elapsed-frequency title":   ("Elapsed time frequency analisys", 16)
      ,"elapsed-frequency label":   ("Frequency", 12)  
      ,"elapsed-frequency y %":     ("Percentage", 12)
      ,"elapsed-frequency y #":     ("Number of samples", 12)
      ,"elapsed-frequency y":       ("", 12)
      ,"elapsed-frequency x":       ("Elapsed (ms)", 12)
              
      ,"tps title":                 ("Transaction per second", 16)
      ,"tps label":                 ("", 12)
      ,"tps y":                     ("Transaction per second", 12)
      ,"tps x":                     ("Time", 12)
              
      ,"quantiles txt title":       ("Quantiles / value table", 12)
      ,"statistics txt title":      ("Statistics:", 12)
              
      ,"stats duration":            ("Begin datetime: {be}\nEnd datetime: {en}\nDuration: {d} (minuti)\n")
      ,"stats sample num":          ("Number of samples: {n}\nSamples in error: {er}\n")
              
      ,"stats elapsed":             ("Elapsed min: {min} (ms)\nElapsed max: {max} (ms)\nElapsed mean: {avg} (ms)\nStandard error: {stde} (ms)\nElapsed median: {med} (ms)\n")
      ,"stats max threads":         ("Max threads num: {th} (ms)\n")
      ,"quantiles":                 ("", 18)
      ,"statistics":                ("", 18)
      ,"notes":                     ("", 18)
    } 
  return ResultOk(labels)

def get_global_statistics(df: pd.DataFrame, errors_df: pd.DataFrame, dec_num: int=2) -> ResultValue:
    log = logging.getLogger('get_global_statistics')
    log.info(" >>")

    global_statistics:dict = {
        "duration": None
        , "max threads": None
        ,"transaction per second": 0
    }
    try:
        # Calculate the test duration in minutes.
        global_statistics["test begin"] = df.index.min()
        global_statistics["test end"] = df.index.max()
        duration = global_statistics["test end"] - global_statistics["test begin"]

        global_statistics["duration"] = round(
            duration.total_seconds() / 60, dec_num)
        global_statistics["duration sec"] = duration.total_seconds()

        # Max number of threads.
        global_statistics["max threads"] = df["allThreads"].max()

        # Total transaction per second.
        global_statistics["transaction per second"] = ( df.shape[0] - errors_df.shape[0]) / global_statistics["duration sec"]
        global_statistics["transaction per second"] = round(global_statistics["transaction per second"], 0)

        # Sample len.
        global_statistics["sample num"] = df.shape[0]
        global_statistics["samples in error"] = errors_df.shape[0]

        # Service time statistics.
        global_statistics["max elapsed"] = df["elapsed"].max()
        global_statistics["min elapsed"] = df["elapsed"].min()
        global_statistics["mean elapsed"] = round(
            df["elapsed"].mean(), dec_num)
        global_statistics["median elapsed"] = round(
            df["elapsed"].median(), dec_num)
        global_statistics["std error elapsed"] = round(
            df["elapsed"].sem(), dec_num)
        global_statistics["quantiles elapsed"] = df["elapsed"].quantile(
            np.arange(0.1, 1, 0.1))

    except Exception as ex:
        log.error(" failed - {ex}".format(ex=ex))
        return ResultKo(ex)
    log.info(" <<")
    return ResultOk(global_statistics)

def get_dataframe_from_csv(data_file: str, ok_codes: List = [200, 202]) -> ResultValue:
    log = logging.getLogger('get_dataframe_from_csv')
    log.info(" >>")
    retuned_df = {}
    try:
        dtype_dict: dict = {
        }

        parse_dates = ["timeStamp"]

        columns = ["elapsed", "responseCode",
                   "responseMessage", "grpThreads", "allThreads"]

        df = pd.read_csv(data_file, sep='#', lineterminator='\n',
                         low_memory=False, dtype=dtype_dict)
        df["timeStamp"] = pd.to_datetime(df["timeStamp"], unit='ms')
        df.set_index("timeStamp", inplace=True)
        df.sort_index(axis=0, ascending=True, inplace=True)
        #df.sort_values(by=["timeStamp"], inplace=True)

        # Get just the columns we need.
        df = df[columns]

        # Collection of all the sample in error.
        errors_df = df.loc[~ df['responseCode'].isin(ok_codes)]

        # Create a dataframe binned and with the average elapsed value and bin size.
        time_delta = 30
        time_unit = 'S'
        td_for_grouper = '{td}{tu}'.format(td=str(time_delta), tu=time_unit)

        binned_elapsed = df.groupby(pd.Grouper(level='timeStamp', freq=td_for_grouper))[
            'elapsed'].agg(['mean', 'count'])
        binned_elapsed["dt centered"] = binned_elapsed.index + \
            pd.offsets.Second(15)

        binned_elapsed['tps'] = binned_elapsed['count'].apply(
            lambda row: row/time_delta)

        retuned_df["df"] = df
        retuned_df["errors_df"] = errors_df
        retuned_df["binned_elapsed"] = binned_elapsed

    except Exception as ex:
        log.error(" failed - {ex}".format(ex=ex))
        return ResultKo(ex)

    log.info(" <<")
    return ResultOk(retuned_df)

def threads_chart(ax: mp.axes.Axes
                ,df: pd.DataFrame
                ,time_limits
                ,labels: dict
                ,color: str = "#009933") -> ResultValue:
    log = logging.getLogger('threads_chart')
    log.info(" >>")
    try:
        x = df.index
        y = df["allThreads"]

        set_axes_common_properties(ax, no_grid=False)

        ax.step(x, y, color=color)
        ax.set_xlim(time_limits)

        minutes = mdates.MinuteLocator(interval=1)
        minutes_fmt = mdates.DateFormatter('%H:%M')

        #seconds = mdates.SecondLocator(bysecond = 30)
        #seconds_fmt = mdates.DateFormatter('%S')

        ax.xaxis.set_major_locator(minutes)
        ax.xaxis.set_major_formatter(minutes_fmt)

        # ax.xaxis.set_minor_locator(seconds)
        # ax.xaxis.set_minor_formatter(seconds_fmt)

        ax.tick_params(axis='x', labelrotation=80)
        remove_tick_lines('x', ax)

        ax.set_title(labels['thread title'][0],
                     fontsize=labels['thread title'][1])
        ax.set_ylabel(labels["thread y"][0],    fontsize=labels['thread y'][1])
        ax.set_xlabel(labels["thread x"][0],    fontsize=labels['thread x'][1])

    except Exception as ex:
        msg = "threads_chart failed - {ex}".format(ex=ex)
        return ResultKo(Exception(msg))

    log.info(" <<")
    return ResultOk(True)

def elapsed_chart(ax: mp.axes.Axes
                 ,df: pd.DataFrame
                 ,errors_df: pd.DataFrame
                 ,time_limits
                 ,labels: dict
                 ,logarithmic=False
                 ,colors: List = ["#0000e6", "#ff1a1a"]) -> ResultValue:
    log = logging.getLogger('elapsed_chart')
    log.info(" >>")
    try:
        x=df.index 
        y=df["elapsed"]

        set_axes_common_properties(ax, no_grid=False)
        ax.set_xlim(time_limits)

        ax.scatter(x, y, color=colors[0], s=3)
        if errors_df is not None and errors_df.shape[0] > 0:
            ax.scatter(errors_df.index, errors_df["elapsed"], color=colors[1], s=3)
    
        minutes = mdates.MinuteLocator(interval = 1)
        minutes_fmt = mdates.DateFormatter('%H:%M')

        ax.xaxis.set_major_locator(minutes)
        ax.xaxis.set_major_formatter(minutes_fmt)
       
        ax.tick_params(axis='x', labelrotation=80)
        remove_tick_lines('x', ax)
        
        if logarithmic:
            ax.set_yscale('log')
            ax.set_title(labels['elapsed title - log'][0], fontsize=labels['elapsed title - log'][1])
        else:
            ax.set_title(labels['elapsed title'][0], fontsize=labels['elapsed title'][1])
        
        ax.set_ylabel(labels["elapsed y"][0],    fontsize=labels['elapsed y'][1])
        ax.set_xlabel(labels["elapsed x"][0],    fontsize=labels['elapsed x'][1])
            
    except Exception as ex:
        msg = "threads_chart failed - {ex}".format(ex=ex)
        return ResultKo(Exception(msg))
    else:
        rv = ResultOk(True)
    log.info(" <<")
    return rv

def main(args: argparse.Namespace) -> ResultValue:
    log = logging.getLogger('Main')
    log.info(" >>")
    rv: ResultValue = ResultKo(Exception("Error"))
    ok_codes = [200, 202]
    labels = create_label_dict()
    try:
        if args.data_frame is not None:
            file_name = args.data_frame[0]
            df_dict = get_dataframe_from_csv(file_name, ok_codes=ok_codes)
            if df_dict.is_ok() == False:
                log.error(df_dict())
            else:
                rv = ResultOk(True)

        elif args.chart is not None:
            chart_name = args.chart[0].lower()
            file_name = args.chart[1]

            df_dict = get_dataframe_from_csv(file_name, ok_codes=ok_codes)
            if df_dict.is_ok() == False:
                log.error(df_dict())
                return ResultKo(df_dict())

            global_stats = get_global_statistics(df=df_dict()["df"], errors_df=df_dict()["errors_df"])
            time_limits = [global_stats()["test begin"] - pd.Timedelta(minutes=0.5) 
                          ,global_stats()["test end"]   + pd.Timedelta(minutes=0.5)]

            fig = plt.figure(figsize=(20, 10))
            gs1 = gridspec.GridSpec(1, 1, hspace=0.2, wspace=0.1, figure=fig)
            ax = []
            ax.append(fig.add_subplot(gs1[0, 0]))
            idx = 0

            if chart_name == "thread":
                rv = threads_chart(ax=ax[idx], df=df_dict()["df"], labels=labels(), time_limits=time_limits)
            elif chart_name == "elapsed":
                rv = elapsed_chart(ax=ax[idx], df=df_dict()["df"], errors_df=df_dict()["errors_df"], labels=labels(), time_limits=time_limits)
            else:
                msg = "Unknown chart name : {c}".format(c=chart_name)
                log.error(msg)
                return ResultKo(Exception(msg))

            if rv.is_ok():
                plt.savefig(os.path.join(os.sep, "tmp", "jMeter-{c}.png".format(c=chart_name))
                                        ,bbox_inches = 'tight'
                                        ,pad_inches = 0.2)

    except Exception as ex:
        log.error("Exception caught - {ex}".format(ex=ex))
        rv = ResultKo(ex)
    log.info(" ({rv}) <<".format(rv=rv))
    return rv

if __name__ == "__main__":
    init_logger('/tmp', "report.log", log_level=logging.DEBUG,
                std_out_log_level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_frame", "-df", nargs=1,
                        help="Create a dataframe from JMeter log file.")
    parser.add_argument("--chart", "-c", nargs=2,
                        help="Create the named chart using the given file name [thread|elapsed].")
    args = parser.parse_args()

    rv = main(args)

    ret_val = os.EX_OK if rv.is_ok() == True else os.EX_USAGE
    sys.exit(ret_val)
