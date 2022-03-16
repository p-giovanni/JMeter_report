import os
import sys
import argparse
import logging
from enum import Enum
from datetime import datetime
import datetime as dt

from typing import Union, Optional, Tuple, List, cast

import numpy as np                      # type: ignore
import pandas as pd                     # type: ignore

import matplotlib as mp                 # type: ignore
from matplotlib import pyplot as plt    # type: ignore
from matplotlib import colors           # type: ignore
import matplotlib.dates as mdates       # type: ignore
import matplotlib.gridspec as gridspec  # type: ignore
import matplotlib.ticker as mticker     # type: ignore

from typing import Any, Tuple, Dict, Union

from common.logger_init import init_logger
from common.result_value import ResultKo, ResultOk, ResultValue
from report.ChartTools import remove_tick_lines
from report.ChartTools import every_nth_tick
from report.ChartTools import autolabel
from report.ChartTools import set_axes_common_properties
from report.ChartTools import text_box

ok_codes = [200, 202]

class TypeOfChart(Enum):
    PERCENTAGE = 1
    FREQUENCIES = 2

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
      ,"stats max threads":         ("Numero massimo di thread: {th} \n")
               
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
              
      ,"stats duration":            ("Begin datetime: {be}\nEnd datetime: {en}\nDuration: {d} (minutes)\n")
      ,"stats sample num":          ("Number of samples: {n}\nSamples in error: {er}\n")
              
      ,"stats elapsed":             ("Elapsed min: {min} (ms)\nElapsed max: {max} (ms)\nElapsed mean: {avg} (ms)\nStandard error: {stde} (ms)\nElapsed median: {med} (ms)\n")
      ,"stats max threads":         ("Max threads num: {th} \n")
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
        df["timeStamp"] = pd.to_datetime(df["timeStamp"]) #, unit='ms')
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
        msg = "elapsed_chart failed - {ex}".format(ex=ex)
        log.error(msg)
        return ResultKo(Exception(msg))
    else:
        rv = ResultOk(True)
    log.info(" <<")
    return rv

def elapsed_binned_chart(ax: mp.axes.Axes
                        ,binned_df: pd.DataFrame
                        ,global_statistics:dict
                        ,labels: dict
                        ,time_limits:List
                        ,colors=["#0000e6", "#d5d5d5", "#c87607"]) -> ResultValue:
    log = logging.getLogger('elapsed_binned_chart')
    log.info(" >>")

    rv: ResultValue = ResultKo(Exception("Error"))
    try:
        x=binned_df.index
        y=binned_df['mean']
        x_snum=binned_df["dt centered"]
        y_snum=binned_df['count']

        set_axes_common_properties(ax, no_grid=False)
        ax.set_xlim(time_limits)

        ax.step(x, y, color=colors[0])
        ax.hlines(y=global_statistics["mean elapsed"]
                 ,xmin=time_limits[0], xmax=time_limits[1]
                 ,linewidth=2
                 ,color=colors[2])
        
        ax.text(x=time_limits[0]
               ,y=global_statistics["mean elapsed"] + 50
               ,s=labels["elapsed-binned mean txt"][0]
               ,color=colors[2]
               ,fontsize=labels["elapsed-binned mean txt"][1])

        minutes = mdates.MinuteLocator(interval = 1)
        minutes_fmt = mdates.DateFormatter('%H:%M')

        ax.xaxis.set_major_locator(minutes)
        ax.xaxis.set_major_formatter(minutes_fmt)
       
        ax.tick_params(axis='x', labelrotation=80)

        ax.set_title(labels['elapsed-binned title'][0], fontsize=labels['elapsed-binned title'][1])
        
        ax.set_ylabel(labels["elapsed-binned y"][0], fontsize=labels['elapsed-binned y'][1])
        ax.set_xlabel(labels["elapsed-binned x"][0], fontsize=labels['elapsed-binned x'][1])
        remove_tick_lines('x', ax)
        
        # Second y axis.
        ax_snum = ax.twinx()
        set_axes_common_properties(ax_snum, no_grid=True)
        
        ax_snum.scatter(x_snum, y_snum, color=colors[1], s=50, alpha=0.9)
        
        ax_snum.set_ylabel(labels["elapsed-binned snum y"][0], fontsize=labels['elapsed-binned snum y'][1])
#        ax_snum.xaxis.set_major_locator(minutes)
        ax_snum.xaxis.set_major_formatter(minutes_fmt)     
        ax_snum.tick_params(axis='x', labelrotation=80)
        remove_tick_lines('x', ax_snum)
        remove_tick_lines('y', ax_snum)
        
    except Exception as ex:
        msg = "elapsed_chart failed - {ex}".format(ex=ex)
        log.error(msg)
        return ResultKo(Exception(msg))

    else:
        rv = ResultOk(True) 
    log.info(" <<")   
    return rv

def elapsed_frequency_chart(ax: mp.axes.Axes
                           ,df: pd.DataFrame
                           ,global_statistics:dict
                           ,labels: dict
                           ,colors:List=["#BEE2F0", "#d5d5d5", "#c87607"]) -> ResultValue:
    log = logging.getLogger('elapsed_frequency_chart')
    log.info(" >>")

    rv: ResultValue = ResultKo(Exception("Error"))
    try:
        frequencies = config_elapsed_frequency_chart(df,global_statistics,labels)
        if frequencies.is_in_error():
            msg = "Frequency calculation failure"
            log.error(msg)
            return ResultKo(Exception(msg))
        else:
            y = frequencies()
            x = frequencies().index.astype(str)
        set_axes_common_properties(ax, no_grid=False)
        ax.set_title(labels["elapsed-frequency title"][0], fontsize=labels["elapsed-frequency title"][1])
        ax.set_ylabel(labels["elapsed-frequency y"][0], fontsize=labels["elapsed-frequency y"][1])
        ax.set_xlabel(labels["elapsed-frequency x"][0], fontsize=labels["elapsed-frequency x"][1])

        width = 0.5
        rects = ax.bar(x, y, width=width, color=colors[0], label=labels["elapsed-frequency label"][0])

        autolabel(rects, ax, 1)

        ax.tick_params(axis='both', labelsize=14)
        ax.set_xticklabels(x, rotation=80)

        remove_tick_lines('y', ax)
        remove_tick_lines('x', ax)
    
    except Exception as ex:
        msg = "elapsed_frequency_chart failed - {ex}".format(ex=ex)
        log.error(msg)
        return ResultKo(Exception(msg))
    else:
        rv = ResultOk(True)
    log.info(" <<")
    return rv
 
def transaction_per_second_chart(ax: mp.axes.Axes
                                ,binned_df:pd.DataFrame
                                ,labels: dict
                                ,time_limits:List
                                ,colors:List=["#0000e6", "#d5d5d5", "#c87607"]) -> ResultValue:
    log = logging.getLogger('transaction_per_second_chart')
    log.info(" >>")

    rv: ResultValue = ResultKo(Exception("Error"))
    try:
        x=binned_df.index
        y=binned_df['tps'].values
        
        set_axes_common_properties(ax, no_grid=False)
        ax.set_xlim(time_limits)

        minutes = mdates.MinuteLocator(interval = 1)
        minutes_fmt = mdates.DateFormatter('%H:%M')

        ax.xaxis.set_major_locator(minutes)
        ax.xaxis.set_major_formatter(minutes_fmt)
       
        ax.tick_params(axis='x', labelrotation=80)

        ax.set_title(labels["tps title"][0], fontsize=labels["tps title"][1])
        ax.set_ylabel(labels["tps y"][0], fontsize=labels["tps y"][1])
        ax.set_xlabel(labels["tps x"][0], fontsize=labels["tps x"][1])

        remove_tick_lines('x', ax)
        remove_tick_lines('y', ax)
        
        ax.step(x, y, color=colors[0])
        
    except Exception as ex:
        msg = "transaction_per_second_chart failed - {ex}".format(ex=ex)
        log.error(msg)
        return ResultKo(Exception(msg))
    else:
        rv = ResultOk(True) 
    log.info(" <<")
    return rv    

def config_elapsed_frequency_chart(df:pd.DataFrame
                                  ,global_statistics:dict
                                  ,labels:dict
                                  ,type_of_chart = TypeOfChart.PERCENTAGE) -> ResultValue:
    log = logging.getLogger('config_elapsed_frequency_chart')
    log.info(" >>")

    rv: ResultValue = ResultKo(Exception("Error"))
    y = None
    try:
        # Set the data set to be visualized.
        bin_step = 250

        for idx in [0, 1, 2, 3]:
            cut_bins = list(range(0, global_statistics["max elapsed"], bin_step))
            bin_size = len(cut_bins)

            df['elapsed binned'] = pd.cut(df['elapsed'], bins=cut_bins, right=True)
            frequencies = df['elapsed binned'].value_counts(sort=False)
            if len(cut_bins) > 100:
                log.info("Too many bins {b}".format(b=frequencies[0]))
                bin_step = int(bin_step * 3)
                continue

            # Quality check.
            #assert df.shape[0] == df['elapsed binned'].value_counts().sum(), "The aggregate form must have the same total number of the total num of sample."

            y = None
            if type_of_chart == TypeOfChart.FREQUENCIES:
                y = frequencies
                labels["elapsed-frequency y"]=labels["elapsed-frequency y #"]
            else:
                y = frequencies.apply(lambda row: round((row / df.shape[0]) * 100, 1))
                labels["elapsed-frequency y"]=labels["elapsed-frequency y %"]
            break
        
    except Exception as ex:
        msg = "config_elapsed_frequency_chart failed - {ex}".format(ex=ex)
        log.error(msg)
        return ResultKo(Exception(msg))
    else:
        rv = ResultOk(y)
    log.info(" <<")
    return rv

def quantiles_box_text(labels:dict, global_statistics:dict) -> ResultValue:
    log = logging.getLogger('quantiles_box_text')
    log.info(" >>")

    rv: ResultValue = ResultKo(Exception("Error"))    
    txt = None
    try:
        txt = "{title}\n\n".format(title=labels["quantiles txt title"][0])
        for quantile, value in zip(global_statistics["quantiles elapsed"].index
                                  ,global_statistics["quantiles elapsed"].values):
            txt = txt + "{q}° => {v} ms\n".format(q=int((quantile*100)), v=int(value))
    except Exception as ex:
        msg = "quantiles_box_text failed - {ex}".format(ex=ex)
        log.error(msg)
        return ResultKo(Exception(msg))    
    else:
        rv = ResultOk(txt)
        
    log.info(" <<")
    return rv

def single_chart(chart_name:str, file_name:str, labels:dict) -> ResultValue:
    log = logging.getLogger('single_chart')
    log.info(" >>")
    try:
        chart_name = chart_name.lower()
                  
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
            rv = threads_chart(ax=ax[idx], df=df_dict()["df"], labels=labels, time_limits=time_limits)
        
        elif chart_name == "elapsed":
            rv = elapsed_chart(ax=ax[idx], df=df_dict()["df"], errors_df=df_dict()["errors_df"], labels=labels, time_limits=time_limits)
        
        elif chart_name == "ebinned":
            rv = elapsed_binned_chart(ax[idx]
                                     ,binned_df=df_dict()["binned_elapsed"]
                                     ,global_statistics=global_stats()
                                     ,labels=labels
                                     ,time_limits=time_limits)
        elif chart_name == "tps":
            rv = transaction_per_second_chart(ax=ax[idx]
                                            ,binned_df=df_dict()["binned_elapsed"]
                                            ,labels=labels
                                            ,time_limits=time_limits)
        elif chart_name == "frequency":
            rv = elapsed_frequency_chart(ax=ax[idx]
                                        ,df=df_dict()["df"]
                                        ,global_statistics=global_stats()
                                        ,labels=labels)
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
                     
    log.info(" <<")
    return rv

def create_report(log_file:str, output_path:str, report_file_name:str, report_type:str, report_title:str) -> ResultValue:
    log = logging.getLogger('create_report')
    log.info(" >>")
    rv: ResultValue = ResultKo(Exception("Error"))
    try:
        report_type = report_type.upper()
        if report_type not in ["PDF", "JPG", "PNG"]:
            msg = "Unknown report type: {rt}".format(rt=report_type)
            log.error(msg)
            return ResultKo(Exception(msg))

        df_dict = get_dataframe_from_csv(log_file, ok_codes=ok_codes)
        if df_dict.is_ok() == False:
            return ResultKo(Exception(df_dict.value))

        df = df_dict()["df"]
        df_errors = df_dict()["errors_df"]
        df_binned = df_dict()["binned_elapsed"]

        global_stats = get_global_statistics(df=df_dict()["df"], errors_df=df_dict()["errors_df"])
        time_limits = [global_stats()["test begin"] - pd.Timedelta(minutes=0.5) 
                      ,global_stats()["test end"]   + pd.Timedelta(minutes=0.5)]

        labels = create_label_dict(language="en")()

        fig0 = plt.figure(figsize=(21, 40)) #, constrained_layout=True)

        gs1 = gridspec.GridSpec(6, 2
                               ,figure=fig0
                               ,hspace=0.25
                               ,wspace=0.01 
                               ,height_ratios=[1, 10, 10, 10, 10, 10]
                               ,width_ratios=[10, 2])
        ax = []

        idx = 0
        ax.append(fig0.add_subplot(gs1[0,0]))
        text = picture_title_text(labels["picture_title"][0] if len(report_title) == 0 else report_title,  global_stats())
        if text.is_ok() == True:
            text_box(ax[idx], text(), fontsize=labels["picture_title"][1])

        idx += 1
        ax.append(fig0.add_subplot(gs1[1,0]))
        threads_chart(ax=ax[idx], df=df, labels=labels, time_limits=time_limits)

        idx += 1
        ax.append(fig0.add_subplot(gs1[1, 1]))
        stats_txt = statistics_box_text(global_stats(), labels)
        if text.is_ok() == True:
            text_box(ax[idx]
                    ,stats_txt()
                    ,fontsize=labels['statistics'][1]
                    ,y=0.35
                    ,x=0.2
                    ,colors=["#e5e5e5", "#000000", "#000000"])

        idx += 1
        ax.append(fig0.add_subplot(gs1[2,0]))
        elapsed_chart(ax=ax[idx], df=df, errors_df=df_errors, labels=labels, time_limits=time_limits)

        idx += 1
        ax.append(fig0.add_subplot(gs1[3,0]))
        elapsed_binned_chart(ax[idx]
                            ,binned_df=df_binned
                            ,global_statistics=global_stats()
                            ,labels=labels
                            ,time_limits=time_limits)

        idx += 1
        ax.append(fig0.add_subplot(gs1[4,0]))
        transaction_per_second_chart(ax=ax[idx]
                                    ,binned_df=df_binned
                                    ,labels=labels
                                    ,time_limits=time_limits)

        idx += 1
        ax.append(fig0.add_subplot(gs1[5,0]))
        elapsed_frequency_chart(ax=ax[idx]
                               ,df=df
                               ,global_statistics=global_stats()
                               ,labels=labels)

        idx += 1
        ax.append(fig0.add_subplot(gs1[5,1]))

        text = quantiles_box_text(labels, global_stats()) 
        if text.is_ok() == True:
            text_box(ax[idx], text(), fontsize=labels['quantiles'][1], colors=["#e5e5e5", "#000000", "#000000"], x=0.2, y=0.5)    

        sample_date = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = "{prefix}-{fi}.{fmt}".format(fi=report_file_name,fmt=report_type, prefix=sample_date)
        plt.savefig(os.path.join(output_path ,file_name)
                   ,format=report_type
                   ,bbox_inches='tight'
                   ,pad_inches=0.5)

    except Exception as ex:
        log.error("Exception (create_report) caught - {ex}".format(ex=ex))
        rv = ResultKo(ex)
    else:
        rv: ResultValue = ResultOk(True)

    log.info(" <<")
    return rv

def main(args: argparse.Namespace) -> ResultValue:
    log = logging.getLogger('Main')
    log.info(" >>")
    rv: ResultValue = ResultKo(Exception("Error"))
    
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
            rv = single_chart(args.chart[0], args.chart[1], labels=labels())
        elif args.report is not None:
            title = "Default title"
            if args.chart_title is not None:
                title = args.chart_title[0]
            rv = create_report(log_file=args.report[0]
                              ,output_path=args.report[1]
                              ,report_file_name=args.report[2]
                              ,report_type=args.report[3]
                              ,report_title=title)

    except Exception as ex:
        log.error("Exception caught - {ex}".format(ex=ex))
        rv = ResultKo(ex)
    log.info(" ({rv}) <<".format(rv=rv))
    return rv

if __name__ == "__main__":
    print("Starting ...")
    init_logger('/Users/ERIZZAG5J/Work/tmp', "jmeter-report.log",log_level=logging.DEBUG, std_out_log_level=logging.WARNING
               ,disable_logging=["urllib3.connectionpool"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_frame", "-df", nargs=1,
                        help="Create a dataframe from JMeter log file.")
    parser.add_argument("--chart_title", "-t", nargs=1,
                        help="The title to be printed on the chart.")
    parser.add_argument("--chart", "-c", nargs=2,
                        help="Create the named chart using the given file name [thread|elapsed|ebinned|frequency].")
    parser.add_argument("--report", "-r", nargs=4,
                        help="log file, output path, report file name,[PDF|JPG|PNG]. Create the report using the given log file and save it in the given path.")
    args = parser.parse_args()

    rv = main(args)

    ret_val = os.EX_OK if rv.is_ok() == True else os.EX_USAGE
    sys.exit(ret_val)
