import os
import re
import sys
import csv
import json
import logging
import argparse
import codecs
import boto3
from typing import List

from logger_init import init_logger
from result_value import ResultKo, ResultOk, ResultValue

job_config = {
    "encoding": 'utf-8'
   ,"delimiter": '#'
   ,"thread_name_col": 'threadName'
   ,"row_print_interval": 5000
}

def getRow(line:str)-> ResultValue:
    log = logging.getLogger('getRow')
    #log.info(" >>")    
    rv:ResultValue = ResultKo(Exception("Error"))
    try:
        row = line.split(job_config["delimiter"])
        rv = ResultOk(row)
    except Exception as ex:
        log.error("Exception caught - {ex}".format(ex=ex))
        rv = ResultKo(ex)
    #log.info(" <<")
    return rv

def getS3Handler(auth_profile:str, bucket_name:str, file_name:str)-> ResultValue:
    log = logging.getLogger('readAndMerge')
    log.info(" >>")
    rv:ResultValue = ResultKo(Exception("Error"))
    try:
        session = boto3.Session(profile_name=auth_profile)
        s3 = session.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=file_name)
        rv = ResultOk(response)
        
    except Exception as ex:
        log.error("Exception caught - {ex}".format(ex=ex))
        rv = ResultKo(ex)

    log.info(" <<")
    return rv

def readAndMergeFromS3(s3_conf:dict
                      ,file_name:str
                      ,write_headers:bool=False
                      ,regExp:re.Pattern=None 
                      ,prefix:str=None
                      ,csv_writer=None) -> ResultValue :
    log = logging.getLogger('readAndMerge')
    log.info(" >>")
    rv:ResultValue = ResultKo(Exception("Error"))
    try:
        #bucket_name = s3_conf["bucket"]
        #session = boto3.Session(profile_name=s3_conf["auth_profile"])
        #s3 = session.client('s3')
        #response = s3.get_object(Bucket=bucket_name, Key=file_name)

        response = getS3Handler(auth_profile=s3_conf["auth_profile"], bucket_name=s3_conf["bucket"], file_name=file_name)
        first = True

        row_counter = 0
        row_total = 0

        for idx, line in enumerate(response()['Body'].iter_lines()):
            do_write = True
            
            line = line.decode('utf-8')
            row = getRow(line)
            if row.is_in_error():
                return rv
            
            row_counter += 1
            if row_counter >= job_config["row_print_interval"]:
                row_total += row_counter
                row_counter = 0
                msg = "Rows written: {n}".format(n=row_total)
                print(msg)
                log.info(msg)

            if first == True:
                first = False
                tn_index = row().index(job_config["thread_name_col"])
                log.info("Thread name column index: {ci}".format(ci=tn_index))
                if write_headers == True:
                    csv_writer.writerow(row())
                    continue
            if regExp is not None:
                row_str = row()[tn_index]
                if regExp.match(row_str) is None:
                    do_write = False            
            if do_write == True:
                thread_name = "{prefix}{current_tn}".format(prefix=prefix, current_tn=row()[tn_index])
                row()[tn_index] = thread_name
                csv_writer.writerow(row())
            log.debug(line)
        log.info("Read completed.")
        
        msg = "Total rows written: {n}".format(n=row_counter + row_total)
        print(msg)
        log.info(msg)

        rv = ResultOk("")

    except Exception as ex:
        log.error("Exception caught - {ex}".format(ex=ex))
        rv = ResultKo(ex)

    log.info(" <<")
    return rv

def readAndMerge(in_file:str
                ,prefix:str
                ,write_headers:bool = False
                ,regExp:re.Pattern = None
                ,csv_writer=None) -> ResultValue :
    log = logging.getLogger('readAndMerge')
    log.info(" >>")
    rv:ResultValue = ResultKo(Exception("Error"))
    try:
        with codecs.open(filename = in_file, mode = "r", encoding = job_config["encoding"]) as fp:
            reader = csv.reader(fp ,delimiter = job_config["delimiter"] ,quotechar = '"')
            
            headers = next(reader) 
            if write_headers == True:
                csv_writer.writerow(headers)

            tn_index = headers.index(job_config["thread_name_col"])
            for row in reader:
                do_write = True
                if regExp is not None:
                    row_str = row[tn_index]
                    if regExp.match(row_str) is None:
                        do_write = False
                if do_write == True:
                    thread_name = "{prefix}{current_tn}".format(prefix=prefix, current_tn=row[tn_index])
                    row[tn_index] = thread_name
                    csv_writer.writerow(row)

        rv = ResultOk("OK")

    except Exception as ex:
        log.error("Exception caught - {ex}".format(ex=ex))
        rv = ResultKo(ex)

    log.info(" <<")
    return rv

def main( args:argparse.Namespace ) -> ResultValue :
    log = logging.getLogger('Main')
    log.info(" >>")
    rv:ResultValue = ResultKo(Exception("Error"))
    try:
        with open(file = args.jm_config, mode = "r", encoding = job_config["encoding"]) as json_file: 
            config = json.load(json_file)
            with open(file = config["merge_file"], mode = "w", encoding = job_config["encoding"]) as mfp:
                writer = csv.writer(mfp, delimiter=job_config["delimiter"])
                first = True
                    
                for element in config["file_list"]:
                    regExp = element.get("regexp")
                    if regExp is not None:
                        regExp = re.compile(regExp)

                    msg = "Working on file: {fi}".format(fi=element["file"])
                    print(msg)
                    log.warning(msg)

                    if config.get("s3") is None:
                        rv = readAndMerge(element["file"]
                                         ,element["thread_prefix"]
                                         ,write_headers=first
                                         ,regExp = regExp
                                         ,csv_writer=writer)
                        if rv.is_in_error():
                            return rv
                    else:
                        rv = readAndMergeFromS3(config.get("s3")
                                               ,file_name=element["file"]
                                               ,write_headers=first
                                               ,prefix=element["thread_prefix"]
                                               ,regExp = regExp
                                               ,csv_writer=writer)
                        if rv.is_in_error():
                            return rv

                        pass
                    if rv.is_in_error():
                        return rv
                    if first == True:
                        first = False
        pass                
    except Exception as ex:
        log.error("Exception caught - {ex}".format(ex=ex))
        rv = ResultKo(ex)
    log.info(" ({rv}) <<".format(rv=rv.is_ok()))
    return rv

if __name__ == "__main__":
    init_logger('/Users/ERIZZAG5J/Work/tmp', "jmeter-merge.log",log_level=logging.ERROR, std_out_log_level=logging.ERROR
               ,disable_logging=["botocore.auth"
                                ,"botocore.endpoint"
                                ,"botocore.parsers"
                                ,"botocore.retryhandler"
                                ,"botocore.session"
                                ,"botocore.loaders"
                                ,"botocore.client"
                                ,"botocore.utils"
                                ,"botocore.hooks"
                                ,"botocore.credentials"
                                ,"urllib3.connectionpool"])
    parser = argparse.ArgumentParser()
    parser.add_argument("jm_config", help="The configuration file.")
    args = parser.parse_args()
    
    rv = main(args)

    ret_val = os.EX_OK if rv.is_ok() == True else os.EX_USAGE
    sys.exit(ret_val)
