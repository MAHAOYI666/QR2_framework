# author: sli
# usage: python3 simsummary.py [daily_ic.parquet] --sdate XXXXXXXX --edate XXXXXXXX

import os 
import numpy as np
import pandas as pd
import pdb
from optparse import OptionParser
from datetime import datetime

def simsum(df,sdate = "20160101",edate = "20250630"):
    if isinstance(df,str):
        df = pd.read_csv(df,sep="\t",index_col = 0)
    df = df.astype(float)
    if df.index.inferred_type == 'integer':
        df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d', errors='coerce')
    df = df[df.index >= sdate]
    df = df[df.index <= edate]

    if "coverage" in df.columns:
        coverage = df['coverage']
        df = df.iloc[:,:-1]
    
    yearly_ic = df.groupby(df.index.year).mean()
    yearly_ir = df.groupby(df.index.year).mean()/df.groupby(df.index.year).std()
    yearly_ir.columns=[x+'IR'for x in yearly_ir.columns]

    yearly_cov = coverage.groupby(coverage.index.year).mean().round(4)
    stats = pd.concat([yearly_ic,yearly_ir,yearly_cov],axis = 1)
    stats['days'] = coverage.groupby(coverage.index.year).apply(lambda x: (x > 0).sum())

    yearly_st = df.groupby(df.index.year).apply(lambda x: x.index.min()).dt.strftime('%Y%m%d')
    yearly_end = df.groupby(df.index.year).apply(lambda x: x.index.max()).dt.strftime('%Y%m%d')
    stats.index = [f"{start}-{end}" for start, end in zip(yearly_st, yearly_end)]

    # global stat
    global_ic = df.mean().to_frame().T
    global_ir = (df.mean()/df.std()).to_frame().T
    global_ir.columns=[x+'IR'for x in global_ir.columns]
    global_cov = coverage.mean().round(4)
    global_stats = pd.concat([global_ic, global_ir], axis=1)
    global_stats['coverage'] = global_cov
    global_stats['days'] = (coverage>0).sum()
    global_stats.index = [f"{df.index.min().strftime('%Y%m%d')}-{df.index.max().strftime('%Y%m%d')}"]
    stats = pd.concat([stats,global_stats])
    return stats.round(4)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-s", "--sdate", dest="sdate", default="20160101", type="str", help="startdate")
    parser.add_option("-e", "--edate", dest="edate", default="20250701", type="str", help="enddate")

    (options, args) = parser.parse_args()
    sdate = datetime.strptime(options.sdate, '%Y%m%d')
    edate = datetime.strptime(options.edate, '%Y%m%d')
    file = os.path.abspath(args[0])

    dailyictable = pd.read_csv(file,sep="\t",index_col = 0)
    print(simsum(dailyictable))
    # yearly stat

