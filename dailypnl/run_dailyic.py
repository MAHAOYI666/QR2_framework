from factorsim.config import NAN_DTYPE, Cache_Path
from factorsim import fast,Memmaper2
from factorsim import operator
import pandas as pd
import numpy as np
import os,argparse
import torch
import pdb

def pro_y(y,start_ds,end_ds,ashare_data_path):
    """
    y 是vwap30_label1d,
    计算方式为vwap30.pct_change(1,fill_method=None).shift(-2)
    提前关闭不在universe和涨跌停的收益
    """
    y = fast.purify(y)
    y = operator.baseUniMask(y,start_ds=start_ds,end_ds=end_ds,path=ashare_data_path,shift_n=-1)
    y = operator.trdMask(y,start_ds=start_ds,end_ds=end_ds,path=ashare_data_path,shift_n=-1)
    return y

def get_backtest_label(ashare_data_path,period,start_time,end_time):
    """
    device: torch.device
    """        
    label = Memmaper2(f"{ashare_data_path}/1d_DailyLabel/DailyLabel.vwap30_label{period}").load(start_ds=start_time,
                                                                                                end_ds=end_time,
                                                                                                df_type=True).dloc[:]

    label[np.isnan(label)]=NAN_DTYPE
    return label

def safe_to_numpy(data):
    """安全地将各种数据类型转换为NumPy数组"""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, (list, tuple)):
        return np.array(data)
    else:
        return data

def _calculate_ic(factor,ashare_data_path="/root/CacheData/AshareCache"):
    if factor.index.nlevels>1:
        factor = factor.reset_index('times',drop=True).sort_index()
    date_idx = factor.index
    end_time = min(date_idx[-1],20240101)
    date_idx = date_idx[date_idx<end_time]
    start_time = date_idx[0]
    end_time = date_idx[-1]
    factor = factor.reindex(index=date_idx)
    x = factor.values
    # IS END

    label_1d = get_backtest_label(ashare_data_path,"1d",start_time,end_time).reindex(index=date_idx).values
    label_5d = get_backtest_label(ashare_data_path,"5d",start_time,end_time).reindex(index=date_idx).values
    
    x_masked = pro_y(torch.tensor(x),start_time,end_time,ashare_data_path).numpy()
    label_1d_masked = pro_y(torch.tensor(label_1d),start_time,end_time,ashare_data_path).numpy()
    label_5d_masked = pro_y(torch.tensor(label_5d),start_time,end_time,ashare_data_path).numpy()

    ic_1d = safe_to_numpy(fast.corr(x_masked,label_1d_masked,dim=-1,keepdims=True))
    ic_perc = safe_to_numpy(fast.corr(fast.perc_long(x_masked),fast.rank(label_1d_masked,dim=-1),dim=-1,keepdims=True))
    ic_rank = safe_to_numpy(fast.corr(fast.rank(x_masked,dim=-1),fast.rank(label_1d_masked,dim=-1),dim=-1,keepdims=True))
    ic_5d = safe_to_numpy(fast.corr(x_masked,label_5d_masked,dim=-1,keepdims=True))
    x_cov = (~np.isnan(x_masked) & ~np.isnan(label_1d_masked)).sum(axis=1).astype(float)
    label_cov = (~np.isnan(label_1d_masked)).sum(axis=1).astype(float)
    label_cov[label_cov==0] = np.nan
    coverage = x_cov/label_cov
    # coverage = ((~factor.isna()) & (abs(factor) > 1e-6)).sum(axis = 1).values[:,np.newaxis]
    daily_ic = np.concatenate([ic_1d,ic_5d,ic_rank,ic_perc,coverage[:,np.newaxis]],axis=-1)
    daily_ic = pd.DataFrame(daily_ic,index=date_idx,columns=["ic","5dic","rankic","percic","coverage"])
    return daily_ic

def loader(factor_path):
    factor = None
    if factor_path.endswith(".parquet"):
        factor = pd.read_parquet(factor_path)
    else:
        factor = Memmaper2(factor_path).load(start_ds=None,end_ds=None,df_type=True).dloc[:]
    return factor

def calculate_ic(factor_path,produce_mode=False,dump=True):
    ashare_data_path="/root/CacheData/AshareCache"
    if produce_mode:
        ashare_data_path = os.path.join(Cache_Path, "AshareCache")
    factor = loader(factor_path)
    daily_ic = _calculate_ic(factor,ashare_data_path)
    outpath = f"{os.path.dirname(factor_path)}/daily_ic"
    if dump:
        daily_ic.to_csv(outpath,sep="\t",na_rep='NAN')
        # pd.read_csv("/root/personal_data/pool_analysis/dump/bp_20250911_01/daily_ic",sep="\t",index_col = 0)
    return daily_ic

    
def main():
    parser = argparse.ArgumentParser(description="factorsim Command Line Interface")
    parser.add_argument("argument", type=str, nargs='+', help="User input")
    args = parser.parse_args()
    calculate_ic(*args.argument)
    


if __name__ == "__main__":
    main()
