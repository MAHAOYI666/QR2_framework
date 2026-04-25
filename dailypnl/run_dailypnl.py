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

def scale_to_book(positions, booksize=1.0):
    """
    将每日持仓数据缩放到指定规模（booksize）。
    确保每日多头持仓（正值）总和为booksize，空头持仓（负值）总和为-booksize。

    参数:
        positions (np.ndarray): 二维数组，形状为(day, stock)，表示每日各股票的持仓。
        booksize (float): 目标规模值，默认为1.0。

    返回:
        np.ndarray: 缩放后的持仓数组，形状与输入相同。
    """
    scaled_positions = np.zeros_like(positions)
    
    for day_idx in range(positions.shape[0]):
        daily_pos = positions[day_idx, :]
        
        # 分离多头和空头持仓
        long_pos = daily_pos[daily_pos > 0]
        short_pos = daily_pos[daily_pos < 0]
        
        # 计算当日多头和空头的总持仓绝对值
        long_sum = np.nansum(long_pos)
        short_sum = -np.nansum(short_pos)
        
        # 避免除零错误：如果某方向无持仓，则缩放后该方向仍为0
        long_scale = booksize / long_sum if long_sum > 0 else 0.0
        short_scale = booksize / short_sum if short_sum > 0 else 0.0
        # 应用缩放
        scaled_daily_pos = np.where(daily_pos > 0, daily_pos * long_scale,
                                   np.where(daily_pos < 0, daily_pos * short_scale, 0))
        
        scaled_positions[day_idx, :] = scaled_daily_pos
    
    return scaled_positions

def pnl(factor,ashare_data_path="/root/CacheData/AshareCache",booksize=1e7, tradecost_ratio=0.0):
    """
    计算多空策略的每日PnL
    
    参数:
        factor: 因子数据
        ashare_data_path: 数据路径
        booksize: 资金规模
        tradecost_ratio: 手续费比例系数(0-1)，实际手续费 = tradecost_ratio × 0.003
    """
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
    x_masked = pro_y(torch.tensor(x),start_time,end_time,ashare_data_path).numpy()
    label_1d_masked = pro_y(torch.tensor(label_1d),start_time,end_time,ashare_data_path).numpy()
    
    x_cov = (~np.isnan(x_masked) & ~np.isnan(label_1d_masked)).sum(axis=1).astype(float)
    label_cov = (~np.isnan(label_1d_masked)).sum(axis=1).astype(float)
    label_cov[label_cov==0] = np.nan
    coverage = x_cov/label_cov
    
    longnum = (x_masked > 0).sum(axis = 1)
    shortnum = (x_masked < 0).sum(axis = 1)
    x_masked_scale2book = scale_to_book(x_masked,booksize)
    
    tvr = np.sum(np.abs(x_masked_scale2book[1:] - x_masked_scale2book[:-1]),axis = 1)/ (booksize*2)
    tvr = np.insert(tvr,0,0)
    pnl_gross = np.nansum(x_masked_scale2book * label_1d_masked,axis = 1)
    
    # 计算手续费：交易金额 × 千三 × tradecost_ratio
    tradecost = tvr * booksize * 2 * 0.003 * tradecost_ratio
    pnl_net = pnl_gross - tradecost
    
    table = np.stack([pnl_net,pnl_gross,tradecost,tvr,longnum,shortnum,coverage],axis=-1)
    table = pd.DataFrame(table,index=date_idx,columns=["pnl","pnl_gross","tradecost","tvr","longnum","shortnum","coverage"])
    return table

def longindex_pnl(factor,ashare_data_path="/root/CacheData/AshareCache",booksize=1e7, long_ratio = 0.1,tradecost_ratio=0.0):
    """
    计算纯多头策略的每日PnL
    
    参数:
        factor: 因子数据
        ashare_data_path: 数据路径
        booksize: 资金规模
        long_ratio: 做多比例，默认0.1（前10%）
        tradecost_ratio: 手续费比例系数(0-1)，实际手续费 = tradecost_ratio × 0.003
    """
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
    x_masked = pro_y(torch.tensor(x),start_time,end_time,ashare_data_path).numpy()
    label_1d_masked = pro_y(torch.tensor(label_1d),start_time,end_time,ashare_data_path).numpy()
    
    x_cov = (~np.isnan(x_masked) & ~np.isnan(label_1d_masked)).sum(axis=1).astype(float)
    label_cov = (~np.isnan(label_1d_masked)).sum(axis=1).astype(float)
    label_cov[label_cov==0] = np.nan
    coverage = x_cov/label_cov
    
    cutoff = np.nanpercentile(x_masked, 100 * (1 - long_ratio), axis=1, keepdims=True)
    x_long = np.where(x_masked > cutoff, x_masked, 0)
    x_long = scale_to_book(x_long,booksize)
    longnum = np.sum(x_long > 0, axis = 1)
    tvr = np.sum(np.abs(x_long[1:] - x_long[:-1]),axis = 1)/ (booksize)
    tvr = np.insert(tvr,0,0)
    longpnl_gross = np.nansum(x_long * label_1d_masked,axis = 1)
    
    # 纯多头只有单边交易
    tradecost = tvr * booksize * 0.003 * tradecost_ratio
    longpnl_net = longpnl_gross - tradecost

    indexwt = Memmaper2(f"{ashare_data_path}/1d_IndexWeight/IndexWeight.000905.SH").load(start_ds=start_time,end_ds=end_time,df_type=True).dloc[:].values
    indexwt = pro_y(torch.tensor(indexwt),start_time,end_time,ashare_data_path).numpy()
    index_portfolio = scale_to_book(indexwt,booksize)
    # longindexpnl = longpnl - np.nansum(indexwt * label_1d_masked,axis = 1)
    index_pnl = np.nansum(index_portfolio * label_1d_masked, axis=1)
    longindexpnl_net = longpnl_net - index_pnl

    table = np.stack([longpnl_net,longpnl_gross,longindexpnl_net,tradecost,tvr,longnum,coverage],axis=-1)
    table = pd.DataFrame(table,index=date_idx,columns=["longpnl","longpnl_gross","longindexpnl","tradecost","tvr","longnum","coverage"])
    return table

def loader(factor_path):
    factor = None
    if factor_path.endswith(".parquet"):
        factor = pd.read_parquet(factor_path)
    else:
        factor = Memmaper2(factor_path).load(start_ds=None,end_ds=None,df_type=True).dloc[:]
    return factor

def run(factor_path,produce_mode=False,dump=True,tradecost_ratio=0.0):
    """
    运行dailypnl分析
    
    参数:
        factor_path: 因子文件路径
        produce_mode: 是否为生产模式
        dump: 是否保存结果
        tradecost_ratio: 手续费比例系数(0-1)，实际手续费 = tradecost_ratio × 0.003
                        默认0表示无手续费，1表示全额千三手续费，0.25表示25%手续费
    """
    ashare_data_path="/root/CacheData/AshareCache"
    if produce_mode:
        ashare_data_path = os.path.join(Cache_Path, "AshareCache")
    factor = loader(factor_path)
    
    daily_pnl = pnl(factor,ashare_data_path,tradecost_ratio=tradecost_ratio)
    daily_longpnl = longindex_pnl(factor,ashare_data_path,tradecost_ratio=tradecost_ratio)

    if dump:
        daily_pnl.to_csv(f"{os.path.dirname(factor_path)}/daily_pnl.tsv",sep="\t")
        daily_longpnl.to_csv(f"{os.path.dirname(factor_path)}/daily_longpnl.tsv",sep="\t")
    # return daily_ic,daily_longpnl
    return daily_pnl,daily_longpnl
    
def main():
    parser = argparse.ArgumentParser(description="Daily PnL Analysis")
    parser.add_argument("factor_path", type=str, help="Factor file path")
    parser.add_argument("--produce_mode", action="store_true", default=False, help="Production mode")
    parser.add_argument("--no_dump", action="store_true", default=False, help="Do not save results")
    parser.add_argument("--tradecost_ratio", type=float, default=0.0, 
                       help="Transaction cost ratio (0-1), actual cost = ratio × 0.003, default=0.0")
    args = parser.parse_args()
    run(args.factor_path, produce_mode=args.produce_mode, dump=not args.no_dump, tradecost_ratio=args.tradecost_ratio)
    


if __name__ == "__main__":
    main()
