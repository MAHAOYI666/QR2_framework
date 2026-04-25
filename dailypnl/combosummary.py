# usage: python3 combosummary.py [daily_pnl.tsv or daily_longpnl.tsv] --mode pnl|longpnl --sdate 20160101 --edate 20250701 --booksize 1e7

import os
import numpy as np
import pandas as pd
from optparse import OptionParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def max_drawdown(cum_pnl):
    """
    计算最大回撤（正值，0~1 之间）
    参数:
        cum_pnl (pd.Series): 累计收益序列
    返回:
        (float, pd.Series): 最大回撤值, 回撤序列
    """
    # 构造资金曲线，从 1 开始
    equity = 1 + cum_pnl
    roll_max = equity.cummax()
    # 正向回撤：相对历史峰值的跌幅比例
    drawdown = (roll_max - equity) / roll_max
    return drawdown.max(), drawdown


def sharpe_ratio(pnl_series, annualize=True):
    """
    计算Sharpe比率
    参数:
        pnl_series (pd.Series or np.ndarray): 日度PnL序列
        annualize (bool): 是否年化，默认True，年化系数为sqrt(252)
    返回:
        float: Sharpe比率值
    """
    if isinstance(pnl_series, pd.Series):
        pnl_values = pnl_series.values
    else:
        pnl_values = pnl_series
    valid_pnl = pnl_values[~np.isnan(pnl_values)]
    if len(valid_pnl) == 0:
        return np.nan
    mean_pnl = np.mean(valid_pnl)
    std_pnl = np.std(valid_pnl, ddof=1)
    if std_pnl == 0:
        return np.nan
    sharpe = mean_pnl / std_pnl
    if annualize:
        sharpe = sharpe * np.sqrt(252)
    return sharpe


def plot_nav_curve(df, mode="pnl", sdate="20160101", edate="20250701", booksize=1e7, output_path=None):
    """
    绘制净值曲线（不考虑复利），回撤曲线使用右侧Y轴
    参数:
        df (pd.DataFrame): daily_pnl 或 daily_longpnl 数据
        mode (str): 'pnl' 或 'longpnl'
        sdate (str): 起始日期
        edate (str): 结束日期
        booksize (float): 资金规模
        output_path (str): 输出文件路径，如果为None则使用默认路径
    """
    df = process_dates(df, sdate, edate)
    fig, ax1 = plt.subplots(figsize=(14, 6))
    if mode.lower() == "longpnl":
        # longpnl模式：绘制longpnl和longindexpnl
        cum_longpnl = df['longpnl'].cumsum() / booksize
        cum_longindexpnl = df['longindexpnl'].cumsum() / booksize
        
        # 净值曲线
        nav_longpnl = 1 + cum_longpnl
        nav_longindexpnl = 1 + cum_longindexpnl
        
        # 计算回撤
        _, dd_longpnl = max_drawdown(cum_longpnl)
        _, dd_longindexpnl = max_drawdown(cum_longindexpnl)
        
        # 左轴：净值曲线
        line1 = ax1.plot(df.index, nav_longpnl, label='Long PnL', linewidth=1.5, color='steelblue')
        line2 = ax1.plot(df.index, nav_longindexpnl, label='Long vs Index', linewidth=1.5, color='orange')
        ax1.set_ylabel('Net Value', fontsize=12, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 右轴：回撤曲线
        ax2 = ax1.twinx()
        ax2.fill_between(df.index, 0, -dd_longpnl, alpha=0.2, color='red', label='Long PnL DD')
        ax2.fill_between(df.index, 0, -dd_longindexpnl, alpha=0.2, color='darkred', label='Long vs Index DD')
        ax2.set_ylabel('Drawdown', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 图例
        ax1.legend(loc='upper left', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)
        
        ax1.set_title('Net Asset Value & Drawdown Curve (No Compounding)', fontsize=14, fontweight='bold')
        
    else:
        # pnl模式：绘制pnl
        cum_pnl = df['pnl'].cumsum() / booksize
        nav = 1 + cum_pnl
        _, dd = max_drawdown(cum_pnl)
        
        # 左轴：净值曲线
        line1 = ax1.plot(df.index, nav, label='Net Value', linewidth=1.5, color='steelblue')
        ax1.set_ylabel('Net Value', fontsize=12, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 右轴：回撤曲线
        ax2 = ax1.twinx()
        ax2.fill_between(df.index, 0, -dd, alpha=0.3, color='red', label='Drawdown')
        ax2.set_ylabel('Drawdown', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 图例
        ax1.legend(loc='upper left', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)
        
        ax1.set_title('Net Asset Value & Drawdown Curve (No Compounding)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = f'nav_curve_{mode}_{sdate}_{edate}.png'
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, f'nav_curve_{mode}_{sdate}_{edate}.png')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"净值曲线已保存至: {output_path}")


def process_dates(df, sdate, edate):
    """标准化索引并裁剪时间"""
    if df.index.inferred_type == 'integer':
        df.index = pd.to_datetime(df.index.astype(str), format='%Y%m%d', errors='coerce')
    elif not np.issubdtype(df.index.dtype, np.datetime64):
        df.index = pd.to_datetime(df.index)
    return df[(df.index >= pd.to_datetime(sdate)) & (df.index <= pd.to_datetime(edate))]


def summarize_pnl(df, sdate="20160101", edate="20250701", booksize=1e7):
    """普通 pnl 汇总"""
    df = process_dates(df, sdate, edate)
    coverage = df.pop('coverage') if 'coverage' in df.columns else None

    yearly_pnl = df['pnl'].groupby(df.index.year).sum() / booksize
    yearly_mean = df.groupby(df.index.year).mean().round(4)
    # 简化列名：ret 表示该段期间的累计收益（按 booksize 归一）
    yearly_mean['ret'] = yearly_pnl

    yearly_dd = []
    yearly_sharpe = []
    for year, group in df.groupby(df.index.year):
        cum_pnl = group['pnl'].cumsum() / booksize
        dd, _ = max_drawdown(cum_pnl)
        yearly_dd.append(dd)
        sr = sharpe_ratio(group['pnl'], annualize=True)
        yearly_sharpe.append(sr)
    yearly_mean['drawdown'] = yearly_dd
    yearly_mean['sharpe'] = yearly_sharpe

    if coverage is not None:
        yearly_mean['coverage'] = coverage.groupby(coverage.index.year).mean().round(4)
        yearly_mean['days'] = coverage.groupby(coverage.index.year).apply(lambda x: (x > 0).sum())
    else:
        yearly_mean['days'] = df.groupby(df.index.year).size()

    yearly_st = df.groupby(df.index.year).apply(lambda x: x.index.min()).dt.strftime('%Y%m%d')
    yearly_end = df.groupby(df.index.year).apply(lambda x: x.index.max()).dt.strftime('%Y%m%d')
    yearly_mean.index = [f"{start}-{end}" for start, end in zip(yearly_st, yearly_end)]

    # 全历史统计
    global_row = {}
    # 全程年收益率的平均值：对各年度 ret 取均值
    global_row['ret'] = yearly_pnl.mean()
    cum_pnl = df['pnl'].cumsum() / booksize
    global_row['drawdown'], _ = max_drawdown(cum_pnl)
    global_row['sharpe'] = sharpe_ratio(df['pnl'], annualize=True)
    global_row.update(df.mean().to_dict())
    if coverage is not None:
        global_row['coverage'] = coverage.mean().round(4)
        global_row['days'] = (coverage > 0).sum()
    else:
        global_row['days'] = len(df)
    global_df = pd.DataFrame([global_row],
        index=[f"{df.index.min().strftime('%Y%m%d')}-{df.index.max().strftime('%Y%m%d')}"])

    return pd.concat([yearly_mean, global_df]).round(4)


def summarize_longpnl(df, sdate="20160101", edate="20250701", booksize=1e7):
    """longpnl 汇总"""
    df = process_dates(df, sdate, edate)
    coverage = df.pop('coverage') if 'coverage' in df.columns else None

    yearly_longpnl = df['longpnl'].groupby(df.index.year).sum() / booksize
    yearly_longindexpnl = df['longindexpnl'].groupby(df.index.year).sum() / booksize

    yearly_mean = df.groupby(df.index.year).mean().round(4)
    yearly_mean['long_ret'] = yearly_longpnl
    yearly_mean['long_excess_ret'] = yearly_longindexpnl

    yearly_dd = []
    yearly_sharpe = []
    for year, group in df.groupby(df.index.year):
        cum_longpnl = group['longpnl'].cumsum() / booksize
        cum_longindexpnl = group['longindexpnl'].cumsum() / booksize
        dd_longpnl, _ = max_drawdown(cum_longpnl)
        dd_longindexpnl, _ = max_drawdown(cum_longindexpnl)
        yearly_dd.append((dd_longpnl, dd_longindexpnl))
        sr_longpnl = sharpe_ratio(group['longpnl'], annualize=True)
        sr_longindexpnl = sharpe_ratio(group['longindexpnl'], annualize=True)
        yearly_sharpe.append((sr_longpnl, sr_longindexpnl))

    dd_df = pd.DataFrame(yearly_dd, columns=['dd_longpnl', 'dd_longindexpnl'], index=yearly_mean.index)
    sharpe_df = pd.DataFrame(yearly_sharpe, columns=['sharpe_longpnl', 'sharpe_longindexpnl'], index=yearly_mean.index)
    yearly_mean = pd.concat([yearly_mean, dd_df, sharpe_df], axis=1)

    if coverage is not None:
        yearly_mean['coverage'] = coverage.groupby(coverage.index.year).mean().round(4)
        yearly_mean['days'] = coverage.groupby(coverage.index.year).apply(lambda x: (x > 0).sum())
    else:
        yearly_mean['days'] = df.groupby(df.index.year).size()

    yearly_st = df.groupby(df.index.year).apply(lambda x: x.index.min()).dt.strftime('%Y%m%d')
    yearly_end = df.groupby(df.index.year).apply(lambda x: x.index.max()).dt.strftime('%Y%m%d')
    yearly_mean.index = [f"{start}-{end}" for start, end in zip(yearly_st, yearly_end)]

    # 全历史统计
    global_row = {}
    # 全程年收益率的平均值：对各年度 long_ret / long_excess_ret 取均值
    global_row['long_ret'] = yearly_longpnl.mean()
    global_row['long_excess_ret'] = yearly_longindexpnl.mean()

    cum_longpnl = df['longpnl'].cumsum() / booksize
    cum_longindexpnl = df['longindexpnl'].cumsum() / booksize
    global_row['dd_longpnl'], _ = max_drawdown(cum_longpnl)
    global_row['dd_longindexpnl'], _ = max_drawdown(cum_longindexpnl)
    global_row['sharpe_longpnl'] = sharpe_ratio(df['longpnl'], annualize=True)
    global_row['sharpe_longindexpnl'] = sharpe_ratio(df['longindexpnl'], annualize=True)
    global_row.update(df.mean().to_dict())

    if coverage is not None:
        global_row['coverage'] = coverage.mean().round(4)
        global_row['days'] = (coverage > 0).sum()
    else:
        global_row['days'] = len(df)

    global_df = pd.DataFrame([global_row],
        index=[f"{df.index.min().strftime('%Y%m%d')}-{df.index.max().strftime('%Y%m%d')}"])

    return pd.concat([yearly_mean, global_df]).round(4)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", "--mode", dest="mode", default="pnl", type="str",
                      help="summary mode: pnl or longpnl")
    parser.add_option("-s", "--sdate", dest="sdate", default="20160101", type="str", help="startdate")
    parser.add_option("-e", "--edate", dest="edate", default="20250701", type="str", help="enddate")
    parser.add_option("-b", "--booksize", dest="booksize", default=1e7, type="float", help="booksize for scaling pnl")
    parser.add_option("-p", "--plot", dest="plot", action="store_true", default=False,
                      help="plot nav curve and save to file")
    parser.add_option("-o", "--output", dest="output", default=None, type="str",
                      help="output path for plot image")

    (options, args) = parser.parse_args()
    file = os.path.abspath(args[0])
    df = pd.read_csv(file, sep="\t", index_col=0)

    if options.mode.lower() == "longpnl":
        summary = summarize_longpnl(df, sdate=options.sdate, edate=options.edate, booksize=options.booksize)
    else:
        summary = summarize_pnl(df, sdate=options.sdate, edate=options.edate, booksize=options.booksize)

    # 设置 pandas 显示选项以显示所有列
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(summary)
    
    # 绘制净值曲线
    if options.plot:
        df_plot = pd.read_csv(file, sep="\t", index_col=0)
        output_path = options.output
        if output_path is None:
            # 默认保存在与输入文件相同的目录
            output_path = file.replace('.tsv', f'_nav_{options.sdate}_{options.edate}.png')
        plot_nav_curve(df_plot, mode=options.mode, sdate=options.sdate, 
                      edate=options.edate, booksize=options.booksize, output_path=output_path)
