# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
from gm.model.storage import context
from gm.model import DictLikeAlgoOrder
from gm.pb.account_pb2 import AlgoOrder
import numpy as np
import pandas as pd
from datetime import  datetime,timezone, timedelta
import datetime
#from sklearn import svm
import copy
import pytz
import datetime
#import alphalens
import numpy as np
import pandas as pd
#import statsmodels.api as sm
#import statsmodels.tsa.stattools as sttools
#from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")
'''
杜邦分析法
1)季-扣非ROE>15%
2)月-现金收入比率>1
3)季-毛利率在行业前10%
4)月-净利润现金含量大于90%
5)月-资产周转(存货and应收账款)都大于前30%
6)季-现金/带息债务 比率>0.5
'''
def init(context):
    # 股票池
    context.symbols_pond = 'SHSE.000300'
    # 设置开仓的最大资金量
    context.ratio = 0.9    
    #扣非ROE
    context.ROE_cut_ratio=15
    #现金收入比率
    context.calsh_income_ratio=1
    #毛利率在行业前几（%）
    context.sale_ratio=0.1
    #净利润现金含量大于(%)
    context.net_cash_in=90
    #资产周转(存货and应收账款)都大于前(%)``
    context.exchange_goods=0.4
    context.exchange_income=0.4
    #现金/带息债务 比率
    context.cash_debt=0.5
    schedule(schedule_func=algo, date_rule='1d', time_rule='09:30:00')
    
def algo(context):
    now = context.now
    now_str = now.strftime('%Y-%m-%d')  
    last_day = get_previous_n_trading_dates(exchange='SHSE', date=now_str, n=1)[0]

    if  pd.Timestamp(last_day).month != now.month :
        symbols_A=stk_get_index_constituents(index=context.symbols_pond , trade_date=last_day)['symbol'].tolist()
        df=stk_get_finance_deriv_pt(symbols_A, fields=['sale_gpm','inv_turnover_rate','acct_rcv_turnover_rate','roe_cut','net_cf_oper_np','int_debt'], rpt_type=12, data_type=None, date=last_day, df=True)

        symbols_B = dylan_cashincome(symbols_A,last_day)                         #现金收入比率>1
        print('现金收入比率剩下{}个股票'.format(len(symbols_B)))  
        symbols_C = dylan_roe(symbols_B,df)                              #扣非ROE>15%
        print('扣非ROE剩下{}个股票'.format(len(symbols_C)))
        symbols_D ,df_all= dylan_sale(symbols_C,last_day,symbols_A,df)                       #毛利率在行业前10% 
        print('毛利率剩下{}个股票'.format(len(symbols_D))) 
        symbols_E = dylan_invturn(symbols_D,last_day,df_all)           #资产周转行业（存货and应收账款）前30%
        print('存货周转剩下{}个股票'.format(len(symbols_E)))         
        symbols_F = dylan_acctturn(symbols_E,last_day,df_all)            #资产周转行业（存货and应收账款）前30%
        print('应收账款周转剩下{}个股票'.format(len(symbols_F)))       
        symbols_G=dylan_net(symbols_F,last_day,df)                              #净利润现金含量大于1
        print('净利润现金含量剩下{}个股票'.format(len(symbols_G)))
        symbols_H=dylan_cashflow(symbols_G,last_day,df)                              #现金/带息债务 比率>0.5
        print('现金债务比率剩下{}个股票'.format(len(symbols_H)))


        positions = get_position('db9caf2a-5014-11f0-bfb8-00163e022aa6')
        for position in positions:
            symbol = position['symbol']
            if symbol not in symbols_H:
                new_price = history_n(symbol=symbol, frequency='1d', count=1, end_time=context.now, fields='open', adjust=ADJUST_PREV, adjust_end_time=context.backtest_end_time, df=False)[0]['open']
                order_info = order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Limit,position_side=PositionSide_Long,price=new_price)
        percent = context.ratio / len(symbols_H)
        for symbol in symbols_H:
            new_price = current(symbols=symbol)[0]['price']
            order_info = order_target_percent(symbol=symbol, percent=percent, order_type=OrderType_Limit,position_side=PositionSide_Long,price=new_price)

def dylan_resort(group, field, ratio):
    threshold = group[field].quantile(1 - ratio)  
    return group[group[field] >= threshold]



def dylan_cashincome(X,last_day):
        df1=stk_get_fundamentals_cashflow_pt(X, rpt_type=12, data_type=None, date=last_day, fields='cash_rcv_sale', df=True)
        df2=stk_get_fundamentals_income_pt(X, rpt_type=12, data_type=None, date=last_day, fields='inc_oper', df=True)
        df=pd.merge(df1, df2, on='symbol')[['symbol','pub_date_x','cash_rcv_sale','inc_oper']]
        df['cash_ratio']=df['cash_rcv_sale']/df['inc_oper']
        Y=df[df['cash_ratio']>context.calsh_income_ratio]['symbol'].tolist()
        return Y

def dylan_roe(X,df):
    df_B=df[df['symbol'].isin(X)]
    df_roe=df_B[df_B['roe_cut']>context.ROE_cut_ratio]
    Y=df_roe['symbol'].tolist()
    return Y

def dylan_sale(X,last_day,symbols_A,df):
    df_C=df[df['symbol'].isin(X)]
    df_industry=stk_get_symbol_industry(symbols_A, source='sw2021', level=2, date=last_day)
    df_all= df_C.merge(df_industry, on='symbol')[['symbol','industry_code','sale_gpm','inv_turnover_rate','acct_rcv_turnover_rate']]
    df_resort= df_all.groupby('industry_code').apply(lambda x: dylan_resort(x, 'sale_gpm', context.sale_ratio)).reset_index(drop=True)
    Y= [symbol for symbol in X if symbol in df_resort['symbol'].values]
    return Y , df_all

def dylan_invturn(X,last_day,df_all):
    df_resort= df_all.groupby('industry_code').apply(lambda x: dylan_resort(x, 'inv_turnover_rate', context.exchange_goods)).reset_index(drop=True)
    Y = [symbol for symbol in X if symbol in df_resort['symbol'].values]
    return Y

def dylan_acctturn(X,last_day,df_all):
    df_resort= df_all.groupby('industry_code').apply(lambda x: dylan_resort(x, 'acct_rcv_turnover_rate', context.exchange_income)).reset_index(drop=True)
    Y = [symbol for symbol in X if symbol in df_resort['symbol'].values]
    return Y

def dylan_net(X,last_day,df):
    df_G=df[df['symbol'].isin(X)]
    Y=df_G[df_G['net_cf_oper_np']>context.net_cash_in]['symbol'].tolist()
    return Y

def dylan_cashflow(X,last_day,df):
        df1=stk_get_fundamentals_balance_pt(X, rpt_type=12, data_type=None, date=last_day, fields='mny_cptl', df=True)
        #df2=stk_get_finance_deriv_pt(X, fields='int_debt', rpt_type=12, data_type=None, date=last_day, df=True)
        
        df_all=pd.merge(df1, df, on='symbol')[['symbol','pub_date_x','mny_cptl','int_debt']]
        df_all['safe']=df_all['mny_cptl']/df_all['int_debt']
        Y=df_all[df_all['safe']>context.cash_debt]['symbol'].tolist()
        return Y





def on_order_status(context, order):
    # 标的代码
    symbol = order['symbol']
    # 委托价格
    price = order['price']
    # 委托数量
    volume = order['volume']
    # 目标仓位
    target_percent = order['target_percent']
    # 查看下单后的委托状态，等于3代表委托全部成交
    status = order['status']
    # 买卖方向，1为买入，2为卖出
    side = order['side']
    # 开平仓类型，1为开仓，2为平仓
    effect = order['position_effect']
    # 委托类型，1为限价委托，2为市价委托
    order_type = order['order_type']
    if status == 3:
        if effect == 1:
            if side == 1:
                side_effect = '开多仓'
            else:
                side_effect = '开空仓'
        else:
            if side == 1:
                side_effect = '平空仓'
            else:
                side_effect = '平多仓'
        order_type_word = '限价' if order_type==1 else '市价'
        print('{}:标的：{}，操作：以{}{}，委托价格：{}，目标仓位：{:.2%}'.format(context.now,symbol,order_type_word,side_effect,price,target_percent))


def on_backtest_finished(context, indicator):
    print('*'*50)
    print('回测已完成，请通过右上角“回测历史”功能查询详情。')



if __name__ == '__main__':
    '''
        strategy_id策略ID, 由系统生成
        filename文件名, 请与本文件名保持一致
        mode运行模式, 实时模式:MODE_LIVE回测模式:MODE_BACKTEST
        token绑定计算机的ID, 可在系统设置-密钥管理中生成
        backtest_start_time回测开始时间
        backtest_end_time回测结束时间
        backtest_adjust股票复权方式, 不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
        backtest_initial_cash回测初始资金
        backtest_commission_ratio回测佣金比例
        backtest_slippage_ratio回测滑点比例
        backtest_match_mode市价撮合模式，以下一tick/bar开盘价撮合:0，以当前tick/bar收盘价撮合：1
        '''
    run(strategy_id='0496e73a-84ac-11f0-82c3-f46b8c5ab5bc',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='c6ed6b1252a6bf9b594592708d5e30389b84334b',
        backtest_start_time='2024-01-01 08:00:00',
        backtest_end_time='2025-09-01 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0007,
        backtest_slippage_ratio=0.00123,
        backtest_match_mode=1)

