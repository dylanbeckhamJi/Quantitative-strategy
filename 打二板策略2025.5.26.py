# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
def init(context):
    # 1、参数设置
    context.max_mv = 520e8                   # 最大市值
    context.min_mv = 70e8                    # 最小市值
    context.min_amount = 7e8                 # 最小成交额
    context.max_amount = 19e8                # 最大成交额
    context.min_call_rate = 0.00             # 最小开盘涨跌幅
    context.max_call_rate = 0.06           # 最大开盘涨跌幅
    context.max_get_profit_rate = 0.04       # 最大获利比例
    context.zy_min_rate = 0.9                # 左压最小比例
    # 2、变量初始化
    context.n_days_limit_up_list = []        # 涨停股列表
    # 3、设置定时任务
    context.sell_time1 = '11:28:00'
    context.sell_time2 = '14:50:00'
    schedule(schedule_func=algo_1, date_rule='1d', time_rule='09:20:00')
    schedule(schedule_func=buy_algo, date_rule='1d', time_rule='09:28:00')
    schedule(schedule_func=sell_algo, date_rule='1d', time_rule=context.sell_time1)
    schedule(schedule_func=sell_algo, date_rule='1d', time_rule=context.sell_time2)
def algo_1(context):
    print('*'*88)
    unsubscribe(symbols='*', frequency='tick')            # 取消所有订阅
    date = context.now.strftime('%Y-%m-%d')               # 当天日期str

    # 1、获取基础股票池（过滤ST、新股、停牌、退市整理期的股票）
    all_stocks,all_stocks_str = get_normal_stocks(context, date,new_days=50)

    # 2、过滤北交所、科创板、创业板
    all_stocks = [code for code in all_stocks if code.startswith('SHSE.60') or code.startswith('SZSE.0')]

    # 2.1、过滤转债股
    if all_stocks:
        zz = get_symbols(sec_type1=1030, skip_suspended=False, skip_st=False, trade_date=date, df=False)
        underlying_symbols = [data['underlying_symbol'] for data in zz]
        all_stocks = list(set(all_stocks)-set(underlying_symbols))

    # 3、计算昨日涨停股、并剔除前2日涨停股
    # 首次运行，先添加前2天的数据
    pre_date_list = get_previous_n_trading_dates(exchange='SHSE', date=date, n=3)
    yesterday = pre_date_list[-1]
    if not context.n_days_limit_up_list:
        for the_date in pre_date_list[::-1][1:]:
            context.n_days_limit_up_list.append(get_zt_stock(all_stocks, the_date))
    # 计算昨日涨停股票
    all_stocks,pre_data = get_zt_stock(all_stocks, yesterday, return_data=True)
    context.n_days_limit_up_list.append(all_stocks) 
    # 过滤前2日涨停股票
    zt2_list = set(context.n_days_limit_up_list[-2] + context.n_days_limit_up_list[-3])
    all_stocks = list(set(all_stocks)-zt2_list)
    # 移除无用的数据
    context.n_days_limit_up_list.pop(0)                        

    # 4、过滤成交额大于19亿、小于7亿
    if all_stocks:
        pre_data_new = pre_data[pre_data['symbol'].isin(all_stocks)].copy()
        all_stocks = pre_data_new[(pre_data_new['amount']>context.min_amount)&(pre_data_new['amount']<context.max_amount)&(pre_data_new['symbol'].isin(all_stocks))]['symbol'].tolist()

    # 5、过滤市值小于70亿，流通市值大于520亿
    if all_stocks:        
        # 流通市值
        mv = stk_get_daily_mktvalue_pt(symbols=all_stocks, fields='tot_mv,a_mv_ex_ltd', trade_date=yesterday, df=True)
        all_stocks = mv[(mv['tot_mv']>context.min_mv)&(mv['a_mv_ex_ltd']<context.max_mv)]['symbol'].tolist()

    # 6、过滤收盘获利比例大于4%
    if all_stocks:
        pre_data_new = pre_data[pre_data['symbol'].isin(all_stocks)].copy()
        pre_data_new['get_profit_rate'] = pre_data_new['close']/(pre_data_new['amount']/pre_data_new['volume'])-1
        all_stocks = pre_data_new[pre_data_new['get_profit_rate']<context.max_get_profit_rate]['symbol'].tolist()

    # 7、过滤开盘涨跌幅（回测中才执行这个步骤，实盘用订阅tick数据的方式过滤)
    if all_stocks and context.mode==MODE_BACKTEST:
        today_data = history(symbol=all_stocks, frequency='1d', start_time=date,  end_time=date, adjust=ADJUST_PREV, adjust_end_time=date, df= True)
        today_data['open_rate'] = today_data['open']/today_data['pre_close']-1
        all_stocks = today_data[(today_data['open_rate']<context.max_call_rate)&(today_data['open_rate']>context.min_call_rate)]['symbol'].tolist()

    # 8、过滤左侧压力位缩量的
    if all_stocks:
        new_stocks = []
        date_list = get_previous_n_trading_dates(exchange='SHSE', date=date, n=101)
        all_data = history_new(security=all_stocks,frequency='1d',start_time=date_list[0],end_time=date_list[-1],fields='symbol,eob,high,volume',skip_suspended=True,fill_missing=None,adjust=ADJUST_PREV,adjust_end_time=None, df=True, type=False)
        for symbol in all_stocks:
            the_all_data = all_data[all_data['symbol'].isin([symbol])]
            prev_high = the_all_data['high'].iloc[-1]  # 计算前一天的高点
            zyts_0 = next((i-1 for i, high in enumerate(the_all_data['high'][-3::-1], 2) if high >= prev_high), 100)  # 计算zyts_0
            zyts = zyts_0+5
            volume_data = the_all_data['volume'][-zyts:]   # 获取高点以来的成交量数据
            # 检查今天的成交量是否同步放大
            if len(volume_data) < 2 or volume_data.iloc[-1] < max(volume_data[:-1])*context.zy_min_rate:
                continue
            new_stocks.append(symbol)
        all_stocks = list(set(all_stocks)&set(new_stocks))

    # 9、剔除伪首板（多日涨停中间有停牌的，会误认为是首板；这边简单地剔除前天停牌的股票，精确的做法可以剔除停牌前涨停的股票）
    if all_stocks:
        before_yesterday = pre_date_list[-2]
        symbols_info = get_symbols(sec_type1=1010, symbols=all_stocks, skip_suspended=False, skip_st=False, trade_date=before_yesterday, df=True)
        all_stocks = symbols_info[symbols_info['is_suspended']==False]['symbol'].tolist()

    # 10、订阅数据
    context.buy_stocks = all_stocks
    print('{} 买入监控股票：{}'.format(context.now, context.buy_stocks))
    
    # 查询持仓
    context.holding_stocks =  [posi['symbol'] for posi in get_position()] 
    # 合并监控的股票和持仓的股票
    subscribe_stocks = list(set(context.buy_stocks)|set(context.holding_stocks))
    # 记录数据
    if subscribe_stocks:
        dicts = get_symbols(sec_type1=1010, symbols=subscribe_stocks, skip_suspended=False, skip_st=False, trade_date=date, df=False)
        pre_data_adjust = history(symbol=subscribe_stocks, frequency='1d', start_time=yesterday,  end_time=yesterday, adjust=ADJUST_PREV, adjust_end_time=date, df= True)
        pre_data_adjust = pre_data_adjust.merge(pd.DataFrame(dicts)[['symbol']], on=['symbol'])
        context.monitor_data = {dic['symbol']:{'upper_limit':dic['upper_limit'], 'lower_limit':dic['lower_limit'],
                                'pre_close':pre_data_adjust[pre_data_adjust['symbol'].isin([dic['symbol']])]['close'].iloc[-1],
                                'pre_volume':pre_data_adjust[pre_data_adjust['symbol'].isin([dic['symbol']])]['volume'].iloc[-1]} for dic in dicts}
    else:
        context.monitor_data = {}
    # 订阅tick数据
    subscribe(symbols=subscribe_stocks, frequency='tick', count=2, unsubscribe_previous=True)



def sell_algo(context):# 所有持仓
    nor_str = context.now.strftime('%H:%M:%S')
    Account_positions = context.account().positions()
    all_symbols = [posi['symbol'] for posi in Account_positions]
    current_data_all = current(symbols=all_symbols)

    # 两种卖出之一：早盘卖出
    if nor_str==context.sell_time1:
        for posi in Account_positions:
            symbol = posi['symbol']
            current_data = list(filter(lambda x:x['symbol']==symbol,current_data_all))[0]
            # 卖出条件，未涨停且有利润(跌停不卖出)
            available_now = posi['volume']-posi['volume_today'] if context.mode==MODE_BACKTEST else posi['available_now']
            sell_cond = available_now>0 and current_data['price']<context.monitor_data[symbol]['upper_limit'] and current_data['price']>posi['vwap']
            if sell_cond and  current_data['price']>context.monitor_data[symbol]['lower_limit']:
                order_volume(symbol=symbol, volume=available_now, side=OrderSide_Sell, order_type=OrderType_Market, price=context.monitor_data[symbol]['lower_limit'], position_effect=PositionEffect_Close)
                
    # 两种卖出之二：尾盘卖出
    elif nor_str==context.sell_time2:
        for posi in Account_positions:
            symbol = posi['symbol']
            current_data = list(filter(lambda x:x['symbol']==symbol,current_data_all))[0]
            # 卖出条件，未涨停(跌停不卖出)
            available_now = posi['volume']-posi['volume_today'] if context.mode==MODE_BACKTEST else posi['available_now']
            sell_cond = current_data['price']<context.monitor_data[symbol]['upper_limit'] and available_now>0
            if sell_cond and  current_data['price']>context.monitor_data[symbol]['lower_limit']:
                order_volume(symbol=symbol, volume=available_now, side=OrderSide_Sell, order_type=OrderType_Market, price=context.monitor_data[symbol]['lower_limit'], position_effect=PositionEffect_Close)
                
                
def buy_algo(context):
    to_buy = context.buy_stocks
    for symbol in to_buy:
        percent = 1/len(to_buy)
        cash = context.account().cash
        # 用市价单进行撮合，即下一个tick的开盘价
        price = context.data(symbol=symbol, frequency='tick', count=1)['open'].iloc[-1]
        volume = cal_stock_buy_volume(symbol,cash['nav']*percent,price)
        if volume>0:
            order_volume(symbol=symbol, volume=volume, side=OrderSide_Buy, order_type=OrderType_Market, price=context.monitor_data[symbol]['upper_limit'], position_effect=PositionEffect_Open)
            unsubscribe(symbols=symbol, frequency='tick')


def on_tick(context,tick):
    symbol = tick['symbol']
    if symbol in context.buy_stocks and symbol not in context.holding_stocks and tick['open']>0 and tick['created_at'].time()<datetime.strptime('09:30:00', '%H:%M:%S').time():
        open_colume_rate = tick['cum_volume']/context.monitor_data[symbol]['pre_volume']
        if open_colume_rate<0.03:
            print('{}:{}集合竞价量能为{},不符合条件，取消订阅'.format(tick['created_at'],symbol,open_colume_rate))
            unsubscribe(symbols=symbol, frequency='tick')
            context.buy_stocks.remove(symbol)
            return
        if context.mode==MODE_LIVE:
            call_rate = tick['open']/context.monitor_data[symbol]['pre_close']-1
            if call_rate>=context.max_call_rate or call_rate<=context.min_call_rate:
                print('{}:{}开盘涨跌幅为{:.2%}，不符合条件,取消订阅'.format(tick['created_at'],symbol,call_rate))
                unsubscribe(symbols=symbol, frequency='tick')
                context.buy_stocks.remove(symbol)
                return


def get_zt_stock(stock_list, date, return_data=False):
    """筛选出某一日涨停的股票"""
    history_data = history(symbol=stock_list, frequency='1d', start_time=date,  end_time=date, adjust=ADJUST_NONE, df= True)
    symbols_info = get_symbols(sec_type1=1010, symbols=stock_list, skip_suspended=False, skip_st=False, trade_date=date, df=True)
    history_data = history_data.merge(symbols_info, on=['symbol'])
    zt_stock = history_data[(history_data['close']==history_data['upper_limit'])]['symbol'].tolist()

    if return_data:
        return zt_stock,history_data
    else:
        return zt_stock


def cal_stock_buy_volume(code,amount,price):
    """计算股票下单数量"""
    Account_cash = get_cash()# 获取账户资金信息
    available_amount = min(amount,Account_cash['available'])                 
    trade_volume = max(int(np.floor(available_amount/price/100)*100),200) if code.startswith('SHSE.68') else max(int(np.floor(available_amount/price/100)*100),100)
    return trade_volume


def get_normal_stocks(context, date,new_days=365,skip_suspended=True, skip_st=True, skip_upper_limit=False, return_info=False):
    """
    获取目标日期date的A股代码（剔除停牌股、ST股、次新股（365天））
    :param date：目标日期
    :param new_days:新股上市天数，默认为365天
    :param skip_suspended:是否剔除停牌股，默认为True
    :param skip_st:是否剔除ST股，默认为True
    :param skip_upper_limit:是否剔除开盘涨停股票，默认为True,仅在回测中生效
    """
    date = pd.Timestamp(date).replace(tzinfo=None)
    next_20date = pd.Timestamp(get_next_n_trading_dates(exchange='SHSE', date=date.strftime('%Y-%m-%d'), n=20)[-1])
    # A股，剔除停牌和ST股票
    stocks_info = get_symbols(sec_type1=1010, sec_type2=101001, skip_suspended=skip_suspended, skip_st=skip_st, trade_date=date.strftime('%Y-%m-%d'), df=True)
    if len(stocks_info)>0:
        stocks_info['listed_date'] = stocks_info['listed_date'].apply(lambda x:x.replace(tzinfo=None))
        stocks_info['delisted_date'] = stocks_info['delisted_date'].apply(lambda x:x.replace(tzinfo=None))
     
        # 剔除次新股和退市股(退市前20个交易日，以过滤退市整理期)
        stocks_info = stocks_info[(stocks_info['listed_date']<=date-timedelta(days=new_days))&(stocks_info['delisted_date']>next_20date)]
        all_stocks = list(stocks_info['symbol'])
        # 剔除开盘涨停股
        if skip_upper_limit and context.mode==MODE_BACKTEST:
            low_price = history(symbol=all_stocks, frequency='1d', start_time=date,  end_time=date, fields='open,symbol', adjust=ADJUST_NONE, df= True)
            stocks_info = stocks_info.merge(low_price,on=['symbol'])
            all_stocks = stocks_info[stocks_info['open']!=stocks_info['upper_limit']]['symbol'].tolist()
    else:
        all_stocks = []
    all_stocks_str = ','.join(all_stocks)
    if return_info:
        return all_stocks,all_stocks_str,stocks_info
    else:
        return all_stocks,all_stocks_str


def history_new(security,frequency,start_time,end_time,fields,skip_suspended=True,fill_missing=None,adjust=ADJUST_PREV,adjust_end_time=None, df=True, type=True, benchmark='SHSE.000300'):
    """
    分区间获取数据（以避免超出数据限制）(start_time和end_date为字符串,fields需包含eob和symbol,单字段)
    :param ：参数同history()参数一致，adjust_end_time默认为回测结束时间：None,注意需要根据不同场景使用end_time或context.backtest_end_time
    :param type：默认为True，输出2维DataFrame（日期*股票）,否则输出1维DataFrame
    """
    Data = pd.DataFrame()
    if frequency=='1d':
        trading_date = pd.Series(get_trading_dates(exchange='SZSE', start_date=start_time, end_date=end_time))
    elif frequency=='tick':
        trading_date = history(benchmark, frequency=frequency, start_time=start_time, end_time=end_time, fields='created_at', skip_suspended=skip_suspended, fill_missing=fill_missing, adjust=adjust, adjust_end_time=adjust_end_time, df=df)
    else:
        trading_date = history(benchmark, frequency=frequency, start_time=start_time, end_time=end_time, fields='bob,eob', skip_suspended=skip_suspended, fill_missing=fill_missing, adjust=adjust, adjust_end_time=adjust_end_time, df=df)
    # 计算合理间隔
    if isinstance(security,str):
        security = security.split(',')
    space = 30000//len(security)
    # 获取数据
    if len(trading_date)<=space:
        Data = history(security, frequency=frequency, start_time=start_time, end_time=end_time, fields=fields, skip_suspended=skip_suspended, fill_missing=fill_missing, adjust=adjust, adjust_end_time=adjust_end_time, df=df)
    else:
        for n in range(int(np.ceil(len(trading_date)/space))):
            start = n*space
            end = start+space
            if end>=len(trading_date):
                if frequency=='1d':
                    data = history(security, frequency=frequency, start_time=trading_date.iloc[start], end_time=trading_date.iloc[-1], fields=fields, skip_suspended=skip_suspended, fill_missing=fill_missing, adjust=adjust, adjust_end_time=adjust_end_time, df=df)
                elif frequency=='tick':
                    data = history(security, frequency=frequency, start_time=trading_date.iloc[start][0], end_time=trading_date.iloc[-1][0], fields=fields, skip_suspended=skip_suspended, fill_missing=fill_missing, adjust=adjust, adjust_end_time=adjust_end_time, df=df)
                else:
                    data = history(security, frequency=frequency, start_time=trading_date.iloc[start][0], end_time=trading_date.iloc[-1][1], fields=fields, skip_suspended=skip_suspended, fill_missing=fill_missing, adjust=adjust, adjust_end_time=adjust_end_time, df=df)
            else:
                if frequency=='1d':
                    data = history(security, frequency=frequency, start_time=trading_date.iloc[start], end_time=trading_date.iloc[end], fields=fields, skip_suspended=skip_suspended, fill_missing=fill_missing, adjust=adjust, adjust_end_time=adjust_end_time, df=df)
                else:
                    data = history(security, frequency=frequency, start_time=trading_date.iloc[start][0], end_time=trading_date.iloc[end][0], fields=fields, skip_suspended=skip_suspended, fill_missing=fill_missing, adjust=adjust, adjust_end_time=adjust_end_time, df=df)
            if len(data)==33000:
                print('请检查返回数据量，可能超过系统限制，缺少数据！！！！！！！！！！')
            Data = pd.concat([Data,data])
    if df and len(Data)>0:
        if frequency=='tick': 
            Data.sort_values(['symbol','created_at'],inplace=True)
            Data.drop_duplicates(subset=['created_at','symbol'],keep='first',inplace=True)
        else:
            Data.sort_values(['symbol','eob'],inplace=True)
            Data.drop_duplicates(subset=['eob','symbol'],keep='first',inplace=True)
        if type:
            if len(Data)>0:
                if frequency=='tick':
                    Data = Data.set_index(['created_at','symbol'])
                else:
                    Data = Data.set_index(['eob','symbol'])
                Data = Data.unstack()
                Data.columns = Data.columns.droplevel(level=0)
    return Data


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
        print('{}:标的：{}，操作：以{}{}，委托价格：{}，委托数量：{}'.format(context.now,symbol,order_type_word,side_effect,price,volume))
    elif status == 8:
        print('{}:拒绝委托：{}'.format(context.now,order))
       

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
    run(strategy_id='',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='',
        backtest_start_time='2023-07-01 08:00:00',
        backtest_end_time='2024-06-25 16:00:00',
        backtest_adjust=ADJUST_NONE,
        backtest_initial_cash=100000,
        backtest_commission_ratio=0.0008,
        backtest_slippage_ratio=0.00123,
        backtest_match_mode=0)
