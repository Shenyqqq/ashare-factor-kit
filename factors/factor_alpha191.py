"""
factors/factor_alpha191.py — 国泰君安 Alpha191 量价因子

来源：.tmp/alphas/alphas191.py（jqdata/多进程基类依赖已剥离）
命名：GTJA_001 ... GTJA_191

约定：
  - returns 使用 clean_ret（禁用裸 pct_change）
  - VWAP = amount/volume；缺省时典型价 (H+L+C)/3，日志标明
  - 基准因子 GTJA_075/181/182 需 market_prices（中证全指）；缺则 skip
  - 截面标准化在 get_alpha191_factors 出口统一 _normalize
  - 跳过：030（FF三因子）、143（SELF）、149（FILTER）、190（未落地）

内存注意
--------
- 无模块级大矩阵缓存；条件面板用 ``ops.empty_like``（勿 ``copy(deep=True)``）。
- ``factor_names`` 白名单只算请求因子；IC 侧应按批调用（``--factor-prefix`` / ``--batch-size``）。
- 基准面板仅在需要 GTJA_075/181/182 时惰性广播，避免无谓全市场复制。
- 全量 187 面板同时驻留会 OOM；默认 registry 分批，勿一次 list 全部结果。

共实现 187 个（191 候选 - 4 skip；基准 3 个有数据时计入）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from numpy import log  # noqa: F401

from factors import alpha_ops as ops
from factors.alpha_ops import compute_vwap
from factors.factor import _normalize

ALPHA191_NAMES = tuple(f"GTJA_{i:03d}" for i in range(1, 192))
SKIP_NAMES = frozenset({"GTJA_030", "GTJA_143", "GTJA_149", "GTJA_190"})
BENCHMARK_NAMES = frozenset({"GTJA_075", "GTJA_181", "GTJA_182"})
SKIP_REASONS = {
    "GTJA_030": "依赖 Fama-French MKT/SMB/HML，本库无",
    "GTJA_143": "SELF 递归定义，源实现返回 0",
    "GTJA_149": "FILTER+基准回归，源实现返回 0",
    "GTJA_190": "源实现返回 0（公式未落地）",
}

Rank = ops.rank
Delta = ops.delta
Delay = ops.delay
Corr = ops.corr
Cov = ops.cov
Sum = ops.ts_sum
Prod = ops.ts_prod
Mean = ops.ts_mean
Std = ops.ts_std
Tsrank = ops.ts_rank
Tsmax = ops.ts_max
Tsmin = ops.ts_min
Sign = ops.sign
Max = ops.maximum
Min = ops.minimum
Rowmax = ops.row_max
Rowmin = ops.row_min
Abs = ops.abs_
Log = ops.log
Sequence = ops.sequence
Regbeta = ops.regbeta
Decaylinear = ops.decay_linear
Wma = ops.wma_gtja
Lowday = ops.lowday
Highday = ops.highday
Count = ops.count
Sumif = ops.sumif


def Sma(sr, n, m):
    return ops.sma_gtja(sr, n, m)


class _GTJAAlphas:
    """内部计算上下文：属性为 date×code 面板。"""

    def __init__(
        self,
        close, open_, high, low, volume, amount, vwap, returns,
        benchmark_open=None, benchmark_close=None,
    ):
        self.close = close
        self.open = open_
        self.high = high
        self.low = low
        self.volume = volume
        self.amount = amount
        self.vwap = vwap
        self.returns = returns
        self.close_prev = close.shift(1)
        self.benchmark_open = benchmark_open
        self.benchmark_close = benchmark_close
    def alpha001(self): #平均1751个数据
        ##### (-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))#### 
        return (-1 * Corr(Rank(Delta(log(self.volume), 1)), Rank(((self.close - self.open) / self.open)), 6))
    
    def alpha002(self): #1783
        ##### -1 * delta((((close-low)-(high-close))/(high-low)),1))####
        return -1*Delta((((self.close-self.low)-(self.high-self.close))/(self.high-self.low)),1) 
    
    def alpha003(self): 
        ##### SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),6) ####
        cond1 = (self.close == Delay(self.close,1))
        cond2 = (self.close > Delay(self.close,1))
        cond3 = (self.close < Delay(self.close,1))
        part = ops.empty_like(self.close)
        part[cond1] = 0
        part[cond2] = self.close - Min(self.low,Delay(self.close,1))
        part[cond3] = self.close - Max(self.high,Delay(self.close,1))
        return Sum(part, 6)
    
    def alpha004(self):  
        #####((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) <((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (VOLUME / MEAN(VOLUME,20))) || ((VOLUME /MEAN(VOLUME,20)) == 1)) ? 1 : (-1 * 1))))
        cond1 = ((Sum(self.close, 8)/8 + Std(self.close, 8)) < Sum(self.close, 2)/2)
        cond2 = ((Sum(self.close, 8)/8 + Std(self.close, 8)) > Sum(self.close, 2)/2)
        cond3 = ((Sum(self.close, 8)/8 + Std(self.close, 8)) == Sum(self.close, 2)/2)
        cond4 = (self.volume/Mean(self.volume, 20) >= 1)
        part = ops.empty_like(self.close)
        part[cond1] = -1
        part[cond2] = 1
        part[cond3] = -1
        part[cond3 & cond4] = 1
        
        return part
    
    def alpha005(self): #1447
        ####(-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))###
        return -1*Tsmax(Corr(Tsrank(self.volume, 5),Tsrank(self.high, 5),5), 3)
    
    def alpha006(self): #1779
        ####(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)### 
        return -1*Rank(Sign(Delta(((self.open * 0.85) + (self.high * 0.15)), 4)))
    
    def alpha007(self): #1782
        ####((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))###
        return ((Rank(Tsmax((self.vwap - self.close), 3)) + Rank(Tsmin((self.vwap - self.close), 3))) * Rank(Delta(self.volume, 3)))
    
    def alpha008(self): #1779
        ####RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)###    
        return Rank(Delta(((((self.high + self.low) / 2) * 0.2) + (self.vwap * 0.8)), 4) * -1)
    
    def alpha009(self): #1790
        ####SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,7,2)###  
        return Sma(((self.high+self.low)/2-(Delay(self.high,1)+Delay(self.low,1))/2)*(self.high-self.low)/self.volume,7,2)
    
    def alpha010(self):    
        ####(RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2),5))###
        cond = (self.returns < 0)
        part = ops.empty_like(self.returns)
        part[cond] = Std(self.returns, 20)
        part[~cond] = self.close
        part = part**2
        
        return Rank(Tsmax(part, 5))
    
    def alpha011(self): #1782
        ####SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,6)###   
        return Sum(((self.close-self.low)-(self.high-self.close))/(self.high-self.low)*self.volume,6)
    
    def alpha012(self): #1779
        ####(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))###   
        return (Rank((self.open - (Sum(self.vwap, 10) / 10)))) * (-1 * (Rank(Abs((self.close - self.vwap)))))
    
    def alpha013(self): #1790
        ####(((HIGH * LOW)^0.5) - VWAP)###
        return (((self.high * self.low)**0.5) - self.vwap)
    
    def alpha014(self): #1776
        ####CLOSE-DELAY(CLOSE,5)###
        return self.close-Delay(self.close,5)
    
    def alpha015(self): #1790
        ####OPEN/DELAY(CLOSE,1)-1###
        return self.open/Delay(self.close,1)-1
    
    def alpha016(self): #1736   
        ####(-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5))###
        return (-1 * Tsmax(Rank(Corr(Rank(self.volume), Rank(self.vwap), 5)), 5))
        
    def alpha017(self): #1776   
        ####RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)###
        return Rank((self.vwap - Tsmax(self.vwap, 15)))**Delta(self.close, 5)
    
    def alpha018(self): #1776   
        ####CLOSE/DELAY(CLOSE,5)###
        return self.close/Delay(self.close,5)  
    
    def alpha019(self):  
        ####(CLOSE<DELAY(CLOSE,5)?(CLOSE-DELAY(CLOSE,5))/DELAY(CLOSE,5):(CLOSE=DELAY(CLOSE,5)?0:(CLOSE-DELAY(CLOSE,5))/CLOSE))###
        cond1 = (self.close < Delay(self.close,5))
        cond2 = (self.close == Delay(self.close,5))
        cond3 = (self.close > Delay(self.close,5))
        part = ops.empty_like(self.close)
        part[cond1] = (self.close-Delay(self.close,5))/Delay(self.close,5)
        part[cond2] = 0
        part[cond3] = (self.close-Delay(self.close,5))/self.close
        
        return part
       
    def alpha020(self): #1773      
        ####(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100###
        return (self.close-Delay(self.close,6))/Delay(self.close,6)*100
    
    def alpha021(self):  #reg？
        ####REGBETA(MEAN(CLOSE,6),SEQUENCE(6))###        
        return Regbeta(Mean(self.close,6), Sequence(6))
    
    def alpha022(self): #1736  
        ####SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)###
        return Sma(((self.close-Mean(self.close,6))/Mean(self.close,6)-Delay((self.close-Mean(self.close,6))/Mean(self.close,6),3)),12,1)
     
    def alpha023(self):  
        ####SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) / (SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1) + SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100###
        cond = (self.close > Delay(self.close,1))
        part1 = ops.empty_like(self.close)
        part1[cond] = Std(self.close,20)
        part1[~cond] = 0
        part2 = ops.empty_like(self.close)
        part2[~cond] = Std(self.close,20)
        part2[cond] = 0
        
        return 100*Sma(part1,20,1)/(Sma(part1,20,1) + Sma(part2,20,1))
        
    def alpha024(self): #1776  
        ####SMA(CLOSE-DELAY(CLOSE,5),5,1)###
        return Sma(self.close-Delay(self.close,5),5,1)
    
    def alpha025(self):  #886  
        ####((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250))))###
        return ((-1 * Rank((Delta(self.close, 7) * (1 - Rank(Decaylinear((self.volume / Mean(self.volume,20)), 9)))))) * (1 + Rank(Sum(self.returns, 250))))
    
    def alpha026(self):   
        ####((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))###
        return ((((Sum(self.close, 7) / 7) - self.close)) + ((Corr(self.vwap, Delay(self.close, 5), 230))))
    
    def alpha027(self):  
        ####WMA((CLOSE-DELAY(CLOSE,3))/DELAY(CLOSE,3)*100+(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*100,12)###
        A = (self.close-Delay(self.close,3))/Delay(self.close,3)*100+(self.close-Delay(self.close,6))/Delay(self.close,6)*100
        return Wma(A, 12)
    
    def alpha028(self):   #1728 
        ####3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(MAX(HIGH,9)-TSMAX(LOW,9))*100,3,1),3,1)###
        return 3*Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1)-2*Sma(Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmax(self.low,9))*100,3,1),3,1)
    
    def alpha029(self):   #1773 
        ####(CLOSE-DELAY(CLOSE,6))/DELAY(CLOSE,6)*VOLUME###
        return (self.close-Delay(self.close,6))/Delay(self.close,6)*self.volume
    
    def alpha031(self):   #1714
        ####(CLOSE-MEAN(CLOSE,12))/MEAN(CLOSE,12)*100###
        return (self.close-Mean(self.close,12))/Mean(self.close,12)*100
    
    def alpha032(self):   #1505
        ####(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3))###
        return (-1 * Sum(Rank(Corr(Rank(self.high), Rank(self.volume), 3)), 3))
    
    def alpha033(self):   #904  数据量较少
        ####((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) *TSRANK(VOLUME, 5))###
        return ((((-1 * Tsmin(self.low, 5)) + Delay(Tsmin(self.low, 5), 5)) * Rank(((Sum(self.returns, 240) - Sum(self.returns, 20)) / 220))) *Tsrank(self.volume, 5))
    
    def alpha034(self):   #1714
        ####MEAN(CLOSE,12)/CLOSE###
        return Mean(self.close,12)/self.close
    
    def alpha035(self):   #1790    (OPEN * 0.65) +(OPEN *0.35)有问题
        ####(MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) +(OPEN *0.35)), 17),7))) * -1)###
        return (Min(Rank(Decaylinear(Delta(self.open, 1), 15)), Rank(Decaylinear(Corr((self.volume), ((self.open * 0.65) +(self.open *0.35)), 17),7))) * -1)
     
    def alpha036(self):   #1714
        ####RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP),6), 2))###
        return Rank(Sum(Corr(Rank(self.volume), Rank(self.vwap),6 ), 2))
    
    def alpha037(self):   #1713
        ####(-1 * RANK(((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY((SUM(OPEN, 5) * SUM(RET, 5)), 10))))###
        return (-1 * Rank(((Sum(self.open, 5) * Sum(self.returns, 5)) - Delay((Sum(self.open, 5) * Sum(self.returns, 5)), 10))))
    
    def alpha038(self):  
        ####(((SUM(HIGH, 20) / 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0)
        cond = ((Sum(self.high, 20) / 20) < self.high)
        part = ops.empty_like(self.close)
        part[cond] = -1 * Delta(self.high, 2)
        part[~cond] = 0
        
        return part
    
    def alpha039(self):   #1666
        ####((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)),SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)###
        return ((Rank(Decaylinear(Delta((self.close), 2),8)) - Rank(Decaylinear(Corr(((self.vwap * 0.3) + (self.open * 0.7)),Sum(Mean(self.volume,180), 37), 14), 12))) * -1)
    
    def alpha040(self):  
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100###
        cond = (self.close > Delay(self.close,1))
        part1 = ops.empty_like(self.close)
        part1[cond] = self.volume
        part1[~cond] = 0
        part2 = ops.empty_like(self.close)
        part2[~cond] = self.volume
        part2[cond] = 0

        return Sum(part1,26)/Sum(part2,26)*100
    
    def alpha041(self):   #1782
        ####(RANK(MAX(DELTA((VWAP), 3), 5))* -1)###
        return (Rank(Tsmax(Delta((self.vwap), 3), 5))* -1)
    
    def alpha042(self):   #1399  数据量较少
        ####((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, VOLUME, 10))###
        return ((-1 * Rank(Std(self.high, 10))) * Corr(self.high, self.volume, 10))
    
    def alpha043(self):  
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),6)###
        cond1 = (self.close > Delay(self.close,1))
        cond2 = (self.close < Delay(self.close,1))
        cond3 = (self.close == Delay(self.close,1))
        part = ops.empty_like(self.close) # pd.Series(np.zeros(self.close.shape))
        part.loc[:, :] = np.nan
        part[cond1] = self.volume
        part[cond2] = -self.volume
        part[cond3] = 0
        
        return Sum(part,6)
    
    def alpha044(self):   #1748
        ####(TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP),3), 10), 15))###
        return (Tsrank(Decaylinear(Corr(((self.low)), Mean(self.volume,10), 7), 6),4) + Tsrank(Decaylinear(Delta((self.vwap),3), 10), 15))
    
    def alpha045(self):   #1070  数据量较少
        ####(RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))###
        return (Rank(Delta((((self.close * 0.6) + (self.open *0.4))), 1)) * Rank(Corr(self.vwap, Mean(self.volume,150), 15)))
    
    def alpha046(self):   #1630
        ####(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)###
        return (Mean(self.close,3)+Mean(self.close,6)+Mean(self.close,12)+Mean(self.close,24))/(4*self.close)
    
    def alpha047(self):   #1759
        ####SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,9,1)###
        return Sma((Tsmax(self.high,6)-self.close)/(Tsmax(self.high,6)-Tsmin(self.low,6))*100,9,1)
    
    def alpha048(self):   #1657
        ####(-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(VOLUME, 5)) / SUM(VOLUME, 20))###
        return (-1*((Rank(((Sign((self.close - Delay(self.close, 1))) + Sign((Delay(self.close, 1) - Delay(self.close, 2)))) + Sign((Delay(self.close, 2) - Delay(self.close, 3)))))) * Sum(self.volume, 5)) / Sum(self.volume, 20))
    
    def alpha049(self):  
        ####SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12) / (SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12) + SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))
        cond = ((self.high + self.low) > (Delay(self.high,1) + Delay(self.low,1)))
        part1 = ops.empty_like(self.close) # pd.Series(np.zeros(self.close.shape))
        part1.loc[:, :] = np.nan
        part1[cond] = 0
        part1[~cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        part2 = ops.empty_like(self.close)
        part2[~cond] = 0
        part2[cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        
        return Sum(part1, 12) / (Sum(part1, 12) + Sum(part2, 12))
    
    def alpha050(self):  
        ####SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))-SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)/(SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))###
        cond = ((self.high + self.low) <= (Delay(self.high,1) + Delay(self.low,1)))
        part1 = ops.empty_like(self.close)
        part1[cond] = 0
        part1[~cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        part2 = ops.empty_like(self.close)
        part2[~cond] = 0
        part2[cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        
        return (Sum(part1, 12) - Sum(part2, 12)) / (Sum(part1, 12) + Sum(part2, 12)) 

    def alpha051(self):  
        ####SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12) / (SUM(((HIGH+LOW)<=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12)+SUM(((HIGH+LOW)>=(DELAY(HIGH,1)+DELAY(LOW,1))?0:MAX(ABS(HIGH-DELAY(HIGH,1)),ABS(LOW-DELAY(LOW,1)))),12))###
        cond = ((self.high + self.low) <= (Delay(self.high,1) + Delay(self.low,1)))
        part1 = ops.empty_like(self.close)
        part1[cond] = 0
        part1[~cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        part2 = ops.empty_like(self.close)
        part2[~cond] = 0
        part2[cond] = Max(Abs(self.high - Delay(self.high,1)), Abs(self.low - Delay(self.low,1)))
        
        return Sum(part1, 12) / (Sum(part1, 12) + Sum(part2, 12))
    
    def alpha052(self):   #1611
        ####SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-L),26)*100###
        return Sum(Max(self.high-Delay((self.high+self.low+self.close)/3,1),0),26)/Sum(Max(Delay((self.high+self.low+self.close)/3,1)-self.low, 0),26)*100
    
    def alpha053(self):  
        ####COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100###
        cond = (self.close > Delay(self.close,1))
        return Count(cond, 12) / 12 * 100
    
    def alpha054(self):   #1729
        ####(-1 * RANK((STD(ABS(CLOSE - OPEN)) + (CLOSE - OPEN)) + CORR(CLOSE, OPEN,10)))###
        return (-1 * Rank(((Abs(self.close - self.open)).std() + (self.close - self.open)) + Corr(self.close, self.open,10)))
    
    def alpha055(self):  #公式有问题
        ###SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2 + ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)
        A = Abs(self.high - Delay(self.close, 1))
        B = Abs(self.low - Delay(self.close, 1))
        C = Abs(self.high - Delay(self.low, 1))
        cond1 = ((A > B) & (A > C))
        cond2 = ((B > C) & (B > A))
        cond3 = ((C >= A) & (C >= B))
        part0 = 16*(self.close + (self.close - self.open)/2 - Delay(self.open,1))
        part1 = ops.empty_like(self.close)
        part1.loc[:, :] = 0
        part1[cond1] = Abs(self.high - Delay(self.close, 1)) + Abs(self.low - Delay(self.close, 1))/2 + Abs(Delay(self.close, 1)-Delay(self.open, 1))/4
        part1[cond2] = Abs(self.low - Delay(self.close, 1)) + Abs(self.high - Delay(self.close, 1))/2 + Abs(Delay(self.close, 1)-Delay(self.open, 1))/4
        part1[cond3] = Abs(self.high - Delay(self.low, 1)) + Abs(Delay(self.close, 1)-Delay(self.open, 1))/4
        part2=Max(Abs(self.high-Delay(self.close,1)),Abs(self.low-Delay(self.close,1)))
        
        return Sum(part0/part1*part2,20)
    
    def alpha056(self):  
        ####(RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19),SUM(MEAN(VOLUME,40), 19), 13))^5)))###
        A = Rank((self.open - Tsmin(self.open, 12)))
        B = Rank((Rank(Corr(Sum(((self.high + self.low) / 2), 19),Sum(Mean(self.volume,40), 19), 13))**5))
        cond = (A < B)
        part = ops.empty_like(self.close)
        part[cond] = 1
        part[~cond] = 0
        return part
    
    def alpha057(self):   #1736
        ####SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)###
        return Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1)
    
    def alpha058(self):  
        ####COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100###
        cond = (self.close > Delay(self.close,1))

        return Count(cond,20)/20*100
        
    
    def alpha059(self):  
        ####SUM((CLOSE=DELAY(CLOSE,1)?0:CLOSE-(CLOSE>DELAY(CLOSE,1)?MIN(LOW,DELAY(CLOSE,1)):MAX(HIGH,DELAY(CLOSE,1)))),20)###
        cond1 = (self.close == Delay(self.close,1))
        cond2 = (self.close > Delay(self.close,1))
        cond3 = (self.close < Delay(self.close,1))
        part = ops.empty_like(self.close)
        part[cond1] = 0
        part[cond2] = self.close - Min(self.low,Delay(self.close,1))
        part[cond3] = self.close - Max(self.low,Delay(self.close,1))
        
        return Sum(part, 20)
    
    def alpha060(self):   #1635
        ####SUM(((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW)*VOLUME,20)###
        return Sum(((self.close-self.low)-(self.high-self.close))/(self.high-self.low)*self.volume,20)

    def alpha061(self):   #1790
        ####(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)###
        return (Max(Rank(Decaylinear(Delta(self.vwap, 1), 12)),Rank(Decaylinear(Rank(Corr((self.low),Mean(self.volume,80), 8)), 17))) * -1)
    
    def alpha062(self):   #1479
        ####(-1 * CORR(HIGH, RANK(VOLUME), 5))###
        return (-1 * Corr(self.high, Rank(self.volume), 5))
    
    def alpha063(self):   #1789
        ####SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100###
        return Sma(Max(self.close-Delay(self.close,1),0),6,1)/Sma(Abs(self.close-Delay(self.close,1)),6,1)*100
    
    def alpha064(self):   #1774
        ####(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)###
        return (Max(Rank(Decaylinear(Corr(Rank(self.vwap), Rank(self.volume), 4), 4)),Rank(Decaylinear(Tsmax(Corr(Rank(self.close), Rank(Mean(self.volume,60)), 4), 13), 14))) * -1)
    
    def alpha065(self):   #1759
        ####MEAN(CLOSE,6)/CLOSE###
        return Mean(self.close,6)/self.close
    
    def alpha066(self):   #1759
        ####(CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)*100###
        return (self.close-Mean(self.close,6))/Mean(self.close,6)*100
    
    def alpha067(self):   #1759
        ####SMA(MAX(CLOSE-DELAY(CLOSE,1),0),24,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),24,1)*100###
        a1 = Sma(Max(self.close-Delay(self.close,1),0),24,1)
        a2 = Sma(Abs(self.close-Delay(self.close,1)),24,1)
        return a1/a2*100
    
    def alpha068(self):   #1790
        ####SMA(((HIGH+LOW)/2-(DELAY(HIGH,1)+DELAY(LOW,1))/2)*(HIGH-LOW)/VOLUME,15,2)###
        return Sma(((self.high+self.low)/2-(Delay(self.high,1)+Delay(self.low,1))/2)*(self.high-self.low)/self.volume,15,2)
    
    def alpha069(self):  
        ####(SUM(DTM,20)>SUM(DBM,20)？ (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20)： (SUM(DTM,20)=SUM(DBM,20)？0： (SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))###
        ####DTM (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
        ####DBM (OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))
        cond1 = (self.open <= Delay(self.open,1))
        cond2 = (self.open >= Delay(self.open,1))
        
        DTM = ops.empty_like(self.close)
        DTM[cond1] = 0
        DTM[~cond1] = Max((self.high-self.open),(self.open-Delay(self.open,1)))
        
        DBM = ops.empty_like(self.close)
        DBM[cond2] = 0
        DBM[~cond2] = Max((self.open-self.low),(self.open-Delay(self.open,1)))
        
        cond3 = (Sum(DTM,20) > Sum(DBM,20))
        cond4 = (Sum(DTM,20)== Sum(DBM,20))
        cond5 = (Sum(DTM,20) < Sum(DBM,20))
        part = ops.empty_like(self.close)
        part[cond3] = (Sum(DTM,20)-Sum(DBM,20))/Sum(DTM,20)
        part[cond4] = 0
        part[cond5] = (Sum(DTM,20)-Sum(DBM,20))/Sum(DBM,20)
        return part
    
    def alpha070(self):   #1759
        ####STD(AMOUNT,6)###
        return Std(self.amount,6)
    
    def alpha071(self):   #1630
        ####(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100###
        return (self.close-Mean(self.close,24))/Mean(self.close,24)*100
    
    def alpha072(self):   #1759
        ####SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)###
        return Sma((Tsmax(self.high,6)-self.close)/(Tsmax(self.high,6)-Tsmin(self.low,6))*100,15,1)
    
    def alpha073(self):   #1729
        ####((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) - RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)###
        return ((Tsrank(Decaylinear(Decaylinear(Corr((self.close), self.volume, 10), 16), 4), 5) - Rank(Decaylinear(Corr(self.vwap, Mean(self.volume,30), 4),3))) * -1) 
    
    def alpha074(self):   #1402
        ####(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))###
        return (Rank(Corr(Sum(((self.low * 0.35) + (self.vwap * 0.65)), 20), Sum(Mean(self.volume,40), 20), 7)) + Rank(Corr(Rank(self.vwap), Rank(self.volume), 6)))
    
    def alpha075(self):  
        ####COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)###
        return Count(((self.close>self.open)&(self.benchmark_close<self.benchmark_open)),50)/Count((self.benchmark_close<self.benchmark_open),50)
    
    def alpha076(self):   #1650
        ####STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)###
        return Std(Abs((self.close/Delay(self.close,1)-1))/self.volume,20)/Mean(Abs((self.close/Delay(self.close,1)-1))/self.volume,20)
    
    def alpha077(self):   #1797
        #### MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)),RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))###
        return  Min(Rank(Decaylinear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20)),Rank(Decaylinear(Corr(((self.high + self.low) / 2), Mean(self.volume,40), 3), 6)))
       
    def alpha078(self):   #1637
        ####((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))###
        return ((self.high+self.low+self.close)/3-Mean((self.high+self.low+self.close)/3,12))/(0.015*Mean(Abs(self.close-Mean((self.high+self.low+self.close)/3,12)),12))
    
    def alpha079(self):   #1789
        ####SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100###
        return Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100
    
    def alpha080(self):   #1776
        ####(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100###
        return (self.volume-Delay(self.volume,5))/Delay(self.volume,5)*100
    
    def alpha081(self):   #1797
        ####SMA(VOLUME,21,2)###
        return Sma(self.volume,21,2)
    
    def alpha082(self):   #1759
        ####SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,20,1)###
        return Sma((Tsmax(self.high,6)-self.close)/(Tsmax(self.high,6)-Tsmin(self.low,6))*100,20,1)
    
    def alpha083(self):   #1766
        ####(-1 * RANK(COVIANCE(RANK(HIGH), RANK(VOLUME), 5)))###
        return (-1 * Rank(Cov(Rank(self.high), Rank(self.volume), 5)))
    
    def alpha084(self):  
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),20)###
        cond1 = (self.close > Delay(self.close,1))
        cond2 = (self.close < Delay(self.close,1))
        cond3 = (self.close == Delay(self.close,1))  
        part = ops.empty_like(self.close)
        part[cond1] = self.volume
        part[cond2] = 0
        part[cond3] = -self.volume 
        return Sum(part, 20)
    
    def alpha085(self):   #1657
        ####(TSRANK((VOLUME / MEAN(VOLUME,20)), 20) * TSRANK((-1 * DELTA(CLOSE, 7)), 8))###
        return (Tsrank((self.volume / Mean(self.volume,20)), 20) * Tsrank((-1 * Delta(self.close, 7)), 8))
    
    def alpha086(self):  
        ####((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) :(((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ?1 : ((-1 * 1) *(CLOSE - DELAY(CLOSE, 1)))))
        A = (((Delay(self.close, 20) - Delay(self.close, 10)) / 10) - ((Delay(self.close, 10) - self.close) / 10))
        cond1 = (A > 0.25)
        cond2 = (A < 0.0)
        cond3 = ((0 <= A) & (A <= 0.25))
        part = ops.empty_like(self.close)
        part[cond1] = -1
        part[cond2] = 1
        part[cond3] = -1*(self.close - Delay(self.close, 1))
        return part

    def alpha087(self):   #1741
        ####((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) /(OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)###
        return ((Rank(Decaylinear(Delta(self.vwap, 4), 7)) + Tsrank(Decaylinear(((((self.low * 0.9) + (self.low * 0.1)) - self.vwap) /(self.open - ((self.high + self.low) / 2))), 11), 7)) * -1)
  
    def alpha088(self):   #1745
        ####(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100###
        return (self.close-Delay(self.close,20))/Delay(self.close,20)*100
    
    def alpha089(self):   #1797
        ####2*(SMA(CLOSE,13,2)-SMA(CLOSE,27,2)-SMA(SMA(CLOSE,13,2)-SMA(CLOSE,27,2),10,2))###
        return 2*(Sma(self.close,13,2)-Sma(self.close,27,2)-Sma(Sma(self.close,13,2)-Sma(self.close,27,2),10,2))
    
    def alpha090(self):   #1745
        ####(RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)###
        return (Rank(Corr(Rank(self.vwap), Rank(self.volume), 5)) * -1)
    
    def alpha091(self):   #1745
        ####((RANK((CLOSE - MAX(CLOSE, 5)))*RANK(CORR((MEAN(VOLUME,40)), LOW, 5))) * -1)###
        return ((Rank((self.close - Tsmax(self.close, 5)))*Rank(Corr((Mean(self.volume,40)), self.low, 5))) * -1)
    
    def alpha092(self):   #1786
        ####(MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)###
        return (Max(Rank(Decaylinear(Delta(((self.close * 0.35) + (self.vwap *0.65)), 2), 3)),Tsrank(Decaylinear(Abs(Corr((Mean(self.volume,180)), self.close, 13)), 5), 15)) * -1)
    
    def alpha093(self):  
        ####SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)###
        cond = (self.open >= Delay(self.open,1))
        part = ops.empty_like(self.close)
        part[cond] = 0
        part[~cond] = Max((self.open-self.low),(self.open-Delay(self.open,1)))
        return Sum(part, 20)
    
    def alpha094(self):  
        ####SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)###
        cond1 = (self.close > Delay(self.close,1))
        cond2 = (self.close < Delay(self.close,1))
        cond3 = (self.close == Delay(self.close,1))
        part = ops.empty_like(self.close)
        part[cond1] = self.volume
        part[cond2] = -1*self.volume
        part[cond3] = 0
        return Sum(part, 30)
    
    def alpha095(self):   #1657
        ####STD(AMOUNT,20)###
        return Std(self.amount,20)
    
    def alpha096(self):   #1736
        ####SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)###
        return Sma(Sma((self.close-Tsmin(self.low,9))/(Tsmax(self.high,9)-Tsmin(self.low,9))*100,3,1),3,1)
    
    def alpha097(self):   #1729
        ####STD(VOLUME,10)###
        return Std(self.volume,10)
    
    def alpha098(self):  
        ####((((DELTA((SUM(CLOSE, 100) / 100), 100) / DELAY(CLOSE, 100)) < 0.05) || ((DELTA((SUM(CLOSE, 100) / 100), 100) /DELAY(CLOSE, 100)) == 0.05)) ? (-1 * (CLOSE - TSMIN(CLOSE, 100))) : (-1 * DELTA(CLOSE, 3)))###
        cond = (Delta(Sum(self.close,100)/100, 100)/Delay(self.close, 100) <= 0.05)
        part = ops.empty_like(self.close)
        part[cond] = -1 * (self.close - Tsmin(self.close, 100))
        part[~cond] = -1 * Delta(self.close, 3)
        return part
    
    def alpha099(self):   #1766
        ####(-1 * Rank(Cov(Rank(self.close), Rank(self.volume), 5)))###
        return (-1 * Rank(Cov(Rank(self.close), Rank(self.volume), 5)))
    
    def alpha100(self):   #1657
        ####Std(self.volume,20)###
        return Std(self.volume,20)
    
    def alpha101(self):  
        ###((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),RANK(VOLUME), 11))) * -1)
        rank1 = Rank(Corr(self.close, Sum(Mean(self.volume,30), 37), 15))
        rank2 = Rank(Corr(Rank(((self.high * 0.1) + (self.vwap * 0.9))),Rank(self.volume), 11))
        cond = (rank1<rank2)
        part = ops.empty_like(self.close)
        part[cond] = 1
        part[~cond] = 0
        return part
    
    def alpha102(self):   #1790
        ####SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100###
        return Sma(Max(self.volume-Delay(self.volume,1),0),6,1)/Sma(Abs(self.volume-Delay(self.volume,1)),6,1)*100
    
    def alpha103(self):  
        ####((20-LOWDAY(LOW,20))/20)*100###
        return ((20-Lowday(self.low,20))/20)*100
    
    def alpha104(self):   #1657
        ####(-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))###
        return (-1 * (Delta(Corr(self.high, self.volume, 5), 5) * Rank(Std(self.close, 20))))
    
    def alpha105(self):   #1729
        ####(-1 * CORR(RANK(OPEN), RANK(VOLUME), 10))###
        return (-1 * Corr(Rank(self.open), Rank(self.volume), 10))
    
    def alpha106(self):   #1745
        ####CLOSE-DELAY(CLOSE,20)###
        return self.close-Delay(self.close,20)
    
    def alpha107(self):   #1790
        ####(((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))###
        return (((-1 * Rank((self.open - Delay(self.high, 1)))) * Rank((self.open - Delay(self.close, 1)))) * Rank((self.open - Delay(self.low, 1))))
    
    def alpha108(self):   #1178   
        ####((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)###
        return ((Rank((self.high - Tsmin(self.high, 2)))**Rank(Corr((self.vwap), (Mean(self.volume,120)), 6))) * -1)
    
    def alpha109(self):   #1797
        ####SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)###
        return Sma(self.high-self.low,10,2)/Sma(Sma(self.high-self.low,10,2),10,2)
    
    def alpha110(self):   #1650
        ####SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100###
        return Sum(Max(self.high-Delay(self.close,1),0),20)/Sum(Max(Delay(self.close,1)-self.low,0),20)*100
      
    def alpha111(self):   #1789
        ####SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)###
        return Sma(self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low),11,2)-Sma(self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low),4,2)
    
    def alpha112(self):  
        ####(SUM((CLOSE-DELAY(CLOSE,1)>0? CLOSE-DELAY(CLOSE,1):0),12) - SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12) + SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100     
        cond = (self.close-Delay(self.close,1) > 0)
        part1 = ops.empty_like(self.close)
        part1[cond] = self.close-Delay(self.close,1)
        part1[~cond] = 0
        part2 = ops.empty_like(self.close)
        part2[~cond] = Abs(self.close-Delay(self.close,1))
        part2[cond] = 0
        return (Sum(part1,12) - Sum(part2,12))/(Sum(part1,12) + Sum(part2,12))*100
    
    def alpha113(self):   #1587
        ####(-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5),SUM(CLOSE, 20), 2))))###
        return (-1 * ((Rank((Sum(Delay(self.close, 5), 20) / 20)) * Corr(self.close, self.volume, 2)) * Rank(Corr(Sum(self.close, 5),Sum(self.close, 20), 2))))
    
    def alpha114(self):   #1751
        ####((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) /(SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))###
        return ((Rank(Delay(((self.high - self.low) / (Sum(self.close, 5) / 5)), 2)) * Rank(Rank(self.volume))) / (((self.high - self.low) /(Sum(self.close, 5) / 5)) / (self.vwap - self.close)))
    
    def alpha115(self):   #1527
        ####(RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) /2), 4), TSRANK(VOLUME, 10), 7)))###
        return (Rank(Corr(((self.high * 0.9) + (self.close * 0.1)), Mean(self.volume,30), 10))**Rank(Corr(Tsrank(((self.high + self.low) /2), 4), Tsrank(self.volume, 10), 7)))
    
    def alpha116(self):  
        ####REGBETA(CLOSE,SEQUENCE,20)###        
        return Regbeta(self.close, Sequence(20))
    
    def alpha117(self):   #1786
        ####((TSRANK(VOLUME, 32) * (1 - TSRANK(((CLOSE + HIGH) - LOW), 16))) * (1 - TSRANK(RET, 32)))###
        return ((Tsrank(self.volume, 32) * (1 - Tsrank(((self.close + self.high) - self.low), 16))) * (1 - Tsrank(self.returns, 32)))
    
    def alpha118(self):   #1657
        ####SUM(HIGH-OPEN,20)/SUM(OPEN-LOW,20)*100###
        return Sum(self.high-self.open,20)/Sum(self.open-self.low,20)*100
    
    def alpha119(self):   #1626
        ####(RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))###
        return (Rank(Decaylinear(Corr(self.vwap, Sum(Mean(self.volume,5), 26), 5), 7)) - Rank(Decaylinear(Tsrank(Tsmin(Corr(Rank(self.open), Rank(Mean(self.volume,15)), 21), 9), 7), 8)))
    
    def alpha120(self):   #1797
        ####(RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))###
        return (Rank((self.vwap - self.close)) / Rank((self.vwap + self.close)))
    
    def alpha121(self):   #972   数据量较少
        ####((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) *-1)###
        return ((Rank((self.vwap - Tsmin(self.vwap, 12)))**Tsrank(Corr(Tsrank(self.vwap, 20), Tsrank(Mean(self.volume,60), 2), 18), 3)) *-1)
    
    def alpha122(self):   #1790
        ####(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)-DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))/DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)###
        return (Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2)-Delay(Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2),1))/Delay(Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2),1)
    
    def alpha123(self):  
        ####((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME,6))) * -1)###
        A = Rank(Corr(Sum(((self.high + self.low) / 2), 20), Sum(Mean(self.volume,60), 20), 9))
        B = Rank(Corr(self.low, self.volume,6))
        cond = (A < B)
        part = ops.empty_like(self.close)
        part[cond] = -1
        part[~cond] = 0
        return part
    
    def alpha124(self):   #1592
        ####(CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)###
        return (self.close - self.vwap) / Decaylinear(Rank(Tsmax(self.close, 30)),2)
     
    def alpha125(self):   #1678
        ####(RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))###
        return (Rank(Decaylinear(Corr((self.vwap), Mean(self.volume,80),17), 20)) / Rank(Decaylinear(Delta(((self.close * 0.5) + (self.vwap * 0.5)), 3), 16)))
    
    def alpha126(self):   #1797
        ####(CLOSE+HIGH+LOW)/3###
        return (self.close+self.high+self.low)/3
    
    def alpha127(self):  #公式有问题，我们假设mean周期为12
        ####(MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2),12)^(1/2)###
        return (Mean((100*(self.close-Tsmax(self.close,12))/(Tsmax(self.close,12)))**2,12))**(1/2)
    
    def alpha128(self):  
        #### 100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)))
        A = (self.high+self.low+self.close)/3
        cond = (A > Delay(A,1))        
        part1 = ops.empty_like(self.close)
        part1[cond] = A*self.volume
        part1[~cond] = 0
        part2 = ops.empty_like(self.close)
        part2[~cond] = A*self.volume
        part2[cond] = 0
        return 100-(100/(1+Sum(part1,14)/Sum(part2,14)))

    def alpha129(self):  
        ####SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)###
        cond = ((self.close-Delay(self.close,1)) < 0)
        part = ops.empty_like(self.close)
        part[cond] = Abs(self.close-Delay(self.close,1))
        part[~cond] = 0
        return Sum(part, 12)
    
    def alpha130(self):   #1657
        ####(RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))###
        return (Rank(Decaylinear(Corr(((self.high + self.low) / 2), Mean(self.volume,40), 9), 10)) / Rank(Decaylinear(Corr(Rank(self.vwap), Rank(self.volume), 7),3)))
    
    def alpha131(self):   #1030   
        ####(RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18))###
        return (Rank(Delta(self.vwap, 1))**Tsrank(Corr(self.close,Mean(self.volume,50), 18), 18))
       
    def alpha132(self):   #1657
        ####MEAN(AMOUNT,20)###
        return Mean(self.amount,20)
    
    def alpha133(self):  
        ####((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100###
        return ((20-Highday(self.high,20))/20)*100-((20-Lowday(self.low,20))/20)*100
    
    def alpha134(self):   #1760
        ####(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME###
        return (self.close-Delay(self.close,12))/Delay(self.close,12)*self.volume
    
    def alpha135(self):   #1744
        ####SMA(DELAY(CLOSE/DELAY(CLOSE,20),1),20,1)###
        return Sma(Delay(self.close/Delay(self.close,20),1),20,1)
    
    def alpha136(self):   #1729
        ####((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))###
        return ((-1 * Rank(Delta(self.returns, 3))) * Corr(self.open, self.volume, 10))
    
    def alpha137(self):  
        ####16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1)))
        A = Abs(self.high- Delay(self.close,1))
        B = Abs(self.low - Delay(self.close,1))
        C = Abs(self.high- Delay(self.low,1))
        D = Abs(Delay(self.close,1)-Delay(self.open,1))          
        cond1 = ((A>B) & (A>C))
        cond2 = ((B>C) & (B>A))
        cond3 = ~cond1 & ~cond2       
        part0 = 16*(self.close + (self.close - self.open)/2 - Delay(self.open,1))
        part1 = ops.empty_like(self.close)
        part1[cond1] = A + B/2 + D/4
        part1[cond2] = B + A/2 + D/4
        part1[cond3] = C + D/4  
        part1.replace({0: None}, inplace=True)
        return part0/part1*Max(A,B)

    def alpha138(self):   #1448
        ####((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)###
        return ((Rank(Decaylinear(Delta((((self.low * 0.7) + (self.vwap *0.3))), 3), 20)) - Tsrank(Decaylinear(Tsrank(Corr(Tsrank(self.low, 8), Tsrank(Mean(self.volume,60), 17), 5), 19), 16), 7)) * -1)
    
    def alpha139(self):   #1729
        ####(-1 * CORR(OPEN, VOLUME, 10))###
        return (-1 * Corr(self.open, self.volume, 10))
    
    def alpha140(self):   #1797
        ####MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))###
        return Min(Rank(Decaylinear(((Rank(self.open) + Rank(self.low)) - (Rank(self.high) + Rank(self.close))), 8)), Tsrank(Decaylinear(Corr(Tsrank(self.close, 8), Tsrank(Mean(self.volume,60), 20), 8), 7), 3))
    
    def alpha141(self):   #1637
        ####(RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9))* -1)###
        return (Rank(Corr(Rank(self.high), Rank(Mean(self.volume,15)), 9))* -1)
    
    def alpha142(self):   #1657
        ####(((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME,20)), 5)))###
        return (((-1 * Rank(Tsrank(self.close, 10))) * Rank(Delta(Delta(self.close, 1), 1))) * Rank(Tsrank((self.volume/Mean(self.volume,20)), 5)))
    
    def alpha144(self):  
        ####SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE,1),20)###
        cond = (self.close<Delay(self.close,1))
        part1 = Abs(self.close/Delay(self.close,1)-1)/self.amount
        return Sumif(part1,20,cond)/Count(cond,20)
    
    def alpha145(self):   #1617
        ####(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100###
        return (Mean(self.volume,9)-Mean(self.volume,26))/Mean(self.volume,12)*100
    
    def alpha146(self):   #1650  
        ####MEAN((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2),20)*((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2))/SMA(((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)-SMA((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1),61,2)))^2,61,2)###
        return Mean((self.close-Delay(self.close,1))/Delay(self.close,1)-Sma((self.close-Delay(self.close,1))/Delay(self.close,1),61,2),20)*((self.close-Delay(self.close,1))/Delay(self.close,1)-Sma((self.close-Delay(self.close,1))/Delay(self.close,1),61,2))/Sma(((self.close-Delay(self.close,1))/Delay(self.close,1)-((self.close-Delay(self.close,1))/Delay(self.close,1)-Sma((self.close-Delay(self.close,1))/Delay(self.close,1),61,2)))**2,61,2)

    def alpha147(self):  
        ####REGBETA(MEAN(CLOSE,12),SEQUENCE(12))###
        return Regbeta(Mean(self.close, 12), Sequence(12))
    
    def alpha148(self):  
        ####((RANK(CORR((OPEN), SUM(MEAN(VOLUME,60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)###
        cond = (Rank(Corr((self.open), Sum(Mean(self.volume,60), 9), 6)) < Rank((self.open - Tsmin(self.open, 14))))
        part = ops.empty_like(self.close)
        part[cond] = -1
        part[~cond] = 0
        return part
    
    def alpha150(self):   #1797
        ####(CLOSE+HIGH+LOW)/3*VOLUME###
        return (self.close+self.high+self.low)/3*self.volume
    
    def alpha151(self):   #1745
        ####SMA(CLOSE-DELAY(CLOSE,20),20,1)###
        return Sma(self.close-Delay(self.close,20),20,1)
    
    def alpha152(self):   #1559
        ####SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1)###
        return Sma(Mean(Delay(Sma(Delay(self.close/Delay(self.close,9),1),9,1),1),12)-Mean(Delay(Sma(Delay(self.close/Delay(self.close,9),1),9,1),1),26),9,1)
    
    def alpha153(self):   #1630
        ####(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4###
        return (Mean(self.close,3)+Mean(self.close,6)+Mean(self.close,12)+Mean(self.close,24))/4
    
    def alpha154(self):  
        ####(((VWAP - MIN(VWAP, 16))) < (CORR(VWAP, MEAN(VOLUME,180), 18)))###
        cond = (((self.vwap - Tsmin(self.vwap, 16))) < (Corr(self.vwap, Mean(self.volume,180), 18)))
        part = ops.empty_like(self.close)
        part[cond] = 1
        part[~cond] = 0
        return part
    
    def alpha155(self):   #1797
        ####SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2)###
        return Sma(self.volume,13,2)-Sma(self.volume,27,2)-Sma(Sma(self.volume,13,2)-Sma(self.volume,27,2),10,2)
    
    def alpha156(self):   #1776
        ####(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)),2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)###
        return (Max(Rank(Decaylinear(Delta(self.vwap, 5), 3)), Rank(Decaylinear(((Delta(((self.open * 0.15) + (self.low *0.85)),2) / ((self.open * 0.15) + (self.low * 0.85))) * -1), 3))) * -1)
    
    def alpha157(self):   #1764
        ####(MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))###
        return (Tsmin(Prod(Rank(Rank(Log(Sum(Tsmin(Rank(Rank((-1 * Rank(Delta((self.close - 1), 5))))), 2), 1)))), 1), 5) + Tsrank(Delay((-1 * self.returns), 6), 5))
    
    def alpha158(self):   #1797
        ####((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE###
        return ((self.high-Sma(self.close,15,2))-(self.low-Sma(self.close,15,2)))/self.close
    
    def alpha159(self):   #1630
        ####((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6)*12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)###
        return ((self.close-Sum(Min(self.low,Delay(self.close,1)),6))/Sum(Max(self.high,Delay(self.close,1))-Min(self.low,Delay(self.close,1)),6)*12*24+(self.close-Sum(Min(self.low,Delay(self.close,1)),12))/Sum(Max(self.high,Delay(self.close,1))-Min(self.low,Delay(self.close,1)),12)*6*24+(self.close-Sum(Min(self.low,Delay(self.close,1)),24))/Sum(Max(self.high,Delay(self.close,1))-Min(self.low,Delay(self.close,1)),24)*6*24)*100/(6*12+6*24+12*24)
    
    def alpha160(self):  
        ####SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)###
        cond = (self.close<=Delay(self.close,1))
        part = ops.empty_like(self.close)
        part[cond] = Std(self.close,20)
        part[~cond] = 0
        return Sma(part, 20, 1)
    
    def alpha161(self):   #1714
        ####MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)###
        return Mean(Max(Max((self.high-self.low),Abs(Delay(self.close,1)-self.high)),Abs(Delay(self.close,1)-self.low)),12)
    
    def alpha162(self):   #1789
        ####(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100,12))###
        return (Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100-Tsmin(Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100,12))/(Sma(Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100,12,1)-Tsmin(Sma(Max(self.close-Delay(self.close,1),0),12,1)/Sma(Abs(self.close-Delay(self.close,1)),12,1)*100,12))
    
    def alpha163(self):   #1657
        ####RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))###
        return Rank(((((-1 * self.returns) * Mean(self.volume,20)) * self.vwap) * (self.high - self.close)))
    
    def alpha164(self):  
        ####SMA(( ((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1) - MIN( ((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1) ,12) )/(HIGH-LOW)*100,13,2)###
        cond = (self.close>Delay(self.close,1))
        part = ops.empty_like(self.close)
        part[cond] = 1/(self.close-Delay(self.close,1))
        part[~cond] = 1

        # 部分无交易或涨停跌停情况下，HIGH=LOW, 此时会有除零问题，使用空值解决
        part2 = self.high-self.low
        part2.replace({0: None}, inplace=True)

        return Sma((part - Tsmin(part,12))/(part2)*100, 13, 2)
    
    def alpha165(self):  # rowmax
        ####MAX(SUMAC(CLOSE-MEAN(CLOSE,48)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,48)))/STD(CLOSE,48)###
        p1 = Rowmax(Sum(self.close-Mean(self.close,48), 48))
        p2 = Rowmin(Sum(self.close-Mean(self.close,48), 48))
        p3 = Std(self.close,48)
        return -1*(1/p3.div(p2, axis = 0)).sub(p1, axis=0)
    
    def alpha166(self):  #公式有问题
        ####-20* ( 20-1 ) ^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)
        p1 = -20* ( 20-1 )**1.5*Sum(self.close/Delay(self.close,1)-1-Mean(self.close/Delay(self.close,1)-1,20),20)
        p2 = ((20-1)*(20-2)*(Sum(Mean(self.close/Delay(self.close,1),20)**2,20))**1.5)
        return p1/p2

    def alpha167(self):  
        ####SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)###
        cond = (self.close > Delay(self.close,1))
        part = ops.empty_like(self.close)
        part[cond] = self.close-Delay(self.close,1)
        part[~cond] = 0
        return Sum(part,12)
    
    def alpha168(self):   #1657
        ####(-1*VOLUME/MEAN(VOLUME,20))###
        return (-1*self.volume/Mean(self.volume,20))
    
    def alpha169(self):   #1610
        ####SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),26),10,1)###
        return Sma(Mean(Delay(Sma(self.close-Delay(self.close,1),9,1),1),12)-Mean(Delay(Sma(self.close-Delay(self.close,1),9,1),1),26),10,1)
    
    def alpha170(self):   #1657
        ####((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) /5))) - RANK((VWAP - DELAY(VWAP, 5))))###
        return ((((Rank((1 / self.close)) * self.volume) / Mean(self.volume,20)) * ((self.high * Rank((self.high - self.close))) / (Sum(self.high, 5) /5))) - Rank((self.vwap - Delay(self.vwap, 5))))
   
    def alpha171(self):   #1789
        ####((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))###
        return ((-1 * ((self.low - self.close) * (self.open**5))) / ((self.close - self.high) * (self.close**5)))
    
    def alpha172(self):  
        ####MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 &HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)
        TR = Max(Max(self.high-self.low,Abs(self.high-Delay(self.close,1))),Abs(self.low-Delay(self.close,1)))
        HD = self.high-Delay(self.high,1)
        LD = Delay(self.low,1)-self.low
        cond1 = ((LD>0) & (LD>HD))
        cond2 = ((HD>0) & (HD>LD)) 
        part1 = ops.empty_like(self.close)
        part1[cond1] = LD
        part1[~cond1] = 0
        part2 = ops.empty_like(self.close)
        part2[cond2] = HD
        part2[~cond2] = 0
        return Mean(Abs(Sum(part1,14)*100/Sum(TR,14)-Sum(part2,14)*100/Sum(TR,14))/(Sum(part1,14)*100/Sum(TR,14)+Sum(part2,14)*100/Sum(TR,14))*100,6)
    
    def alpha173(self):   #1797
        ####3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)###
        return 3*Sma(self.close,13,2)-2*Sma(Sma(self.close,13,2),13,2)+Sma(Sma(Sma(Log(self.close),13,2),13,2),13,2)
    
    def alpha174(self):  
        ####SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)###
        cond = (self.close>Delay(self.close,1))
        part = ops.empty_like(self.close)
        part[cond] = Std(self.close,20)
        part[~cond] = 0
        return Sma(part,20,1)
    
    def alpha175(self):   #1759
        ####MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),6)###
        return Mean(Max(Max((self.high-self.low),Abs(Delay(self.close,1)-self.high)),Abs(Delay(self.close,1)-self.low)),6)
    
    def alpha176(self):   #1678
        ####CORR(RANK(((CLOSE - TSMIN(LOW, 12)) / (TSMAX(HIGH, 12) - TSMIN(LOW,12)))), RANK(VOLUME), 6)###
        return Corr(Rank(((self.close - Tsmin(self.low, 12)) / (Tsmax(self.high, 12) - Tsmin(self.low,12)))), Rank(self.volume), 6)
    
    def alpha177(self):  
        ####((20-HIGHDAY(HIGH,20))/20)*100###
        return ((20-Highday(self.high,20))/20)*100
    
    def alpha178(self):   #1790
        ####(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME###
        return (self.close-Delay(self.close,1))/Delay(self.close,1)*self.volume
    
    def alpha179(self):   #1421   数据量较少
        ####(RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))###
        return (Rank(Corr(self.vwap, self.volume, 4)) *Rank(Corr(Rank(self.low), Rank(Mean(self.volume,50)), 12)))
    
    def alpha180(self):  #指标有问题
        ####((MEAN(VOLUME,20) < VOLUME) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 *VOLUME)))
        cond = (Mean(self.volume,20) < self.volume)
        part = ops.empty_like(self.close)
        part[cond] = (-1 * Tsrank(Abs(Delta(self.close, 7)), 60)) * Sign(Delta(self.close, 7)) 
        part[~cond] = -1 * self.volume
        return part
    
    def alpha181(self):   #1532  公式有问题，假设后面的sum周期为20
        ####SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-(BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^2,20)/SUM((BANCHMARKINDEXCLOSE-MEAN(BANCHMARKINDEXCLOSE,20))^3)###
        return Sum(((self.close/Delay(self.close,1)-1)-Mean((self.close/Delay(self.close,1)-1),20))-(self.benchmark_close-Mean(self.benchmark_close,20))**2,20)/Sum(((self.benchmark_close-Mean(self.benchmark_close,20))**3),20)
    
    def alpha182(self):  
        ####COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20###
        return Count((((self.close>self.open) & (self.benchmark_close>self.benchmark_open)) | ((self.close<self.open) & (self.benchmark_close<self.benchmark_open))),20)/20
    
    def alpha183(self):  
        ###MAX(SUMAC(CLOSE-MEAN(CLOSE,24)))-MIN(SUMAC(CLOSE-MEAN(CLOSE,24)))/STD(CLOSE,24)###
        p1 = Rowmax(Sum(self.close-Mean(self.close,24), 24))
        p2 = Rowmin(Sum(self.close-Mean(self.close,24), 24))
        p3 = Std(self.close,24)
        return -1*(1/p3.div(p2, axis = 0)).sub(p1, axis=0)
    
    def alpha184(self):   #983   数据量较少
        ####(RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))###
        return (Rank(Corr(Delay((self.open - self.close), 1), self.close, 200)) + Rank((self.open - self.close)))
    
    def alpha185(self):   #1797
        ####RANK((-1 * ((1 - (OPEN / CLOSE))^2)))###
        return Rank((-1 * ((1 - (self.open / self.close))**2)))
    
    def alpha186(self):  
        ####(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2
        TR = Max(Max(self.high-self.low,Abs(self.high-Delay(self.close,1))),Abs(self.low-Delay(self.close,1)))
        HD = self.high-Delay(self.high,1)
        LD = Delay(self.low,1)-self.low
        cond1 = ((LD>0) & (LD>HD))
        cond2 = ((HD>0) & (HD>LD)) 
        part1 = ops.empty_like(self.close)
        part1[cond1] = LD
        part1[~cond1] = 0
        part2 = ops.empty_like(self.close)
        part2[cond2] = HD
        part2[~cond2] = 0
        return (Mean(Abs(Sum(part1,14)*100/Sum(TR,14)-Sum(part2,14)*100/Sum(TR,14))/(Sum(part1,14)*100/Sum(TR,14)+Sum(part2,14)*100/Sum(TR,14))*100,6)+Delay(Mean(Abs(Sum(part1,14)*100/Sum(TR,14)-Sum(part2,14)*100/Sum(TR,14))/(Sum(part1,14)*100/Sum(TR,14)+Sum(part2,14)*100/Sum(TR,14))*100,6),6))/2
    
    def alpha187(self):  
        ####SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)###
        cond = (self.open<=Delay(self.open,1))
        part = ops.empty_like(self.close)
        part[cond] = 0
        part[~cond] = Max((self.high-self.open),(self.open-Delay(self.open,1)))
        return Sum(part,20) 
    
    def alpha188(self):   #1797
        ####((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100###
        return ((self.high-self.low-Sma(self.high-self.low,11,2))/Sma(self.high-self.low,11,2))*100
    
    def alpha189(self):   #1721
        ####MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)###
        return Mean(Abs(self.close-Mean(self.close,6)),6)
    
    def alpha191(self):   #1721
        ####((CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)###
        return ((Corr(Mean(self.volume,20), self.low, 5) + ((self.high + self.low) / 2)) - self.close)
    
_METHOD_BY_NAME = {
    "GTJA_001": "alpha001",
    "GTJA_002": "alpha002",
    "GTJA_003": "alpha003",
    "GTJA_004": "alpha004",
    "GTJA_005": "alpha005",
    "GTJA_006": "alpha006",
    "GTJA_007": "alpha007",
    "GTJA_008": "alpha008",
    "GTJA_009": "alpha009",
    "GTJA_010": "alpha010",
    "GTJA_011": "alpha011",
    "GTJA_012": "alpha012",
    "GTJA_013": "alpha013",
    "GTJA_014": "alpha014",
    "GTJA_015": "alpha015",
    "GTJA_016": "alpha016",
    "GTJA_017": "alpha017",
    "GTJA_018": "alpha018",
    "GTJA_019": "alpha019",
    "GTJA_020": "alpha020",
    "GTJA_021": "alpha021",
    "GTJA_022": "alpha022",
    "GTJA_023": "alpha023",
    "GTJA_024": "alpha024",
    "GTJA_025": "alpha025",
    "GTJA_026": "alpha026",
    "GTJA_027": "alpha027",
    "GTJA_028": "alpha028",
    "GTJA_029": "alpha029",
    "GTJA_031": "alpha031",
    "GTJA_032": "alpha032",
    "GTJA_033": "alpha033",
    "GTJA_034": "alpha034",
    "GTJA_035": "alpha035",
    "GTJA_036": "alpha036",
    "GTJA_037": "alpha037",
    "GTJA_038": "alpha038",
    "GTJA_039": "alpha039",
    "GTJA_040": "alpha040",
    "GTJA_041": "alpha041",
    "GTJA_042": "alpha042",
    "GTJA_043": "alpha043",
    "GTJA_044": "alpha044",
    "GTJA_045": "alpha045",
    "GTJA_046": "alpha046",
    "GTJA_047": "alpha047",
    "GTJA_048": "alpha048",
    "GTJA_049": "alpha049",
    "GTJA_050": "alpha050",
    "GTJA_051": "alpha051",
    "GTJA_052": "alpha052",
    "GTJA_053": "alpha053",
    "GTJA_054": "alpha054",
    "GTJA_055": "alpha055",
    "GTJA_056": "alpha056",
    "GTJA_057": "alpha057",
    "GTJA_058": "alpha058",
    "GTJA_059": "alpha059",
    "GTJA_060": "alpha060",
    "GTJA_061": "alpha061",
    "GTJA_062": "alpha062",
    "GTJA_063": "alpha063",
    "GTJA_064": "alpha064",
    "GTJA_065": "alpha065",
    "GTJA_066": "alpha066",
    "GTJA_067": "alpha067",
    "GTJA_068": "alpha068",
    "GTJA_069": "alpha069",
    "GTJA_070": "alpha070",
    "GTJA_071": "alpha071",
    "GTJA_072": "alpha072",
    "GTJA_073": "alpha073",
    "GTJA_074": "alpha074",
    "GTJA_075": "alpha075",
    "GTJA_076": "alpha076",
    "GTJA_077": "alpha077",
    "GTJA_078": "alpha078",
    "GTJA_079": "alpha079",
    "GTJA_080": "alpha080",
    "GTJA_081": "alpha081",
    "GTJA_082": "alpha082",
    "GTJA_083": "alpha083",
    "GTJA_084": "alpha084",
    "GTJA_085": "alpha085",
    "GTJA_086": "alpha086",
    "GTJA_087": "alpha087",
    "GTJA_088": "alpha088",
    "GTJA_089": "alpha089",
    "GTJA_090": "alpha090",
    "GTJA_091": "alpha091",
    "GTJA_092": "alpha092",
    "GTJA_093": "alpha093",
    "GTJA_094": "alpha094",
    "GTJA_095": "alpha095",
    "GTJA_096": "alpha096",
    "GTJA_097": "alpha097",
    "GTJA_098": "alpha098",
    "GTJA_099": "alpha099",
    "GTJA_100": "alpha100",
    "GTJA_101": "alpha101",
    "GTJA_102": "alpha102",
    "GTJA_103": "alpha103",
    "GTJA_104": "alpha104",
    "GTJA_105": "alpha105",
    "GTJA_106": "alpha106",
    "GTJA_107": "alpha107",
    "GTJA_108": "alpha108",
    "GTJA_109": "alpha109",
    "GTJA_110": "alpha110",
    "GTJA_111": "alpha111",
    "GTJA_112": "alpha112",
    "GTJA_113": "alpha113",
    "GTJA_114": "alpha114",
    "GTJA_115": "alpha115",
    "GTJA_116": "alpha116",
    "GTJA_117": "alpha117",
    "GTJA_118": "alpha118",
    "GTJA_119": "alpha119",
    "GTJA_120": "alpha120",
    "GTJA_121": "alpha121",
    "GTJA_122": "alpha122",
    "GTJA_123": "alpha123",
    "GTJA_124": "alpha124",
    "GTJA_125": "alpha125",
    "GTJA_126": "alpha126",
    "GTJA_127": "alpha127",
    "GTJA_128": "alpha128",
    "GTJA_129": "alpha129",
    "GTJA_130": "alpha130",
    "GTJA_131": "alpha131",
    "GTJA_132": "alpha132",
    "GTJA_133": "alpha133",
    "GTJA_134": "alpha134",
    "GTJA_135": "alpha135",
    "GTJA_136": "alpha136",
    "GTJA_137": "alpha137",
    "GTJA_138": "alpha138",
    "GTJA_139": "alpha139",
    "GTJA_140": "alpha140",
    "GTJA_141": "alpha141",
    "GTJA_142": "alpha142",
    "GTJA_144": "alpha144",
    "GTJA_145": "alpha145",
    "GTJA_146": "alpha146",
    "GTJA_147": "alpha147",
    "GTJA_148": "alpha148",
    "GTJA_150": "alpha150",
    "GTJA_151": "alpha151",
    "GTJA_152": "alpha152",
    "GTJA_153": "alpha153",
    "GTJA_154": "alpha154",
    "GTJA_155": "alpha155",
    "GTJA_156": "alpha156",
    "GTJA_157": "alpha157",
    "GTJA_158": "alpha158",
    "GTJA_159": "alpha159",
    "GTJA_160": "alpha160",
    "GTJA_161": "alpha161",
    "GTJA_162": "alpha162",
    "GTJA_163": "alpha163",
    "GTJA_164": "alpha164",
    "GTJA_165": "alpha165",
    "GTJA_166": "alpha166",
    "GTJA_167": "alpha167",
    "GTJA_168": "alpha168",
    "GTJA_169": "alpha169",
    "GTJA_170": "alpha170",
    "GTJA_171": "alpha171",
    "GTJA_172": "alpha172",
    "GTJA_173": "alpha173",
    "GTJA_174": "alpha174",
    "GTJA_175": "alpha175",
    "GTJA_176": "alpha176",
    "GTJA_177": "alpha177",
    "GTJA_178": "alpha178",
    "GTJA_179": "alpha179",
    "GTJA_180": "alpha180",
    "GTJA_181": "alpha181",
    "GTJA_182": "alpha182",
    "GTJA_183": "alpha183",
    "GTJA_184": "alpha184",
    "GTJA_185": "alpha185",
    "GTJA_186": "alpha186",
    "GTJA_187": "alpha187",
    "GTJA_188": "alpha188",
    "GTJA_189": "alpha189",
    "GTJA_191": "alpha191",
}


def _broadcast_benchmark(market_prices, prices: pd.DataFrame):
    """将基准收盘价广播为 date×code（仅基准因子需要时调用）。"""
    if market_prices is None:
        return None, None
    if isinstance(market_prices, pd.DataFrame):
        bm_close = market_prices.iloc[:, 0].reindex(prices.index).ffill()
    else:
        bm_close = market_prices.reindex(prices.index).ffill()
    bm_open = bm_close.shift(1)
    close_arr = np.broadcast_to(
        bm_close.to_numpy(dtype=np.float64)[:, None], prices.shape
    ).copy()
    open_arr = np.broadcast_to(
        bm_open.to_numpy(dtype=np.float64)[:, None], prices.shape
    ).copy()
    return (
        pd.DataFrame(close_arr, index=prices.index, columns=prices.columns),
        pd.DataFrame(open_arr, index=prices.index, columns=prices.columns),
    )


def iter_alpha191_factors(
    prices: pd.DataFrame,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
    clean_ret: pd.DataFrame = None,
    market_prices: pd.DataFrame = None,
    factor_names: set[str] | list[str] | None = None,
):
    """
    流式产出 Alpha191 因子 (name, normalized_panel)。

    逐个计算 → normalize → yield，不在内存中累积全部面板（~187 张会 OOM）。
    """
    import gc

    wanted = set(factor_names) if factor_names is not None else None

    def _want(name: str) -> bool:
        return wanted is None or name in wanted

    if any(x is None for x in (open_, high, low, volume)):
        logger.warning("Alpha191 跳过：缺少 open/high/low/volume")
        return

    names = [n for n in _METHOD_BY_NAME if _want(n)]
    if not names:
        return

    need_bm = any(n in BENCHMARK_NAMES for n in names)

    vwap, vwap_note = compute_vwap(amount, volume, high, low, prices)
    if vwap is None:
        logger.warning("Alpha191 跳过：无法构建 VWAP")
        return
    if vwap_note != "amount/volume":
        logger.info(f"Alpha191 VWAP 近似: {vwap_note}")

    amount_panel = amount
    if amount_panel is None:
        amount_panel = prices * volume
        logger.info("Alpha191 amount 近似: close*volume")

    returns = clean_ret if clean_ret is not None else prices.pct_change()

    bm_close_df = bm_open_df = None
    if need_bm:
        bm_close_df, bm_open_df = _broadcast_benchmark(market_prices, prices)

    ctx = _GTJAAlphas(
        close=prices, open_=open_, high=high, low=low,
        volume=volume, amount=amount_panel, vwap=vwap, returns=returns,
        benchmark_open=bm_open_df, benchmark_close=bm_close_df,
    )

    n_out = 0
    skipped = []
    for name in names:
        if name in SKIP_NAMES:
            skipped.append(name)
            continue
        if name in BENCHMARK_NAMES and bm_close_df is None:
            skipped.append(name)
            continue
        meth = _METHOD_BY_NAME[name]
        try:
            result = getattr(ctx, meth)()
            if result is None:
                skipped.append(name)
                continue
            if isinstance(result, pd.Series):
                skipped.append(name)
                continue
            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame(result, index=prices.index, columns=prices.columns)
            if result.isna().all(axis=None):
                skipped.append(name)
                continue
            panel = _normalize(result)
            del result
            yield name, panel
            n_out += 1
            if n_out % 20 == 0:
                gc.collect()
        except Exception as e:
            logger.warning(f"Alpha191 {name} 计算失败: {e}")
            skipped.append(name)

    del ctx
    if bm_close_df is not None:
        del bm_close_df, bm_open_df

    subset = "" if wanted is None else f" (白名单 {len(names)} 个)"
    logger.info(
        f"Alpha191 因子: 流式完成 {n_out} 个{subset}；"
        f"skip/失败 {len(skipped)}；硬跳过 {sorted(SKIP_NAMES)}"
    )


def get_alpha191_factors(
    prices: pd.DataFrame,
    open_: pd.DataFrame = None,
    high: pd.DataFrame = None,
    low: pd.DataFrame = None,
    volume: pd.DataFrame = None,
    amount: pd.DataFrame = None,
    clean_ret: pd.DataFrame = None,
    market_prices: pd.DataFrame = None,
    factor_names: set[str] | list[str] | None = None,
) -> dict:
    """
    返回 Alpha191 因子字典（已截面标准化）。

    跳过 SKIP_NAMES；基准类在 market_prices 缺失时跳过。
    警告：全量 dict 会 OOM；内存敏感请用 ``iter_alpha191_factors`` 或小白名单。
    """
    return dict(
        iter_alpha191_factors(
            prices=prices, open_=open_, high=high, low=low,
            volume=volume, amount=amount, clean_ret=clean_ret,
            market_prices=market_prices, factor_names=factor_names,
        )
    )
