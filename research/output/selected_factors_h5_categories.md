# IC 入选因子分类（h5）

类别说明：
- **普通因子**：稠密池全样本过线（IC/ICIR/t/FDR/corr）→ `factors`
- **稀疏因子**：语义稀疏池独立轨道（同向IC胜率 + 触发日截面胜率，按 sign(mean_IC) 对齐），仅建议 ridge 注入
- **新兴因子**：全样本未过 IC/ICIR 门 + 近窗 FDR∧ICIR∧lift；**仅观察**（`factors_emerging`），不进 ML 主池
- **衰减因子**（警示标签，仍在 factors 池）：R 塌缩 ∧ |ICIR_recent| 弱 ∧ |IC_recent| 弱（合取；近窗与 barra pure 对齐）
- **风格逆转**（警示标签，可与衰减叠加）：近一季多数强 IC 与全样本符号相反

## 普通因子（45）
- GTJA_099
- WQ_055
- GTJA_042  [衰减因子]
- GTJA_016
- GTJA_064
- GTJA_032
- GTJA_140  [衰减因子]
- A158_CORD20  [衰减因子]
- GTJA_105
- GTJA_179  [衰减因子]
- WQ_012
- WQ_026
- A158_CORR20  [衰减因子]
- GTJA_074  [衰减因子]
- A158_VSUMN60
- A158_QTLD60
- GTJA_156
- A158_VMA60  [衰减因子]
- GTJA_001
- A158_LOW0  [衰减因子]
- A158_IMAX30  [衰减因子]
- WQ_071
- GTJA_145  [衰减因子]
- GTJA_121  [衰减因子]
- A158_VSUMN10  [衰减因子]
- GTJA_092  [衰减因子]
- A158_VSUMP5
- WQ_037
- GTJA_029
- GTJA_086  [衰减因子]
- GTJA_048  [衰减因子]
- A158_IMIN20  [衰减因子]
- WQ_066  [衰减因子]
- GTJA_058
- GTJA_045  [衰减因子]
- GTJA_044  [衰减因子]
- A158_VMA5  [衰减因子]
- GTJA_131
- GTJA_006
- A158_RESI10
- GTJA_039  [衰减因子]
- 动量_120d
- GTJA_125
- GTJA_142  [衰减因子]
- GTJA_091  [衰减因子]

## 稀疏因子（11）
- 龙虎榜连续上榜
- 连板数
- 涨跌停净强度_20d
- 龙虎榜换手上榜_20d
- 龙虎榜涨幅上榜_20d
- 龙虎榜跌幅上榜规避_20d
- 解禁定增压力_60d
- 龙虎榜净买占比_20d
- 未来30日解禁次数
- 龙虎榜机构净买入_20d
- 大宗机构接盘_20d

## 新兴因子（3）
- 营收增长率
- GTJA_013
- WQ_098

## 衰减因子（23）
- GTJA_042
- GTJA_140
- A158_CORD20
- GTJA_179
- A158_CORR20
- GTJA_074
- A158_VMA60
- A158_LOW0
- A158_IMAX30
- GTJA_145
- GTJA_121
- A158_VSUMN10
- GTJA_092
- GTJA_086
- GTJA_048
- A158_IMIN20
- WQ_066
- GTJA_045
- GTJA_044
- A158_VMA5
- GTJA_039
- GTJA_142
- GTJA_091

## 风格逆转（0）
