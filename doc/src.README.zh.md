# WithdrawChan 核心模块 README

[English](../src/README.md) | [简体中文](src.README.zh.md)

本目录是 WithdrawChan 的金额推荐规则：输入这次至少要提领的金额，以及可选的上限，返回一个随机推荐的提领金额。计算只用这两个输入和公开常量，不需要余额、池子快照或链上数据。推荐理由可见 [仓库 README](README.zh.md)。

`python/` 和 `js/` 分别是这条规则的 Python 和 TypeScript 实现。下文以 Python 为例，TypeScript 的写法见[两种实现的符号对照](#两种实现的符号对照)。

## 符号对照表

| 公式  | 代码                 | 含义                                                                   |
| --- | ------------------ | -------------------------------------------------------------------- |
| $X$ | `x`                | 至少需要提领的金额，须大于 0                                                      |
| $M$ | `m`                | 最多愿意提领的金额，须是 $D$ 的整数倍且不小于 $X$；省略时取 $\lceil 2X \rceil_D$（`default_m`） |
| $Y$ | 返回值                | 推荐金额，$X \le Y \le M$                                                 |
| $D$ | `d`                | 取整单位，须是 1、10、100 这样的 10 的幂，调用时须显式传入；各类资产的取值见 [`units.json`](../src/units.json)，用 `unit(code, decimals)` 换算成最小单位 |
| $q$ | `q`，默认 `Q` $= 1/2$ | 每次再上一级台阶的概率                                                           |
| $p$ | `p`，默认 `P` $= 0$   | 附加随机小数的概率                                                            |

注意：
- 金额都以资产最小单位的整数表示，例如 1 USDT 记为 $10^6$
- $X$、$M$、$Y$ 指扣除中继费之前的协议提领额（`withdrawnValue`），不是实际到账金额。
- $\lceil v \rceil_D = \lceil v/D \rceil \cdot D$ 表示把 $v$ 向上取到 $D$ 的整数倍。

## 规则解释

1. 构造阶梯 $c_0 < c_1 < \cdots$。第一级台阶是 $c_0 = \lceil X \rceil_D$；第二级台阶把 $c_0$ 向上取到一位有效数字（例如 44 USDT 取到 50 USDT），即 $\lceil c_0 \rceil_{10^e}$，其中 $e+1$ 是 $c_0$ 以最小单位计的十进制位数；之后依次是 $10^{e+1}, 10^{e+2}, \ldots$。去掉重复金额，只保留不超过 $M$ 的部分，最高的一级台阶记为 $c_J$。$M$ 本身不会成为一级台阶。
2. 从 $c_0$ 出发，在每级台阶上以概率 $q$ 再上一级台阶，以概率 $1-q$ 停下；走到 $c_J$ 时直接停下。停下时的金额记为 $Y_0$。
3. 以概率 $p$ 从 $\{0, \ldots, D-1\}$ 中均匀抽取 $U$：若 $Y_0 + U \le M$，则 $Y = Y_0 + U$，否则 $Y = Y_0$。没有抽取时 $Y = Y_0$。（当前 $p = 0$，所以这一步 **实际上不会发生**）

`distribution` 返回的就是下面这个精确分布：

$$
P(Y_0=c_j)=\begin{cases} q^{j}(1-q) & j<J \\ q^{J} & j=J \end{cases}
\qquad
T(y\mid v)=\begin{cases}
(1-p)+\dfrac{p}{D}\bigl(1+\max(0,\ v+D-1-M)\bigr) & y=v \\[4pt]
\dfrac{p}{D} & v<y\le\min(v+D-1,\ M) \\[4pt]
0 & \text{其他}
\end{cases}
$$

$$P(Y=y\mid X,M)=\sum_{j=0}^{J} P(Y_0=c_j)\, T(y\mid c_j)$$

**上限的影响**：在 $p = 0$、$X$ 和 $D$ 固定时，同一个输出 $y$ 在任意两个不小于 $y$ 的合法上限下出现的概率，比值介于 $1-q$ 和 $1/(1-q)$ 之间，当前也就是 1/2 到 2。在 Privacy Pools v1 与模拟观察者实验中，增加概率计算的过程可以增大最小熵，详情见 [`lab/README.md`](lab.README.zh.md)。

## 示例

下表以 USDT 为例：$D = 1$ USDT，$q = 1/2$，$p = 0$。

| $X$   | $M$      | 阶梯                       | 输出及概率                                      |
| ----- | -------- | ------------------------ | ------------------------------------------ |
| 437   | 874（默认）  | 437, 500                 | 437：1/2，500：1/2                            |
| 520   | 1040（默认） | 520, 600, 1000           | 520：1/2，600：1/4，1000：1/4                   |
| 43.21 | 87（默认）   | 44, 50                   | 44：1/2，50：1/2                              |
| 43.21 | 44       | 44                       | 44：1                                       |
| 0.3   | 1（默认）    | 1                        | 1：1                                        |
| 43.21 | 10000    | 44, 50, 100, 1000, 10000 | 44：1/2，50：1/4，100：1/8，1000：1/16，10000：1/16 |

完整列表见 [`examples.json`](../src/examples.json)，两种实现的测试都将其包含在内。

## 调用方法

```python
from withdrawchan import distribution, recommend, unit

USDT = 10**6
d = unit("USD-stable", 6)                  # 10**6
recommend(43_210_000, 120 * USDT, d=d)     # 例如 50000000
distribution(43_210_000, 120 * USDT, d=d)
# {44000000: Fraction(1, 2), 50000000: Fraction(1, 4), 100000000: Fraction(1, 4)}
```

`recommend` 默认用 `secrets.randbelow` 取随机数。如果已经有现成的阶梯，可以直接调用 `walk` 和 `walk_distribution`；`decimal_rungs` 和 `default_m` 分别单独返回阶梯和默认上限。

## 两种实现的符号对照

| Python | TypeScript |
| --- | --- |
| `recommend(x, m=None, *, d, q=Q, p=P, rng=secrets.randbelow)` | `recommend(x, m \| undefined, { d, q?, p?, rng? })` |
| `distribution(...)` → `dict[int, Fraction]` | `distribution(...)` → `Map<bigint, Fraction>` |
| `walk`、`walk_distribution` | `walk`、`walkDistribution` |
| `decimal_rungs`、`default_m` | `decimalRungs`、`defaultM` |
| `UNITS`、`unit(code, decimals)` | `UNITS`、`unit(code, decimals)` |
| `AmountError(ValueError)` | `AmountError(RangeError)` |
| 金额 `int`，概率 `fractions.Fraction` | 金额 `bigint`，概率 `{ num, den }` |
| `secrets.randbelow` | `secureRng`（`crypto.getRandomValues` 加拒绝采样） |

两种实现使用随机数的方式相同。`rng(n)` 返回 $[0, n)$ 中的均匀整数；一枚正面概率为 $a/b$ 的硬币调用一次 `rng(b)`，结果小于 $a$ 就算正面；概率为 0 或 1 时不调用。抽取随机小数时再调用一次 `rng(D)`。因此给两边输入同一串随机数，得到的 $Y$ 也相同。

[`vectors.json`](../src/vectors.json) 由 Python 穷举所有随机数路径生成，记录了每条路径的调用序列、输出和精确分布，这是两套测试的数据基础。Python 的测试框架是 pytest，TypeScript 用 Node 内置的 `node:test`。构建、测试和重新生成向量的步骤见 [`SKILL.md`](../src/SKILL.md)。
