# WithdrawChan Core Module README

[English](README.md) | [简体中文](../doc/src.README.zh.md)

This directory contains WithdrawChan's withdrawal amount recommendation rule. Given the minimum amount to withdraw this time and an optional upper limit, it returns a randomly recommended withdrawal amount. The calculation uses only these two inputs and public constants. It does not need the balance, a pool snapshot or on-chain data. For the rationale behind the recommendation, see the [repository README](../README.md).

`python/` and `js/` contain the Python and TypeScript implementations of this rule, respectively. The examples below use Python. For the TypeScript equivalents, see [Implementation correspondence](#implementation-correspondence).

## Notation

| Symbol | Code | Meaning |
| --- | --- | --- |
| $X$ | `x` | Minimum amount to withdraw. Must be greater than 0 |
| $M$ | `m` | Maximum amount the user is willing to withdraw. Must be a multiple of $D$ and no less than $X$. If omitted, defaults to $\lceil 2X \rceil_D$ (`default_m`) |
| $Y$ | Return value | Recommended amount, $X \le Y \le M$ |
| $D$ | `d` | Rounding unit. Must be a power of 10, such as 1, 10 or 100, and must be passed explicitly. See [`units.json`](units.json) for asset-specific values. Use `unit(code, decimals)` to convert them into the asset's smallest unit |
| $q$ | `q`, default `Q` $= 1/2$ | Probability of climbing from each rung to the next |
| $p$ | `p`, default `P` $= 0$ | Probability of adding a random amount below one rounding unit |

Notes:
- All amounts are represented as integers in the asset's smallest unit. For example, 1 USDT is represented as $10^6$.
- $X$, $M$, and $Y$ refer to the protocol withdrawal amount (`withdrawnValue`) before deducting the relayer fee, not the amount actually received.
- $\lceil v \rceil_D = \lceil v/D \rceil \cdot D$ means rounding $v$ up to the nearest multiple of $D$.

## How the rule works

1. Construct a sequence of rungs $c_0 < c_1 < \cdots$. The first rung is $c_0 = \lceil X \rceil_D$. The second rounds $c_0$ up to one significant digit (for example, 44 USDT becomes 50 USDT), namely $\lceil c_0 \rceil_{10^e}$, where $e+1$ is the number of decimal digits in $c_0$ expressed in the asset's smallest unit. Subsequent rungs are $10^{e+1}, 10^{e+2}, \ldots$. Remove duplicate amounts and retain only those no greater than $M$. Denote the highest remaining rung by $c_J$. $M$ itself is not added to the sequence.
2. Start at $c_0$. At each rung, climb to the next rung with probability $q$, or stop with probability $1-q$. Stop automatically upon reaching $c_J$. Denote the amount at which the process stops by $Y_0$.
3. With probability $p$, sample $U$ uniformly from $\{0, \ldots, D-1\}$. If $Y_0 + U \le M$, set $Y = Y_0 + U$. Otherwise, set $Y = Y_0$. If no sample is taken, set $Y = Y_0$. (Currently $p = 0$, so this step **does not occur in practice**.)

`distribution` returns the following exact distribution:

$$
P(Y_0=c_j)=\begin{cases} q^{j}(1-q) & j<J \\ q^{J} & j=J \end{cases}
\qquad
T(y\mid v)=\begin{cases}
(1-p)+\dfrac{p}{D}\bigl(1+\max(0,\ v+D-1-M)\bigr) & y=v \\[4pt]
\dfrac{p}{D} & v<y\le\min(v+D-1,\ M) \\[4pt]
0 & \text{otherwise}
\end{cases}
$$

$$P(Y=y\mid X,M)=\sum_{j=0}^{J} P(Y_0=c_j)\, T(y\mid c_j)$$

**Effect of the upper limit:** When $p = 0$ and $X$ and $D$ are fixed, the ratio of the probabilities of obtaining the same output $y$ under any two valid upper limits no smaller than $y$ lies between $1-q$ and $1/(1-q)$. With the current $q$, that is 1/2 to 2. In experiments on Privacy Pools v1 with simulated observers, adding the random walk can increase min-entropy. See [`lab/README.md`](../lab/README.md) for details.

## Examples

The following examples use USDT, with $D = 1$ USDT, $q = 1/2$, and $p = 0$.

| $X$ | $M$ | Rungs | Outputs and probabilities |
| --- | --- | --- | --- |
| 437 | 874 (default) | 437, 500 | 437: 1/2, 500: 1/2 |
| 520 | 1040 (default) | 520, 600, 1000 | 520: 1/2, 600: 1/4, 1000: 1/4 |
| 43.21 | 87 (default) | 44, 50 | 44: 1/2, 50: 1/2 |
| 43.21 | 44 | 44 | 44: 1 |
| 0.3 | 1 (default) | 1 | 1: 1 |
| 43.21 | 10000 | 44, 50, 100, 1000, 10000 | 44: 1/2, 50: 1/4, 100: 1/8, 1000: 1/16, 10000: 1/16 |

For the full list, see [`examples.json`](examples.json), which is included in the tests for both implementations.

## Usage

```python
from withdrawchan import distribution, recommend, unit

USDT = 10**6
d = unit("USD-stable", 6)                  # 10**6
recommend(43_210_000, 120 * USDT, d=d)     # e.g. 50000000
distribution(43_210_000, 120 * USDT, d=d)
# {44000000: Fraction(1, 2), 50000000: Fraction(1, 4), 100000000: Fraction(1, 4)}
```

By default, `recommend` draws random numbers using `secrets.randbelow`. If the rungs have already been constructed, call `walk` and `walk_distribution` directly. `decimal_rungs` and `default_m` return the rungs and the default upper limit, respectively.

## Implementation correspondence

| Python | TypeScript |
| --- | --- |
| `recommend(x, m=None, *, d, q=Q, p=P, rng=secrets.randbelow)` | `recommend(x, m \| undefined, { d, q?, p?, rng? })` |
| `distribution(...)` → `dict[int, Fraction]` | `distribution(...)` → `Map<bigint, Fraction>` |
| `walk`, `walk_distribution` | `walk`, `walkDistribution` |
| `decimal_rungs`, `default_m` | `decimalRungs`, `defaultM` |
| `UNITS`, `unit(code, decimals)` | `UNITS`, `unit(code, decimals)` |
| `AmountError(ValueError)` | `AmountError(RangeError)` |
| Amounts are `int`, probabilities are `fractions.Fraction` | Amounts are `bigint`, probabilities are `{ num, den }` |
| `secrets.randbelow` | `secureRng` (`crypto.getRandomValues` with rejection sampling) |

Both implementations use random numbers in the same way. `rng(n)` returns an integer uniformly distributed over $[0, n)$. A coin with probability $a/b$ of heads calls `rng(b)` once and counts a result below $a$ as heads. Probabilities 0 and 1 make no call. Sampling the random amount makes one more call, `rng(D)`. Thus, feeding the same sequence of random numbers into both implementations produces the same $Y$.

[`vectors.json`](vectors.json) is generated by Python through exhaustive enumeration of all random-number paths. It records each path's sequence of calls, output, and exact distribution, and serves as the test data for both implementations. Python uses pytest, while TypeScript uses Node's built-in `node:test`. For instructions on building, testing, and regenerating the vectors, see [`SKILL.md`](SKILL.md).
