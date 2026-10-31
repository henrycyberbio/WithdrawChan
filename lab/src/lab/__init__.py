"""Reproducible experiments on withdrawal-amount privacy in privacy pools.

Notation shared by every module.  Amounts are integers in the asset's smallest unit.

People

- deposit: one positive deposit made before the train/holdout split, owned by one person who
  withdraws it in parts.  Deposits are indexed `i = 0 .. N-1` (`n` in code).
- `b`: a deposit's initial balance, which is public.
- kind: `NON` withdraws without the tool, following the historical model; `KEEP` uses the
  tool while `r >= threshold` and then stops, leaving the remainder in the pool; `DRAIN` uses
  the tool likewise and then withdraws the whole remainder below `threshold` at once.
- adopter: a `KEEP` or `DRAIN` person.  `rho` is the adoption ratio and
  `K = floor(rho*N + 1/2)` the number of adopters (`k` in code).

One withdrawal

- `r`: the deposit's remaining balance before the withdrawal.
- `x`: the need, the least amount to withdraw; a non-adopter sends exactly `x`.
- `y`: the public withdrawal amount, the withdrawal the chain shows.
- event: one public withdrawal; its source is the index of the deposit that made it.

The tool (see :mod:`withdrawchan`)

- `d`: the rounding unit, in the asset's smallest unit; `ceil_d(v)`/`floor_d(v)` round `v` up/down to a
  multiple of `d`.
- `m`: the maximum, the largest amount to withdraw, set by the maximum model: `double` gives
  `min(ceil_d(2x), floor_d(r))`, `balance` gives `floor_d(r)`.  When `m < ceil_d(x)` the
  tool cannot be used and `y = x`.
- rungs: the amounts a walk can stop at, starting at `c0 = ceil_d(x)`.
- `q`, `p`: the probability of climbing one rung and of adding the random decimal below `d`.
- rule: `need` (`y = x`); `decstep`/`decsmooth` (decimal rungs, `q = 1`/`1/2`);
  `popstep`/`popsmooth` (rungs at popular historical amounts, `q = 1`/`1/2`).
- `threshold`: adopters use the tool only while `r >= threshold`.
- `quantum`: an adopter's need `x` is uniform over the multiples of `quantum` in
  `[quantum, r]`.

The historical (no-tool) model of the next withdrawal `x` from balance `r`

- `pi`: probability of withdrawing everything, `x = r`.
- `s`: otherwise, probability of a matched-tail amount: `x < r` with
  `x % unit == r % unit != 0`, which leaves a multiple of `unit` in the pool.
- `w`: otherwise, probability of a uniform `x` in `[1, r]`; the rest follows the demand `g`.
- `g`: the demand, the empirical distribution of training withdrawal amounts.  Version `hi`
  counts every amount; `lo` first removes, per amount, as many withdrawals as there are earlier
  deposits of that amount.
- `unit`: the tail unit of `s`.  `step` (`G`): the grid the `grid` historical model puts
  the demand density and the uniform part on.
- `L(x | r)`: the unnormalised likelihood of `x`; `Z(r) = sum_{x=1..r} L(x | r)` and
  `P(x | r) = L(x | r) / Z(r)`.

Observers and scores

- posterior: an observer's probability for each deposit being the source of one event, given the
  public amounts so far (`historical`, `trained-logit`, `group`) or the whole public stream
  (`combinatorial`).
- observers: `historical` scores each deposit by the historical model on its initial balance;
  `trained-logit` is a conditional logit on amount features; `combinatorial` links each withdrawal to
  the deposits that can have sourced it, from the public amounts alone.
- `group`: a particle posterior over which deposits have withdrawn, run with a fixed 1000 particles.
  It is scored only when a config's `observers` list names `group`; the shipped configs do not, so
  enabling it is an explicit choice.  It knows the rounding rule and the path's adopter kind, hence
  the path-wide `KEEP`/`DRAIN` stopping policy, but not the adopter count, the adopter identities,
  the event kinds or the true sources.  Because it uses that stopping policy, it is a separate
  kind-aware attack comparison rather than a public observer: its rows appear in `groups` and
  `paired`, and it never enters `best`, whose envelope stays the strongest public observer's.
- `p_true`, `top1`, `max_posterior`: the posterior of the true source, whether the top guess
  is the true source (split evenly over ties), and the largest posterior, i.e. the `V` in the
  per-observation min-entropy `-log2 V`.
"""
