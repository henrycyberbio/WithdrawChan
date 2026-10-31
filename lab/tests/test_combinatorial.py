"""The `combinatorial` observer's linking, tail narrowing, passes and fallback, hand-computed."""

from __future__ import annotations

from fractions import Fraction

from lab.observers.combinatorial import CombinatorialObserver


def test_full_withdrawal_links_and_closes_a_deposit() -> None:
    # GIVEN: Three deposits and two events where the first withdrawal equals deposit 0 exactly.
    observer = CombinatorialObserver((100, 40, 90), (100, 30), Fraction(0), 10)
    # WHEN: The whole stream is linked.
    # THEN: Deposit 0 is linked to event 0 and closed, and the later event 30 keeps only the two
    # deposits whose remaining capacity covers it, one candidate short of the three deposits.
    assert observer.link.tolist() == [0, -1]
    assert observer.closed.tolist() == [True, False, False]
    assert observer._sets[1].tolist() == [1, 2]


def test_nonzero_tail_narrows_to_the_matching_remainders() -> None:
    # GIVEN: Three deposits, none a near match for 7, of which two leave a remainder of 7 at unit 10.
    observer = CombinatorialObserver((17, 27, 100), (7,), Fraction(0), 10)
    # WHEN: The amount is narrowed.
    # THEN: Only the deposits whose remaining capacity shares the tail 7 stay candidates.
    assert observer._sets[0].tolist() == [0, 1]


def test_link_in_a_later_pass_after_another_deposit_closes() -> None:
    # GIVEN: Two events where the second withdrawal equals deposit 0 exactly.
    observer = CombinatorialObserver((100, 60), (50, 100), Fraction(0), 10)
    # WHEN: The linking runs to its fixpoint.
    # THEN: Event 0 is ambiguous in the first pass and links only once event 1 has closed deposit 0.
    assert observer.link.tolist() == [1, 0]
    assert observer.closed.tolist() == [True, False]


def test_posterior_is_uniform_over_the_candidates() -> None:
    # GIVEN: A withdrawal with two matching-tail candidates out of three deposits.
    observer = CombinatorialObserver((17, 27, 100), (7,), Fraction(0), 10)
    # WHEN: Scoring the event.
    # THEN: The posterior is uniform over the two candidates and zero on the third deposit.
    assert observer.posterior(7).tolist() == [0.5, 0.5, 0.0]


def test_tolerance_closes_a_nearly_drained_deposit() -> None:
    # GIVEN: A withdrawal of 95 against deposits of 100 and 1000.
    linked = CombinatorialObserver((100, 1000), (95,), Fraction(1, 10), 10)
    # WHEN: The tolerance is one tenth, the 100-deposit is a single near match.
    # THEN: It links and closes, while without tolerance the amount stays ambiguous.
    assert (linked.link.tolist(), linked.closed.tolist()) == ([0], [True, False])
    loose = CombinatorialObserver((100, 1000), (95,), Fraction(0), 10)
    assert (loose.link.tolist(), loose._sets[0].tolist()) == ([-1], [0, 1])


def test_empty_candidate_set_falls_back_uniformly() -> None:
    # GIVEN: Two deposits drained before a third withdrawal that neither can source.
    observer = CombinatorialObserver((100, 90), (90, 100, 50), Fraction(0), 10)
    # WHEN: Scoring that last event.
    observer.update(90)
    observer.update(100)
    posterior = observer.posterior(50)
    # THEN: The posterior is uniform over the deposits with b >= 50 and one fallback is counted.
    assert posterior.tolist() == [0.5, 0.5]
    assert observer.fallback_count == 1


def test_an_amount_above_int64_links_exactly() -> None:
    # GIVEN: Two balances above int64 that differ by one wei, where float64 cannot tell them apart.
    big = 2**63 + 5
    # WHEN: A withdrawal equal to the first one is linked.
    observer = CombinatorialObserver((big, big + 1, 10), (big,), Fraction(0), 10)
    # THEN: Only the exactly equal deposit is a candidate, so it links, closes and owns the posterior.
    assert float(big) == float(big + 1)
    assert observer.link.tolist() == [0]
    assert observer.closed.tolist() == [True, False, False]
    assert observer.posterior(big).tolist() == [1.0, 0.0, 0.0]
