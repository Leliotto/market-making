import numpy as np
import pytest

from mm.backtest import run_backtest
from mm.calibration import _poisson_rate, estimate_intensity, estimate_intensity_trades, estimate_sigma
from mm.data import Market
from mm.strategies import AvellanedaStoikov, State, Symmetric
from mm.synthetic import make_market


def tiny_market(trades):
    """Flat book 99.99 / 100.01 every 100 ms for 1 s, plus the given (t, price, side) trades."""
    n = 10
    t = np.arange(n, dtype=np.int64) * 100
    tr = np.array(trades, dtype=float).reshape(-1, 3)
    return Market(t, np.full((n, 1), 99.99), np.ones((n, 1)), np.full((n, 1), 100.01), np.ones((n, 1)),
                  tr[:, 0].astype(np.int64), tr[:, 1], np.ones(len(tr)), tr[:, 2].astype(np.int8))


def test_calibration_recovers_known_parameters():
    m = make_market(hours=4, sigma=0.02, A=2.0, k=20.0, seed=3)
    assert estimate_sigma(m) == pytest.approx(0.02, rel=0.05)
    fit = estimate_intensity(m, latency_ms=0)
    assert fit.k == pytest.approx(20.0, rel=0.15)
    assert fit.A == pytest.approx(2.0, rel=0.15)
    naive = estimate_intensity_trades(m)
    assert naive.k == pytest.approx(20.0, rel=0.15)


def test_poisson_rate_mle_with_unequal_intervals():
    rng = np.random.default_rng(0)
    dt = rng.choice([0.05, 0.1, 0.3], size=200_000)
    lam = 3.0
    hit = rng.random(dt.size) < 1 - np.exp(-lam * dt)
    assert _poisson_rate(dt[hit], dt.sum()) == pytest.approx(lam, rel=0.03)


def test_window_fit_is_consistent_with_backtest_fills():
    """A quote at distance d should be filled about lambda(d) times per second in the backtest."""
    m = make_market(hours=2, sigma=0.002, A=2.0, k=20.0, seed=5, tick=0.001, half_spread=0.0005)
    fit = estimate_intensity(m, latency_ms=0)
    d = 0.05
    res = run_backtest(m, Symmetric(half_spread=d, q_max=1e9), 1.0, 0.001, latency_ms=0)
    fills_per_s_per_side = len(res.fills) / m.duration_s / 2
    expected = fit.A * np.exp(-fit.k * d)
    assert fills_per_s_per_side == pytest.approx(expected, rel=0.2)


def test_as_formulas_and_skew():
    s = AvellanedaStoikov(gamma=1.0, sigma=0.1, k=10.0, horizon_s=100)
    assert s.reservation_price(100.0, q=2) == pytest.approx(100.0 - 2 * 1.0 * 0.01 * 100)
    assert s.total_spread() == pytest.approx(1.0 * 0.01 * 100 + 2.0 * np.log(1.1))
    flat = s.quote(State(0, 100.0, 99.0, 101.0, q=0, tick=0.01))
    long = s.quote(State(0, 100.0, 99.0, 101.0, q=3, tick=0.01))
    assert long[0] < flat[0] and long[1] < flat[1]  # long inventory: quote lower to sell it


def test_quotes_never_cross_and_respect_limits():
    s = Symmetric(half_spread=0.0, q_max=5)
    bid, ask = s.quote(State(0, 100.0, 99.99, 100.01, q=0, tick=0.01))
    assert bid <= 100.0 and ask >= 100.0 and bid < ask
    assert s.quote(State(0, 100.0, 99.99, 100.01, q=5, tick=0.01))[0] is None


def test_through_fill_needs_price_beyond_our_quote():
    strat = Symmetric(half_spread=0.01)       # bid 99.99, ask 100.01
    at_quote = tiny_market([(300, 99.99, -1)])
    beyond = tiny_market([(300, 99.98, -1)])
    assert len(run_backtest(at_quote, strat, 1.0, 0.01, fill_mode="through").fills) == 0
    assert len(run_backtest(at_quote, strat, 1.0, 0.01, fill_mode="touch").fills) == 1
    res = run_backtest(beyond, strat, 1.0, 0.01, fill_mode="through")
    assert len(res.fills) == 1 and res.fills["price"][0] == pytest.approx(99.99)
    assert res.inventory[-1] == pytest.approx(1.0)


def test_round_trip_earns_the_spread():
    strat = Symmetric(half_spread=0.01)
    res = run_backtest(tiny_market([(300, 99.98, -1), (600, 100.02, 1)]), strat, 1.0, 0.01, latency_ms=0)
    assert res.inventory[-1] == pytest.approx(0.0)
    assert res.pnl[-1] == pytest.approx(0.02)


def test_fees_reduce_pnl():
    strat = Symmetric(half_spread=0.01)
    mk = tiny_market([(300, 99.98, -1), (600, 100.02, 1)])
    free = run_backtest(mk, strat, 1.0, 0.01, latency_ms=0).pnl[-1]
    paid = run_backtest(mk, strat, 1.0, 0.01, latency_ms=0, maker_fee_bps=1.0).pnl[-1]
    assert paid == pytest.approx(free - 1e-4 * (99.99 + 100.01))


def test_accounting_identity_on_synthetic_market():
    m = make_market(hours=0.5, seed=7)
    res = run_backtest(m, AvellanedaStoikov(20.0, 0.02, 20.0, 60), 1.0, 0.01)
    f = res.fills
    assert res.inventory[-1] == pytest.approx(np.sum(f["side"] * f["qty"]))
    assert res.cash[-1] == pytest.approx(-np.sum(f["side"] * f["qty"] * f["price"]))


def test_hac_t_stat_is_more_cautious_with_autocorrelation():
    from mm.metrics import hac_t_stat
    rng = np.random.default_rng(1)
    iid = rng.normal(0.1, 1.0, 5000)
    plain = iid.mean() / (iid.std(ddof=1) / np.sqrt(len(iid)))
    assert hac_t_stat(iid) == pytest.approx(plain, rel=0.1)
    ar = np.zeros(5000)
    eps = rng.normal(0.1, 1.0, 5000)
    for i in range(1, 5000):
        ar[i] = 0.7 * ar[i - 1] + eps[i]
    plain_ar = ar.mean() / (ar.std(ddof=1) / np.sqrt(len(ar)))
    assert abs(hac_t_stat(ar)) < abs(plain_ar)


def test_microprice_leans_to_thin_side_and_moves_quotes():
    heavy_bid = State(0, 100.005, 100.0, 100.01, q=0, tick=0.01, bid_qty=30, ask_qty=1)
    assert heavy_bid.microprice > heavy_bid.mid
    mid_as = AvellanedaStoikov(gamma=1.0, sigma=0.01, k=50.0, horizon_s=60)
    micro_as = AvellanedaStoikov(gamma=1.0, sigma=0.01, k=50.0, horizon_s=60, center="micro")
    assert micro_as.quote(heavy_bid)[0] >= mid_as.quote(heavy_bid)[0]
    assert micro_as.quote(heavy_bid)[1] >= mid_as.quote(heavy_bid)[1]
    assert micro_as.name.startswith("AS-micro")


def test_fit_on_first_ticks_only():
    m = make_market(hours=4, sigma=0.002, A=2.0, k=20.0, seed=9, tick=0.001, half_spread=0.0005)
    deltas = (np.arange(40) + 0.5) * 0.001
    fit = estimate_intensity(m, latency_ms=0, deltas=deltas)
    assert fit.deltas.max() < 0.041
    assert fit.k == pytest.approx(20.0, rel=0.25)


def test_skew_parameterisation_means_ticks_per_lot():
    s = AvellanedaStoikov.from_skew(0.5, tick=0.01, sigma=5.0, k=9.0, horizon_s=60)
    assert (100.0 - s.reservation_price(100.0, q=1)) == pytest.approx(0.005)
    assert s.name == "AS(skew=0.5t)"


def test_paired_comparison_sign():
    from mm.metrics import paired_comparison
    m = make_market(hours=1, seed=11)
    narrow = run_backtest(m, Symmetric(half_spread=0.05, q_max=5), 1.0, 0.01)
    same = run_backtest(m, Symmetric(half_spread=0.05, q_max=5), 1.0, 0.01)
    c = paired_comparison(narrow, same)
    assert c["diff"] == pytest.approx(0.0)
