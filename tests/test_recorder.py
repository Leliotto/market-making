import json

import numpy as np

from mm.data import load_market
from mm.recorder import HourlyWriter, parse_message

# shapes copied from Binance's combined-stream payloads
DEPTH = {"stream": "btcusdt@depth10@100ms",
         "data": {"lastUpdateId": 1, "bids": [["100.00", "1.5"], ["99.99", "2"]],
                  "asks": [["100.01", "0.7"], ["100.02", "3"]]}}
TRADE_SELL = {"stream": "btcusdt@aggTrade",
              "data": {"e": "aggTrade", "E": 2, "s": "BTCUSDT", "a": 5, "p": "99.98", "q": "0.25",
                       "f": 1, "l": 2, "T": 1700000000123, "m": True, "M": True}}


def test_parse_messages():
    sym, book = parse_message(DEPTH, levels=10)
    assert sym == "btcusdt" and book["type"] == "book"
    assert book["b"][0] == [100.0, 1.5] and book["a"][0] == [100.01, 0.7]
    sym, trade = parse_message(TRADE_SELL, levels=10)
    assert trade["s"] == -1 and trade["p"] == 99.98 and trade["q"] == 0.25  # buyer was maker: seller aggressed
    assert parse_message({"stream": "btcusdt@kline_1m", "data": {}}, 10) is None


def test_write_and_load_round_trip(tmp_path):
    w = HourlyWriter(tmp_path, "btcusdt")
    t0 = 1_700_000_000_000
    for i in range(5):
        _, book = parse_message(DEPTH, 10)
        book["t"] = t0 + 100 * i
        w.write(book["t"], book)
    _, trade = parse_message(TRADE_SELL, 10)
    trade["t"] = t0 + 250
    w.write(trade["t"], trade)
    w.close()
    files = list((tmp_path / "btcusdt").glob("*.jsonl.gz"))
    assert len(files) == 1
    m = load_market(tmp_path, "btcusdt")
    assert len(m.book_t) == 5 and len(m.trade_t) == 1
    assert np.allclose(m.mid, 100.005)
    assert m.trade_side[0] == -1


def test_loader_skips_truncated_line(tmp_path):
    d = tmp_path / "ethusdt"
    d.mkdir()
    good = json.dumps({"t": 1, "type": "book", "b": [[10.0, 1.0]], "a": [[10.1, 1.0]]})
    (d / "20261108_10.jsonl").write_text(good + "\n" + good[:20])
    m = load_market(tmp_path, "ethusdt")
    assert len(m.book_t) == 1
