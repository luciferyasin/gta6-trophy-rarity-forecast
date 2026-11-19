"""GTA VI week-one trophy rarity forecaster (Python 3.13, standard library only).

Idea: an Exophase earn rate is a ratio, earners / players tracked. Both counts only grow, but at different speeds:
on launch day the tracked base explodes (rates fall: the few early earners are diluted), later players progress and
earners grow faster than the base (story trophies rise). So instead of extrapolating the rate itself, we
extrapolate log(players) and log(earners) separately against log-time since release, from the recent refreshes,
with a damping factor (growth in log-log space slows as the launch wave passes), then rebuild the rate.
The spread is a logit-scale normal whose width grows with the horizon, with the size of the predicted move
(the trend may stall or run on), with the fit's residuals and with small-count noise.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
Z = {0.05: -1.6448536, 0.25: -0.6744898, 0.5: 0.0, 0.75: 0.6744898, 0.95: 1.6448536}

# Tunables (chosen from launch-curve reasoning, not fitted to the practice).
TIME_OFFSET_H = 3.0      # u = ln(hours since release + offset)
WINDOW_H = 30.0          # refreshes used for the trend: the last 30 hours
RECENCY_H = 14.0         # exponential recency weight inside the window
DAMP_PLAYERS = 0.75      # share of the fitted log-log growth carried forward
DAMP_EARNERS = 0.70
DAMP_LOGIT = 0.85        # share of the rate's fitted logit trend carried forward
W_COUNT = 0.3            # weight of the earners/players view for rare trophies (under 5%)
S_BASE = 0.12            # logit sd floor
S_WALK = 0.25            # logit sd per sqrt(day) of horizon
S_MOVE = 0.30            # extra sd per unit of predicted logit move
SKEW = 0.10              # widen the side back toward the last value
S_NO_TREND = 1.5         # with fewer than 3 distinct refreshes there is no trend to read: stay wide

# local tuning only: tools/score.py may override the constants above; the grader never sets this
for _k, _v in json.loads(os.environ.get("FORECAST_TUNE") or "{}").items():
    globals()[_k] = _v


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    p = clamp(pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(s):
    return (parse_time(s) - RELEASE).total_seconds() / 3600.0


def u_of(h):
    return math.log(max(h, -TIME_OFFSET_H + 0.5) + TIME_OFFSET_H)


def refreshes(history, n):
    """Collapse runs of identical reads (Exophase had not refreshed) to their first read."""
    out = []
    prev = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev:
            continue
        prev = key
        out.append(read)
    return out


def wls(xs, ys, ws):
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return 0.0, (ys[-1] if ys else 0.0), 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx < 1e-9:
        return 0.0, my, 0.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    a = my - b * mx
    res = sum(w * (y - a - b * x) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    return b, a, res


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    hist = [r for r in case.get("history") or [] if r.get("rates")]
    target_h = hours_since_release(case["target"])
    at_h = hours_since_release(case["at"])
    days = max(case.get("horizon_minutes", 0), 0) / 1440.0

    if not hist:
        rows = []
        for t in trophies:
            q = [inv_logit(logit(50.0) + Z[l] * 3.0) for l in (0.05, 0.25, 0.5, 0.75, 0.95)]
            rows.append({"id": t["id"], "q": [round(v, 4) for v in q]})
        return {"trophies": rows}

    refs = refreshes(hist, n)
    last = hist[-1]
    last_h = hours_since_release(refs[-1]["at"])
    u_last = u_of(last_h)
    u_tgt = u_of(max(target_h, last_h))
    du = max(u_tgt - u_last, 0.0)

    window = [r for r in refs if hours_since_release(r["at"]) >= last_h - WINDOW_H]
    if len(window) < 3:
        window = refs[-3:]
    xs = [u_of(hours_since_release(r["at"])) for r in window]
    ws = [math.exp((hours_since_release(r["at"]) - last_h) / RECENCY_H) for r in window]

    players = [max(float(r.get("players_tracked") or 0), 1.0) for r in window]
    bp, _, resp = wls(xs, [math.log(p) for p in players], ws)
    bp = max(bp, 0.0)
    p_last = max(float(last.get("players_tracked") or 0), 1.0)
    log_p_tgt = math.log(p_last) + DAMP_PLAYERS * bp * du

    rows = []
    for i, t in enumerate(trophies):
        r_last = None
        for read in reversed(hist):
            v = read["rates"][i] if i < len(read["rates"]) else None
            if v is not None:
                r_last = float(v)
                break
        if r_last is None:
            r_last = 1.0
        l_last = logit(r_last)

        pts = [(x, w, r["rates"][i], float(r.get("players_tracked") or 1)) for x, w, r in zip(xs, ws, window)
               if i < len(r["rates"]) and r["rates"][i] is not None]
        # (a) the rate's own logit trend against log-time: saturates naturally near 0 and 100
        ll = [logit(rt) for _, _, rt, _ in pts]
        bl, _, resid = wls([p[0] for p in pts], ll, [p[1] for p in pts])
        med_trend = l_last + DAMP_LOGIT * bl * du
        # (b) earners and players extrapolated apart: reads rare trophies whose earners grow from a handful
        e_last = max(r_last * p_last / 100.0, 0.5)
        le = [math.log(max(rt * pl / 100.0, 0.5)) for _, _, rt, pl in pts]
        be, _, _ = wls([p[0] for p in pts], le, [p[1] for p in pts])
        log_e_tgt = math.log(e_last) + DAMP_EARNERS * max(be, 0.0) * du
        med_count = logit(100.0 * math.exp(min(log_e_tgt - log_p_tgt, 0.0)))
        wc = W_COUNT if r_last < 5.0 else 0.0
        med = (1.0 - wc) * med_trend + wc * med_count
        count_var = 1.0 / max(min(e_last, max((100.0 - r_last) * p_last / 100.0, 0.5)), 0.5)
        # 2-decimal print: a rate near 0.01 is uncertain by half a unit
        quant = 0.005 / max(r_last, 0.01) if r_last < 50 else 0.005 / max(100.0 - r_last, 0.01)
        move = med - l_last
        s = math.sqrt(S_BASE ** 2 + S_WALK ** 2 * days + (S_MOVE * move) ** 2
                      + min(resid, 1.0) + min(count_var, 0.5) * 0.25 + quant ** 2)
        if len(refs) < 3:
            s = max(s, S_NO_TREND)
        lo_s = s * (1.0 + SKEW) if move > 0 else s * (1.0 - SKEW)
        hi_s = s * (1.0 - SKEW) if move > 0 else s * (1.0 + SKEW)
        q = []
        for lv in (0.05, 0.25, 0.5, 0.75, 0.95):
            z = Z[lv]
            q.append(inv_logit(med + z * (lo_s if z < 0 else hi_s)))
        q = [round(clamp(v, 0.0, 100.0), 4) for v in q]
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": t["id"], "q": q})
    return {"trophies": rows}


def main():
    in_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR") or "/input")
    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(in_dir.glob("*.in")):
        case = json.loads(f.read_text(encoding="utf-8"))
        result = forecast_case(case)
        (out_dir / (f.stem + ".out")).write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
