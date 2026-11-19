"""Local checks for the forecaster: validate output shape, score pinball loss on the logit scale, and run a
synthetic launch-week simulation (players grow fast then slow, earners progress) to see the spread is honest.

    python3 tools/score.py            # runs main.py on the kit's practice cases and a synthetic week
"""

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def logit(x):
    p = min(max(x, 0.01), 99.99) / 100.0
    return math.log(p / (1 - p))


def pinball(y, q):
    tot = 0.0
    for tau, qq in zip(LEVELS, q):
        d = logit(y) - logit(qq)
        tot += 2 * max(tau * d, (tau - 1) * d)
    return tot / 5


def validate(case, out):
    assert set(out) == {"trophies"}
    ids = sorted(r["id"] for r in out["trophies"])
    assert ids == sorted(t["id"] for t in case["trophies"]), "ids"
    for r in out["trophies"]:
        assert set(r) == {"id", "q"} and len(r["q"]) == 5
        assert all(0 <= v <= 100 for v in r["q"]) and all(a <= b for a, b in zip(r["q"], r["q"][1:]))


def run(cases):
    with tempfile.TemporaryDirectory() as tin, tempfile.TemporaryDirectory() as tout:
        for c in cases:
            Path(tin, c["id"] + ".in").write_text(json.dumps(c))
        env = dict(os.environ, MAKEFUN_INPUT_DIR=tin, MAKEFUN_OUTPUT_DIR=tout)
        subprocess.run([sys.executable, "-P", "-s", str(ROOT / "main.py")], env=env, check=True)
        return {c["id"]: json.loads(Path(tout, c["id"] + ".out").read_text()) for c in cases}


def iso(h):
    from datetime import datetime, timedelta, timezone
    return (datetime(2026, 11, 19, tzinfo=timezone.utc) + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")


def simulate(seed, cut_h=59.5, target_h=168.0, n=55):
    """A toy launch week: tracked players P(h) saturating, trophy i needs H_i hours of play; players' play time
    grows with their days since joining. Returns (case, truth) with a refresh every ~2 hours."""
    import random
    rnd = random.Random(seed)
    hardness = [rnd.lognormvariate(math.log(12), 1.1) for _ in range(n)]
    hardness[1] = 0.05  # the opening trophy
    pace = rnd.uniform(2.5, 5.0)          # hours of play per day for a typical player
    p_inf = rnd.uniform(1.5e5, 6e5)
    tau = rnd.uniform(40, 110)

    def players(h):
        return 4000 + p_inf * (1 - math.exp(-h / tau))

    cheaters = rnd.randint(2, 8)          # launch-night accounts that hold every trophy
    weekend = rnd.uniform(1.0, 1.6)       # play pace multiplier from Saturday 21 November

    fatigue = rnd.uniform(60, 400)        # hours over which a player's daily play time fades

    def played(h, j):
        # hours of play from joining at j to h: faster over the weekend (h 48..96), fading with time since joining
        steps = 12
        tot = 0.0
        for k in range(steps):
            x = j + (h - j) * (k + 0.5) / steps
            m = (weekend if 48.0 <= x < 96.0 else 1.0) * math.exp(-(x - j) / fatigue)
            tot += m
        return pace * (h - j) / 24.0 * tot / steps

    def rate(i, h):
        # mixture over join times: integrate earners over cohorts joined in [0, h]
        steps = 40
        tot = 0.0
        pw = 0.0
        for k in range(steps):
            j = h * (k + 0.5) / steps
            w = math.exp(-j / tau)
            play = played(h, j)
            frac = 1 - math.exp(-((play / hardness[i]) ** 2.0))
            tot += w * frac
            pw += w
        real = tot / pw
        p = players(h)
        return round(100 * min(1.0, (real * (p - cheaters) + cheaters) / p), 2)

    hist = []
    h = 4.5
    while h < cut_h:
        hist.append({"slot": iso(h), "at": iso(h), "players_tracked": int(players(h)), "trophies_earned": 0,
                     "rates": [rate(i, h) for i in range(n)]})
        h += rnd.uniform(1.5, 3.0)
    trophies = [{"id": i + 1, "name": f"t{i+1}", "description": "", "tier": "bronze", "points": 15, "secret": False}
                for i in range(n)]
    case = {"id": f"s{seed:02d}", "at": iso(cut_h), "horizon_minutes": int((target_h - cut_h) * 60),
            "target": iso(target_h), "levels": list(LEVELS), "trophies": trophies, "history": hist}
    truth = [rate(i, target_h) for i in range(n)]
    return case, truth


def baseline(case):
    d = case["horizon_minutes"] / 1440
    last = case["history"][-1]["rates"]
    z = (-1.6448536, -0.6744898, 0, 0.6744898, 1.6448536)
    return [[100 / (1 + math.exp(-(logit(r) + zz * 0.5 * math.sqrt(d)))) for zz in z] for r in last]


def curve_baseline(case):
    """The house curve baseline as the rules describe it: logit rate ~ ln(1 + hours) over the last 24 h, by LS."""
    from datetime import datetime, timezone
    rel = datetime(2026, 11, 19, tzinfo=timezone.utc)
    hh = lambda s: (datetime.fromisoformat(s.replace("Z", "+00:00")) - rel).total_seconds() / 3600
    reads = case["history"]
    last_h = hh(reads[-1]["at"])
    win = [r for r in reads if hh(r["at"]) >= last_h - 24]
    d = case["horizon_minutes"] / 1440
    xt = math.log(1 + hh(case["target"]))
    z = (-1.6448536, -0.6744898, 0, 0.6744898, 1.6448536)
    rows = []
    for i in range(len(case["trophies"])):
        xs = [math.log(1 + max(hh(r["at"]), 0)) for r in win]
        ys = [logit(r["rates"][i]) for r in win]
        if len(xs) < 3:
            m, res = ys[-1], 0.0
        else:
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs) or 1e-9
            b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            a = my - b * mx
            m = a + b * xt
            res = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys)) / len(xs)
        s = math.sqrt(0.25 * d + res)
        rows.append([100 / (1 + math.exp(-(m + zz * s))) for zz in z])
    return rows


def main():
    kit = ROOT.parent.parent / "gta6-trophy-rarity-forecast-01-kit" / "input"
    practice = [json.loads(p.read_text()) for p in sorted(kit.glob("p*.in"))]
    outs = run(practice)
    for c in practice:
        validate(c, outs[c["id"]])
    print("practice: valid", len(practice), "cases")

    sims = [simulate(s) for s in range(int(os.environ.get("SIMS") or 16))]
    outs = run([c for c, _ in sims])
    ours, base, curve, cov = [], [], [], []
    for c, truth in sims:
        validate(c, outs[c["id"]])
        rows = {r["id"]: r["q"] for r in outs[c["id"]]["trophies"]}
        bl = baseline(c)
        cb = curve_baseline(c)
        for i, y in enumerate(truth):
            q = rows[i + 1]
            ours.append(pinball(y, q))
            base.append(pinball(y, bl[i]))
            curve.append(pinball(y, cb[i]))
            cov.append(q[0] <= y <= q[4])
    print(f"synthetic week: ours {sum(ours)/len(ours):.4f}  baseline {sum(base)/len(base):.4f}  "
          f"curve {sum(curve)/len(curve):.4f}  90% coverage {sum(cov)/len(cov):.2f}")


if __name__ == "__main__":
    main()
