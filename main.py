"""GTA VI trophy rarity forecast (make.fun: gta6-trophy-rarity-forecast-01).

Reads one case file per run from the input directory and writes, for every
trophy in the case, five quantiles (5/25/50/75/95%) of the trophy's Exophase
earn rate -- the percent of tracked players who hold it -- at the case's
target instant.

Method: a two-cohort count model, fitted per case from the case's own read
history. A trophy's earn rate is earners / players_tracked, so the forecast
is built in counts and divided at the end:

  * The recorded page only changes when Exophase refreshes it, so the
    history is first collapsed to the distinct snapshots it contains.
  * Between two snapshots the tracked pool grows by dP and a trophy's
    earners by dE. The ratio of summed dE to summed dP over the snapshots
    where the pool grew is the trophy's *marginal earn share* m: the chance
    a newly tracked player already holds the trophy. It is shrunk toward
    the pooled share of the trophy's tier (platinum/gold/silver/bronze),
    because one trophy's transitions are few and noisy. Transitions are
    weighted by recency (one-day half-life): the player mix of two days
    ago is not the player mix of the target week.
  * Over the snapshots where the pool barely moved, earners still grow:
    the players already tracked keep playing. The mean log growth of the
    earner count per day over those spans is the trophy's *progression
    rate* g, likewise shrunk toward its tier's.
  * The pool itself is extrapolated on a log-linear fit of players_tracked
    against time, damped toward the whole-window slope, so a launch-week
    surge is projected forward without exploding.
  * Earners at the target = earners now, grown by progression over the
    horizon, plus the pool's growth times the marginal share. The median
    rate is that count over the projected pool. The platinum's row is
    then capped at the rarest other trophy's, level by level: its
    holders are a subset of every other trophy's holders.

The spread is a normal spread on the logit scale (the scale the contest
scores on), combining a random-walk term that grows with the horizon, the
trophy's own observed logit-step volatility between snapshots, and a flat
estimation term; it is clamped so no trophy is forecast with a dishonestly
narrow or absurdly wide band.

Everything is a deterministic function of the case file: no randomness, no
network, standard library only. Two runs on the same case write the same
bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Random-walk scale of a rate's logit, per square-root day (the house
# baseline's own scale; our band adds volatility and estimation terms).
WALK_SIGMA = 0.5
# Shrinkage prior strength for the marginal share, in "players" units.
SHARE_PRIOR_PLAYERS = 0.25
# Shrinkage prior strength for the progression rate, in days.
PROG_PRIOR_DAYS = 2.0
# A snapshot-to-snapshot pool move below this fraction counts as "pool
# barely moved", so its earner growth is read as progression.
FLAT_POOL_FRACTION = 0.02


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    return 100.0 / (1.0 + math.exp(-x))


def round4(x: float) -> float | int:
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def snapshots(case: dict) -> list[dict]:
    """The history collapsed to the reads where the page actually changed.

    Exophase refreshes a game's counts on its own schedule, so consecutive
    reads often repeat one page; keeping the repeats would overweight the
    flat spans and hide the step dynamics the forecast is built from.
    """
    out: list[dict] = []
    for read in case["history"]:
        snap = {
            "at": parse_time(read["at"]),
            "players": read["players_tracked"],
            "rates": read["rates"],
        }
        if out and out[-1]["players"] == snap["players"] and out[-1]["rates"] == snap["rates"]:
            continue
        out.append(snap)
    return out


def tier_groups(case: dict) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, trophy in enumerate(case["trophies"]):
        groups.setdefault(trophy["tier"], []).append(i)
    return groups


def pool_growth_per_day(snaps: list[dict]) -> float:
    """Log-linear slope of players_tracked per day, damped.

    Fits ln(players) on days since the first snapshot over the whole
    window and over the last day, then blends them: launch-week growth
    decays, so the damped blend is the honest projection.
    """
    if len(snaps) < 2:
        return 0.0
    t0 = snaps[0]["at"]

    def slope(rows: list[dict]) -> float:
        xs = [(r["at"] - t0).total_seconds() / 86400.0 for r in rows]
        ys = [math.log(max(1, r["players"])) for r in rows]
        n = len(rows)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx

    whole = slope(snaps)
    last_day = [r for r in snaps if (snaps[-1]["at"] - r["at"]).total_seconds() <= 86400.0]
    recent = slope(last_day) if len(last_day) >= 2 else whole
    blended = 0.5 * whole + 0.5 * recent
    return max(0.0, 0.8 * blended)


def trophy_stats(snaps: list[dict], idx: int, weights: list[float]) -> dict:
    """One trophy's transition sums over the distinct snapshots.

    Each transition is weighted by recency (see transition_weights): in
    launch week the player mix and the pace of play change by the day, so
    a transition from an hour ago says more about the target than one from
    two days ago.
    """
    d_pool_growth = 0.0      # weighted sum of dP over pool-growing transitions
    d_earn_growth = 0.0      # weighted sum of dE over those same transitions
    prog_log_growth = 0.0    # weighted sum of ln(E2/E1) over flat-pool spans
    prog_days = 0.0          # weighted days those flat transitions span
    step_sq = 0.0            # weighted squared logit steps, for volatility
    step_days = 0.0
    steps = 0
    for k, (a, b) in enumerate(zip(snaps, snaps[1:])):
        ra, rb = a["rates"][idx], b["rates"][idx]
        if ra is None or rb is None:
            continue
        days = (b["at"] - a["at"]).total_seconds() / 86400.0
        if days <= 0:
            continue
        w = weights[k]
        ea = ra / 100.0 * a["players"]
        eb = rb / 100.0 * b["players"]
        dp = b["players"] - a["players"]
        if dp > 0:
            d_pool_growth += w * dp
            d_earn_growth += w * max(0.0, eb - ea)
        if abs(dp) <= FLAT_POOL_FRACTION * max(1, a["players"]) and ea > 0 and eb > 0:
            prog_log_growth += w * math.log(eb / ea)
            prog_days += w * days
        step = logit(rb) - logit(ra)
        step_sq += w * step * step
        step_days += w * days
        steps += 1
    return {
        "d_pool": d_pool_growth,
        "d_earn": d_earn_growth,
        "prog_log": prog_log_growth,
        "prog_days": prog_days,
        "step_sq": step_sq,
        "step_days": step_days,
        "steps": steps,
    }


def transition_weights(snaps: list[dict]) -> list[float]:
    """Recency weight per transition: a half-life of one day back from the
    last snapshot, so the freshest transition weighs 1 and one a day older
    weighs a half."""
    if len(snaps) < 2:
        return []
    end = snaps[-1]["at"]
    return [
        0.5 ** (((end - b["at"]).total_seconds() / 86400.0) / 1.0)
        for b in snaps[1:]
    ]


def forecast_case(case: dict) -> dict:
    snaps = snapshots(case)
    horizon_days = case["horizon_minutes"] / 1440.0
    last = snaps[-1]
    players_now = max(1, last["players"])
    groups = tier_groups(case)
    weights = transition_weights(snaps)
    stats = [trophy_stats(snaps, i, weights) for i in range(len(case["trophies"]))]

    # Tier pools: the shrinkage targets for the share and the progression.
    tier_share: dict[str, float] = {}
    tier_prog: dict[str, float] = {}
    for tier, idxs in groups.items():
        dp = sum(stats[i]["d_pool"] for i in idxs)
        de = sum(stats[i]["d_earn"] for i in idxs)
        tier_share[tier] = de / dp if dp > 0 else 0.0
        pl = sum(stats[i]["prog_log"] for i in idxs)
        pd = sum(stats[i]["prog_days"] for i in idxs)
        tier_prog[tier] = pl / pd if pd > 0 else 0.0

    growth = pool_growth_per_day(snaps)
    pool_target = players_now * math.exp(growth * horizon_days)
    pool_target = min(pool_target, players_now * 12.0)
    pool_gain = max(0.0, pool_target - players_now)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        if rate is None:  # fall back to the trophy's last recorded rate
            for snap in reversed(snaps):
                if snap["rates"][i] is not None:
                    rate = snap["rates"][i]
                    break
        if rate is None:
            rate = 0.0
        st = stats[i]
        tier = trophy["tier"]

        # Marginal share of newly tracked players already holding it.
        prior = SHARE_PRIOR_PLAYERS * players_now
        if st["d_pool"] > 0:
            share = (st["d_earn"] + prior * tier_share[tier]) / (st["d_pool"] + prior)
        else:
            share = rate / 100.0
        share = min(1.0, max(0.0, share))

        # Progression of the players already tracked.
        if st["prog_days"] > 0:
            prog = (st["prog_log"] + PROG_PRIOR_DAYS * tier_prog[tier]) / (
                st["prog_days"] + PROG_PRIOR_DAYS
            )
        else:
            prog = tier_prog[tier]
        prog = min(0.6, max(0.0, prog))

        earners_now = rate / 100.0 * players_now
        earners_target = earners_now * math.exp(prog * horizon_days) + pool_gain * share
        median = clamp_rate(100.0 * earners_target / pool_target)

        # Spread on the logit scale: random walk + own volatility + a flat
        # estimation term, combined in quadrature and clamped.
        walk = WALK_SIGMA * math.sqrt(horizon_days)
        if st["steps"] >= 2 and st["step_days"] > 0:
            vol = math.sqrt(st["step_sq"] / st["step_days"]) * math.sqrt(horizon_days)
        else:
            vol = 0.0
        sigma = math.sqrt(walk * walk + (0.7 * vol) ** 2 + 0.15 ** 2)
        sigma = min(2.4, max(0.35, sigma))

        centre = logit(median)
        rows.append(
            {"id": trophy["id"], "q": [round4(inv_logit(centre + z * sigma)) for z in Z]}
        )

    # The platinum's holders are a subset of every other trophy's holders
    # (it is awarded for earning all the rest), so its rate can never pass
    # the rarest other trophy's. Cap its row, level by level, at the
    # lowest other row; the cap of non-decreasing rows is non-decreasing.
    platinum = [r for r, t in zip(rows, case["trophies"]) if t["tier"] == "platinum"]
    others = [r for r, t in zip(rows, case["trophies"]) if t["tier"] != "platinum"]
    if platinum and others:
        caps = [min(r["q"][k] for r in others) for k in range(len(Z))]
        row = platinum[0]
        row["q"] = [min(v, caps[k]) for k, v in enumerate(row["q"])]
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case["levels"] != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case['levels']}, not {LEVELS}")
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
