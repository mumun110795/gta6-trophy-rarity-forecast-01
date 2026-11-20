#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case's target time.

For every case file in the input directory this writes one forecast file:
for each trophy, five quantiles (5/25/50/75/95) of its earn rate, in
percent, at the case's target.

Model, per trophy
-----------------
A rate is earners / players_tracked.  The two parts move differently, so
they are modelled separately and divided at the end:

* Players: the tracked base grows fast after launch and slows down.
  Three curves are fitted to the distinct player counts in the history
  -- a power law in log-log space, an exponential approach to an
  asymptote chosen by grid search, and the last day's growth decaying
  geometrically -- and their target projections are combined by median.

* Earners: across the history's refresh points (reads where the page
  actually changed), a trophy's earner count is regressed on the player
  count in log-log space, earners ~ players ** beta, by weighted least
  squares (weights grow with the player count, which is also where the
  page's 2-decimal rates carry the least rounding noise).  beta = 1
  holds the rate steady, beta < 1 dilutes it as the base broadens.  Each
  trophy's beta is shrunk toward the pooled beta fitted on the page's
  total trophies-earned count, by the precision of its own fit.

The median forecast is projected earners over projected players.  The
other quantiles spread around it on the logit scale, with a width that
grows with the horizon (a random-walk term), a floor for the timing
uncertainty of Exophase's own refreshes, and extra width where a
trophy's trend is weakly identified or it has never been earned.

Pure standard library, no network, fully deterministic.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles at LEVELS.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

RATE_LO = 0.01    # scoring clamps rates into [0.01, 99.99]; stay inside
RATE_HI = 99.99
BETA_LO = -0.5    # sane bounds for the earners/players elasticity
BETA_HI = 2.0
BETA_PRIOR = 0.85  # mild dilution when nothing identifies the trend


# ---------------------------------------------------------------- helpers

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(rate):
    """Logit of a percent rate, with the rate clamped for safety."""
    p = clamp(rate, RATE_LO, RATE_HI) / 100.0
    return math.log(p / (1.0 - p))


def logistic_pct(x):
    """Inverse logit back to percent, overflow-safe."""
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp to epoch seconds (or None)."""
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.  Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return (my - b * mx, b)


# ------------------------------------------------------- history handling

def refresh_points(history):
    """Collapse the read history to the reads where the page changed.

    Exophase refreshes in jumps: between refreshes every read repeats the
    same players count and rates, and those repeats carry no new
    information, so consecutive duplicates are dropped.
    """
    points = []
    prev_key = None
    for read in history:
        if not isinstance(read, dict):
            continue
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        key = (players, tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        points.append(read)
    return points


def player_growth_points(points):
    """(hours since first read, players) at each distinct player count,
    timed by the first read that showed that count."""
    out = []
    seen = set()
    t0 = None
    for read in points:
        t = parse_time(read.get("slot") or read.get("at"))
        if t is None:
            continue
        if t0 is None:
            t0 = t
        players = read.get("players_tracked")
        if not isinstance(players, (int, float)) or players <= 0:
            continue
        if players in seen:
            continue
        seen.add(players)
        out.append(((t - t0) / 3600.0, float(players)))
    return out, (t0 or 0.0)


def project_players(points, target_epoch):
    """Project players_tracked at the target epoch.

    Combines (by median) a power-law fit, a saturating-exponential fit
    and a decaying-daily-growth projection, all made on the distinct
    player counts.  Falls back to the last count whenever the history
    shows no usable growth.
    """
    gp, t0 = player_growth_points(points)
    if not gp:
        return None
    p_last = gp[-1][1]
    if target_epoch is None:
        return p_last
    h_target = (target_epoch - t0) / 3600.0
    if len(gp) < 2 or h_target <= gp[-1][0]:
        return p_last
    # Fit on the recent regime when it is rich enough: the launch spike
    # says little about next week's growth, and old points would drag
    # the fits.  The last 48 hours of distinct counts, when they span at
    # least 12 hours, are the regime being extrapolated.
    recent = [pt for pt in gp if pt[0] >= gp[-1][0] - 48.0]
    if len(recent) >= 3 and recent[-1][0] - recent[0][0] >= 12.0:
        gp = recent

    projections = []

    # Power law: ln P = a + b * ln(1 + h/24).  b < 1 is decelerating
    # growth with no hard ceiling.
    xs = [math.log1p(h / 24.0) for h, _ in gp]
    ys = [math.log(p) for _, p in gp]
    fit = wls_line(xs, ys, [1.0] * len(xs))
    if fit is not None:
        a, b = fit
        if b > 0:
            projections.append(math.exp(a + b * math.log1p(h_target / 24.0)))

    # Saturating exponential: P = A - B * exp(-k h).  Grid over the
    # asymptote A; for each, ln(A - P) is linear in h.  Keep the A whose
    # fit best reproduces the observed counts in log space.
    p_max = max(p for _, p in gp)
    best = None
    for factor in (1.15, 1.3, 1.5, 1.8, 2.2, 2.7, 3.3, 4.0, 5.0,
                   6.5, 8.0, 11.0, 15.0, 20.0, 30.0):
        big_a = p_max * factor
        xs2 = [h for h, _ in gp]
        ys2 = [math.log(big_a - p) for _, p in gp]
        fit2 = wls_line(xs2, ys2, [1.0] * len(xs2))
        if fit2 is None:
            continue
        a2, k_neg = fit2
        if k_neg >= 0:
            continue
        sse = 0.0
        ok = True
        for h, p in gp:
            pred = big_a - math.exp(a2 + k_neg * h)
            if pred <= 0:
                ok = False
                break
            sse += (math.log(pred) - math.log(p)) ** 2
        if not ok:
            continue
        if best is None or sse < best[0]:
            best = (sse, big_a, a2, k_neg)
    if best is not None:
        _, big_a, a2, k_neg = best
        pred = big_a - math.exp(a2 + k_neg * h_target)
        if pred > 0:
            projections.append(pred)

    # Decaying daily growth: the multiplicative growth of the last day
    # of history, with the daily factor decaying geometrically toward 1
    # -- a launch spike does not persist at full strength for a week.
    h_last, p_last_v = gp[-1]
    if h_target > h_last:
        p_day_ago = gp[0][1]
        for h, p in gp:
            if h <= h_last - 24.0:
                p_day_ago = p
            else:
                break
        window = min(24.0, h_last) if h_last > 0 else 0.0
        if window > 0 and p_day_ago > 0:
            g_day = (p_last_v / p_day_ago) ** (24.0 / window)
            g_day = clamp(g_day, 1.0, 8.0)
            factor = 1.0
            g = g_day
            left = (h_target - h_last) / 24.0
            while left > 0:
                step = min(1.0, left)
                factor *= g ** step
                g = 1.0 + (g - 1.0) * 0.75
                left -= step
            projections.append(p_last_v * factor)

    if not projections:
        return p_last
    projections.sort()
    mid = len(projections) // 2
    if len(projections) % 2:
        geo = projections[mid]
    else:
        geo = math.sqrt(projections[mid - 1] * projections[mid])
    return clamp(geo, p_last, p_last * 40.0)


# ------------------------------------------------------------- betas

def pooled_beta(points):
    """Elasticity of the page's total earners to the player count.

    Fitted on trophies_earned over the refresh points.  Returns None
    when the totals cannot identify it.
    """
    xs, ys, ws = [], [], []
    p_ref = None
    for read in points:
        players = read.get("players_tracked")
        total = read.get("trophies_earned")
        if not isinstance(players, (int, float)) or players <= 0:
            continue
        if not isinstance(total, (int, float)) or total <= 0:
            continue
        p_ref = players
        xs.append(math.log(players))
        ys.append(math.log(total))
        ws.append(float(players))
    if p_ref is None or len(set(xs)) < 2:
        return None
    ws = [w / p_ref for w in ws]
    fit = wls_line(xs, ys, ws)
    if fit is None:
        return None
    return clamp(fit[1], BETA_LO, BETA_HI)


def trophy_beta(obs, beta_pool):
    """A trophy's earners/players elasticity, shrunk to the pooled one.

    obs is the trophy's (players, rate) observations at refresh points.
    Returns (beta, weight) where weight is the share the trophy's own
    fit carries.  The share follows the fit's own precision: the slope's
    standard error from the regression residuals, against a prior that
    true betas scatter about the pooled one with sd 0.3.  A clean fit
    over a wide range of player counts (a universal trophy whose earners
    track the player count one for one, say) is believed; a noisy or
    narrow one falls back to the pool.
    """
    xs, ys, ws = [], [], []
    p_ref = None
    for players, rate in obs:
        if rate is None or players <= 0:
            continue
        earners = rate / 100.0 * players
        if earners <= 0:
            continue
        p_ref = players
        xs.append(math.log(players))
        ys.append(math.log(earners))
        ws.append(float(players))
    if p_ref is None:
        return beta_pool, 0.0
    ws = [w / p_ref for w in ws]
    n = len(xs)
    if len(set(xs)) < 2:
        return beta_pool, 0.0
    fit = wls_line(xs, ys, ws)
    if fit is None:
        return beta_pool, 0.0
    a, b = fit
    own = clamp(b, BETA_LO, BETA_HI)
    # Slope standard error from the residuals.  The residual floor keeps
    # the page's 2-decimal rounding from reading as a perfect fit.
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    s2 = max(sum(r * r for r in resid) / dof, 0.03 ** 2)
    mx = sum(xs) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return beta_pool, 0.0
    se2 = s2 / sxx
    w = 0.09 / (0.09 + se2)
    w = min(w, n / (n + 1.0))
    if n == 2:
        w = min(w, 0.45)
    return w * own + (1.0 - w) * beta_pool, w


# ------------------------------------------------------------- forecast

def spread_sigma(horizon_days, weight, never_earned):
    """Logit-scale spread of a trophy's forecast.

    A random-walk term growing with the horizon, a floor for Exophase
    refresh timing, extra width when the trophy's own trend is weakly
    identified, and more when nobody has earned it yet.
    """
    var = (0.55 * math.sqrt(max(horizon_days, 0.0))) ** 2
    var += 0.30 ** 2
    var += (0.35 * (1.0 - weight)) ** 2
    if never_earned:
        var += 0.45 ** 2
    return math.sqrt(var)


def quantiles(median, sigma):
    centre = logit_pct(median)
    qs = []
    for z in Z:
        q = logistic_pct(centre + z * sigma)
        qs.append(clamp(q, RATE_LO, RATE_HI))
    qs = [round(q, 4) for q in qs]
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def fallback_row(rate, horizon_days):
    """Last-rate row with a plain horizon spread (used if the model
    cannot run for a trophy, and as the whole-case safety net)."""
    if rate is None:
        rate = 1.0
    sigma = 0.5 * math.sqrt(max(horizon_days, 0.0)) + 0.30
    return quantiles(clamp(rate, RATE_LO, RATE_HI), sigma)


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    points = refresh_points(history)

    horizon_min = case.get("horizon_minutes")
    if not isinstance(horizon_min, (int, float)) or horizon_min < 0:
        horizon_min = 0
    horizon_days = horizon_min / 1440.0
    target_epoch = parse_time(case.get("target"))

    p_target = project_players(points, target_epoch)
    beta_pool = pooled_beta(points)
    if beta_pool is None:
        beta_pool = BETA_PRIOR

    # Last read (the raw history's, not only refresh points) supplies the
    # reference players count and the fallback rates.
    last_read = history[-1] if history else {}
    p_last = last_read.get("players_tracked")
    if not isinstance(p_last, (int, float)) or p_last <= 0:
        p_last = None
        for read in reversed(points):
            cand = read.get("players_tracked")
            if isinstance(cand, (int, float)) and cand > 0:
                p_last = cand
                break
    if p_target is None:
        p_target = p_last
    growth = 1.0
    if p_last and p_target:
        growth = p_target / p_last

    rows = []
    for idx, trophy in enumerate(trophies):
        tid = trophy.get("id", idx + 1) if isinstance(trophy, dict) else idx + 1
        obs = []
        r_last = None
        p_at_last = None
        for read in points:
            rates = read.get("rates") or []
            rate = rates[idx] if idx < len(rates) else None
            players = read.get("players_tracked")
            if isinstance(players, (int, float)) and players > 0:
                obs.append((float(players), rate))
                if rate is not None:
                    r_last = rate
                    p_at_last = float(players)
        # Rates from the raw last read win over refresh-point bookkeeping.
        last_rates = last_read.get("rates") or []
        if idx < len(last_rates) and last_rates[idx] is not None:
            r_last = last_rates[idx]
            if isinstance(last_read.get("players_tracked"), (int, float)) \
                    and last_read["players_tracked"] > 0:
                p_at_last = float(last_read["players_tracked"])

        if r_last is None or p_at_last is None:
            rows.append({"id": tid,
                         "q": fallback_row(None, horizon_days)})
            continue

        beta, weight = trophy_beta(obs, beta_pool)
        earners_last = r_last / 100.0 * p_at_last
        if earners_last > 0:
            earners_t = earners_last * growth ** beta
        else:
            # Never earned (or earned below the page's 2-decimal floor):
            # project a half-earner pseudocount on the pooled trend.
            earners_t = 0.5 * growth ** beta_pool
        if p_target:
            median = 100.0 * earners_t / p_target
        else:
            median = r_last
        median = clamp(median, RATE_LO, RATE_HI)
        sigma = spread_sigma(horizon_days, weight, earners_last <= 0)
        rows.append({"id": tid, "q": quantiles(median, sigma)})
    return {"trophies": rows}


def safe_forecast_case(case):
    """forecast_case with a last-rate safety net: a case must always
    produce a valid forecast file, never a crash."""
    try:
        return forecast_case(case)
    except Exception:
        trophies = case.get("trophies") or []
        history = case.get("history") or []
        horizon_min = case.get("horizon_minutes")
        horizon_days = (horizon_min / 1440.0
                        if isinstance(horizon_min, (int, float))
                        and horizon_min >= 0 else 0.0)
        last_rates = (history[-1].get("rates") or []) if history else []
        rows = []
        for idx, trophy in enumerate(trophies):
            tid = trophy.get("id", idx + 1) if isinstance(trophy, dict) \
                else idx + 1
            rate = last_rates[idx] if idx < len(last_rates) else None
            rows.append({"id": tid, "q": fallback_row(rate, horizon_days)})
        return {"trophies": rows}


# ------------------------------------------------------------------ main

def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        pass
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        stem = os.path.basename(path)[:-3]
        try:
            with open(path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
        except Exception:
            continue
        result = safe_forecast_case(case)
        out_path = os.path.join(out_dir, stem + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False,
                      separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
