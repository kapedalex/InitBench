#!/usr/bin/env python3
"""Search-and-grab loop for a VerInf 400B box, with a HARD SPEND CAP.

Replaces the throwaway hunt.py that ran in a scratchpad on Aug 2-3 and burned
~$50 unattended after the Claude session hit its weekly limit. Every failure
mode from that post-mortem is capped here:

  old failure                                   -> what stops it now
  ------------------------------------------------------------------------
  700x20s = 3.9h blanket watch, twice           -> --preflight-timeout (30m)
                                                   applies until PRE-FLIGHT PASS;
                                                   the long --work-timeout only
                                                   starts once the box is proven good
  box_log() swallowed errors -> unreachable     -> --max-err-polls consecutive
  box rode the full 3.9h                           errors -> destroy
  r.terminate() skipped the runner's            -> runner now traps SIGTERM
  `finally: destroy`                               (vast_run_verinf._install_term_handlers)
                                                   and we wait for it to unwind
  destroy() 429 -> printed, box kept billing    -> _destroy() fixed-interval
                                                   retries + DEBT_FILE
  decisions on offer price (1.6-2.6x too low)   -> spend accounted at the LIVE
                                                   instance rate; --max-real-dph
  no budget / attempt / give-up limit           -> --max-spend, --max-attempts,
                                                   --max-create-failures
  62 consecutive HTTP 400 creates (no balance)  -> balance checked up front and
                                                   after each create failure
  one network blip killed the loop              -> main loop is exception-safe
  nothing knew the session had died             -> heartbeat file for vast_watchdog.py

Spend is persisted, so a restart continues the same budget instead of resetting it.

Usage:
  python3 vast_hunt.py --max-spend 15 --max-attempts 6 \
      --min-vram 44 --min-ram 48 --min-disk 6000 --max-dph 1.5 --max-real-dph 3.0
"""
import argparse
import json
import os
import signal
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vast_run_verinf import _destroy, load_key, real_dph, vast_req  # noqa: E402

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
EXCLUDE = ("Blackwell", "RTX PRO 6000", "RTX 5090", "RTX 5080", "RTX 5060", "B200", "B300")


class Budget:
    """Cumulative spend, charged at the REAL rate of whatever box is live.

    Persisted to disk: killing and restarting the hunter must not hand it a
    fresh budget."""

    def __init__(self, cap, path):
        self.cap = cap
        self.path = Path(path)
        self.spent = 0.0
        if self.path.exists():
            try:
                self.spent = float(json.loads(self.path.read_text()).get("spent", 0.0))
                print(f"[budget] resuming: ${self.spent:.2f} already spent of ${cap:.2f}")
            except Exception:
                pass

    def charge(self, dollars):
        self.spent += max(0.0, dollars)
        try:
            self.path.write_text(json.dumps({"spent": round(self.spent, 4),
                                             "updated": int(time.time())}))
        except Exception:
            pass

    @property
    def left(self):
        return self.cap - self.spent

    def exhausted(self, headroom=0.0):
        return self.left <= headroom


def account_balance(key):
    try:
        u = vast_req("/users/current/", key)
        return float(u.get("credit") or 0.0) + float(u.get("balance") or 0.0)
    except Exception as e:
        print(f"[balance] could not read: {e}")
        return None


def best_offer(key, args):
    body = {"num_gpus": {"eq": 1}, "gpu_ram": {"gte": args.min_vram * 1024},
            "disk_space": {"gte": args.min_disk}, "cpu_ram": {"gte": args.min_ram * 1024},
            "rentable": {"eq": True}, "verified": {"eq": True}, "type": "on-demand",
            "order": [["dph_total", "asc"]], "limit": 400}
    req = urllib.request.Request("https://console.vast.ai/api/v0/bundles/",
                                 data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}",
                                          "Content-Type": "application/json"})
    offers = json.load(urllib.request.urlopen(req, timeout=25)).get("offers", [])
    ok = [o for o in offers
          if (o.get("num_gpus") or 9) == 1
          and (o.get("dph_total") or 9e9) <= args.max_dph
          and not any(x in (o.get("gpu_name") or "") for x in EXCLUDE)]
    return sorted(ok, key=lambda o: o.get("dph_total", 9e9))[0] if ok else None


def instance_info(key, inst):
    """Live instance record, or None if it is gone."""
    try:
        i = vast_req(f"/instances/{inst}/", key).get("instances")
        return i or None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def box_log(key, inst, path="suite.log"):
    """(text, status) from the box. status 'gone' means the instance vanished;
    'err:*' means WE could not reach it — the caller must count those, because
    an unreachable box still bills."""
    try:
        i = instance_info(key, inst)
        if not i:
            return "", "gone"
        ip = i.get("public_ipaddr")
        hp = (i.get("ports") or {}).get("8080/tcp", [{}])[0].get("HostPort")
        tok = i.get("jupyter_token")
        if not (ip and hp and tok):
            return "", i.get("actual_status") or "pending"
        url = f"https://{ip}:{hp}/api/contents/{path}?content=1&token={tok}"
        c = json.load(urllib.request.urlopen(url, timeout=15, context=CTX)).get("content", "")
        return c, i.get("actual_status")
    except urllib.error.HTTPError as e:
        # 404 on the log file just means the run has not written it yet — that is
        # a healthy box, not an unreachable one.
        return "", ("pending" if e.code == 404 else f"err:HTTP{e.code}")
    except Exception as e:
        return "", f"err:{type(e).__name__}"


def stop_runner(proc, grace=180):
    """SIGTERM the runner and WAIT — it now traps the signal and destroys its own
    instance in `finally`. Only SIGKILL if it refuses to unwind."""
    if proc.poll() is not None:
        return
    print("  stopping runner (SIGTERM -> it destroys its own box)", flush=True)
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=grace)
        print("  runner unwound cleanly", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  runner still alive after {grace}s — SIGKILL", flush=True)
        proc.kill()
        proc.wait(timeout=30)
    except Exception as e:
        print(f"  stop_runner: {e}", flush=True)


def hunt(args, key, budget, log, mine):
    """`mine` is the set of instance ids this hunter created — the sweep at exit
    destroys exactly those and nothing else."""
    attempts = 0
    create_failures = 0

    while True:
        if budget.exhausted(headroom=args.min_headroom):
            log(f"BUDGET: ${budget.spent:.2f} of ${budget.cap:.2f} spent "
                f"(headroom ${args.min_headroom:.2f}) — STOP")
            return "budget"
        if attempts >= args.max_attempts:
            log(f"ATTEMPTS: {attempts}/{args.max_attempts} used — STOP")
            return "attempts"
        if create_failures >= args.max_create_failures:
            log(f"CREATE FAILED {create_failures}x in a row — STOP (check balance / filters)")
            return "create_failures"

        try:
            offer = best_offer(key, args)
        except Exception as e:
            log(f"offer query failed ({type(e).__name__}: {e}) — retry in 5min")
            time.sleep(300)
            continue

        if not offer:
            log(f"no matching offer, waiting 5min (budget left ${budget.left:.2f})")
            time.sleep(300)
            continue

        attempts += 1
        log(f"attempt {attempts}/{args.max_attempts}: {offer.get('gpu_name')} "
            f"{(offer.get('disk_space') or 0)/1000:.1f}TB @{(offer.get('disk_bw') or 0)/1000:.0f}GB/s "
            f"offer ${offer.get('dph_total'):.2f}/h {offer.get('geolocation','?')} "
            f"| budget left ${budget.left:.2f}")

        outcome, inst, proc = None, None, None
        try:
            proc = subprocess.Popen(
                ["python3", "vast_run_verinf.py", "--env-file", args.env_file, "--mavrun",
                 "--min-vram", str(args.min_vram), "--min-ram", str(args.min_ram),
                 "--min-disk", str(args.min_disk), "--disk-gb", str(args.disk_gb),
                 "--max-dph", str(args.max_dph), "--max-real-dph", str(args.max_real_dph),
                 "--max-wait", str(args.work_timeout)],
                cwd=str(Path(__file__).parent),
                env={**os.environ, "MAV_LAYERS": str(args.mav_layers),
                     "MAV_MIN_READ_GBPS": str(args.min_read_gbps)},
                stdout=open(args.runner_log, "w"), stderr=subprocess.STDOUT)

            # ── find the instance id the runner created ──────────────────────
            for _ in range(48):          # 4 min: pick offer + PUT /asks
                time.sleep(5)
                txt = Path(args.runner_log).read_text(errors="replace")
                if "Instance:" in txt:
                    inst = txt.split("Instance:")[1].split()[0].strip()
                    break
                if proc.poll() is not None:
                    break
            if not inst:
                create_failures += 1
                tail = Path(args.runner_log).read_text(errors="replace")[-400:]
                log(f"  no instance created ({create_failures}/{args.max_create_failures}) "
                    f"| runner tail: {tail.strip()[-200:]}")
                bal = account_balance(key)
                if bal is not None and bal <= 0:
                    log(f"  BALANCE ${bal:.2f} <= 0 — that is why creates fail. STOP.")
                    return "no_balance"
                stop_runner(proc)
                time.sleep(60)
                continue

            create_failures = 0
            mine.add(str(inst))
            # persist immediately: if this process dies right here, vast_watchdog.py
            # still knows which box to kill
            try:
                with open(args.mine_file, "a") as fh:
                    fh.write(f"{inst}\n")
            except Exception:
                pass
            log(f"  instance {inst} created")

            # ── the real billed rate, charged from here on ───────────────────
            info = instance_info(key, inst)
            rate = real_dph(info)[0] if info else args.max_real_dph
            if rate <= 0:
                rate = offer.get("dph_total") or args.max_real_dph
            log(f"  real rate ${rate:.2f}/h (offer said ${offer.get('dph_total'):.2f}/h)")

            # ── watch: SHORT cap until pre-flight passes, long cap only after ─
            t0 = time.time()
            deadline = t0 + args.preflight_timeout
            phase = "preflight"
            err_streak = 0
            last_charge = t0

            while True:
                now = time.time()
                budget.charge((now - last_charge) / 3600.0 * rate)
                last_charge = now

                if budget.exhausted():
                    outcome = f"budget_exceeded(${budget.spent:.2f})"
                    break
                if now > deadline:
                    outcome = f"timeout_{phase}({int(now-t0)}s)"
                    break

                text, status = box_log(key, inst)
                if status == "gone":
                    outcome = "gone"
                    break
                if str(status).startswith("err:"):
                    err_streak += 1
                    if err_streak >= args.max_err_polls:
                        outcome = f"unreachable_{err_streak}x({status})"
                        break
                else:
                    err_streak = 0

                if "PREFLIGHT_FAIL" in text:
                    outcome = "preflight_fail"
                    break
                if "DOWNLOAD FAILED" in text:
                    outcome = "download_fail"
                    break
                # a FAILED prove ends the run too — without this the box just
                # rides the watchdog to timeout, paying for a finished job
                if "prove exit=" in text and "prove exit=0" not in text:
                    outcome = "prove_failed"
                    break
                if "MAV DONE" in text and "prove returned" not in text:
                    outcome = "run_finished_no_timing"
                    break
                if phase == "preflight" and "PRE-FLIGHT PASS" in text:
                    phase = "work"
                    deadline = time.time() + args.work_timeout
                    log(f"  PRE-FLIGHT PASS after {int(now-t0)}s — long watch armed "
                        f"({args.work_timeout}s, ~${args.work_timeout/3600*rate:.2f})")
                if "prove returned" in text:
                    secs = text.split("prove returned (")[1].split("s)")[0]
                    outcome = ("prove", float(secs))
                    break

                time.sleep(args.poll_s)

            budget.charge((time.time() - last_charge) / 3600.0 * rate)

        except Exception as e:
            outcome = f"hunter_error({type(e).__name__}: {e})"
        finally:
            if proc is not None:
                stop_runner(proc)
            if inst:
                _destroy(inst, key)          # backstop; 404 = runner already did it

        if isinstance(outcome, tuple):
            secs = outcome[1]
            log(f"\n*** PROVE DONE on {offer.get('gpu_name')}: {secs:.1f}s = {secs/60:.1f} min "
                f"— spent ${budget.spent:.2f} ***")
            text, _ = box_log(key, inst)
            for ln in (text or "").splitlines():
                if any(x in ln for x in ("prove returned", "peakGPU", "PROVE", "disk BW",
                                         "PRE-FLIGHT PASS")):
                    log("  | " + ln[:130])
            return "success"

        log(f"  attempt {attempts} outcome={outcome} | spent ${budget.spent:.2f} "
            f"of ${budget.cap:.2f}")
        time.sleep(args.cooldown_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-file", default="/home/riftuser/VerInf/.env")
    ap.add_argument("--max-spend", type=float, required=True,
                    help="HARD budget in $ for this hunt, charged at the live instance rate")
    ap.add_argument("--max-attempts", type=int, default=6, help="give up after N rentals")
    ap.add_argument("--max-create-failures", type=int, default=5,
                    help="give up after N consecutive failures to create an instance")
    ap.add_argument("--min-headroom", type=float, default=3.0,
                    help="do not start another rental with less than this much budget left")
    ap.add_argument("--preflight-timeout", type=int, default=1800,
                    help="s to reach PRE-FLIGHT PASS (setup dead-time ~18min). This is the cap "
                         "that stops 4h of paying for a box that never reports anything.")
    ap.add_argument("--work-timeout", type=int, default=14400,
                    help="s for the actual run, armed ONLY after PRE-FLIGHT PASS")
    ap.add_argument("--max-err-polls", type=int, default=15,
                    help="consecutive unreachable polls before destroying (default 15 x poll-s)")
    ap.add_argument("--poll-s", type=int, default=20)
    ap.add_argument("--cooldown-s", type=int, default=90)
    ap.add_argument("--min-vram", type=int, default=44)
    ap.add_argument("--min-ram", type=int, default=48)
    ap.add_argument("--min-disk", type=int, default=6000)
    ap.add_argument("--disk-gb", type=int, default=6000)
    ap.add_argument("--max-dph", type=float, default=1.5, help="max OFFER $/h (understates the bill)")
    ap.add_argument("--max-real-dph", type=float, default=3.0,
                    help="max ACTUAL $/h (compute+storage) — the cap that matters")
    ap.add_argument("--mav-layers", type=int, default=48)
    ap.add_argument("--min-read-gbps", type=float, default=1.5)
    ap.add_argument("--state-dir", default=str(Path.home() / ".vast_hunt"))
    args = ap.parse_args()

    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    args.runner_log = str(state / "runner.log")
    args.mine_file = str(state / "mine.txt")
    logfile = state / "hunt.log"
    heartbeat = state / "heartbeat"

    def log(m):
        line = f"[{time.strftime('%m-%d %H:%M')}] {m}"
        print(line, flush=True)
        try:
            with open(logfile, "a") as fh:
                fh.write(line + "\n")
            heartbeat.write_text(str(int(time.time())))
        except Exception:
            pass

    key = load_key(args.env_file)
    budget = Budget(args.max_spend, state / "spend.json")
    MINE = set()          # instance ids this hunter created

    bal = account_balance(key)
    if bal is not None:
        log(f"account balance ${bal:.2f}")
        if bal <= 0:
            log("balance <= 0 — vast will reject every create (HTTP 400). Top up first. STOP.")
            return 2
    log(f"hunt start: budget ${budget.left:.2f} left of ${budget.cap:.2f}, "
        f"max {args.max_attempts} rentals, preflight cap {args.preflight_timeout}s, "
        f"real-rate cap ${args.max_real_dph:.2f}/h")

    # Signals must not skip the sweep either — the whole point is that nothing
    # this hunter rented outlives it.
    def _bail(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _bail)
        except (ValueError, OSError):
            pass

    reason = "error"
    try:
        reason = hunt(args, key, budget, log, MINE)
    except KeyboardInterrupt as e:
        reason = f"interrupted({e})"
    finally:
        log(f"hunt end: reason={reason}, spent ${budget.spent:.2f} of ${budget.cap:.2f}")
        # last line of defence: destroy every instance THIS hunter rented.
        # Anything else on the account is somebody else's and is only reported.
        for inst in sorted(MINE):
            if not _destroy(inst, key):
                log(f"SWEEP: could NOT destroy {inst} — see {Path.home()/'.vast_undestroyed'}")
        try:
            live = vast_req("/instances/", key).get("instances", [])
            stray = [i["id"] for i in live if str(i["id"]) not in MINE]
            log(f"SWEEP: {len(live)} live instance(s) left on the account"
                + (f" — NOT mine, left alone: {stray}" if stray else " — clean"))
        except Exception as e:
            log(f"SWEEP CHECK FAILED ({e}) — verify https://console.vast.ai/instances/ MANUALLY")
    return 0 if reason == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
