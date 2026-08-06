#!/usr/bin/env python3
"""Host-side dead-man's switch for vast.ai rentals.

The Aug 2-3 incident: the Claude session hit its weekly limit at 19:15, the
detached hunter kept renting boxes for another 19 hours, and nothing on the host
knew the supervisor was gone. This process is that missing piece — it outlives
nothing: it only watches, and kills rentals whose owner has stopped reporting.

Scope is deliberately narrow. It destroys ONLY:
  - instances listed in the hunter's mine.txt (boxes the hunter rented), and
    only once the hunter's heartbeat has been stale for --stale-after seconds;
  - instances in ~/.vast_undestroyed (boxes a DELETE previously failed on).
Anything else on the account is somebody else's and is reported, never touched.

Run it next to any unattended hunt:
  nohup python3 vast_watchdog.py > ~/.vast_hunt/watchdog.log 2>&1 &

Or one-shot, to check/clean up after the fact:
  python3 vast_watchdog.py --once
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vast_run_verinf import DEBT_FILE, _destroy, load_key, real_dph, vast_req  # noqa: E402


def read_ids(path):
    p = Path(path)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        # mine.txt is bare ids; the debt file is "ts<TAB>id<TAB>reason"
        out.add(line.split("\t")[1] if "\t" in line else line)
    return out


def heartbeat_age(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return time.time() - float(p.read_text().strip())
    except Exception:
        return time.time() - p.stat().st_mtime


def sweep(key, args, log):
    try:
        live = vast_req("/instances/", key).get("instances", [])
    except Exception as e:
        log(f"cannot list instances ({e}) — will retry")
        return
    live_ids = {str(i["id"]): i for i in live}

    debts = read_ids(DEBT_FILE)
    mine = read_ids(args.mine_file)
    age = heartbeat_age(args.heartbeat)

    burn = sum(real_dph(i)[0] for i in live)
    log(f"live={len(live)} (${burn:.2f}/h) mine={len(mine & set(live_ids))} "
        f"debt={len(debts)} heartbeat="
        + ("absent" if age is None else f"{int(age)}s old"))

    # 1) boxes a DELETE previously failed on — always fair game, they are known leaks
    for inst in debts:
        if inst in live_ids:
            log(f"DEBT: destroying {inst} (a previous DELETE failed)")
            _destroy(inst, key)
        else:
            log(f"DEBT: {inst} already gone — clearing")
            _destroy(inst, key)   # 404 path clears it from the debt file

    # 2) the hunter's boxes, once the hunter has stopped reporting
    orphans = [i for i in mine if i in live_ids]
    if not orphans:
        return
    if age is None:
        log(f"{len(orphans)} tracked instance(s) live but NO heartbeat file — "
            f"treating owner as dead")
    elif age < args.stale_after:
        log(f"{len(orphans)} tracked instance(s) live, owner alive "
            f"({int(age)}s < {args.stale_after}s) — leaving alone")
        return
    else:
        log(f"OWNER DEAD: heartbeat {int(age)}s old > {args.stale_after}s — "
            f"destroying {len(orphans)} tracked instance(s)")
    for inst in orphans:
        rate = real_dph(live_ids[inst])[0]
        log(f"  destroying {inst} (${rate:.2f}/h)")
        _destroy(inst, key)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-file", default="/home/riftuser/VerInf/.env")
    ap.add_argument("--state-dir", default=str(Path.home() / ".vast_hunt"))
    ap.add_argument("--stale-after", type=int, default=1800,
                    help="s without a heartbeat before the owner counts as dead (default 30min)")
    ap.add_argument("--interval", type=int, default=600, help="s between checks (default 10min)")
    ap.add_argument("--once", action="store_true", help="single check, then exit")
    args = ap.parse_args()

    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    args.mine_file = state / "mine.txt"
    args.heartbeat = state / "heartbeat"

    def log(m):
        print(f"[{time.strftime('%m-%d %H:%M')}] {m}", flush=True)

    key = load_key(args.env_file)
    log(f"watchdog up: stale-after={args.stale_after}s interval={args.interval}s "
        f"state={state}")
    while True:
        try:
            sweep(key, args, log)
        except Exception as e:
            log(f"sweep error ({type(e).__name__}: {e}) — continuing")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
