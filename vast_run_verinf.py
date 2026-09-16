"""
VAST.ai runner for the VerInf prover VALIDATION campaign (companion to the
prover-remote-validation skill). Adapted from vast_run.py.

Flow: pick a VerInf-suitable offer within budget -> create instance -> upload the
VerInf repo as ONE tarball -> bootstrap (uv sync + build Rust verifier) -> run
analysis/bench/remote_suite.py at medium scale -> download remote_results ->
DESTROY the instance.

Differences from vast_run.py (VerInf downloads NO model, and needs nvcc + Rust):
  - offer filter: gpu_ram>=40GB, cpu_ram>=64GB, rentable+verified, sorted by
    price, honoring --max-dph. NO fast-network filter (nothing is downloaded).
  - CUDA *devel* image (nvcc) — the custom kernels JIT-compile on first prove.
  - bootstrap installs uv, runs `uv sync` (pinned torch cu126), builds the Rust
    verifier (cargo) so the ACCEPT gate works.
  - payload = repo tarball, not per-file uploads.

I could NOT test this end-to-end (no VAST key in the dev env). The vast/ws/jupyter
mechanics are copied verbatim from the proven vast_run.py; the VerInf-specific
bootstrap is the shakedown risk — watch onstart.log + bootstrap.log on the first
run. Requires the `vastai` CLI on PATH and VAST_AI_KEY (env or --env-file).

Usage:
  export VAST_AI_KEY=...            # or --env-file /path/to/.env
  python3 vast_run_verinf.py --max-dph 1.20 --matrix medium
  python3 vast_run_verinf.py --min-vram 48 --matrix medium --reps 3
"""
import argparse
import base64
import io
import json
import os
import signal
import socket
import ssl
import struct
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
VERINF = Path(os.environ.get("VERINF_ROOT", "/home/riftuser/VerInf"))
LOCAL_RESULTS = VERINF / "analysis" / "bench" / "remote_results"
# Image MUST have python+pip (vast pip-installs jupyter into it at launch) AND
# nvcc (devel tag) for the custom CUDA kernels. A raw nvidia/cuda image has no
# python -> vast can't set up jupyter -> instance stays Offline ("Template not
# found"). The pytorch *-devel* image (proven in vast_run.py) has both. uv sync
# still installs our pinned torch cu126 into its own venv, so the base torch
# version doesn't matter; we only need its python + nvcc.
IMAGE = os.environ.get("VERINF_IMAGE", "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel")
DISK_GB = 60          # repo + uv env + cargo target; no model weights (override --disk-gb)

# Billed DEAD TIME before the first useful measurement — this is vast's real
# "first-rental is expensive" effect (not a fee; it's setup time at full $/h):
#   image pull (~5GB devel) 2-4m + uv sync (torch cu126 ~2.5GB) 3-6m +
#   cargo build --release 2-4m + first CUDA-kernel JIT compile 2-4m.
SETUP_MIN = 18          # conservative; the fixed cost you amortize by BATCHING
GATES_MIN = 4           # the soundness gates (small, fixed)
# per-config A/B prove COUNT: ab_gpu_softmax = 2 modes*reps, ab_witness_spill =
# 3 modes*reps, each preceded by a build (~folded into avg-prove-s).
PROVES_PER_CONFIG = lambda reps: (2 + 3) * reps

_MATRIX_CONFIGS = {   # mirror remote_suite.MATRICES sizes for the estimate
    "smoke": 1, "phone": 2, "medium": 4,
}
# repo paths to EXCLUDE from the tarball (build junk / caches / prior results)
TAR_EXCLUDE = {".git", "__pycache__", ".venv", "venv", "node_modules", "target",
               "remote_results", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TAR_SECRET_SUFFIXES = {".key", ".pem", ".secret", ".wcommit"}

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def load_key(env_file):
    if os.environ.get("VAST_AI_KEY"):
        return os.environ["VAST_AI_KEY"]
    if env_file and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            if line.startswith("VAST_AI_KEY") and "=" in line:
                return line.partition("=")[2].strip()
    raise SystemExit("No VAST_AI_KEY (set env var or --env-file)")


def vast_req(path, key, method="GET", data=None):
    req = urllib.request.Request(
        f"https://console.vast.ai/api/v0{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


# ── billing safety net ────────────────────────────────────────────────────────
# Post-mortem of the Aug 2-3 hunt: ~$50 went to instances nobody was watching.
# Three mechanisms failed at once, and all three are handled here:
#   1. the hunter SIGTERM'd this runner -> Python exits WITHOUT running `finally`
#      -> the "always destroy" net never fired.  _install_term_handlers() turns
#      SIGTERM/SIGINT/SIGHUP into SystemExit so `finally` DOES run.
#   2. a single DELETE that returned 429 was only printed, and the box kept
#      billing.  _destroy() retries at a FIXED interval and, if it still fails,
#      records the id in DEBT_FILE so a sweeper (vast_watchdog.py) finishes the job.
#   3. decisions used the OFFER price, which understated the real billed rate by
#      1.6-2.6x (L40S offer $0.64/h -> actual $2.45 compute + $0.20 storage).
#      real_dph() reads the price off the LIVE instance instead.
DEBT_FILE = Path(os.environ.get("VAST_DEBT_FILE", Path.home() / ".vast_undestroyed"))


def _install_term_handlers():
    """Make a kill signal raise SystemExit so `finally: destroy` still runs.

    Without this, `terminate()` from a parent process kills us instantly and the
    rented instance keeps billing until somebody notices."""
    def _bail(signum, _frame):
        raise SystemExit(f"received signal {signum} — unwinding (instance will be destroyed)")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _bail)
        except (ValueError, OSError):
            pass   # not on the main thread / signal unavailable — best effort


def _note_debt(instance, reason):
    """Record an instance we could NOT destroy, so it is never silently forgotten."""
    try:
        with open(DEBT_FILE, "a") as fh:
            fh.write(f"{int(time.time())}\t{instance}\t{reason}\n")
        print(f"!! instance {instance} written to {DEBT_FILE} — it may STILL BE BILLING")
    except Exception as e:
        print(f"!! could not even record the debt ({e}); DESTROY {instance} MANUALLY")


def _clear_debt(instance):
    """Drop an id from the debt file once it is confirmed gone."""
    try:
        if not DEBT_FILE.exists():
            return
        keep = [ln for ln in DEBT_FILE.read_text().splitlines()
                if ln.split("\t")[1:2] != [str(instance)]]
        DEBT_FILE.write_text("".join(ln + "\n" for ln in keep))
    except Exception:
        pass


def _destroy(instance, key, tries=6, gap=20):
    """Destroy an instance, retrying at a FIXED interval. 404 counts as success
    (already gone). Returns True if the box is confirmed down."""
    if not instance:
        return True
    last = None
    for attempt in range(1, tries + 1):
        try:
            vast_req(f"/instances/{instance}/", key, "DELETE")
            print(f"Destroyed {instance}")
            _clear_debt(instance)
            return True
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code == 404:
                print(f"Instance {instance} already gone (404)")
                _clear_debt(instance)
                return True
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        print(f"  destroy {instance} failed ({last}) — retry {attempt}/{tries} in {gap}s", flush=True)
        if attempt < tries:
            time.sleep(gap)
    _note_debt(instance, last or "unknown")
    return False


def real_dph(info):
    """Actual billed $/h of a LIVE instance: compute + storage.

    The offer's dph_total is not what you pay. vast reports the storage line
    under different keys depending on the endpoint version, so try the explicit
    hourly figure first and fall back to `storage_cost`, which matched the
    observed bill (6TB -> ~$0.20-0.33/h). Both raw fields are printed so a
    mismatch is visible in the log instead of silently mispricing the run."""
    comp = info.get("dph_total") or 0.0
    stor = info.get("storage_total_cost")
    if stor is None:
        stor = info.get("storage_cost") or 0.0
    return float(comp) + float(stor), float(comp), float(stor)


# ── jupyter contents + websocket helpers (verbatim from vast_run.py) ───────────
def http_req(url, token, method="GET", data=None, timeout=30):
    req = urllib.request.Request(url, data=data, method=method,
        headers={"Authorization": f"token {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.status, r.read()


def upload_file(base_url, token, remote_path, content_bytes):
    payload = json.dumps({"name": Path(remote_path).name, "path": remote_path,
        "type": "file", "format": "base64",
        "content": base64.b64encode(content_bytes).decode()}).encode()
    status, _ = http_req(f"{base_url}/api/contents/{remote_path}", token, "PUT", payload)
    return status in (200, 201)


def read_remote_file(base_url, token, remote_path):
    try:
        req = urllib.request.Request(f"{base_url}/api/contents/{remote_path}?format=base64",
            headers={"Authorization": f"token {token}"})
        with urllib.request.urlopen(req, timeout=60, context=CTX) as r:
            return base64.b64decode(json.load(r)["content"])
    except Exception:
        return None


def list_remote_dir(base_url, token, path):
    try:
        req = urllib.request.Request(f"{base_url}/api/contents/{path}",
            headers={"Authorization": f"token {token}"})
        with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
            return json.load(r).get("content", [])
    except Exception:
        return []


def ws_connect(ip, port, path, token):
    key = base64.b64encode(uuid.uuid4().bytes).decode()
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    s = socket.create_connection((ip, int(port)), timeout=30)
    s = ctx.wrap_socket(s, server_hostname=ip)
    s.send((f"GET {path} HTTP/1.1\r\nHost: {ip}:{port}\r\nUpgrade: websocket\r\n"
            f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
            f"Authorization: token {token}\r\n\r\n").encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += s.recv(4096)
    assert b"101" in resp, f"Handshake failed: {resp[:200]}"
    return s


def ws_send(s, msg):
    data = msg.encode(); length = len(data); mask = uuid.uuid4().bytes[:4]
    masked = bytes(data[i] ^ mask[i % 4] for i in range(length))
    frame = bytearray([0x81])
    if length < 126: frame.append(0x80 | length)
    elif length < 65536: frame += bytearray([0x80 | 126]) + struct.pack(">H", length)
    else: frame += bytearray([0x80 | 127]) + struct.pack(">Q", length)
    frame += mask + masked
    s.send(bytes(frame))


def ws_recv(s):
    def rd(n):
        b = b""
        while len(b) < n:
            c = s.recv(n - len(b))
            if not c: return None
            b += c
        return b
    h = rd(2)
    if not h: return None
    op = h[0] & 0x0f
    if op == 8: return None
    ln = h[1] & 0x7f
    if ln == 126: ln = struct.unpack(">H", rd(2))[0]
    elif ln == 127: ln = struct.unpack(">Q", rd(8))[0]
    data = rd(ln)
    return data.decode("utf-8", "replace") if op in (1, 2) and data else ("" if op in (1, 2) else None)


def run_on_kernel(ip, ext_port, kernel_id, token, code, timeout=120):
    s = ws_connect(ip, int(ext_port), f"/api/kernels/{kernel_id}/channels", token)
    msg_id = str(uuid.uuid4())
    ws_send(s, json.dumps({"header": {"msg_id": msg_id, "username": "user",
        "session": str(uuid.uuid4()), "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {}, "content": {"code": code, "silent": False,
        "store_history": True, "user_expressions": {}, "allow_stdin": False,
        "stop_on_error": False}, "buffers": [], "channel": "shell"}))
    s.settimeout(30); deadline = time.time() + timeout; out = []
    while time.time() < deadline:
        try:
            m = ws_recv(s)
            if m is None: break
            msg = json.loads(m)
            if msg.get("parent_header", {}).get("msg_id") != msg_id: continue
            mt = msg.get("header", {}).get("msg_type", ""); c = msg.get("content", {})
            if mt == "stream":
                print(c.get("text", ""), end="", flush=True); out.append(c.get("text", ""))
            elif mt in ("display_data", "execute_result"):
                t = c.get("data", {}).get("text/plain", ""); print(t, flush=True); out.append(t)
            elif mt == "error":
                print(f"ERROR: {c.get('ename')}: {c.get('evalue')}")
            elif mt == "execute_reply":
                print(f"\n[Done: {c.get('status')}]"); s.close()
                return "".join(out), c.get("status")
        except socket.timeout:
            print(".", end="", flush=True)
    s.close(); return "".join(out), "timeout"


# ── offer selection (VerInf filter, via HTTP bundles API — no CLI needed) ──────
def _env_val(env_file, key):
    """Read one KEY=value out of the .env (same parser as the vast key)."""
    if os.environ.get(key):
        return os.environ[key]
    if env_file and Path(env_file).exists():
        for line in Path(env_file).read_text().splitlines():
            if line.startswith(key) and "=" in line:
                return line.partition("=")[2].strip()
    return None


def pick_offer(key, min_vram_gb, min_ram_gb, max_dph, min_disk_gb=0, allow_unverified=False, geo=None, min_disk_bw=0, gpu_substr=None, min_inet=0, max_inet_cost=None):
    # NOTE: order + limit must live INSIDE the q JSON — passing them as separate
    # URL params (the old runbook form) makes vast return HTTP 400.
    q = {"gpu_ram": {"gte": min_vram_gb * 1024}, "cpu_ram": {"gte": min_ram_gb * 1024},
         "rentable": {"eq": True}, "num_gpus": {"eq": 1},
         "order": [["dph_total", "asc"]], "limit": 25}
    if not allow_unverified:
        q["verified"] = {"eq": True}
    if max_dph:
        q["dph_total"] = {"lte": max_dph}
    if min_disk_gb:
        q["disk_space"] = {"gte": min_disk_gb}
    if min_disk_bw:
        q["disk_bw"] = {"gte": min_disk_bw}
    if min_inet:
        # Mbit/s down. The S5b chain pulls ~243 GB of GGUF before it can start,
        # so a slow box is billed dead time.
        q["inet_down"] = {"gte": min_inet}
    path = "/bundles?q=" + urllib.parse.quote(json.dumps(q))
    offers = vast_req(path, key).get("offers", [])
    for o in offers:
        o.setdefault("id", o.get("ask_contract_id"))
    # Blackwell (sm_120) GPUs have no kernels in torch 2.6/cu126 -> cudaErrorNoKernelImage.
    EXCLUDE = ("Blackwell", "RTX PRO 6000", "RTX 5090", "RTX 5080", "RTX 5060", "B200", "B300")
    ok = [o for o in offers
          if (o.get("gpu_ram") or 0) >= min_vram_gb * 1024
          and (o.get("cpu_ram") or 0) >= min_ram_gb * 1024
          and (o.get("disk_space") or 0) >= min_disk_gb
          and (o.get("disk_bw") or 0) >= min_disk_bw
          and (o.get("inet_down") or 0) >= min_inet
          and (max_inet_cost is None or (o.get("inet_down_cost") or 0) <= max_inet_cost)
          and not any(x in (o.get("gpu_name") or "") for x in EXCLUDE)
          and (o.get("num_gpus") or 9) == 1
          and (geo is None or geo in (o.get("geolocation") or ""))
          and o.get("rentable") and (max_dph is None or (o.get("dph_total") or 9e9) <= max_dph)
          and (not gpu_substr
               or any(g.strip().lower() in (o.get("gpu_name") or "").lower()
                      for g in gpu_substr.split(",")))]
    ok.sort(key=lambda o: o.get("dph_total", 9e9))
    if not ok:
        raise SystemExit(f"No offer with >={min_vram_gb}GB VRAM, >={min_ram_gb}GB RAM, "
                         f"<=${max_dph}/h. Loosen --max-dph / --min-vram.")
    print("Candidate offers (cheapest first):")
    for o in ok[:8]:
        print(f"  ID:{o['id']} {o.get('gpu_name','?')} VRAM:{o.get('gpu_ram',0)}MB "
              f"RAM:{o.get('cpu_ram',0)}MB disk:{o.get('disk_space',0):.0f}GB "
              f"net:{o.get('inet_down',0):.0f}Mb/s dl:${o.get('inet_down_cost',0):.3f}/GB ${o.get('dph_total',0):.3f}/h {o.get('geolocation','?')}")
    return ok[0]


def estimate_session(matrix, reps, dph, avg_prove_s, gpu_name=""):
    """Print the setup-vs-run economics so you can decide destroy-now vs batch.
    The point: setup is a FIXED cost; a single short experiment pays it in full,
    so batch as much as possible into one rental."""
    n_cfg = _MATRIX_CONFIGS.get(matrix, 1)
    run_min = GATES_MIN + n_cfg * PROVES_PER_CONFIG(reps) * avg_prove_s / 60.0
    total_min = SETUP_MIN + run_min
    setup_frac = SETUP_MIN / total_min
    cost = total_min / 60.0 * dph
    print(f"\n=== session cost estimate ({matrix}, reps={reps}"
          f"{', ' + gpu_name if gpu_name else ''}) ===")
    print(f"  setup dead-time : ~{SETUP_MIN:>4} min  (image+uv sync+cargo+kernel JIT — billed, no results)")
    print(f"  gates + A/Bs    : ~{run_min:>6.0f} min  ({n_cfg} configs x {PROVES_PER_CONFIG(reps)} proves x {avg_prove_s}s)")
    print(f"  TOTAL           : ~{total_min:>6.0f} min  ≈ ${cost:.2f} @ ${dph:.3f}/h")
    print(f"  setup is {100*setup_frac:.0f}% of the bill — "
          + ("BATCH more experiments in this rental to amortize it."
             if setup_frac > 0.25 else "run-time dominates; destroy-after is fine."))
    print(f"  NOTE: the spill needs >=2 GPU types -> plan exactly that many rentals,")
    print(f"        each running the FULL batch (never re-rent for one lever).")
    return total_min, cost


def make_tarball() -> bytes:
    """One sanitized tarball: source only, never local credentials/models."""
    buf = io.BytesIO()
    def flt(ti: tarfile.TarInfo):
        path = Path(ti.name)
        parts = set(path.parts)
        basename = path.name.lower()
        sensitive = (
            basename.startswith(".env")
            or basename in {"credentials", ".netrc"}
            or path.suffix.lower() in TAR_SECRET_SUFFIXES
            or path.suffix.lower() in {".gguf", ".safetensors"}
        )
        return None if parts & TAR_EXCLUDE or sensitive else ti
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(VERINF, arcname="VerInf", filter=flt)
    data = buf.getvalue()
    print(f"repo tarball: {len(data)/1e6:.1f} MB")
    return data


# ── bootstrap + run scripts (remote) ──────────────────────────────────────────
BOOTSTRAP = r"""
set -e
cd /workspace
# $SECONDS auto-increments; L logs elapsed seconds per phase so we SEE the cost.
L(){ echo "[boot +${SECONDS}s] $1"; }
L "unpack tarball"; tar xzf verinf.tar.gz
L "gpu:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
L "apt (git/curl/build)"; export DEBIAN_FRONTEND=noninteractive
apt-get update -q && apt-get install -y -q git curl build-essential >/dev/null 2>&1 || true
L "install uv"; curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || pip install -q uv
export PATH="$HOME/.local/bin:$PATH"
L "install rust"; curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null 2>&1 || true
source "$HOME/.cargo/env" 2>/dev/null || true
cd /workspace/VerInf
L "uv sync (torch cu126) — START"; uv sync 2>&1 | tail -4; L "uv sync — DONE"
L "cargo build verifier — START"; ( cd verifier && cargo build --release 2>&1 | tail -4 ) || echo "WARN: cargo build failed — ACCEPT gate may not run"; L "cargo build — DONE"
L "BOOTSTRAP TOTAL ${SECONDS}s"; echo "$SECONDS" > /workspace/BOOTSTRAP_DONE
"""

RUN_TMPL = r"""
set -e
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
export VERINF_PYRUN="uv run --project /workspace/VerInf python3"
nohup uv run --project /workspace/VerInf python3 analysis/bench/remote_suite.py \
    --matrix {matrix} --reps {reps} > /workspace/suite.log 2>&1 &
echo "suite PID $!"; echo "running"
"""

# rho=2 optimization-validation run (prove+Rust-verify, baseline vs optimized)
RUN_TMPL_OPT = r"""
set -e
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
nohup bash analysis/bench/optrun_remote.sh > /workspace/suite.log 2>&1 &
echo "optrun PID $!"; echo "running"
"""

# Full Maverick 400B run: download the UD-Q4_K_XL GGUF, then demo_maverick_full
# at optimized rho=2/T=17 + disk-spill. MAV_LAYERS controls scale (2 = de-risk,
# 48 = full 400B).
RUN_TMPL_MAV = r"""
set -e
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
export MAV_LAYERS={layers}
nohup bash analysis/bench/optrun_mav.sh > /workspace/suite.log 2>&1 &
echo "mavrun PID $!"; echo "running"
"""

# S5 admission probe: gates + production-geometry kernel rates, no model download.
RUN_TMPL_ADMISSION = r"""
set -e
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
nohup bash analysis/bench/admission_remote.sh > /workspace/suite.log 2>&1 &
echo "admission PID $!"; echo "running"
"""

# S5b/S6: the real 400B run — fetch the GGUF on the box, enroll, measure the
# model stages, prove, verify. One rental for the whole chain.
RUN_TMPL_S5B = r"""
set -e
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
export HF_TOKEN="{hf_token}"
export MODEL_DIR=/workspace/gguf
export MIN_MBPS="{min_mbps}"
# hf_transfer/huggingface_hub are download-only tooling, not prover deps, so
# they are installed here rather than pinned in the project. gguf IS a prover
# dep and is in pyproject/uv.lock now; the fallback stays as a belt so a lock
# mismatch cannot cost another rental.
uv pip install --project /workspace/VerInf 'huggingface_hub==0.34.4' \
  'hf_transfer==0.1.9' gguf >/dev/null 2>&1 || \
  pip install 'huggingface_hub==0.34.4' 'hf_transfer==0.1.9' gguf >/dev/null 2>&1 || true
uv run --project /workspace/VerInf python3 -c "import gguf, huggingface_hub" \
  || {{ echo "FATAL: gguf/huggingface_hub missing after install"; exit 1; }}
nohup bash analysis/bench/s5b_remote.sh > /workspace/suite.log 2>&1 &
echo "s5b PID $!"; echo "running"
"""

# Real 2,596-claim sampled audit: download GGUF, enroll once, then time the
# one-pass audit. The remote script publishes campaign_results.json last.
RUN_TMPL_SAMPLED = r"""
set -e
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
export MODEL_DIR=/workspace/gguf
export MIN_MBPS="{min_mbps}"
uv run --project /workspace/VerInf python3 -c "import gguf" \
  || {{ echo "FATAL: project gguf dependency missing"; exit 1; }}
nohup bash analysis/bench/sampled_audit_full_remote.sh > /workspace/suite.log 2>&1 &
echo "sampled-audit PID $!"; echo "running"
"""

# spill A/B on a fast-NVMe box (fadvise forces disk reads; toy model, no download)
RUN_TMPL_SPILLAB = r"""
set -e
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
export AB_D={ab_d} AB_SEQ={ab_seq} AB_NL={ab_nl}
nohup bash analysis/bench/spillab_remote.sh > /workspace/suite.log 2>&1 &
echo "spillab PID $!"; echo "running"
"""

# WC-LCRL-STC FINAL: the full bridged 400B proof (wc_final_remote.sh).
RUN_TMPL_WCFINAL = r"""
set -e
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
export HF_TOKEN={hf_token}
nohup bash analysis/bench/wc_final_remote.sh > /workspace/suite.log 2>&1 &
echo "wcfinal PID $!"; echo "running"
"""

# WC-LCRL-STC over the REAL full Maverick GGUF (download on the box).
RUN_TMPL_WCMAV = r"""
set -e
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
export HF_TOKEN={hf_token}
nohup bash analysis/bench/wc_mav_remote.sh > /workspace/suite.log 2>&1 &
echo "wcmav PID $!"; echo "running"
"""

# WC-LCRL-STC bridge validation: wc test suite + production-geometry bench.
RUN_TMPL_WC = r"""
set -e
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/.cargo/env" 2>/dev/null || true
cd /workspace/VerInf
export VERINF_ROOT=/workspace/VerInf
nohup bash analysis/bench/wc_remote.sh > /workspace/suite.log 2>&1 &
echo "wc PID $!"; echo "running"
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-file", default=str(Path(__file__).parent / ".env"))
    ap.add_argument("--min-vram", type=int, default=40, help="min GPU VRAM (GB)")
    ap.add_argument("--min-ram", type=int, default=64, help="min host RAM (GB, for spill)")
    ap.add_argument("--max-dph", type=float, default=1.20, help="max $/hour")
    ap.add_argument("--matrix", default="medium", choices=["smoke", "phone", "medium"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-wait", type=int, default=4 * 3600, help="max s to wait for the suite")
    ap.add_argument("--avg-prove-s", type=float, default=120.0,
                    help="avg seconds per prove for the cost estimate (V100 Qwen-0.5B ~90s; "
                         "a faster rented card is lower)")
    ap.add_argument("--keep-alive", action="store_true",
                    help="do NOT destroy after the suite — leave the instance up to run a "
                         "few more short experiments. YOU must destroy it (prints the command).")
    ap.add_argument("--dry-run", action="store_true", help="pick offer + print cost estimate, don't rent")
    ap.add_argument("--min-disk", type=int, default=0, help="min HOST disk (GB) to filter offers — e.g. 8000")
    ap.add_argument("--disk-gb", type=int, default=DISK_GB, help="CONTAINER disk to REQUEST (GB); default 60. gguf needs ~400, full spill ~8000")
    ap.add_argument("--allow-unverified", action="store_true", help="allow community (unverified) hosts — needed for rare big-disk boxes")
    ap.add_argument("--gpu-substr", default=None,
                    help="comma-separated GPU name substrings to accept, e.g. "
                         "'4090,L40S,A6000' — the probe needs a card that could "
                         "plausibly BE the production card, not the cheapest one")
    ap.add_argument("--geo", default=None, help="require this substring in geolocation, e.g. US")
    ap.add_argument("--min-inet", type=int, default=0,
                    help="min advertised download Mbit/s (S5b pulls ~243GB)")
    ap.add_argument("--min-disk-bw", type=int, default=0, help="min disk bandwidth MB/s (e.g. 4000 for NVMe)")
    ap.add_argument("--spillab", action="store_true", help="run the spill A/B (spillab_remote.sh)")
    ap.add_argument("--mavrun", action="store_true",
                    help="download Maverick GGUF + run demo_maverick_full (optrun_mav.sh); MAV_LAYERS env sets scale")
    ap.add_argument("--s5b", action="store_true",
                    help="the real 400B chain on the box: download GGUF, smoke, "
                         "witness-only (Sz + model stages), enroll, admission "
                         "report, prove, verify. Needs --disk-gb >= 500.")
    ap.add_argument("--sampled", action="store_true",
                    help="real Maverick 2,596-claim sampled audit: download, "
                         "enroll, then one timed audit pass")
    ap.add_argument("--admission", action="store_true",
                    help="S5 probe: run the gates + production-geometry kernel "
                         "rates for the admission report (no model download)")
    ap.add_argument("--wc", action="store_true",
                    help="run the WC-LCRL-STC bridge validation (wc_remote.sh)")
    ap.add_argument("--wcfinal", action="store_true",
                    help="WC-LCRL-STC FINAL: full bridged Maverick proof "
                         "(needs --disk-gb >= 400, ~48GB VRAM)")
    ap.add_argument("--wcmav", action="store_true",
                    help="WC-LCRL-STC over the REAL full Maverick GGUF "
                         "(downloads ~243 GB on the box; needs --disk-gb >= 400)")
    ap.add_argument("--optrun", action="store_true",
                    help="run the rho=2 optimization-validation (optrun_remote.sh) instead of the campaign")
    ap.add_argument("--max-inet-cost", type=float, default=None,
                    help="max $/GB the host may charge for internet DOWNLOAD. "
                         "Payloads that pull the 243 GB GGUF (--s5b, --wcmav) "
                         "default this to 0.002 - the Aug 14 lesson: two paid "
                         "downloads on $0.026/GB hosts cost $12.55 while the "
                         "GPU hours cost $0.22.")
    ap.add_argument("--max-real-dph", type=float, default=None,
                    help="hard cap on the ACTUAL billed rate (compute+storage) of the live "
                         "instance. The offer price understates this by 1.6-2.6x, so this is "
                         "the cap that matters. Default: 2.5x --max-dph. Over cap -> destroy "
                         "immediately, before any paid work.")
    args = ap.parse_args()
    KEY = load_key(args.env_file)
    print("VAST credentials: loaded")
    _install_term_handlers()
    max_real_dph = args.max_real_dph if args.max_real_dph is not None else 2.5 * args.max_dph
    max_inet_cost = args.max_inet_cost
    if max_inet_cost is None and (args.s5b or args.sampled or args.wcmav or args.wcfinal):
        max_inet_cost = 0.002          # big-download payloads: free-ingress hosts only

    # phase timer — prints elapsed wall-seconds at each stage so we SEE where time
    # goes (setup dead-time vs the actual suite). Timestamps also go to onstart.log
    # (image/boot) and bootstrap output ($SECONDS per step) and suite.log.
    T0 = time.time()
    def lg(m): print(f"[+{int(time.time()-T0):5d}s] {m}", flush=True)

    lg("picking offer")
    chosen = pick_offer(KEY, args.min_vram, args.min_ram, args.max_dph, args.min_disk, args.allow_unverified, args.geo, args.min_disk_bw, args.gpu_substr, args.min_inet, max_inet_cost)
    print(f"\nSelected: {chosen['id']} {chosen['gpu_name']} ${chosen['dph_total']:.3f}/h")
    estimate_session(args.matrix, args.reps, chosen["dph_total"], args.avg_prove_s,
                     chosen.get("gpu_name", ""))
    if args.dry_run:
        print("\n--dry-run: stopping before renting."); return

    tarball = make_tarball()   # build BEFORE renting so a pack error costs nothing

    lg("creating instance")
    onstart = f"#!/bin/bash\nexec > >(tee /root/onstart.log) 2>&1\necho onstart $(date -u)\n"
    payload = {"client_id": "me", "image": IMAGE, "disk": args.disk_gb,
               "onstart": onstart, "runtype": "jupyter_direc ssh_direc"}
    res = vast_req(f"/asks/{chosen['id']}/", KEY, "PUT", json.dumps(payload).encode())
    INSTANCE = res["new_contract"]
    print(f"Instance: {INSTANCE}")

    t_boot0 = None
    try:
        info = _wait_running(INSTANCE, KEY); lg("instance running (booted)")

        # PRICE GATE — the offer price is not the bill. Check what we are ACTUALLY
        # being charged before spending 18 min of setup dead-time on this box.
        rate, comp, stor = real_dph(info)
        print(f"  REAL rate: compute ${comp:.2f}/h + storage ${stor:.2f}/h = ${rate:.2f}/h "
              f"(offer said ${chosen['dph_total']:.2f}/h) · num_gpus={info.get('num_gpus')} "
              f"· disk={info.get('disk_space')}GB", flush=True)
        if rate > max_real_dph:
            raise RuntimeError(f"PRICE GATE: real ${rate:.2f}/h > cap ${max_real_dph:.2f}/h "
                               f"— destroying before any paid work")
        if (info.get("num_gpus") or 1) != 1:
            raise RuntimeError(f"PRICE GATE: got num_gpus={info.get('num_gpus')} (paying for all "
                               f"of them, using one) — destroying")

        IP = info["public_ipaddr"]; PORT = info["ports"]["8080/tcp"][0]["HostPort"]
        TOKEN = info["jupyter_token"]; BASE = f"https://{IP}:{PORT}"
        _wait_jupyter(BASE, TOKEN); lg("jupyter ready")
        KID = _bash_kernel(BASE, TOKEN)

        lg("uploading repo tarball")
        assert upload_file(BASE, TOKEN, "verinf.tar.gz", tarball), "tarball upload failed"

        lg("BOOTSTRAP start (uv sync + cargo build)"); t_boot0 = time.time()
        run_on_kernel(IP, PORT, KID, TOKEN, BOOTSTRAP, timeout=2400)
        boot_dur = int(time.time() - t_boot0)
        lg(f"BOOTSTRAP done — {boot_dur}s of setup dead-time")

        lg("launching validation suite"); t_suite0 = time.time()
        # PF_* = what the offer ADVERTISED; the on-box pre-flight gate measures the real
        # hardware and aborts (no paid workload) if reality doesn't match. Never run on a
        # machine whose specs don't match what we rented.
        pf = {
            "PF_GPU": (chosen.get("gpu_name") or "").split()[0] if chosen.get("gpu_name") else "",
            "PF_NUM_GPUS": chosen.get("num_gpus") or "",
            "PF_VRAM_GB": round((chosen.get("gpu_ram") or 0) / 1024, 1) or "",
            "PF_RAM_GB": round((chosen.get("cpu_ram") or 0) / 1024, 1) or "",
            "PF_CPUS": chosen.get("cpu_cores") or chosen.get("cpu_cores_effective") or "",
            "PF_DISK_BW_GBPS": round((chosen.get("disk_bw") or 0) / 1000, 2) or "",
        }
        PF_EXPORTS = "".join(f'export {k}="{v}"\n' for k, v in pf.items() if v != "")
        launch = (RUN_TMPL_SAMPLED.format(
                      min_mbps=os.environ.get("MIN_MBPS", "120")) if args.sampled
                  else RUN_TMPL_S5B.format(
                      hf_token=_env_val(args.env_file, "HF_TOKEN") or "",
                      min_mbps=os.environ.get("MIN_MBPS", "40")) if args.s5b
                  else RUN_TMPL_ADMISSION if args.admission
                  else RUN_TMPL_SPILLAB.format(ab_d=os.environ.get("AB_D","1024"), ab_seq=os.environ.get("AB_SEQ","2048"), ab_nl=os.environ.get("AB_NL","4")) if args.spillab
                  else RUN_TMPL_MAV.format(layers=os.environ.get("MAV_LAYERS", "2")) if args.mavrun
                  else RUN_TMPL_WCFINAL.format(hf_token=_env_val(args.env_file, "HF_TOKEN") or "") if args.wcfinal
                  else RUN_TMPL_WCMAV.format(hf_token=_env_val(args.env_file, "HF_TOKEN") or "") if args.wcmav
                  else RUN_TMPL_WC if args.wc
                  else RUN_TMPL_OPT if args.optrun
                  else RUN_TMPL.format(matrix=args.matrix, reps=args.reps))
        launch = launch.replace("cd /workspace/VerInf",
                                PF_EXPORTS + "cd /workspace/VerInf", 1)
        run_on_kernel(IP, PORT, KID, TOKEN, launch, timeout=120)

        _poll(BASE, TOKEN, args.max_wait); lg("suite finished")
        _download(BASE, TOKEN); lg("results downloaded")
        print(f"\n=== TIMING: total {int(time.time()-T0)}s · setup(pick→boot→upload+bootstrap) "
              f"{int(t_suite0-T0)}s · bootstrap alone {boot_dur}s · "
              f"suite {int(time.time()-t_suite0)}s (per-step in onstart.log/suite.log) ===")
    finally:
        if args.keep_alive:
            print("\n" + "=" * 64)
            print(f"  --keep-alive: instance {INSTANCE} LEFT RUNNING (still billing"
                  f" ${chosen['dph_total']:.3f}/h).")
            print( "  Batch more experiments now, THEN destroy it yourself:")
            print(f"    vastai destroy instance {INSTANCE}")
            print( "  (setup is already paid — this is the cheap window to add short runs.)")
            print("=" * 64)
        else:
            # Runs on success, on error, AND on SIGTERM/SIGINT/SIGHUP (see
            # _install_term_handlers) — a parent that kills us can no longer
            # leave a rented box behind.
            print("\nDestroying instance (always, even on error or signal)")
            if not _destroy(INSTANCE, KEY):
                print(f"!! MANUALLY destroy instance {INSTANCE} to stop billing!")


def _wait_running(instance, key, timeout=600):
    t0 = time.time()
    while True:
        info = vast_req(f"/instances/{instance}/", key)["instances"]
        st = info.get("actual_status", "unknown")
        print(f"  status: {st} ({int(time.time()-t0)}s)", flush=True)
        if st == "running": return info
        if st in ("error", "exited", "failed"): raise RuntimeError(f"bad state: {st}")
        if time.time() - t0 > timeout:
            raise TimeoutError(f"instance not 'running' after {timeout}s (stuck at '{st}') "
                               "— image likely lacks python/jupyter; destroying")
        time.sleep(15)


def _wait_jupyter(base, token):
    while True:
        try:
            req = urllib.request.Request(f"{base}/api/kernelspecs",
                headers={"Authorization": f"token {token}"})
            with urllib.request.urlopen(req, timeout=10, context=CTX) as r:
                if r.status == 200:
                    print("Jupyter ready"); return
        except Exception as e:
            print(f"  waiting jupyter: {e}", flush=True)
        time.sleep(15)


def _bash_kernel(base, token):
    req = urllib.request.Request(f"{base}/api/kernels",
        data=json.dumps({"name": "bash"}).encode(), method="POST",
        headers={"Authorization": f"token {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        return json.load(r)["id"]


def _poll(base, token, max_wait):
    print("\nPolling for completion")
    deadline = time.time() + max_wait
    while time.time() < deadline:
        # remote_suite writes remote_results/<host>/campaign_results.json at the end
        hosts = list_remote_dir(base, token, "VerInf/analysis/bench/remote_results")
        for h in hosts:
            if h.get("type") == "directory":
                done = read_remote_file(base, token, f"{h['path']}/campaign_results.json")
                if done:
                    print("\ncampaign_results.json found — suite done."); return
        log = read_remote_file(base, token, "suite.log")
        if log:
            print(f"  [{int((time.time()-(deadline-max_wait))/60)}m] "
                  f"...{log[-240:].decode('utf-8','replace').strip()}")
        time.sleep(120)
    print("WARNING: suite timed out; downloading whatever exists.")


def _download(base, token):
    print("\nDownloading remote_results")
    hosts = list_remote_dir(base, token, "VerInf/analysis/bench/remote_results")
    for h in hosts:
        if h.get("type") != "directory": continue
        dest = LOCAL_RESULTS / Path(h["path"]).name
        dest.mkdir(parents=True, exist_ok=True)
        for item in list_remote_dir(base, token, h["path"]):
            if item.get("type") == "file":
                c = read_remote_file(base, token, item["path"])
                if c:
                    (dest / Path(item["path"]).name).write_bytes(c)
                    print(f"  {Path(item['path']).name} ({len(c)}B)")
    print(f"saved under {LOCAL_RESULTS}")


if __name__ == "__main__":
    main()
