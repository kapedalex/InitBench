"""
VAST.ai runner for model family pair experiments.
Mirrors vast_run.py: spins up instance → uploads scripts → runs original then abliterated
→ downloads logs to ../logs/<family>/ → destroys instance.

Usage:
    python3 vast_run_family.py gemma2_9b
    python3 vast_run_family.py deepseek_r1_14b
    python3 vast_run_family.py llama3_8b
    python3 vast_run_family.py qwen2_5_72b
    python3 vast_run_family.py mistral_7b
"""
import base64
import json
import socket
import ssl
import struct
import sys
import time
import urllib.request
import uuid
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────

LOCAL   = Path(__file__).parent          # model_family_experiments/
PROJECT = LOCAL.parent                   # InitBench/

env = {}
for line in (PROJECT / ".env").read_text().splitlines():
    if "=" in line:
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()

VAST_KEY = env["VAST_AI_KEY"]
HF_TOKEN = env.get("HF_TOKEN", "")
print(f"VAST_KEY: {VAST_KEY[:16]}...")
print(f"HF_TOKEN: {HF_TOKEN[:8]}..." if HF_TOKEN else "HF_TOKEN: not set!")

# ── per-family config ──────────────────────────────────────────────────────────
# min_vram_mb : minimum GPU VRAM to look for on VAST.ai
# disk_gb     : instance disk size
# use_4bit    : pass USE_4BIT=1 env var to the scripts (needed for 70B+ models)
# apply_moe_patch : patch transformers moe.py for multi-GPU expert dispatch
# original_cache  : HuggingFace cache dir name for original model (to delete after run)
# gguf_file   : for GGUF-only models; passed as GGUF_FILE env var

FAMILY_CONFIG = {
    "gemma2_9b": {
        "min_vram_mb":    24576,
        "disk_gb":        60,
        "use_4bit":       False,
        "apply_moe_patch": False,
        "original_script":   "run_gemma2_9b_original.py",
        "abliterated_script": "run_gemma2_9b_abliterated.py",
        "original_cache":    "models--google--gemma-2-9b-it",
        "abliterated_cache": "models--IlyaGusev--gemma-2-9b-it-abliterated",
        "gguf_file": "",
    },
    "deepseek_r1_14b": {
        "min_vram_mb":    40960,
        "disk_gb":        80,
        "use_4bit":       False,
        "apply_moe_patch": False,
        "original_script":   "run_deepseek_r1_14b_original.py",
        "abliterated_script": "run_deepseek_r1_14b_abliterated.py",
        "original_cache":    "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B",
        "abliterated_cache": "models--huihui-ai--DeepSeek-R1-Distill-Qwen-14B-abliterated-v2",
        "gguf_file": "",
    },
    "llama3_8b": {
        "min_vram_mb":    16384,
        "disk_gb":        40,
        "use_4bit":       False,
        "apply_moe_patch": False,
        "original_script":   "run_llama3_8b_original.py",
        "abliterated_script": "run_llama3_8b_abliterated.py",
        "original_cache":    "models--meta-llama--Meta-Llama-3-8B-Instruct",
        "abliterated_cache": "models--Orenguteng--Llama-3-8B-Lexi-Uncensored",
        "gguf_file": "",
    },
    "qwen2_5_72b": {
        "min_vram_mb":    40960,
        "disk_gb":        200,
        "use_4bit":       True,
        "apply_moe_patch": False,
        "original_script":   "run_qwen2_5_72b_original.py",
        "abliterated_script": "run_qwen2_5_72b_abliterated.py",
        "original_cache":    "models--Qwen--Qwen2.5-72B-Instruct",
        "abliterated_cache": "models--huihui-ai--Qwen2.5-72B-Instruct-abliterated",
        "gguf_file": "",
    },
    "mistral_7b": {
        "min_vram_mb":    16384,
        "disk_gb":        40,
        "use_4bit":       False,
        "apply_moe_patch": False,
        "original_script":   "run_mistral_7b_original.py",
        "abliterated_script": "run_mistral_7b_abliterated.py",
        "original_cache":    "models--mistralai--Mistral-7B-Instruct-v0.3",
        "abliterated_cache": "models--mradermacher--Mistral-7B-Instruct-v0.3-abliterated-GGUF",
        "gguf_file": "Mistral-7B-Instruct-v0.3-abliterated.Q4_K_M.gguf",
    },
}

# ── SSL / Jupyter helpers (verbatim from vast_run.py) ─────────────────────────

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def vast_req(path, method="GET", data=None):
    req = urllib.request.Request(
        f"https://console.vast.ai/api/v0{path}",
        data=data, method=method,
        headers={"Authorization": f"Bearer {VAST_KEY}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def get_instance(instance_id):
    return vast_req(f"/instances/{instance_id}/")["instances"]


def http_req(url, token, method="GET", data=None, timeout=30):
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"token {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.status, r.read()


def upload_file(base_url, token, remote_path, content_bytes):
    payload = json.dumps({
        "name":    Path(remote_path).name,
        "path":    remote_path,
        "type":    "file",
        "format":  "base64",
        "content": base64.b64encode(content_bytes).decode(),
    }).encode()
    try:
        status, _ = http_req(f"{base_url}/api/contents/{remote_path}", token,
                             method="PUT", data=payload)
        return status in (200, 201)
    except Exception as e:
        print(f"  upload_file {remote_path}: {e}")
        return False


def create_dir(base_url, token, path):
    payload = json.dumps({"type": "directory", "path": path}).encode()
    try:
        status, _ = http_req(f"{base_url}/api/contents/{path}", token,
                             method="PUT", data=payload)
        return status in (200, 201)
    except Exception as e:
        print(f"  create_dir {path}: {e}")
        return False


def read_remote_file(base_url, token, remote_path):
    try:
        req = urllib.request.Request(
            f"{base_url}/api/contents/{remote_path}?format=base64",
            headers={"Authorization": f"token {token}"},
        )
        with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
            d = json.load(r)
            return base64.b64decode(d["content"])
    except Exception:
        return None


def list_remote_dir(base_url, token, path):
    try:
        req = urllib.request.Request(
            f"{base_url}/api/contents/{path}",
            headers={"Authorization": f"token {token}"},
        )
        with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
            return json.load(r).get("content", [])
    except Exception:
        return []


def ws_connect(ip, port, path, token):
    key = base64.b64encode(uuid.uuid4().bytes).decode()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    s = socket.create_connection((ip, int(port)), timeout=30)
    s = ctx.wrap_socket(s, server_hostname=ip)
    s.send((
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {ip}:{port}\r\n"
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
        f"Authorization: token {token}\r\n\r\n"
    ).encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += s.recv(4096)
    assert b"101" in resp, f"Handshake failed: {resp[:200]}"
    return s


def ws_send(s, msg):
    data = msg.encode()
    length = len(data)
    mask_key = uuid.uuid4().bytes[:4]
    masked = bytes(data[i] ^ mask_key[i % 4] for i in range(length))
    frame = bytearray([0x81])
    if length < 126:
        frame.append(0x80 | length)
    elif length < 65536:
        frame += bytearray([0x80 | 126]) + struct.pack(">H", length)
    else:
        frame += bytearray([0x80 | 127]) + struct.pack(">Q", length)
    frame += mask_key + masked
    s.send(bytes(frame))


def ws_recv(s):
    def read_exact(n):
        buf = b""
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf
    header = read_exact(2)
    if not header:
        return None
    opcode = header[0] & 0x0f
    if opcode == 8:
        return None
    length = header[1] & 0x7f
    if length == 126:
        length = struct.unpack(">H", read_exact(2))[0]
    elif length == 127:
        length = struct.unpack(">Q", read_exact(8))[0]
    data = read_exact(length)
    if opcode in (1, 2):
        return data.decode("utf-8", errors="replace") if data else ""
    return None


def run_on_kernel(ip, ext_port, kernel_id, token, code, timeout=14400):
    path = f"/api/kernels/{kernel_id}/channels"
    s = ws_connect(ip, int(ext_port), path, token)
    msg_id = str(uuid.uuid4())
    ws_send(s, json.dumps({
        "header":        {"msg_id": msg_id, "username": "user",
                          "session": str(uuid.uuid4()),
                          "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content":       {"code": code, "silent": False, "store_history": True,
                          "user_expressions": {}, "allow_stdin": False, "stop_on_error": False},
        "buffers": [], "channel": "shell",
    }))
    s.settimeout(60)
    deadline = time.time() + timeout
    output = []
    while time.time() < deadline:
        try:
            msg_str = ws_recv(s)
            if msg_str is None:
                break
            msg = json.loads(msg_str)
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mtype   = msg.get("header", {}).get("msg_type", "")
            content = msg.get("content", {})
            if mtype == "stream":
                text = content.get("text", "")
                print(text, end="", flush=True)
                output.append(text)
            elif mtype in ("display_data", "execute_result"):
                text = content.get("data", {}).get("text/plain", "")
                print(text, flush=True)
                output.append(text)
            elif mtype == "error":
                print(f"ERROR: {content.get('ename')}: {content.get('evalue')}")
                for tb in content.get("traceback", []):
                    print(tb)
            elif mtype == "execute_reply":
                print(f"\n[Done: {content.get('status')}]")
                s.close()
                return "".join(output), content.get("status")
        except socket.timeout:
            print(".", end="", flush=True)
    s.close()
    return "".join(output), "timeout"


# ── launch code builders ───────────────────────────────────────────────────────

MOE_PATCH = r"""
python3 - << 'PYEOF'
import transformers.integrations.moe as m
src = open(m.__file__).read()
patches = [
    (
        'torch.mm(input[start:end], weight[i], out=output[start:end])',
        'torch.mm(input[start:end], weight[i].to(input.device), out=output[start:end])',
        'weight[i].to(input.device)',
    ),
    (
        'selected_weights = self.gate_up_proj[expert_ids]',
        'selected_weights = self.gate_up_proj[expert_ids.to(self.gate_up_proj.device)]',
        'gate_up_proj expert_ids.to(device)',
    ),
    (
        'selected_weights = self.gate_down_proj[expert_ids]',
        'selected_weights = self.gate_down_proj[expert_ids.to(self.gate_down_proj.device)]',
        'gate_down_proj expert_ids.to(device)',
    ),
]
changed = False
for old, new, label in patches:
    if old in src:
        src = src.replace(old, new)
        changed = True
        print(f'[moe patch] {label}')
    elif new in src:
        print(f'[moe patch] already applied: {label}')
    else:
        print(f'[moe patch] pattern not found (ok): {label}')
if changed:
    open(m.__file__, 'w').write(src)
PYEOF
"""


def build_setup_code(cfg, family):
    """Runs synchronously: writes env, applies patches, mkdir."""
    moe_patch = MOE_PATCH if cfg["apply_moe_patch"] else ""
    return f"""
cd /workspace/inspect_project
python3 write_env.py
mkdir -p logs/{family}
python3 -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
{moe_patch}
echo "SETUP OK"
"""


def build_launch_code(cfg, family):
    """Launches experiments in background via nohup (returns immediately)."""
    use_4bit_export = "export USE_4BIT=1" if cfg["use_4bit"] else ""
    gguf_export = f'export GGUF_FILE="{cfg["gguf_file"]}"' if cfg["gguf_file"] else ""
    orig_script  = cfg["original_script"]
    ablit_script = cfg["abliterated_script"]
    orig_cache   = cfg["original_cache"]

    script_body = (
        "#!/bin/bash\n"
        "exec > >(tee /workspace/inspect_project/experiment.log) 2>&1\n"
        "export HF_HOME=/workspace/.huggingface\n"
        "export HF_TOKEN=$(python3 -c \"\n"
        "d={}\n"
        "[d.update({k.strip(): v.strip()}) for line in open('/workspace/inspect_project/.env') if '=' in line for k,_,v in [line.partition('=')]]\n"
        "print(d.get('HF_TOKEN',''))\n"
        "\")\n"
        f"{use_4bit_export}\n"
        f"{gguf_export}\n"
        f"echo '=== {orig_script} started: '$(date -u)\n"
        f"cd /workspace/inspect_project && python3 {orig_script} 2>&1 | tee logs/{family}/original_stdout.log\n"
        f"echo '=== {orig_script} done: '$(date -u)\n"
        f"echo '=== Removing original model cache ==='\n"
        f"rm -rf /workspace/.huggingface/hub/{orig_cache}\n"
        f"echo '=== {ablit_script} started: '$(date -u)\n"
        f"cd /workspace/inspect_project && python3 {ablit_script} 2>&1 | tee logs/{family}/abliterated_stdout.log\n"
        f"echo '=== {ablit_script} done: '$(date -u)\n"
        "echo 'DONE' > /workspace/inspect_project/DONE\n"
        "echo '=== ALL DONE: '$(date -u)\n"
    )

    return (
        f"cat > /workspace/run_experiments.sh << 'HEREDOC'\n"
        f"{script_body}"
        "HEREDOC\n"
        "chmod +x /workspace/run_experiments.sh\n"
        "nohup /workspace/run_experiments.sh &\n"
        "echo \"Launched PID: $!\"\n"
    )


def _destroy(instance_id):
    req = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{instance_id}/",
        method="DELETE",
        headers={"Authorization": f"Bearer {VAST_KEY}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print("Destroy:", json.load(r))
    except Exception as e:
        print(f"Destroy failed: {e}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 vast_run_family.py <family>")
        print("Families:", ", ".join(FAMILY_CONFIG))
        sys.exit(1)

    family = sys.argv[1]
    if family not in FAMILY_CONFIG:
        print(f"Unknown family: {family!r}. Available: {', '.join(FAMILY_CONFIG)}")
        sys.exit(1)

    cfg = FAMILY_CONFIG[family]

    if cfg["apply_moe_patch"] and not cfg["gguf_file"] and "mistral_small" in family:
        pass  # gguf_file check for abliterated only — warn but don't block
    if cfg["gguf_file"] == "" and "mistral_small" in family:
        print("WARNING: mistral_small abliterated needs GGUF_FILE set in FAMILY_CONFIG.")

    local_logs = PROJECT / "logs" / family
    local_logs.mkdir(parents=True, exist_ok=True)

    # ── find offer ────────────────────────────────────────────────────────────
    print(f"\nLooking for GPU offers (≥{cfg['min_vram_mb']}MB VRAM, fast net)...")
    import subprocess as _sp
    VASTAI = "/home/riftuser/gaia_pipeline/.venv/bin/vastai"
    _cli = _sp.run(
        [VASTAI, "search", "offers", "--raw"],
        capture_output=True, text=True, timeout=30,
    )
    if _cli.returncode != 0:
        raise RuntimeError(f"vastai search error: {_cli.stderr}")
    all_offers = json.loads(_cli.stdout)
    offers = [
        o for o in all_offers
        if (o.get("gpu_ram") or 0) >= cfg["min_vram_mb"] and o.get("rentable")
    ]
    offers.sort(key=lambda o: o.get("dph_total", 9999))
    for o in offers:
        o.setdefault("id", o.get("ask_contract_id"))
    offers = [
        o for o in all_offers
        if (o.get("gpu_ram") or 0) >= cfg["min_vram_mb"] and o.get("rentable")
    ]
    offers.sort(key=lambda o: o.get("dph_total", 9999))
    for o in offers:
        o.setdefault("id", o.get("ask_contract_id"))

    fast = [o for o in offers if o.get("inet_down", 0) > 5000]
    if not fast:
        print("No fast offers; falling back to all offers")
        fast = offers
    if not fast:
        sys.exit("ERROR: no suitable offers found")

    print("Fast offers (>5Gbps):")
    for o in fast[:6]:
        print(f"  ID:{o['id']} GPU:{o['gpu_name']} VRAM:{o['gpu_ram']}MB "
              f"net:{o['inet_down']:.0f}Mbps ${o['dph_total']:.3f}/h loc:{o['geolocation']}")

    ONSTART = (
        "#!/bin/bash\n"
        "exec > >(tee /root/onstart.log) 2>&1\n"
        "echo \"Setup started: $(date -u)\"\n"
        "pip install -q python-dotenv sentencepiece protobuf accelerate transformers huggingface_hub bitsandbytes gguf\n"
        "echo \"SETUP_DONE\" > /root/SETUP_DONE\n"
        "echo \"Setup done: $(date -u)\"\n"
    )
    CUDA_TEST = (
        "python3 << 'PYEOF'\n"
        "import torch, sys\n"
        "try:\n"
        "    t = torch.zeros(10).cuda()\n"
        "    t.fill_(1.0)\n"
        "    assert t.sum().item() == 10.0\n"
        "    print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())\n"
        "    print('CUDA_KERNEL_TEST_PASSED')\n"
        "except Exception as e:\n"
        "    print('CUDA_KERNEL_TEST_FAILED:', e)\n"
        "    sys.exit(1)\n"
        "PYEOF\n"
    )

    # ── find compatible instance (loop over offers until CUDA test passes) ────
    INSTANCE_ID = BASE_URL = TOKEN = IP = EXT_PORT = KERNEL_ID = None
    for offer_idx, chosen in enumerate(fast):
        OFFER_ID = chosen["id"]
        print(f"\nTrying offer {offer_idx+1}/{len(fast)}: {OFFER_ID} "
              f"GPU:{chosen['gpu_name']} ${chosen['dph_total']:.3f}/h")

        payload = {
            "client_id": "me",
            "image":     "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
            "disk":      cfg["disk_gb"],
            "onstart":   ONSTART,
            "runtype":   "jupyter_direc ssh_direc",
        }
        result = vast_req(f"/asks/{OFFER_ID}/", method="PUT",
                          data=json.dumps(payload).encode())
        _iid = result["new_contract"]
        print(f"Instance created: {_iid}")

        # wait for running
        print("Waiting for instance to start...")
        while True:
            info = get_instance(_iid)
            status = info.get("actual_status", "unknown")
            print(f"  status: {status}", flush=True)
            if status == "running":
                break
            if status in ("error", "exited", "failed"):
                print(f"Instance entered bad state: {status}, skipping.")
                break
            time.sleep(15)
        if info.get("actual_status") != "running":
            continue

        _ip       = info["public_ipaddr"]
        _port     = info["ports"]["8080/tcp"][0]["HostPort"]
        _token    = info["jupyter_token"]
        _base_url = f"https://{_ip}:{_port}"
        print(f"Jupyter: {_base_url}  token: {_token[:8]}...")

        # wait for Jupyter
        print("Waiting for Jupyter...")
        jupyter_ok = False
        for _ in range(60):
            try:
                req = urllib.request.Request(
                    f"{_base_url}/api/kernelspecs",
                    headers={"Authorization": f"token {_token}"},
                )
                with urllib.request.urlopen(req, timeout=10, context=CTX) as r:
                    if r.status == 200:
                        print("Jupyter ready!")
                        jupyter_ok = True
                        break
            except Exception as e:
                print(f"  waiting: {e}", flush=True)
            time.sleep(15)
        if not jupyter_ok:
            print("Jupyter did not start, skipping.")
            _destroy(_iid)
            continue

        # create kernel + wait for onstart
        print("Waiting for onstart pip install...")
        req = urllib.request.Request(
            f"{_base_url}/api/kernels",
            data=json.dumps({"name": "bash"}).encode(), method="POST",
            headers={"Authorization": f"token {_token}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
            _kid = json.load(r)["id"]
        print(f"Kernel: {_kid}")

        wait_code = (
            "for i in $(seq 1 120); do\n"
            "    [ -f /root/SETUP_DONE ] && echo 'SETUP_DONE found!' && break\n"
            "    sleep 5\n"
            "done\n"
            "[ -f /root/SETUP_DONE ] || echo 'WARNING: SETUP_DONE not found'\n"
        )
        run_on_kernel(_ip, _port, _kid, _token, wait_code, timeout=720)

        # CUDA compatibility test
        print("Testing CUDA compatibility...")
        cuda_out, _ = run_on_kernel(_ip, _port, _kid, _token, CUDA_TEST, timeout=30)
        if "CUDA_KERNEL_TEST_PASSED" in cuda_out:
            print("CUDA OK — proceeding with this instance.")
            INSTANCE_ID, BASE_URL, TOKEN, IP, EXT_PORT, KERNEL_ID = (
                _iid, _base_url, _token, _ip, _port, _kid
            )
            break
        else:
            print(f"CUDA incompatible on {chosen['gpu_name']} — destroying and trying next.")
            _destroy(_iid)

    if INSTANCE_ID is None:
        sys.exit("ERROR: no CUDA-compatible GPU found among available offers")

    # ── upload files ──────────────────────────────────────────────────────────
    print("\nUploading files...")
    create_dir(BASE_URL, TOKEN, "inspect_project")
    create_dir(BASE_URL, TOKEN, "inspect_project/logs")
    create_dir(BASE_URL, TOKEN, f"inspect_project/logs/{family}")

    for script_name in [cfg["original_script"], cfg["abliterated_script"]]:
        upload_file(BASE_URL, TOKEN,
                    f"inspect_project/{script_name}",
                    (LOCAL / script_name).read_bytes())
        print(f"  uploaded: {script_name}")

    env_content   = (PROJECT / ".env").read_text()
    write_env_code = (
        "import pathlib\n"
        f"pathlib.Path('/workspace/inspect_project/.env').write_text({repr(env_content)})\n"
        "print('env written')\n"
    )
    upload_file(BASE_URL, TOKEN, "inspect_project/write_env.py",
                write_env_code.encode())
    print("  uploaded: write_env.py")
    print("All files uploaded.")

    # ── setup (sync: write env, patches, mkdir) ───────────────────────────────
    print("\nRunning setup...")
    run_on_kernel(IP, EXT_PORT, KERNEL_ID, TOKEN, build_setup_code(cfg, family), timeout=120)

    # ── launch experiments in background ──────────────────────────────────────
    print(f"\nLaunching {family} pair in background (original → abliterated)...")
    run_on_kernel(IP, EXT_PORT, KERNEL_ID, TOKEN, build_launch_code(cfg, family), timeout=60)

    # ── poll for completion ───────────────────────────────────────────────────
    print("\nPolling for completion...")
    POLL_INTERVAL = 120
    MAX_WAIT = 6 * 3600
    deadline = time.time() + MAX_WAIT

    while time.time() < deadline:
        done = read_remote_file(BASE_URL, TOKEN, "inspect_project/DONE")
        if done is not None:
            print("\nDONE file found!")
            break

        elapsed = int(time.time() - (deadline - MAX_WAIT))
        print(f"\n[+{elapsed // 60}m] progress...", flush=True)

        for log_name in [
            f"inspect_project/logs/{family}/original_stdout.log",
            f"inspect_project/logs/{family}/abliterated_stdout.log",
            "inspect_project/experiment.log",
        ]:
            content = read_remote_file(BASE_URL, TOKEN, log_name)
            if content:
                tail = content[-300:].decode("utf-8", errors="replace").strip()
                print(f"  {log_name.split('/')[-1]} tail: ...{tail[-200:]}")

        time.sleep(POLL_INTERVAL)
    else:
        print("WARNING: timed out after 6h")

    # ── download logs ─────────────────────────────────────────────────────────
    print("\nDownloading logs...")
    items = list_remote_dir(BASE_URL, TOKEN, f"inspect_project/logs/{family}")
    remote_files = [item["path"] for item in items if item.get("type") == "file"]

    # always grab the stdout tails too
    for extra in [
        f"inspect_project/logs/{family}/original_stdout.log",
        f"inspect_project/logs/{family}/abliterated_stdout.log",
    ]:
        if extra not in remote_files:
            remote_files.append(extra)

    for remote_path in remote_files:
        content = read_remote_file(BASE_URL, TOKEN, remote_path)
        if content:
            local_name = Path(remote_path).name
            (local_logs / local_name).write_bytes(content)
            print(f"  {local_name} ({len(content)} bytes)")
        else:
            print(f"  not found: {remote_path}")

    # ── destroy instance ──────────────────────────────────────────────────────
    print("\nDestroying instance...")
    _destroy(INSTANCE_ID)

    print(f"\n=== DONE. Logs saved to {local_logs} ===")
    for f in sorted(local_logs.iterdir()):
        print(f"  {f.name} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
