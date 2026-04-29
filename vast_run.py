"""
VAST.ai runner: launches GPU instance, runs patched experiments, downloads logs, destroys instance.
"""
import base64
import json
import socket
import ssl
import struct
import time
import urllib.request
import uuid
from pathlib import Path

LOCAL = Path("/root/inspect_project")
LOGS_DIR = LOCAL / "logs"
LOGS_DIR.mkdir(exist_ok=True)

env = {}
for line in (LOCAL / ".env").read_text().splitlines():
    if "=" in line:
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()

VAST_KEY = env["VAST_AI_KEY"]
HF_TOKEN = env.get("HF_TOKEN", "")
print(f"VAST_KEY loaded: {VAST_KEY[:16]}...")
print(f"HF_TOKEN loaded: {HF_TOKEN[:8]}..." if HF_TOKEN else "HF_TOKEN: not found!")

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def vast_req(path, method="GET", data=None):
    req = urllib.request.Request(
        f"https://console.vast.ai/api/v0{path}",
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {VAST_KEY}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def get_instance(instance_id):
    return vast_req(f"/instances/{instance_id}/")["instances"]


def http_req(url, token, method="GET", data=None, timeout=30):
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Authorization": f"token {token}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.status, r.read()


def upload_file(base_url, token, remote_path, content_bytes):
    payload = json.dumps({
        "name": Path(remote_path).name,
        "path": remote_path,
        "type": "file",
        "format": "base64",
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
            headers={"Authorization": f"token {token}"}
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
            headers={"Authorization": f"token {token}"}
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


def run_on_kernel(ip, ext_port, kernel_id, token, code, timeout=120):
    """Send code to kernel and collect output until execute_reply or timeout."""
    path = f"/api/kernels/{kernel_id}/channels"
    s = ws_connect(ip, int(ext_port), path, token)
    msg_id = str(uuid.uuid4())
    ws_send(s, json.dumps({
        "header": {"msg_id": msg_id, "username": "user",
                   "session": str(uuid.uuid4()),
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": True,
                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": False},
        "buffers": [], "channel": "shell",
    }))
    s.settimeout(30)
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
            mtype = msg.get("header", {}).get("msg_type", "")
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


print("\nFinding GPU offers (≥40GB VRAM, fast network)")

req = urllib.request.Request(
    'https://console.vast.ai/api/v0/bundles/?q={"gpu_ram":{"gte":40960},'
    '"rentable":{"eq":true},"verified":{"eq":true}}&order=dph_total+asc&limit=20',
    headers={"Authorization": f"Bearer {VAST_KEY}"}
)
with urllib.request.urlopen(req, timeout=15) as r:
    offers = json.load(r)["offers"]

fast_offers = [o for o in offers if o.get("inet_down", 0) > 5000]
if not fast_offers:
    print("No fast offers found, using all offers")
    fast_offers = offers

print("Fast offers (>5Gbps):")
for o in fast_offers[:8]:
    print(f"  ID:{o['id']} GPU:{o['gpu_name']} VRAM:{o['gpu_ram']}MB "
          f"net:{o['inet_down']:.0f}Mbps ${o['dph_total']:.3f}/h loc:{o['geolocation']}")

# Known-full machines to skip if you have specific preferences
SKIP_OFFERS = {31683443}
candidates = [o for o in fast_offers if o["id"] not in SKIP_OFFERS]
if not candidates:
    candidates = fast_offers
candidates.sort(key=lambda o: o.get("dph_total", 9999))
chosen = candidates[0]
OFFER_ID = chosen["id"]
print(f"Selected offer: {OFFER_ID} GPU:{chosen['gpu_name']} VRAM:{chosen['gpu_ram']}MB "
      f"${chosen['dph_total']:.3f}/h net:{chosen.get('inet_down',0):.0f}Mbps loc:{chosen['geolocation']}")

print("\nCreating instance")

ONSTART = '''#!/bin/bash
exec > >(tee /root/onstart.log) 2>&1
echo "Setup started: $(date -u)"
pip install -q python-dotenv sentencepiece protobuf accelerate transformers huggingface_hub triton
echo "SETUP_DONE" > /root/SETUP_DONE
echo "Setup done: $(date -u)"
'''

payload = {
    "client_id": "me",
    "image": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    "disk": 120,
    "onstart": ONSTART,
    "runtype": "jupyter_direc ssh_direc",
}

result = vast_req(f"/asks/{OFFER_ID}/", method="PUT", data=json.dumps(payload).encode())
INSTANCE_ID = result["new_contract"]
print(f"Instance created: {INSTANCE_ID}")

print("\n=== STEP 3: Waiting for instance to be running ===")

while True:
    info = get_instance(INSTANCE_ID)
    status = info.get("actual_status", "unknown")
    print(f"  Status: {status}", flush=True)
    if status == "running":
        break
    if status in ("error", "exited", "failed"):
        raise RuntimeError(f"Instance entered bad state: {status}")
    time.sleep(15)

IP = info["public_ipaddr"]
EXT_PORT = info["ports"]["8080/tcp"][0]["HostPort"]
TOKEN = info["jupyter_token"]
BASE_URL = f"https://{IP}:{EXT_PORT}"
print(f"Jupyter: {BASE_URL}  token: {TOKEN[:8]}...")

while True:
    try:
        req = urllib.request.Request(
            f"{BASE_URL}/api/kernelspecs",
            headers={"Authorization": f"token {TOKEN}"}
        )
        with urllib.request.urlopen(req, timeout=10, context=CTX) as r:
            if r.status == 200:
                print("Jupyter ready!")
                break
    except Exception as e:
        print(f"  waiting: {e}", flush=True)
    time.sleep(15)


print("\nCreating bash kernel")

req = urllib.request.Request(
    f"{BASE_URL}/api/kernels",
    data=json.dumps({"name": "bash"}).encode(),
    method="POST",
    headers={"Authorization": f"token {TOKEN}", "Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
    kernel = json.load(r)
    KERNEL_ID = kernel["id"]
print(f"Kernel: {KERNEL_ID}")

print("\nUploading files")

create_dir(BASE_URL, TOKEN, "inspect_project")
create_dir(BASE_URL, TOKEN, "inspect_project/logs")

upload_file(BASE_URL, TOKEN, "inspect_project/run_gpt_oss_20b.py",
            (LOCAL / "_patched_run_gpt_oss_20b.py").read_bytes())
print("  uploaded: run_gpt_oss_20b.py")

upload_file(BASE_URL, TOKEN, "inspect_project/run_gpt_oss_20b_heretic.py",
            (LOCAL / "_patched_run_gpt_oss_20b_heretic.py").read_bytes())
print("  uploaded: run_gpt_oss_20b_heretic.py")

env_content = (LOCAL / ".env").read_text()
write_env_code = (
    "import pathlib\n"
    f"pathlib.Path('/workspace/inspect_project/.env').write_text({repr(env_content)})\n"
    "print('env written')\n"
)
upload_file(BASE_URL, TOKEN, "inspect_project/write_env.py", write_env_code.encode())
print("  uploaded: write_env.py")
print("All files uploaded.")

print("\nWaiting for onstart (pip install)")

WAIT_CODE = """
df -h /workspace
echo "Waiting for SETUP_DONE..."
for i in $(seq 1 120); do
    if [ -f /root/SETUP_DONE ]; then
        echo "SETUP_DONE found!"
        break
    fi
    sleep 5
done
[ -f /root/SETUP_DONE ] || echo "WARNING: SETUP_DONE not found after 10min"
python3 -c "import torch, torchvision; print('torch:', torch.__version__, 'torchvision:', torchvision.__version__, 'cuda:', torch.cuda.is_available())"
"""
run_on_kernel(IP, EXT_PORT, KERNEL_ID, TOKEN, WAIT_CODE, timeout=720)


print("\nLaunching experiments in background")

LAUNCH_CODE = r"""
cd /workspace/inspect_project
python3 write_env.py
mkdir -p logs

cat > /workspace/run_experiments.sh << 'SCRIPT'
#!/bin/bash
set -e
export HF_HOME=/workspace/.huggingface
export HF_TOKEN=$(python3 -c "
d={}
[d.update({k.strip(): v.strip()}) for line in open('/workspace/inspect_project/.env') if '=' in line for k,_,v in [line.partition('=')]]
print(d.get('HF_TOKEN',''))
")
echo "HF_TOKEN set: ${HF_TOKEN:0:8}..."

# Check triton version
python3 -c "import triton; print('triton version:', triton.__version__)" || echo "triton not found"

python3 - << 'PYEOF'
import transformers.integrations.moe as m
src = open(m.__file__).read()
old = 'torch.mm(input[start:end], weight[i], out=output[start:end])'
new = 'torch.mm(input[start:end], weight[i].to(input.device), out=output[start:end])'
if old in src:
    open(m.__file__, 'w').write(src.replace(old, new))
    print('[patch] moe.py patched: weight[i].to(input.device)')
elif new in src:
    print('[patch] moe.py already patched')
else:
    print('[patch] WARNING: pattern not found in moe.py')
PYEOF

cd /workspace/inspect_project
echo "=== run_gpt_oss_20b.py started: $(date -u) ===" | tee -a logs/run_gpt_oss_20b_stdout.log
python3 run_gpt_oss_20b.py 2>&1 | tee -a logs/run_gpt_oss_20b_stdout.log
echo "=== run_gpt_oss_20b_heretic.py started: $(date -u) ===" | tee -a logs/run_gpt_oss_20b_heretic_stdout.log
python3 run_gpt_oss_20b_heretic.py 2>&1 | tee -a logs/run_gpt_oss_20b_heretic_stdout.log
echo "=== ALL DONE: $(date -u) ===" | tee /workspace/inspect_project/DONE_LOG
echo "DONE" > /workspace/inspect_project/DONE
SCRIPT

chmod +x /workspace/run_experiments.sh
nohup /workspace/run_experiments.sh > /workspace/nohup_main.log 2>&1 &
echo "Launched PID: $!"
echo "Experiments running in background."
"""

out, status = run_on_kernel(IP, EXT_PORT, KERNEL_ID, TOKEN, LAUNCH_CODE, timeout=60)
print(f"Launch status: {status}")


print("\nPolling for completion")

POLL_INTERVAL = 120  # seconds
MAX_WAIT = 6 * 3600  # 6 hours
deadline = time.time() + MAX_WAIT
last_log1_size = 0
last_log2_size = 0

while time.time() < deadline:
    # Check DONE file
    done = read_remote_file(BASE_URL, TOKEN, "inspect_project/DONE")
    if done is not None:
        print("\nDONE file found! Experiments completed.")
        break

    # Show progress via log tail
    elapsed = int(time.time() - (deadline - MAX_WAIT))
    print(f"\n[+{elapsed//60}m] Checking progress...", flush=True)

    for logname in ["run_gpt_oss_20b_stdout.log", "run_gpt_oss_20b_heretic_stdout.log"]:
        log = read_remote_file(BASE_URL, TOKEN, f"inspect_project/logs/{logname}")
        if log:
            size = len(log)
            tail = log[-400:].decode("utf-8", errors="replace").strip()
            print(f"  {logname} ({size}B): ...{tail[-200:]}")

    nohup = read_remote_file(BASE_URL, TOKEN, "nohup_main.log")
    if nohup:
        tail = nohup[-200:].decode("utf-8", errors="replace").strip()
        print(f"  nohup_main.log tail: {tail}")

    time.sleep(POLL_INTERVAL)
else:
    print("WARNING: Timed out waiting for experiments after 6 hours!")


print("\nDownloading all logs")

items = list_remote_dir(BASE_URL, TOKEN, "inspect_project/logs")
if items:
    remote_files = [item["path"] for item in items if item.get("type") == "file"]
else:
    remote_files = [
        "inspect_project/logs/run_gpt_oss_20b_stdout.log",
        "inspect_project/logs/run_gpt_oss_20b_heretic_stdout.log",
    ]

# Also grab nohup_main.log and DONE_LOG
extra_files = ["nohup_main.log", "inspect_project/DONE_LOG"]

for remote_path in remote_files + extra_files:
    local_name = Path(remote_path).name
    content = read_remote_file(BASE_URL, TOKEN, remote_path)
    if content:
        (LOGS_DIR / local_name).write_bytes(content)
        print(f"  Downloaded: {local_name} ({len(content)} bytes)")
    else:
        print(f"  Not found: {remote_path}")


print("\nDestroying instance")

req = urllib.request.Request(
    f"https://console.vast.ai/api/v0/instances/{INSTANCE_ID}/",
    method="DELETE",
    headers={"Authorization": f"Bearer {VAST_KEY}"}
)
with urllib.request.urlopen(req, timeout=15) as r:
    print("Destroy:", json.load(r))

print("\n=== DONE. Logs saved to /root/inspect_project/logs/ ===")
print("Files:")
for f in sorted(LOGS_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size} bytes)")
