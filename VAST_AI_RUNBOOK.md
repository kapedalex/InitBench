# VAST.ai Runbook: запуск run_gpt_oss_20b*.py

## Контекст задачи
Запустить два скрипта на GPU-инстансе VAST.ai, сохранить логи в `/root/inspect_project/logs/`, уничтожить инстанс.

Скрипты (патченые, без Docker):
- `/root/inspect_project/_patched_run_gpt_oss_20b.py`
- `/root/inspect_project/_patched_run_gpt_oss_20b_heretic.py`

Ключи — в `/root/inspect_project/.env`.

---

## КРИТИЧЕСКИЕ ФАКТЫ (выученные болью)

### 1. VAST.ai team account → нет SSH
`vastai create ssh-key` → ошибка "Team SSH keys not supported".
**Решение**: только Jupyter REST API + WebSocket.

### 2. Jupyter — HTTPS, не HTTP
```
WRONG:  http://IP:PORT/api/...
RIGHT: https://IP:PORT/api/...
```
Сертификат самоподписанный → `ssl.CERT_NONE` обязательно.

### 3. Jupyter root = `/workspace/`, а НЕ `/root/`
Файлы, загруженные через `PUT /api/contents/inspect_project/foo.py`,
попадают в `/workspace/inspect_project/foo.py`.

Онстарт-скрипт работает в `/root/`. Это разные директории!

**Вывод**: НЕ используй onstart-вотчер с триггер-файлом.
Вместо этого выполняй код напрямую через WebSocket-ядро Jupyter.

### 4. Порт Jupyter = 8080 внутри → маппится наружу
Инфо об инстансе через API:
```python
ports = info["ports"]  # {"8080/tcp": [{"HostPort": "XXXXX"}]}
jupyter_port = ports["8080/tcp"][0]["HostPort"]
jupyter_token = info["jupyter_token"]  # не instance_api_key!
```

### 5. Правильный образ Docker
```
pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```
Содержит: `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`.
Onstart должен доустановить только: `python-dotenv sentencepiece protobuf accelerate`

Почему 2.6.0:
- `openai/gpt-oss-20b` требует `torch.accelerator` — появился в 2.6.0
- `torchvision::nms` не работает при несовпадении версий torch/torchvision

### 6. Docker внутри VAST.ai НЕ работает
`dockerd` падает с `iptables permission denied` или `unshare not permitted`.
**Решение**: использовать патченые скрипты `_patched_*.py`, которые
заменяют Docker-вызовы на `subprocess.run(["bash", "-c", command])`.

### 7. .env нельзя загружать как скрытый файл через Jupyter API
`PUT /api/contents/inspect_project/.env` → 404 "Cannot create hidden file".
**Решение**: загрузить Python-хелпер `write_env.py` и выполнить его через ядро:
```python
# write_env.py
import pathlib
pathlib.Path('/workspace/inspect_project/.env').write_text("""ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
VAST_AI_KEY=...""")
```
НЕ использовать heredoc в bash — спецсимволы в ключах (`+`, `=`) ломают heredoc.

### 8. GPU с достаточным VRAM
- `openai/gpt-oss-20b`: MXFP4, ~10-20GB VRAM (работает на любом современном GPU)
- `p-e-w/gpt-oss-20b-heretic`: без MXFP4, float16 ≈ 40GB VRAM
- **Минимум: 40GB VRAM**. A100 80GB — идеально.

### 9. Скорость сети — критична
HuggingFace модель ~40GB. При 100 Mbps = 53 мин, при 1 Gbps = 5 мин.
Искать: `inet_down > 5000 Mbps`.
Проверенные быстрые локации: Arizona (A100, 21951 Mbps), Germany (RTX6000Ada, 9249 Mbps).

---

## ПРАВИЛЬНЫЙ АЛГОРИТМ

### Шаг 1: Найти GPU-инстанс
```python
import json, urllib.request

VAST_KEY = "<VAST_API_KEY — see .env>"

req = urllib.request.Request(
    'https://console.vast.ai/api/v0/bundles/?q={"gpu_ram":{"gte":40960},'
    '"rentable":{"eq":true},"verified":{"eq":true}}&order=dph_total+asc&limit=20',
    headers={"Authorization": f"Bearer {VAST_KEY}"}
)
with urllib.request.urlopen(req, timeout=15) as r:
    offers = json.load(r)["offers"]

for o in offers:
    if o.get("inet_down", 0) > 5000:
        print(f"ID:{o['id']} GPU:{o['gpu_name']} VRAM:{o['gpu_ram']}MB "
              f"net:{o['inet_down']:.0f}Mbps ${o['dph_total']:.3f}/h loc:{o['geolocation']}")
```

### Шаг 2: Создать инстанс
```python
ONSTART = '''#!/bin/bash
exec > >(tee /root/onstart.log) 2>&1
echo "Setup started: $(date -u)"
pip install -q python-dotenv sentencepiece protobuf accelerate
echo "SETUP_DONE" > /root/SETUP_DONE
echo "Setup done: $(date -u)"
'''

OFFER_ID = 31683443  # A100 80GB Arizona — проверенный, менять если недоступен

payload = {
    "client_id": "me",
    "image": "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    "disk": 80,
    "onstart": ONSTART,
    "runtype": "jupyter_direc ssh_direc",
}

req = urllib.request.Request(
    f"https://console.vast.ai/api/v0/asks/{OFFER_ID}/",
    data=json.dumps(payload).encode(),
    method="PUT",
    headers={"Authorization": f"Bearer {VAST_KEY}", "Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=15) as r:
    result = json.load(r)
    INSTANCE_ID = result["new_contract"]
    print("Instance:", INSTANCE_ID)
```

### Шаг 3: Дождаться running + получить Jupyter URL
```python
import time, ssl

def get_instance(instance_id):
    req = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{instance_id}/",
        headers={"Authorization": f"Bearer {VAST_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)["instances"]

# Ждём running
while True:
    info = get_instance(INSTANCE_ID)
    if info["actual_status"] == "running":
        break
    time.sleep(15)

# Достаём Jupyter координаты
IP = info["public_ipaddr"]
EXT_PORT = info["ports"]["8080/tcp"][0]["HostPort"]
TOKEN = info["jupyter_token"]
BASE_URL = f"https://{IP}:{EXT_PORT}"

print(f"Jupyter: {BASE_URL}  token: {TOKEN}")
```

### Шаг 4: Дождаться Jupyter (HTTPS, CERT_NONE)
```python
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

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
    except Exception:
        pass
    time.sleep(15)
```

### Шаг 5: Создать bash-ядро
```python
req = urllib.request.Request(
    f"{BASE_URL}/api/kernels",
    data=json.dumps({"name": "bash"}).encode(),
    method="POST",
    headers={"Authorization": f"token {TOKEN}", "Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
    kernel = json.load(r)
    KERNEL_ID = kernel["id"]
print("Kernel:", KERNEL_ID)
```

### Шаг 6: Загрузить файлы через Jupyter API
```python
import base64
from pathlib import Path

def upload_file(remote_path, content_bytes):
    payload = json.dumps({
        "name": Path(remote_path).name,
        "path": remote_path,
        "type": "file",
        "format": "base64",
        "content": base64.b64encode(content_bytes).decode(),
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/contents/{remote_path}",
        data=payload, method="PUT",
        headers={"Authorization": f"token {TOKEN}", "Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        return r.status in (200, 201)

def create_dir(path):
    payload = json.dumps({"type": "directory", "path": path}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/contents/{path}",
        data=payload, method="PUT",
        headers={"Authorization": f"token {TOKEN}", "Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15, context=CTX) as r:
            return r.status in (200, 201)
    except:
        return False

# Создать директории
create_dir("inspect_project")
create_dir("inspect_project/logs")

# Загрузить патченые скрипты (как run_gpt_oss_20b.py, без _patched_ префикса)
LOCAL = Path("/root/inspect_project")
upload_file("inspect_project/run_gpt_oss_20b.py",
            (LOCAL / "_patched_run_gpt_oss_20b.py").read_bytes())
upload_file("inspect_project/run_gpt_oss_20b_heretic.py",
            (LOCAL / "_patched_run_gpt_oss_20b_heretic.py").read_bytes())

# Загрузить write_env.py (записывает .env безопасно через Python)
env_content = (LOCAL / ".env").read_text()
write_env_code = f"import pathlib\npathlib.Path('/workspace/inspect_project/.env').write_text({repr(env_content)})\nprint('env written')\n"
upload_file("inspect_project/write_env.py", write_env_code.encode())

print("All files uploaded")
```

### Шаг 7: Выполнить эксперименты через WebSocket
```python
import struct, socket, uuid

def ws_connect(ip, port, path, token):
    key = base64.b64encode(uuid.uuid4().bytes).decode()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    s = socket.create_connection((ip, port), timeout=30)
    s = ctx.wrap_socket(s, server_hostname=ip)
    s.send((
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {ip}:{port}\r\n"
        f"Upgrade: websocket\r\nConnection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
        f"Authorization: token {token}\r\n\r\n"
    ).encode())
    resp = b""
    while b"\r\n\r\n" not in resp:
        resp += s.recv(4096)
    assert b"101" in resp, f"Handshake failed: {resp[:200]}"
    return s

def ws_send(s, message):
    data = message.encode()
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
    """Run code on bash kernel, stream stdout. Returns when execute_reply received."""
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
    s.settimeout(60)
    deadline = time.time() + timeout
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
                print(content.get("text", ""), end="", flush=True)
            elif mtype in ("display_data", "execute_result"):
                print(content.get("data", {}).get("text/plain", ""), flush=True)
            elif mtype == "error":
                print(f"ERROR: {content.get('ename')}: {content.get('evalue')}")
            elif mtype == "execute_reply":
                print(f"\n[Done: {content.get('status')}]")
                break
        except socket.timeout:
            print(".", end="", flush=True)
    s.close()

# Запустить эксперименты (один большой bash-скрипт)
EXPERIMENT_CODE = """
cd /workspace/inspect_project

# Записать .env безопасно
python3 write_env.py

# Проверить torch
python3 -c "import torch, torchvision; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# Запустить первый скрипт
echo "=== run_gpt_oss_20b.py: $(date -u) ==="
python3 run_gpt_oss_20b.py 2>&1 | tee logs/run_gpt_oss_20b_stdout.log

# Запустить второй скрипт
echo "=== run_gpt_oss_20b_heretic.py: $(date -u) ==="
python3 run_gpt_oss_20b_heretic.py 2>&1 | tee logs/run_gpt_oss_20b_heretic_stdout.log

echo "=== ALL DONE: $(date -u) ==="
echo "DONE" > /workspace/inspect_project/DONE
"""

run_on_kernel(IP, EXT_PORT, KERNEL_ID, TOKEN, EXPERIMENT_CODE, timeout=14400)
```

### Шаг 8: Скачать логи
```python
def read_file(remote_path):
    req = urllib.request.Request(
        f"{BASE_URL}/api/contents/{remote_path}?format=base64",
        headers={"Authorization": f"token {TOKEN}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
            d = json.load(r)
            return base64.b64decode(d["content"])
    except:
        return None

LOCAL_LOGS = Path("/root/inspect_project/logs")
LOCAL_LOGS.mkdir(exist_ok=True)

for remote, local_name in [
    ("inspect_project/logs/run_gpt_oss_20b_stdout.log", "run_gpt_oss_20b_stdout.log"),
    ("inspect_project/logs/run_gpt_oss_20b_heretic_stdout.log", "run_gpt_oss_20b_heretic_stdout.log"),
]:
    content = read_file(remote)
    if content:
        (LOCAL_LOGS / local_name).write_bytes(content)
        print(f"Downloaded: {local_name} ({len(content)} bytes)")
```

### Шаг 9: Уничтожить инстанс
```python
req = urllib.request.Request(
    f"https://console.vast.ai/api/v0/instances/{INSTANCE_ID}/",
    method="DELETE",
    headers={"Authorization": f"Bearer {VAST_KEY}"}
)
with urllib.request.urlopen(req, timeout=15) as r:
    print("Destroy:", json.load(r))
```

---

## БЫСТРАЯ ШПАРГАЛКА

| Что | Правильно | Неправильно |
|-----|-----------|-------------|
| Протокол Jupyter | HTTPS | HTTP |
| SSL верификация | CERT_NONE | по умолчанию |
| Jupyter root | `/workspace/` | `/root/` |
| Токен Jupyter | `info["jupyter_token"]` | `info["instance_api_key"]` |
| Загрузка .env | через Python write_text() | через heredoc, PUT /api/contents/.env |
| Docker в контейнере | НЕ работает (unshare denied) | — |
| Torch версия | 2.6.0 | 2.5.x (нет torch.accelerator) |
| Onstart-вотчер | НЕ использовать (смотрит /root/, файлы в /workspace/) | — |
| Запуск кода | WebSocket на /api/kernels/{id}/channels | — |

## СТАТУСЫ

- VAST_KEY: `<VAST_API_KEY — see .env>`
- Патченые скрипты готовы: `/root/inspect_project/_patched_run_gpt_oss_20b.py`, `_patched_run_gpt_oss_20b_heretic.py`
- Эксперименты ещё НЕ прогнаны успешно
- Проверенный offer: A100 80GB PCIe, Arizona, ID=31683443, $0.802/h, 21951 Mbps
