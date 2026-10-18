"""vLLM sm75 test v16 — enforce_eager; alagbo clean-batch run.

Injected at push time: @@CF_ACCOUNT@@ @@CF_DB@@ @@CF_KEY@@ @@HF_TOKEN@@
Rows: vllm_sm75 (status+notes+result), vllm_sm75_hb (30s heartbeat).
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

CF_ACCT = "@@CF_ACCOUNT@@"
CF_DB = "@@CF_DB@@"
CF_KEY = "@@CF_KEY@@"
HF_TOKEN = "@@HF_TOKEN@@"
T0 = time.time()
ROW = "vllm_sm75"
HB = "vllm_sm75_hb"
MODEL = "Qwen/Qwen3.5-4B"
NOTES = []
PHASE = ["INIT"]
HBN = [0]


def d1(sql, params=()):
    req = urllib.request.Request(
        "https://api.cloudflare.com/client/v4/accounts/%s/d1/database/%s/query" % (CF_ACCT, CF_DB),
        data=json.dumps({"sql": sql, "params": list(params)}).encode(),
        method="POST",
        headers={"Authorization": "Bearer " + CF_KEY, "Content-Type": "application/json",
                 "User-Agent": "vllm3"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def put(row, status, info):
    try:
        d1("INSERT INTO kaggle_gpu_state (id,status,tunnel_url,api_key,model,gpu_info,updated_at) "
           "VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
           "gpu_info=excluded.gpu_info, updated_at=excluded.updated_at;",
           [row, str(status)[:80], "", "", MODEL, json.dumps(info)[:16000], time.time()])
    except Exception as e:
        print("D1FAIL", row, repr(e), flush=True)


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def save(status, detail="", extra=None):
    info = {"status": status, "detail": str(detail)[:16000], "notes": NOTES[-24:],
            "elapsed": round(time.time() - T0, 1), "pid": os.getpid()}
    if extra:
        info.update(extra)
    put(ROW, status, info)


def phase(name, detail=""):
    PHASE[0] = name
    NOTES.append({"phase": name, "t": round(time.time() - T0, 1), "d": str(detail)[:400]})
    log("PHASE %s" % name)
    save("RUNNING_" + name, str(detail))


def heartbeat():
    while True:
        time.sleep(30)
        HBN[0] += 1
        put(HB, "HB", {"phase": PHASE[0], "n": HBN[0], "t": round(time.time() - T0, 1),
                       "pid": os.getpid()})


threading.Thread(target=heartbeat, daemon=True).start()


def sh(cmd, timeout=3600):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.time() - t0
    except subprocess.TimeoutExpired as e:
        return 124, ((e.stdout or "") + (e.stderr or ""))[-3000:], time.time() - t0
    except Exception as e:
        return 1, str(e), time.time() - t0


# ---------------------------------------------------------------- phases
phase("ENV")
rc, out, dt = sh("python3 -V; nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader; "
                 "free -g|head -2; df -h /kaggle/working|tail -1")
save("RUNNING_GPU_INFO", out)

rc, out, dt = sh("python3 - <<'PY'\n"
                 "import torch\n"
                 "print('torch', torch.__version__, 'cuda', torch.version.cuda)\n"
                 "print('cap', torch.cuda.get_device_capability(0), 'bf16', torch.cuda.is_bf16_supported(),\n"
                 "      'name', torch.cuda.get_device_name(0))\n"
                 "PY")
phase("CAPABILITY", out)

rc, out, dt = sh("pip show vllm 2>/dev/null|head -2; pip show transformers|head -2")
if "Name: vllm" not in out:
    phase("INSTALL")
    rc, out, dt = sh("pip install --quiet vllm 2>&1|tail -4", timeout=1500)
    if rc != 0:
        save("FAIL_INSTALL", out[-4000:])
        sys.exit(1)

rc, out, dt = sh("python3 -c \"import vllm;print('vllm',vllm.__version__)\"")
m = re.search(r"vllm\s+([\d.]+)", out)
VER = m.group(1) if m else "?"
phase("VERSION", out)

rc, out, _ = sh("pip show flashinfer-jit-cache >/dev/null 2>&1 && echo PRESENT || echo ABSENT")
if "PRESENT" in out:
    sh("pip uninstall -y flashinfer-jit-cache 2>&1|tail -1", timeout=300)

# patch vLLM source: neuter the CC<8 bf16 hard-block (file-level: spawn workers see it)
phase("PATCH_SM75")
rc, out, _ = sh("python3 - <<'PY2'\n"
                "import re, pathlib\n"
                "p = pathlib.Path('/usr/local/lib/python3.13/dist-packages/vllm/platforms/cuda.py')\n"
                "s = p.read_text()\n"
                "m = re.search(r'def check_if_supports_dtype', s)\n"
                "assert m, 'method not found'\n"
                "start = m.start()\n"
                "m2 = re.search(r'\\n    def ', s[start+10:])\n"
                "end = start + 10 + m2.start() if m2 else len(s)\n"
                "head = s[start:end].splitlines()[0]\n"
                "new = head + '\\n        return  # bf16 CC check bypassed for sm75 Turing emulation\\n'\n"
                "p.write_text(s[:start] + new + s[end:])\n"
                "import py_compile\n"
                "py_compile.compile(str(p), doraise=True)\n"
                "print('PATCHED_OK', head)\n"
                "PY2", timeout=120)
if rc != 0 or "PATCHED_OK" not in out:
    save("FAIL_PATCH", out[-1500:])
    sys.exit(1)
save("RUNNING_PATCH_OK", out[:400])

# download: in-process, per-file, xet disabled (xet crashed session in v1)
phase("DOWNLOAD")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HOME"] = "/kaggle/working/hf_cache"
try:
    from huggingface_hub import list_repo_files, hf_hub_download

    files = list_repo_files(MODEL, token=HF_TOKEN)
    sfiles = [f for f in files if f.endswith((".safetensors", ".json", ".txt", ".model"))]
    NOTES.append({"files": sfiles[:8], "n": len(sfiles)})
    got = []
    for f in sfiles:
        p0 = time.time()
        hf_hub_download(MODEL, f, token=HF_TOKEN, cache_dir="/kaggle/working/hf_cache")
        got.append(f)
        put(ROW, "RUNNING_DL_FILE", {"file": f, "got": len(got), "total": len(sfiles),
                                     "t": round(time.time() - T0, 1), "notes": NOTES[-6:]})
        log("dl %s %.1fs" % (f, time.time() - p0))
    phase("DOWNLOAD_OK", "%d files" % len(got))
except Exception as e:
    import traceback
    save("FAIL_DOWNLOAD", str(e) + "\n" + traceback.format_exc()[-1500:])
    sys.exit(1)

# pre-serve import fix: pip's vllm pulls torch-cu130 while kaggle torchaudio=12.8
phase("PRE_SERVE")
rc, out, _ = sh("python3 -c \"import vllm.entrypoints.llm; print('vllm-import-ok')\"", timeout=300)
if rc != 0:
    sh("pip uninstall -y torchaudio 2>&1|tail -1", timeout=180)
    rc, out, _ = sh("python3 -c \"import vllm.entrypoints.llm; print('vllm-import-ok-after-uninstall')\"", timeout=300)
    if rc != 0:
        sh("pip install --quiet torchaudio 2>&1|tail -2", timeout=600)
        rc, out, _ = sh("python3 -c \"import vllm.entrypoints.llm; print('vllm-import-ok-after-reinstall')\"", timeout=300)
    if rc != 0:
        save("FAIL_IMPORT", out[-4000:])
        sys.exit(1)
save("RUNNING_PRE_SERVE_OK", out[-500:])


# preflight: /dev/shm + worker env
phase("PREFLIGHT")
sh_df = sh("df -h /dev/shm /tmp; mount | grep -E '/dev/shm' | head -2")[1]
sh_rem = sh("mount -o remount,size=4G /dev/shm 2>&1 && echo REMOUNT_OK || echo REMOUNT_FAIL")[1]
sh_df2 = sh("df -h /dev/shm | tail -1")[1]
save("RUNNING_PREFLIGHT", "df:"+sh_df[:300]+" remount:"+sh_rem[:120]+" after:"+sh_df2[:200])

# serve — spawn-safe (guarded) script + full log capture
phase("SERVE")
SERVE_SRC = """
import os, time, json

def main():
    # sm75 bf16 bypass: vLLM hard-blocks bf16 below CC 8.0 although torch
    # emulates bf16 on Turing (is_bf16_supported()==True). Patch the check.
    from vllm.platforms.cuda import CudaPlatform
    _orig = getattr(CudaPlatform, 'check_if_supports_dtype', None)
    def _allow(self, dtype):
        import torch
        if 'bfloat16' in str(dtype) and not torch.cuda.is_bf16_supported():
            raise ValueError('bf16 requested but torch reports unsupported')
    CudaPlatform.check_if_supports_dtype = _allow
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model='Qwen/Qwen3.5-4B', dtype='bfloat16', tensor_parallel_size=2,
              gpu_memory_utilization=0.92, max_model_len=4096,
              trust_remote_code=True, download_dir='/kaggle/working/hf_cache',
              skip_mm_profiling=True, enforce_eager=True)
    t_load = time.time()-t0
    sp = SamplingParams(max_tokens=64, temperature=0.7)
    t1 = time.time()
    outs = llm.generate(['Write a one-line greeting from a friendly robot.'], sp)
    t_gen = time.time()-t1
    ntok = len(outs[0].outputs[0].token_ids)
    print('VLLM_RESULT ' + json.dumps({'t_load_s': round(t_load,1), 't_gen_s': round(t_gen,2),
          'tokens': ntok, 'tok_per_s': round(ntok/t_gen,2),
          'text': outs[0].outputs[0].text[:200]}), flush=True)

if __name__ == '__main__':
    main()
"""
with open("/kaggle/working/serve.py", "w") as fh:
    fh.write(SERVE_SRC)
rc, out, dt = sh("HF_TOKEN=%s HF_HUB_DISABLE_XET=1 HF_HOME=/kaggle/working/hf_cache "
                 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
                 "VLLM_LOGGING_LEVEL=INFO NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=0 NCCL_DEBUG=WARN python3 /kaggle/working/serve.py "
                 "> /kaggle/working/vllm_serve.log 2>&1; echo RC=$?" % HF_TOKEN, timeout=3000)
log_tail = sh("tail -c 10000 /kaggle/working/vllm_serve.log")[1]
log_head = sh("head -c 6000 /kaggle/working/vllm_serve.log")[1]
errlines = sh("grep -n -m 30 -Ei 'error|not supported|notimplemented|driver|cuda err|bfloat|"
              "qwen3_5|failed|bootstrap' /kaggle/working/vllm_serve.log")[1]
NOTES.append({"serve_rc": out.strip()[-40:]})
mm = re.search(r"VLLM_RESULT (\{.*\})", log_head + log_tail)
if mm and out.strip().endswith("RC=0"):
    save("DONE", log_tail[-4000:], extra={"result": json.loads(mm.group(1)), "vllm": VER})
    print("SUCCESS", mm.group(1), flush=True)
    sys.exit(0)


# full log in chunks (rows vllm_sm75_l1..l3)
try:
    blob = open("/kaggle/working/vllm_serve.log","rb").read()
    txt = blob.decode("utf-8","replace")
    n = len(txt)
    for i in range(3):
        part = txt[i*n//3:(i+1)*n//3]
        put("vllm_sm75_l%d"%(i+1), "LOG%d"% (i+1), {"off": i*n//3, "len": len(part), "part": part[:15000]})
except Exception as e:
    save("LOGCAP_FAIL", str(e))

cuda_post = sh("python3 - <<'PY'\nimport torch\n"
               "try:\n    print('cuda-ok', torch.zeros(4,device='cuda').sum().item())\n"
               "except Exception as e:\n    print('cuda-fail',repr(e))\n"
               "print('torch',torch.__version__)\nPY", timeout=180)[1]
low = (log_head + log_tail + errlines + out).lower()
tag = "FAIL_SERVE"
# OOM first: 'bfloat16' appears in every config dump and FA2's
# "not supported" line used to mis-tag OOM as FAIL_DTYPE_BF16 (v9).
if "out of memory" in low:
    tag = "FAIL_OOM"
elif "skipframe" in low or "bf16 is not supported" in low or "inductor" in low and "error" in low:
    tag = "FAIL_COMPILE"   # torch._inductor bf16 gate (v13); fixed via enforce_eager
elif "bfloat16" in low and ("not supported" in low or "unsupported" in low) \
        and "out of memory" not in low and "profile" not in low:
    tag = "FAIL_DTYPE_BF16"
elif "qwen3_5" in low and ("not supported" in low or "unknown" in low or "keyerror" in low):
    tag = "FAIL_ARCH_QWEN3_5"
elif "driver version is insufficient" in low:
    tag = "FAIL_DRIVER"
save(tag, "ERRLINES:\n" + errlines[:3000] + "\n\nHEAD:\n" + log_head[:3500] +
     "\n\nTAIL:\n" + log_tail[-4500:], extra={"cuda_post": cuda_post[:400], "vllm": VER})
sys.exit(1)
