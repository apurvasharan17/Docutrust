# =============================================================================
# DocuTrust — expose Kaggle Ollama to the Flask app on your Mac
# Paste as ONE new cell in docutrust-logprob-probe.ipynb, AFTER cell 8
# (`ollama serve` running + qwen2.5:7b-instruct pulled). Needs: Settings → Internet ON.
# =============================================================================
import os, re, subprocess, time, requests

CF = "/kaggle/working/cloudflared"
LOG = "/kaggle/working/cloudflared.log"

# 1) Download cloudflared once (no account / token needed for quick tunnels)
if not os.path.exists(CF):
    subprocess.run(["wget", "-q", "-O", CF,
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"],
                   check=True)
    os.chmod(CF, 0o755)

# 2) Sanity check: Ollama answers locally
print("local ollama:", requests.get("http://localhost:11434/api/tags", timeout=5).json())

# 3) Start the tunnel in the background.
#    --http-host-header makes Ollama see "localhost" so it doesn't reject the request.
log = open(LOG, "w")
tunnel = subprocess.Popen(
    [CF, "tunnel", "--url", "http://localhost:11434",
     "--http-host-header", "localhost:11434", "--no-autoupdate"],
    stdout=log, stderr=subprocess.STDOUT)

# 4) Wait for the public URL
url = None
for _ in range(60):
    time.sleep(1)
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(LOG).read())
    if m:
        url = m.group(0)
        break

if url is None:
    print("Tunnel did not come up — check", LOG)
else:
    time.sleep(3)  # DNS for the new hostname takes a moment
    print("\nPublic Ollama URL:", url)
    print("Through tunnel:", requests.get(url + "/api/tags", timeout=20).status_code)
    print("\nOn your Mac (from the Docutrust folder):")
    print("  export DOCUTRUST_OLLAMA_URL=" + url)
    print("  python webapp/app.py")

# Keep this notebook session alive during the demo. Stop the tunnel with: tunnel.terminate()
# The URL is public while it runs — terminate it after the review.
