"""Export only the simulation's known, nonsecret result artifacts via Actions logs.

This supports clients whose artifact-download CDN is unavailable. It never
reads environment values into the archive, repository files, or runner logs.
GitHub authentication is handled by gh for read-only artifact access.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
from time import monotonic, sleep
import zipfile

REPO = "sohamtech-uk/cherryq-qubo-qaoa"
RUN_ID = "37916961742"


def gh(*args):
    result = subprocess.run(["gh", *args], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Read-only GitHub artifact request failed; response omitted")
    return result.stdout


def main():
    started = monotonic()
    while True:
        run = json.loads(gh("api", f"repos/{REPO}/actions/runs/{RUN_ID}"))
        if run["status"] == "completed":
            break
        if monotonic() - started > 12000:
            raise TimeoutError("Source simulation has not completed")
        sleep(30)
    gh("run", "download", RUN_ID, "--repo", REPO, "--pattern", "resonance-*", "--dir", "source-results")
    allowed = {"comparison.json", "comparison.csv", "mock-inventory.json", "mock-architecture.json",
               "iqm-garnet-error-profile.json", "failure.json", "progress.json"}
    for schedule in ("original", "round-robin"):
        allowed.add(f"{schedule}-frozen.qpy")
        for seed in (42, 7, 123):
            allowed.add(f"{schedule}-garnet-route-{seed}.qpy")
    root = Path("source-results")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files or any(p.name not in allowed or p.is_symlink() for p in files):
        raise RuntimeError("Unexpected or missing result artifact; refusing export")
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
    raw = data.getvalue()
    Path("CherryQ_Resonance_Raw_Results.zip").write_bytes(raw)
    print("CHERRYQ_ARCHIVE_METADATA " + json.dumps({"source_run_id": RUN_ID,
          "source_conclusion": run["conclusion"], "bytes": len(raw),
          "sha256": hashlib.sha256(raw).hexdigest(), "files": len(files)}), flush=True)
    encoded = base64.b64encode(raw).decode("ascii")
    for index in range(0, len(encoded), 8192):
        print("CHERRYQ_ARCHIVE_CHUNK " + encoded[index:index + 8192], flush=True)
    print("CHERRYQ_ARCHIVE_END", flush=True)


if __name__ == "__main__":
    main()
