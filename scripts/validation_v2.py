"""`python -m validation` with protocol v2 (validation/protocols/v2.json), the common architecture set.

    python scripts/validation_v2.py run --task regbench --model cs_mamba --profile smoke --seeds 0 --out ...

validation/runner.py is not modified: runner.PROTOCOL is re-pointed before main() runs. Every runner
function reads that module global (protocol load, protocol_sha256, evaluation_sha256), so v2 results
carry v2's hashes while v1 runs, which hash validation/*.py, keep theirs. This file is deliberately
outside validation/ for that reason.

parity/tomita5/tomita6: cs_ models need the TFL checkout patched with third_party/tfl-common-set.patch.
--tfl-root defaults to <repo>/third_party/tfl-v2 (setup.sh creates it), not $PROXY_VENDOR_ROOT/tfl,
so v2 never touches the shared vendor directory.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from validation import runner  # noqa: E402

runner.PROTOCOL = ROOT / "validation/protocols/v2.json"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] in (["check"], ["run"]) and not any(a == "--tfl-root" or a.startswith("--tfl-root=") for a in argv):
        argv += ["--tfl-root", str(ROOT / "third_party/tfl-v2")]
    return runner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
