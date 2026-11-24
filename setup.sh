#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  parity|formal)
    tfl_target="$PWD/third_party/tfl"
    tfl_commit=$(awk '{print $3}' third_party/tfl.commit)
    if [ ! -d "$tfl_target" ]; then
      git clone https://github.com/satwik77/Transformer-Formal-Languages.git "$tfl_target"
      git -C "$tfl_target" checkout --detach "$tfl_commit"
    fi
    test "$(git -C "$tfl_target" rev-parse HEAD)" = "$tfl_commit"
    # The upstream pin stores some Python sources as CRLF. The published
    # patches and the historical evaluation checkout use LF.
    python - "$tfl_target" <<'PYTHON'
from pathlib import Path
import sys

for source in (Path(sys.argv[1]) / 'src').rglob('*.py'):
    contents = source.read_bytes()
    if b'\r\n' in contents:
        source.write_bytes(contents.replace(b'\r\n', b'\n'))
PYTHON
    for patch in tfl.patch tfl-common-set.patch; do
      if git -C "$tfl_target" apply --check "$PWD/third_party/$patch"; then
        git -C "$tfl_target" apply "$PWD/third_party/$patch"
      else
        git -C "$tfl_target" apply --reverse --check "$PWD/third_party/$patch"
      fi
    done
    ;;
  retnet)
    python -m pip install --no-deps --target "$PWD/third_party/fla_vendor" flash-linear-attention==0.3.1 einops==0.8.1
    ;;
  *)
    echo 'Usage: bash setup.sh formal | parity | retnet' >&2
    exit 2
    ;;
esac
