#!/usr/bin/env bash
# Sync this checkout to the CUDA host and run a command there.
#
#   tools/remote.sh build            configure + build (preset cuda-release)
#   tools/remote.sh test             build + ctest
#   tools/remote.sh bench            build + run all benchmarks
#   tools/remote.sh run <cmd...>     run an arbitrary command in the remote checkout
#
# Settings come from tools/remote.env (git-ignored, see tools/remote.env.example)
# or the environment: CURF_REMOTE (ssh destination), CURF_REMOTE_DIR, CURF_PRESET.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"
[[ -f "$here/remote.env" ]] && source "$here/remote.env"

: "${CURF_REMOTE:?set CURF_REMOTE in tools/remote.env (e.g. user@host)}"
: "${CURF_REMOTE_DIR:=work/cuRF-Stream}"
: "${CURF_PRESET:=cuda-release}"
SSH_OPTS=(${CURF_SSH_OPTS:-})

ssh "${SSH_OPTS[@]}" "$CURF_REMOTE" "mkdir -p $CURF_REMOTE_DIR"
rsync -az --delete \
    --exclude '/build/' --exclude '/cmake-build-*/' --exclude '/.git/' --exclude '/.idea/' \
    --exclude '/tools/remote.env' --exclude '__pycache__/' --exclude '/model/artifacts/' \
    -e "ssh ${SSH_OPTS[*]}" "$root/" "$CURF_REMOTE:$CURF_REMOTE_DIR/"

# The remote login shell may be zsh; run everything under bash explicitly.
remote() {
    local cmd="$1"
    ssh "${SSH_OPTS[@]}" "$CURF_REMOTE" "bash -lc $(printf '%q' "
        set -euo pipefail
        [[ -f ~/opt/tensorrt-11.4/env.sh ]] && source ~/opt/tensorrt-11.4/env.sh
        cd $CURF_REMOTE_DIR
        $cmd")"
}

build="cmake --preset $CURF_PRESET >/dev/null && cmake --build --preset $CURF_PRESET"
bin="build/$CURF_PRESET/bench"

case "${1:-build}" in
    build) remote "$build" ;;
    test)  remote "$build && ctest --preset $CURF_PRESET" ;;
    bench) remote "$build && $bin/curf_bench_source && $bin/curf_bench_unpack" ;;
    run)   shift; remote "$*" ;;
    *)     echo "usage: $0 {build|test|bench|run <cmd...>}" >&2; exit 2 ;;
esac
