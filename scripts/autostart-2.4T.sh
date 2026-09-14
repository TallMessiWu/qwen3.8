#!/usr/bin/env bash
# Cron entry point for the four-node 2.4T service.  Runs on the HOST, once per
# machine, and does three things: make sure the serving container is up, pick
# the rank this machine owns, and launch that rank's 2.4T-N.sh inside the
# container through an interactive shell so ~/.bashrc is sourced.
#
# All four machines run an identical copy of this script and an identical
# crontab line.  The rank is derived from the machine's own IPv4 address, so
# nothing here is per-machine; override with NODE_RANK to force one.
#
# This is a restart, not a health check: 2.4T-N.sh calls npu-cleaner.sh, which
# SIGKILLs every process holding an NPU.  Firing this while the service is
# healthy therefore kills and relaunches it, which is the intended behaviour.
# Point cron at it on whatever cadence the service should be recycled on, and
# use --check first to prove the container plumbing works.

set -euo pipefail

# cron hands over a nearly empty PATH, and docker lives in /usr/bin or
# /usr/local/bin depending on how it was installed.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

CONTAINER_NAME="${CONTAINER_NAME:-hajimi-vllm}"
CONTAINER_USER="${CONTAINER_USER:-root}"
# The directory as seen INSIDE the container.  /home is bind-mounted straight
# from the host, so the same path works on both sides.
SCRIPTS_DIR="${SCRIPTS_DIR:-/home/hajimi/qwen3.5/scripts}"
LOG_DIR="${LOG_DIR:-$SCRIPTS_DIR/logs}"
LOG_KEEP_DAYS="${LOG_KEEP_DAYS:-7}"

# Keep this table in step with the LOCAL_IP defaults in 2.4T-{0..3}.sh;
# scripts/tests/test_script_defaults.py pins that they agree.
NODE0_IP="${NODE0_IP:-141.61.52.179}"
NODE1_IP="${NODE1_IP:-141.61.52.183}"
NODE2_IP="${NODE2_IP:-141.61.52.187}"
NODE3_IP="${NODE3_IP:-141.61.52.191}"

DP_RPC_PORT="${DP_RPC_PORT:-13389}"
# Node 0 binds the DP handshake port before it loads any weights, so ranks 1-3
# wait seconds here, not the whole model load.  zmq connect() tolerates a peer
# that is not listening yet, so a timeout is a warning rather than a failure.
WAIT_NODE0_SECONDS="${WAIT_NODE0_SECONDS:-300}"
# A flat head start for ranks 1-3, applied before the probe below.  The probe
# asks whether a TCP connect succeeds, and a middlebox that accepts on node 0's
# behalf answers yes while node 0 is still down, which would collapse the wait
# to nothing.  This part costs a fixed 20s and assumes nothing about the network.
STAGGER_SECONDS="${STAGGER_SECONDS:-20}"
CONTAINER_START_TIMEOUT="${CONTAINER_START_TIMEOUT:-60}"
# npu-cleaner.sh only reaps processes that hold an NPU; the API-server frontend
# does not, and a survivor still owning port 13389 makes node 0's next bind
# fail.  This clears those before relaunching.
KILL_STALE="${KILL_STALE:-1}"

CHECK_ONLY=0

usage() {
    printf '%s\n' \
        "Usage: bash $0 [--check] [--rank N]" \
        "" \
        "Options:" \
        "  --check       Verify container, rank, script path and that ~/.bashrc" \
        "                is sourced by the launch shell, then exit without" \
        "                touching the service.  Run this first on a new box." \
        "  --rank N      Force the node rank instead of deriving it from the" \
        "                machine's IPv4 address (same as NODE_RANK=N)." \
        "  -h, --help    Show this help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h | --help)
            usage
            exit 0
            ;;
        --check)
            CHECK_ONLY=1
            shift
            ;;
        --rank)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "ERROR: --rank requires a value." >&2
                exit 2
            fi
            NODE_RANK="$2"
            shift 2
            ;;
        --rank=*)
            NODE_RANK="${1#*=}"
            shift
            ;;
        *)
            echo "ERROR: unknown option '$1'." >&2
            usage >&2
            exit 2
            ;;
    esac
done

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { printf 'RED: %s\n' "$*" >&2; exit 2; }

host_ipv4s() {
    ip -o -4 addr show 2>/dev/null | awk '{split($4, a, "/"); print a[1]}'
}

resolve_rank() {
    local ip
    while read -r ip; do
        case "$ip" in
            "$NODE0_IP") echo 0; return 0 ;;
            "$NODE1_IP") echo 1; return 0 ;;
            "$NODE2_IP") echo 2; return 0 ;;
            "$NODE3_IP") echo 3; return 0 ;;
        esac
    done < <(host_ipv4s)
    return 1
}

if [[ -n "${NODE_RANK:-}" ]]; then
    rank="$NODE_RANK"
elif ! rank="$(resolve_rank)"; then
    die "this machine owns none of $NODE0_IP $NODE1_IP $NODE2_IP $NODE3_IP" \
        "(saw: $(host_ipv4s | tr '\n' ' ')); pass --rank N or fix the IP table."
fi
case "$rank" in
    0 | 1 | 2 | 3) ;;
    *) die "node rank must be 0, 1, 2, or 3; got '$rank'." ;;
esac

launcher="2.4T-$rank.sh"
command -v docker >/dev/null 2>&1 || die "docker is not installed or not in PATH."

mkdir -p "$LOG_DIR"

# Two cron firings must never overlap: vLLM takes minutes to come up, and a
# second run would npu-cleaner the first one mid-load.  Non-blocking, so the
# late one gives up instead of queueing.
if ((! CHECK_ONLY)); then
    exec 9>"$LOG_DIR/.autostart.lock"
    if ! flock -n 9; then
        log "another autostart run is still in progress; skipping this tick."
        exit 0
    fi
fi

log "node rank $rank, container '$CONTAINER_NAME', launcher $SCRIPTS_DIR/$launcher"

# --- container ------------------------------------------------------------
if ! state="$(docker container inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)"; then
    die "container '$CONTAINER_NAME' does not exist. Create it once with" \
        "scripts/setup/create-container.sh; this script never runs 'docker run'" \
        "because that would skip the vLLM-Ascend install."
fi

if [[ "$state" != "true" ]]; then
    log "container is not running; starting it."
    docker start "$CONTAINER_NAME" >/dev/null
    deadline=$((SECONDS + CONTAINER_START_TIMEOUT))
    until [[ "$(docker container inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == "true" ]]; do
        if ((SECONDS >= deadline)); then
            die "container '$CONTAINER_NAME' did not reach running state within ${CONTAINER_START_TIMEOUT}s."
        fi
        sleep 1
    done
    log "container started."
else
    log "container is already running."
fi

if ! docker exec --user "$CONTAINER_USER" "$CONTAINER_NAME" \
    test -r "$SCRIPTS_DIR/$launcher"; then
    die "$SCRIPTS_DIR/$launcher is missing or unreadable inside the container." \
        "Set SCRIPTS_DIR if the checkout lives elsewhere (the repo this script" \
        "ships from is /home/hajimi/qwen3.8)."
fi

# --- ~/.bashrc check ------------------------------------------------------
# bash sources ~/.bashrc only for interactive shells, which is why the launch
# below uses `bash -ic` and not `bash -c` (never sources it) or `bash -lc`
# (sources the profile files instead).  --check proves that rather than
# assuming it, by diffing what the two shells actually end up with.
probe_env() {
    # $1: the bash flags to probe under.  Dumps the parts of the environment
    # ~/.bashrc is expected to move: working directory, PATH and proxy.
    docker exec --user "$CONTAINER_USER" "$CONTAINER_NAME" \
        bash "$1" 'printf "interactive=%s\nPWD=%s\nPATH=%s\nhttp_proxy=%s\n" \
            "$(if [[ $- == *i* ]]; then echo yes; else echo no; fi)" \
            "$PWD" "$PATH" "${http_proxy:-<unset>}"' 2>/dev/null
}

if ((CHECK_ONLY)); then
    plain="$(probe_env -c || true)"
    rc="$(probe_env -ic || true)"

    [[ -n "$rc" ]] || die "'bash -ic' produced no output inside '$CONTAINER_NAME'."

    log "bash -c  (~/.bashrc NOT sourced):"
    printf '%s\n' "$plain" | sed 's/^/    /'
    log "bash -ic (the shell this script launches the service with):"
    printf '%s\n' "$rc" | sed 's/^/    /'
    echo

    if ! printf '%s\n' "$rc" | grep -q '^interactive=yes'; then
        die "'bash -ic' did not yield an interactive shell, so ~/.bashrc is never sourced."
    fi
    if ! docker exec --user "$CONTAINER_USER" "$CONTAINER_NAME" bash -c 'test -s ~/.bashrc'; then
        die "~/.bashrc is missing or empty for user '$CONTAINER_USER' in '$CONTAINER_NAME'."
    fi
    if [[ "$plain" == "$rc" ]]; then
        printf 'RED: both shells ended up identical, so ~/.bashrc changed nothing.\n'
        printf '     It is being sourced (the shell is interactive and the file is\n'
        printf '     non-empty), but it sets nothing up. Read it before relying on it.\n'
        exit 1
    fi

    printf 'GREEN: container up, rank %s, %s readable, and `bash -ic` sources a\n' \
        "$rank" "$launcher"
    printf '       ~/.bashrc that does change the environment (compare the two dumps above).\n'
    exit 0
fi

# --- rendezvous ordering --------------------------------------------------
if [[ "$rank" != "0" ]]; then
    if ((STAGGER_SECONDS > 0)); then
        log "giving node 0 a ${STAGGER_SECONDS}s head start."
        sleep "$STAGGER_SECONDS"
    fi
    log "waiting up to ${WAIT_NODE0_SECONDS}s for node 0 at $NODE0_IP:$DP_RPC_PORT."
    node0_up=1
    deadline=$((SECONDS + WAIT_NODE0_SECONDS))
    until timeout 2 bash -c "exec 3<>/dev/tcp/$NODE0_IP/$DP_RPC_PORT" 2>/dev/null; do
        if ((SECONDS >= deadline)); then
            node0_up=0
            break
        fi
        sleep 5
    done
    if ((node0_up)); then
        log "node 0 is listening on $DP_RPC_PORT; proceeding."
    else
        log "WARNING: node 0 never answered on $DP_RPC_PORT within ${WAIT_NODE0_SECONDS}s."
        log "WARNING: starting anyway, since zmq retries the handshake. If this rank"
        log "WARNING: stalls, look at node 0 rather than at this machine."
    fi
fi

# --- launch ---------------------------------------------------------------
if [[ "$KILL_STALE" == "1" ]]; then
    log "clearing stale vllm processes inside the container."
    docker exec --user "$CONTAINER_USER" "$CONTAINER_NAME" \
        bash -c "pkill -9 -f 'vllm serve' || true" >/dev/null 2>&1 || true
    sleep 3
fi

log_file="$LOG_DIR/2.4T-node$rank-$(date '+%Y%m%d-%H%M%S').log"

# --detach so the server outlives this cron process.  Output is redirected
# INSIDE the container, because a detached exec discards its own stdout.
docker exec --detach --user "$CONTAINER_USER" "$CONTAINER_NAME" \
    bash -ic "cd '$SCRIPTS_DIR' && exec bash './$launcher' >>'$log_file' 2>&1"

find "$LOG_DIR" -maxdepth 1 -name '2.4T-node*.log' -mtime "+$LOG_KEEP_DAYS" -delete 2>/dev/null || true

log "launched $launcher in '$CONTAINER_NAME'."
printf 'GREEN: node %s launching; follow it with\n    tail -f %s\n' "$rank" "$log_file"
