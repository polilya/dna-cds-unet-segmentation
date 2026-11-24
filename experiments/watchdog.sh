#!/bin/sh
# Keep the sweep alive: restart it if the process dies or its log stops changing.
# Failed runs (claimed but without results.json) are retried up to MAX_RETRIES times.
# Every action is logged to runs/watchdog.log; the script exits when no runs are left.
#
#   sh experiments/watchdog.sh                  # E2-E5
#   sh experiments/watchdog.sh --groups best    # arguments are passed to sweep.py
SWEEP_ARGS="$*"
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
LOG=runs/sweep_log.txt
WLOG=runs/watchdog.log
STALL_MIN=60  # a full test evaluation (gated, ungated, plain) writes nothing for ~30 min
MAX_RETRIES=2
say() { echo "$(date '+%Y-%m-%d %H:%M') $*" | tee -a "$WLOG"; }

# The sweep needs the project's environment (numpy 2); a shell without `conda activate` would run the base Python
# and every run would fail at once. Put the env's bin first on PATH if the current python is not it.
ENV_NAME=${CDSSEG_ENV:-cdsseg}
if ! python -c "import numpy, torch; numpy.trapezoid" 2>/dev/null; then
    env_bin="$(conda info --base 2>/dev/null)/envs/$ENV_NAME/bin"
    if [ -x "$env_bin/python" ]; then
        PATH="$env_bin:$PATH"; export PATH
        say "using conda env $ENV_NAME ($env_bin/python)"
    else
        say "python here lacks numpy>=2/torch and no conda env $ENV_NAME was found: run 'conda activate $ENV_NAME' first"
        exit 1
    fi
fi

running() { pgrep -f "^python experiments/sweep.py" >/dev/null; }  # only the sweep itself, not shells that mention it

release_stale_claims() {
    for d in runs/*/; do
        [ -f "$d/claimed" ] && [ ! -f "$d/results.json" ] || continue
        [ -f "$d/deferred" ] || [ -f "$d/paused_by_battery" ] && continue  # held back on purpose, not failed
        n=$(cat "$d/retries" 2>/dev/null || echo 0)
        if [ "$n" -lt "$MAX_RETRIES" ]; then
            echo $((n + 1)) > "$d/retries"
            rm -f "$d/claimed"
            say "retry $((n + 1))/$MAX_RETRIES for $(basename "$d")"
        fi
    done
}

start() {
    release_stale_claims
    todo=$(python experiments/sweep.py $SWEEP_ARGS --dry-run 2>/dev/null)
    left=$(echo "$todo" | awk '/runs to do/{print $1}')
    # count only this group's runs that are claimed but unfinished (failed for good, or held back with
    # `deferred` unless the sweep runs with --take-deferred)
    claimed_left=0
    for r in $(echo "$todo" | awk 'NR > 1 {print $1}'); do
        [ -f "runs/$r/claimed" ] || continue
        case "$SWEEP_ARGS" in *--take-deferred*) [ -f "runs/$r/deferred" ] && continue ;; esac
        claimed_left=$((claimed_left + 1))
    done
    if [ "${left:-0}" -le "$claimed_left" ]; then
        say "DONE: no runs left (${claimed_left} gave up after retries)"
        exit 0
    fi
    say "starting sweep (${left} runs to do)"
    nohup caffeinate -i python experiments/sweep.py $SWEEP_ARGS >> "$LOG" 2>&1 &
    sleep 30
}

say "watchdog started"
while true; do
    if ! running; then
        say "sweep not running"
        start
    else
        age=$(( ( $(date +%s) - $(stat -f %m "$LOG") ) / 60 ))
        if [ "$age" -ge "$STALL_MIN" ]; then
            say "STALL: log unchanged for ${age} min, restarting"
            pkill -f "^python experiments/sweep.py"
            sleep 10
            start
        fi
    fi
    sleep 60
done
