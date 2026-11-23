#!/bin/sh
# Stop a running sweep from starting new runs when the battery drops below MIN percent.
# The run in progress is never interrupted: the guard stops the watchdog (so nothing restarts the
# sweep) and claims every run of the group that has not started yet, so the sweep skips them and
# exits after the current run. Claims it placed are marked with a `paused_by_battery` file.
#
#   sh experiments/battery_guard.sh v2            # watch the v2 group
#   sh experiments/battery_guard.sh v2 --resume   # remove the guard's claims and restart the watchdog
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
GROUP=$1
MIN=15
say() { echo "$(date '+%Y-%m-%d %H:%M') battery guard: $*" >> runs/watchdog.log; }

if [ "$2" = "--resume" ]; then
    for f in runs/*/paused_by_battery; do
        [ -f "$f" ] || continue
        d=$(dirname "$f")
        rm -f "$f" "$d/claimed"
        rmdir "$d" 2>/dev/null
    done
    say "resumed group $GROUP"
    nohup sh experiments/watchdog.sh --groups "$GROUP" >> runs/watchdog_"$GROUP".out 2>&1 &
    exit 0
fi

while true; do
    charge=$(pmset -g batt | grep -Eo '[0-9]+%' | head -1 | tr -d '%')
    if [ -n "$charge" ] && [ "$charge" -lt "$MIN" ]; then
        pkill -f "watchdog.sh --groups $GROUP"
        python experiments/sweep.py --groups "$GROUP" --dry-run | awk 'NR > 1 {print $1}' | while read -r run; do
            d="runs/$run"
            [ -f "$d/claimed" ] && continue
            mkdir -p "$d" && touch "$d/claimed" "$d/paused_by_battery"
        done
        say "battery at ${charge}%: watchdog stopped, unstarted runs of $GROUP paused; the current run finishes"
        exit 0
    fi
    sleep 60
done
