#!/bin/bash
# The animate pod's start command: pipeline/runpod_pod.py passes this file's text as `bash -c` (so the pod always runs the
# repo's version). It starts the template's /start.sh (ComfyUI on :8188) in the background, then deletes this pod once
# ComfyUI has had nothing to do for $IDLE_SECONDS, so a pod never outlives a studio that died or forgot it.
# Idle = /queue empty and the newest /history entry unchanged. Until ComfyUI has answered once (the boot) the limit is at least
# BOOT_SECONDS, so a short IDLE_SECONDS can't kill a pod that is still starting, yet one that never comes up still goes away;
# a ComfyUI that stops answering later counts as idle.
# Deleting uses what RunPod puts in every pod: runpodctl with a pod-scoped key (v2 syntax, then v1), else the REST API with
# that key; if deleting is refused it stops the pod (a stopped pod bills no GPU; the next ensure_up() terminates it).
/start.sh &
last="" since=$(date +%s) up=""
idle=${IDLE_SECONDS:-120} boot=${BOOT_SECONDS:-600}
while sleep 30; do
    busy=$(curl -sf --max-time 10 localhost:8188/queue \
        | python3 -c 'import json, sys; q = json.load(sys.stdin); print(len(q["queue_running"]) + len(q["queue_pending"]))' 2>/dev/null)
    [ -n "$busy" ] && [ -z "$up" ] && up=1 since=$(date +%s)  # ComfyUI answered: the idle clock starts now
    newest=$(curl -sf --max-time 10 'localhost:8188/history?max_items=1' | md5sum)
    limit=$idle; [ -z "$up" ] && (( boot > idle )) && limit=$boot
    if [ "${busy:-0}" != 0 ] || [ "$newest" != "$last" ]; then
        last=$newest since=$(date +%s)
    elif (( $(date +%s) - since >= limit )); then
        echo "watchdog: idle for $(( limit / 60 )) min${up:+ after work}, deleting pod $RUNPOD_POD_ID"
        runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID" \
            || curl -sf -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" \
            || runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID" \
            || echo "watchdog: could not delete or stop the pod, trying again in 30 s"
    fi
done
