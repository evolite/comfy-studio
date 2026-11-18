# contrib

- `systemd/`: user units. Copy them to `~/.config/systemd/user/`, change `%h/comfy-studio` to where you cloned the repo, then
  `systemctl --user daemon-reload && systemctl --user enable --now comfy-studio comfy-queue comfy-monitor comfyui-restart.timer`.
  `comfy-queue` runs the jobs the web UI submits (`webui/server/queue_daemon.py`); restarting `comfy-studio` leaves them running.
  `comfyui-restart` restarts the local ComfyUI unit named by `COMFYUI_SERVICE` at 04:30 when it is idle.
- `kde-widgets/`: KDE Plasma 6 widgets for `dashboard.py` (local job, Comfy Cloud job, RunPod pod). Install one with
  `kpackagetool6 -t Plasma/Applet -i kde-widgets/org.comfycloud.status`.
