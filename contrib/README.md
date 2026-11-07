# contrib

- `systemd/`: user units. Copy them to `~/.config/systemd/user/`, change `%h/comfy-studio` to where you cloned the repo, then
  `systemctl --user daemon-reload && systemctl --user enable --now comfy-studio comfy-monitor comfyui-restart.timer`.
  `comfyui-restart` restarts the local ComfyUI unit named by `COMFYUI_SERVICE` at 04:30 when it is idle.
- `kde-widgets/`: KDE Plasma 6 widgets for `dashboard.py` (local job, Comfy Cloud job, RunPod pod). Install one with
  `kpackagetool6 -t Plasma/Applet -i kde-widgets/org.comfycloud.status`.
