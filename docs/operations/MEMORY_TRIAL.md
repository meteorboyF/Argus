# ARGUS NVMe swap trial (2026-10-21)

The user requested virtual memory / NVMe swap after two llama-server CUDA
allocation failures. This is an experiment, not a T1 acceptance workaround.
The rig has 7607 MiB DRAM, a 476.9 GiB NVMe root device (ext4, 399 GiB free
at inspection), and six zram swap devices totalling 3803 MiB at priority 5.
`reports/memory-before-nvme-swap-2026-10-21.json` records the starting state.

NVIDIA documents that swapping pageable CPU memory can free physical DRAM
for the integrated GPU; `cudaMemGetInfo` does not count all reclaimable swap
capacity. It does not make NVMe equivalent to physical RAM:
https://docs.nvidia.com/cuda/archive/12.6.3/cuda-for-tegra-appnote/index.html#estimating-total-allocatable-device-memory-on-an-integrated-gpu-device

## Preparation and activation

Prepare an 8 GiB file at `/opt/argus/swapfile.nvme-8g`, mode 0600, with real
allocated blocks (`dd` with exclusive creation), then `mkswap`. Do not
reformat an existing path. Check its size, mode, `blkid -p`, and `filefrag`.
The coding agent can do this unprivileged because `/opt/argus` belongs to argus.
Preparation must happen outside the measured pre-swap soak.

Fardeen must type the activation command. This requirement is from
JETSON_PROMPT_04 section 2 rule 4: sudo is a specific system operation run by
Fardeen; never request or store a password.

```bash
sudo chown root:root /opt/argus/swapfile.nvme-8g && sudo swapon --priority 0 /opt/argus/swapfile.nvme-8g
swapon --show
```

Priority 0 leaves existing priority-5 zram preferred. Keep swappiness at its
current value (60) for this trial, so only the extra backing capacity changes.
The first activation is temporary: do not edit `/etc/fstab` or disable zram.

## Compare

Keep the same llama batch/microbatch (256), model, camera roles, CPU caps,
render rate, workload, and desktop applications. Start with llama-server
healthy; do not benchmark against an absent server. Save post-activation
`swapon --show`, zram statistics, MemAvailable, `/proc/vmstat` and swappiness.

```bash
python3 scripts/profile_fast_loop.py --offscreen --phases combined --seconds 600 --out reports/soak-nvme-swap-2026-10-21.json
```

Compare against `reports/soak-automated-offscreen-2026-10-21.json`:

- all completed and still-open depth outages over 250 ms;
- fast tick median/p95 and source-age rejects;
- peak ARGUS + encoder + llama RSS, minimum MemAvailable;
- swap-in/out and major-fault deltas (system-wide, includes the desktop);
- maximum GR3D and `doctor.oc_events()` delta;
- real Gemma/Whisper/OCR operation counts, errors, and latency percentiles.

The automated workload uses synthetic speech and a busy synthetic page. It
renders and records without opening a window. It does not replace the guided
voice/reading soak with Fardeen, or calibrated hazard acceptance. Extra swap
never changes the <=5.5 GiB resident target or permits depth gaps / CPU fallback.

## Removal

Keep ARGUS stopped and enough memory headroom before asking Fardeen to disable
swap; moving occupied swap back into DRAM can otherwise exhaust memory.

```bash
sudo swapoff /opt/argus/swapfile.nvme-8g
```

Confirm the file is absent from `/proc/swaps` before removing it. No reboot
configuration was changed by this trial.
