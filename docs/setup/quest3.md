# Meta Quest 3 · setup record

The headset's exact state, so every measurement can be traced to it and every change can be undone. Update this file whenever anything on the headset changes, and log the reason in `notes/decisions.md`. Nothing on the headset is changed without approval.

## Device

| | |
|---|---|
| OS build (`ro.build.fingerprint`) | `oculus/eureka/eureka:14/UP1A.231005.007.A1/52433670048800520:user/abl_signing_keys:release,amss_signing_keys:release,release-keys` |
| Developer mode | on |
| Recorded on | 2026-09-27, run `20260927_A1_r001` |

## Baseline, before any change

Run `20260927_A1_r001`, 2026-09-27: 8.7 minutes after a reboot, idle in Home with no app open, connected by USB. The files are in `runs/20260927_A1_r001/raw/`: three package lists, `meminfo.txt`, `cpuinfo.txt` and `top.txt`. PowerShell saved them as UTF-16, not UTF-8.

- **CPU:** about 2% of capacity in use; no background process above about 1%.
- **Memory:** 7.6 GiB in total and 4.6 GiB free: 4.1 GiB available, plus 0.55 GiB in cached apps that Android drops on demand.
- **Used 3.0 GiB:** native OS services 0.8, Home shell 0.7, Android system server 0.2, VR runtime and boundary 0.2; the rest is mostly system UI, the Store and cached apps.
- **Packages:** 170 system, 8 user-installed, 1 already disabled (`com.meta.credentialsmanager`).
- **Unneeded apps that were running** (TV, People, Help Center, Avatar editor, Gallery, Media player, Guidebook): about 0.23 GiB, mostly cached.
- **To watch:** the Store ran as a foreground process, at about 180 MB.

## Changes

One row per change, in the order the changes were made. None: O5 was closed with no changes (D26), because the processes busy while our app runs are the XR system itself (run `20260928_A1_r007`).

| # | Date | What changed | Command or setting used | How to undo | Measured effect | Approved |
|---|---|---|---|---|---|---|

## Restoring

Undo changes in reverse order, using the "How to undo" column. A factory reset is the last resort: it erases everything on the headset.
