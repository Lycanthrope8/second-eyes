# Limitations

Running list, kept from day one. It feeds the Limitations sections of the papers. Add an entry as soon as you notice one.

| Date | Phase | Limitation | Evidence (run ID, media) | Affects |
|---|---|---|---|---|
| 2026-09-30 | A1.7d | OVR Metrics' GPU utilization covers the whole GPU, including the system's boundary and compositor, so it varies with where the wearer sits: 5 points between two model-off runs at the same clock | `20260929_A1_r017`, `20260929_A1_r020` | Paper 1 method; the agreement check (O17) |
| 2026-09-30 | A1.7d | OVR Metrics writes no rows while the app is frozen, so frame rates count those seconds as frameless (0 fps): an approximation of what the wearer sees | `20260929_A1_r019`, `20260930_A1_r021` | frame-rate results |
| 2026-09-30 | A1.7d | All headset measurements so far are plugged in (D22): no battery drain, and the heat includes charging | A1.6 onwards | Gate A's battery item; heat |
| 2026-09-30 | A1.7d | With the model loaded, the headset has 350–600 MB free, and Android pages the idle model out within minutes | `20260929_A1_r018`, `20260929_A1_r019` | memory headroom for speech and the detector |
