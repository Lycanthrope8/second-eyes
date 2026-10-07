# Detector schedules (D89)

Cost-balanced slice schedules written by `python -m perception.detector schedule <profiling run>`: the scheduling
steps dispatched in each consecutive frame, with the identity they were made for (model hash, runtime, backend, the
runtime's layer count and layer-type hash), the per-frame GPU budget, the predicted frames and latency, and any step
that alone exceeds the budget. Push one to the headset's `files/schedules/` folder and it joins the steps-per-frame
choices on the left thumbstick; the app refuses a schedule that does not match the loaded model and runtime.

Only the schedule frozen after tuning is committed here (`notes/decisions.md` names it); candidates are working files.
