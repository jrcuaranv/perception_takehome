# Part 1 — Calibration Report

**Time spent:**
5 hours
**AI tools used:**
Claude Code
## Summary

## 1. What's wrong

Six issues. Two are sync faults specific to one scene each (a contiguous mis-pairing in scene_a, a frozen depth stream in scene_c); one is the ordinary single-frame slip present in every scene, below the level worth acting on; one is a scene_b intrinsics inconsistency that's confidently detected but not fixable from this data alone; one is a world-frame axis convention that differs from the README (harmless once known); and one is that the recorded colour↔depth registration (intrinsics and `T_color_depth`) is measurably worse than assuming pixel alignment in scenes b and c. 

| # | Scene | Issue | Magnitude | Frames affected | How found | Confidence |
|---|-------|-------|-----------|-----------------|-----------|------------|
| 1 | a | Depth paired 3 frames (0.3 s) late | depth 0.3 s stale; edge corr 0.06 vs 0.32 on clean frames | 135–372 (238, contiguous) | Sync offset plot (`sync_offset.png`); shift sweep; RGB/depth edge correlation | High |
| 2 | c | Depth frozen: one depth image reused | 48 frames with byte-identical depth; stamp gap grows to 4.8 s | 58–105 (48 of 756) | Sync offset plot (`sync_offset.png`), 54 duplicate depth stamps; pixel comparison | High |
| 3 | all | Isolated 1-frame depth slips (0.1 s) | ≈2 cm at median speed, ≈6–8 cm at p99 | a 9, b 16, c ≈16, d 23 (outside the two blocks) | Sync offset plot (`sync_offset.png`) | High that they exist; low impact |
| 4 | b | Colour `cx, cy` inconsistent with depth `cx, cy` | recorded values imply a (−12.8, −14.1) px depth→colour shift; images align at (0, 0) | all frames | RGB/depth edge correlation swept over 2D shift (pose-free): corr 0.327 at (0,0) vs 0.064 at the calib-predicted shift | High that one or both values are wrong; detected, not fixable from this data |
| 5 | all | World frame is not y-up as documented; measured "up" is dominated by +x | floor/gravity direction ≈89° from the documented y-axis, ≈14° from +x| all frames | Trajectory plot (`trajectories.png`) + RANSAC floor-plane fit from unprojected points | High; convention only, not an error worth fixing |
| 6 | b, c (a, d marginal) | Recorded colour↔depth registration (intrinsics + `T_color_depth`) is worse than assuming pixel alignment | Warping depth into colour with the recorded calibration drops edge correlation 0.326 → 0.065 (b) and 0.287 → 0.055 (c); a: 0.258 → 0.212, d: 0.294 → 0.267 | all frames | RGB/depth edge correlation, raw vs warped, on identical valid pixels (`check_aligned_depth.py`) | High that the recorded values are wrong for b and c; a and d differences are small and not established as errors |

### Issue 1 — scene_a: 3-frame depth lag

Frames 135–372 pair each colour frame with a depth frame captured 3 frames earlier (offset −0.3 s in every one of the 238 frames). Depth is never ahead of colour in any scene. Colour frame *i*'s depth is recorded in row *i*+3 for frames 135–369; for 370–372 no row holds it. The log's claim was checked against the data, not taken on trust: sweeping the depth-row shift k from −6 to +6, RGB/depth edge correlation (no poses involved) peaks at k=+3 (0.32 vs ≤0.15 at any other k; 0.06 as recorded), in both slow- and fast-rotating frames. Reprojection residual also minimises at k=3 (0.028 vs 0.082 m at k=0), but it is rotation-sensitive (its floor rises 0.010 → 0.040 m from low to high rotation, and k=3 vs k=4 is nearly tied at high rotation), so edge correlation is the primary evidence. Worth acting on: at the median 0.24 m/s and typical pan rates, a 0.3 s lag misaligns depth against colour by several cm and several degrees.

### Issue 2 — scene_c: frozen depth

From frame 58 to 105 the recorder kept re-pairing the depth image of frame 57 while colour advanced (the offset ramps from −0.1 s to −4.8 s, then snaps back to 0). All 48 depth images are byte-identical (max pixel difference 0; 1 distinct depth stamp) while consecutive normal frames differ in ≈99% of pixels. Frame 57 is legitimate (its depth stamp is its own colour stamp). No depth row holds these frames' true stamps, so the correct depth was never recorded. Worth acting on: any 3D lift from these frames lands on the wrong geometry.

### Issue 3 — isolated single-frame slips

Every scene has short runs (1–10 frames) where depth is 0.1 s behind colour, and duplicated depth stamps from the same events (scene_c also has 6 two-frame repeats outside the frozen block). The README says every recorder drops frames. Not worth acting on: 0.1 s is ≈2 cm at the median speed and ≈6–8 cm at p99 (max 0.8 m/s), comparable to depth noise (≈1–3 cm residual on clean frames). scene_b and scene_d have nothing worse than this and are the "accept and ship" case for sync.

### Issue 4 — scene_b: colour principal point disagrees with depth

`calib.json` gives scene_b a colour `(cx, cy)` of (115.09, 81.82), 13–14 px off image centre; every other scene is within ~1 px of centre, and the depth `(cx, cy)` (127.933, 95.933, identical in all four scenes) implies depth pixels should land ~13 px away from their matching colour pixels given the near-identity `T_color_depth`. That's testable without poses: sweep a 2D shift of the depth image against RGB and compare gradient-edge correlation. The images peak at shift (0, 0) — corr 0.327 — not at the calibration-predicted (−12.8, −14.1), where corr drops to 0.064. So the two recorded principal points can't both be right.

What this doesn't tell us is which one is wrong, or what the correct absolute value is — the edge test only constrains the *relative* colour-depth offset. Recovering the true colour `cx, cy` from this dataset isn't reliable: self-calibration via bundle adjustment over the poses is possible in principle, but can be poorly constrained or degenerate depending on camera motion and scene geometry. This is a detect-but-can't-fix finding (§3).

### Issue 5 — world frame is x-up, not y-up as documented

The README states "the world frame is the robot's map frame, y up." Plotting each scene's trajectory with a camera-axis triad every 16th frame alongside the world-frame axes (`trajectories.png`) shows the camera triads consistently tilted relative to the labelled y-axis; RANSAC-fitting the dominant plane from unprojected, world-transformed depth points confirms it quantitatively. For the given scenes, the floor normal is ≈(0.97, 0–0.02, 0.22–0.27) — 89° from the documented +y axis and only ≈14° from +x. So "up" is not the y-axis; it's closest to +x, tilted a further ~14° toward +z.

This isn't a defect in the calibration — it's a labelling mismatch between the README and the data, and it costs nothing to work around once known. It matters only because any Part 2/3 code that assumes `+y` is vertical (e.g. "on the floor" reasoning, height filtering, up-vector priors) will silently misbehave; the fix is to use the measured direction (or refit it per scene) instead of hardcoding an axis.

### Issue 6 — recorded colour↔depth registration is worse than pixel alignment

`calib.json` gives depth intrinsics, colour intrinsics and `T_color_depth`, so the principled way to get depth at a colour pixel is to warp the depth image into the colour camera (`aligned_depth_to_color()` in `perception/geometry.py`: raw units × `scale_m`, unproject with the depth intrinsics, apply `T_color_depth`, project with the colour intrinsics, z-buffer). I tested that against simply treating colour pixel (u, v) and depth pixel (u, v) as the same point, using RGB/depth edge correlation (pose-free) on 30 well-paired frames per scene. Both columns are computed on the same pixels (those where the warped depth is valid, eroded 5×5, so holes and empty borders don't bias either):

| scene | raw depth, pixel-aligned | warped with recorded calibration | valid pixels |
|---|---|---|---|
| a | 0.258 | 0.212 | 0.95 |
| b | 0.326 | **0.065** | 0.86 |
| c | 0.287 | **0.055** | 0.86 |
| d | 0.294 | 0.267 | 0.99 |

If the recorded calibration were right, warping should help or at least not hurt. In b and c it destroys the alignment, so the recorded values disagree with the images. The likely sources differ by scene, from the numbers already in this report: in b the colour principal point implies a (−12.8, −14.1) px shift (Issue 4, an intrinsics error); in c the two principal points agree to ≈1 px, so the loss must come from the 1.79° `T_color_depth` rotation, which is ≈6–7 px at f ≈ 213 (an extrinsics error, inferred from magnitude, not isolated by a separate sweep). In a and d the warp costs only 0.03–0.05; that could be small calibration error, or just resampling loss from the nearest-pixel forward warp, and I have not separated the two, so I do not call either scene wrong. The pixel-aligned assumption itself is supported by the shift sweep in Issue 4: correlation peaks at (0, 0) in all four scenes.

## 2. What I did about each issue

| # | Issue | Action (fixed / worked around / flagged) | Corrected value | Before → after on which measurement |
|---|-------|------------------------------------------|-----------------|-------------------------------------|
| 1 | scene_a lag | Fixed in `data/scene_a_fixed/` (`variant/scene_a/sync_fixed.json`): re-paired by timestamp, depth row *i*+3 for 135–369; frames 370–372 zero depth (none recorded) | shift k=+3 | Edge correlation 0.063 → 0.356 (clean control 0.316, unchanged); reprojection residual 0.070 → 0.028 m (rotation-sensitive, secondary) |
| 2 | scene_c frozen depth | Worked around in `data/scene_c_fixed/` (`variant/scene_c/sync_fixed.json`): depth zeroed for 58–105; RGB, poses and indices kept so detections still align | n/a (not recoverable) | 48 identical images → 48 explicit "no depth" frames; other 708 depth files byte-identical to original |
| 3 | isolated slips | Flagged, not fixed | n/a | not measured; below the noise floor |
| 4 | scene_b principal point | Flagged, not fixed (not recoverable from this data) | n/a | Edge-alignment peak at (0,0), corr 0.327 vs 0.064 at the calib-predicted shift — confirms the disagreement, doesn't resolve it |
| 5 | world frame not y-up | Flagged, not "fixed" — it's a convention, not an error. Downstream (Parts 2/3) uses the measured up direction, not the README's stated y-axis | up ≈ (0.97, 0, 0.24), i.e. mostly +x | Floor-normal RANSAC agrees across a, b, d to within ~1° (89° from +y, 14° from +x) |
| 6 | colour↔depth registration | Worked around: Parts 2/3 treat depth as pixel-aligned with colour (colour pixel (u, v) reads depth (u, v), unprojected with the depth intrinsics); the recorded colour intrinsics and `T_color_depth` are not used. `aligned_depth_to_color()` is kept for when a corrected calibration exists | n/a (corrected values not recoverable from this data) | Edge correlation, pixel-aligned vs warped with recorded calibration: b 0.326 vs 0.065, c 0.287 vs 0.055 (a, d: 0.258 vs 0.212, 0.294 vs 0.267) |

## 3. What I cannot verify with this data

| What | Why not | What one measurement or fixture would let me |
|------|---------|----------------------------------------------|
| Absolute metric scale | Depth scale and trajectory scale are degenerate against each other: every depth-vs-pose check here (e.g.reprojection residual) only tests whether the two *agree*, and multiplying depth by k while shrinking the trajectory by 1/k leaves every one of those checks unchanged. So a scene could be self-consistent and still uniformly wrong in metres, and nothing in `calib.json`, the poses or the images breaks that tie | One object of known length (a door, a ruler, a checkerboard square, a tape-measured wall segment) seen in the frame, measured once in world units and compared to its size in the point cloud/boxes |
| Camera trajectory accuracy (absolute position/orientation error, drift over the run) | No independent reference trajectory is provided | A handful of surveyed fiducial markers (or an external tracking rig / total station) at known world coordinates, revisited at multiple points during the recording, so position error can be measured directly instead of inferred from RGB-D self-consistency |

## 4. One capture-time check

One online check per frame, run as the rig records, would have caught every sync fault above (not the scene_b intrinsics issue — that needs a calibration-target capture, not a runtime check):

```
baseline = none          # per recording

for each frame i:
    gap = |depth_stamp[i] - colour_stamp[i]|
    flag if gap > 0.05 s                                    # not paired close in time

    flag if depth[i] == depth[i-1] and rgb[i] changed        # detects frozen depth, reused frame

    corr = edge_correlation(rgb[i], depth[i])                 # pose-free, captures color-depth misalignment
    baseline = corr if baseline is none else 0.95*baseline + 0.05*corr
    flag if corr < 0.5 * baseline                             # misregistered: lag, bad extrinsic/principal point

if any flag: quarantine the recording before it reaches a map
```

The first two are exact hard thresholds — they catch scene_a's and scene_c's faults on every affected frame. The third is the general-purpose one: no ground truth or poses needed, it degrades under lag, frozen depth, or a bad principal point/extrinsic, and its baseline is per-recording, so it never assumes any two rigs match.

## Method

- Sync: `scripts/plot_sync.py` (depth − colour stamp per frame), `scripts/check_depth_frozen.py` (hash raw depth PNGs), `scripts/find_sync_shift.py` (shift sweep), `scripts/check_rotation_bias.py` (sweep stratified by rotation), `scripts/fix_scene_a_sync.py`, `scripts/fix_scene_c_sync.py`, `scripts/verify_scene_a_fix.py`. Intrinsics/extrinsics: `scripts/check_principal_point.py`, `scripts/check_aligned_depth.py` (uses `aligned_depth_to_color()`). World frame: `scripts/plot_trajectories.py`. Plots in `deliverables/plots/`.
- Each scene is judged against its own data; nothing assumes the four units share a rig.

## What I didn't do

- Did not correct the isolated 1-frame slips (below noise) or frames 370–372 of scene_a (no depth recorded).
- Did not remove scene_c's frozen frames from the sequence; depth is zeroed so indices stay aligned.
