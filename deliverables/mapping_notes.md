# Part 3 — Mapping Notes

**Time spent:**
6 Hours
**AI tools used:**
Claude Code.

## Summary

I build the map one frame at a time. Every detection is lifted to a 3D estimate of the object's centre. It then either joins the nearest same-class instance (if it falls within that class's size radius) or starts a new one. At the end I keep only instances with enough support. It mostly works: on scene_b it finds 78% of the surveyed objects with purity 0.96. It only half works on scene_a (52% recall), and both scenes still have false instances (36 and 12), mostly detector noise. Fragmentation is 1.2 (a) and 1.8 (b), where 1.0 would be perfect.

## Result (`uv run python score.py`, dev scenes)

| Scene | Instances | Recall | Fragmentation | False instances | ID switches | Observation purity |
|-------|-----------|--------|---------------|-----------------|-------------|--------------------|
| a | 55 (31 boxes) | 0.52 | 1.19 | 36 | 24 | 0.91 |
| b | 44 (23 boxes) | 0.78 | 1.78 | 12 | 17 | 0.96 |
| c (test) | 30 | – | – | – | – | – |
| d (test) | 57 | – | – | – | – | – |


## How detections become instances

Detections come from the same lifting as Part 2 (pixel-aligned depth, near-surface point, world frame via the pose, using the fixed scene_a and scene_c depth). Each frame is cleaned first: drop scores below 0.15, class-wise 2D NMS, and merge duplicate boxes on one object in 3D.

Each instance keeps a running centre, a position weight (computed as the cumulative detection score for each instance), the list of `[frame, detection_index]` observations, and a Beta score (alpha and beta) for its confidence.

### Association

1. **Push to the centre.** A lifted point is on the object's near surface, so I move it one class radius further along the viewing ray (radii are rough per-class half-sizes, like 0.4 m for a chair and 1.0 m for a sofa). This makes views from opposite sides agree.
2. **Collapse duplicate boxes.** The detector often puts several boxes on one object. Within a frame, same-class detections whose centres are closer than one class radius count as one object: the best-scoring one is used, and the others' detections are kept as observations of the same instance.
3. **Match within a gate.** A detection can join a same-label instance whose centre is within 1.5× the class radius. Within a frame, matches are one-to-one, closest pairs first.
4. **Update on a match.** The centre becomes a score-weighted average, the position weight grows by the score, and alpha/beta get `+score` and `+(1 − score)`.
5. **Otherwise start a new instance.** The confidence is alpha / (alpha + beta).

### Merging and splitting

There is no splitting. After all frames, I merge same-label instances whose centres ended up within the gate, closest pair first. Finally I report an instance only if it was seen in at least 3 frames, has confidence ≥ 0.3, and its detector scores add up to at least 5.


### What carries identity when geometry can't

**What I used**
- **Class label.** A detection can only join an instance with the same label, so a chair never merges with a stool or a table. The cost is that one appliance detected as both "oven" and "stove" becomes two instances (see failure modes).
- **Size prior.** Each class has a rough radius. It sets the gate (how far apart two views of one object can be, 1.5× the radius), it moves each lifted surface point one radius further along the viewing ray towards the object's centre so opposite views of a large object land together, and it defines when two same-frame boxes count as duplicates (closer than one radius).
- **Detector score.** It weights the running centre and feeds the confidence and the support filter, so weak detections count for less.

**What I tried and dropped: "two detections in one frame are different objects".** It's tempting, but the detector's redundant boxes break it.

**What I could have used, and didn't**
- **Appearance.** Visual feature embeddings of the crop. This is the strongest cue, since it doesn't depend on where the point lands, and it's what I'd add first for neighbouring same-class objects and for drift, where the same object shows up in two places.
- **Instance segmentation.** This can help separete objects instances directly from the 2D observations.
- **Re-projection.** Project each instance into the current frame and compare it with the detection boxes. This would help with visibility and occlusion, and could attach detections that had no usable depth.

## Remaining failure modes

| Failure| Example (scene, frames) | Cause | Fix I'd try |
|---------|-------------------------|-------|-------------|
| Remaining fragmentation | Cabinets: 22 instances for 11 boxes (a) and 18 for 13 (b). scene_a refrigerator: 7 instances for one box | Centre errors bigger than the gate; the cabinet radius is small (0.30 m) and a run of cabinets gives several detections along it | Estimate the size of the object, not only the center|
| One object, two labels | Oven and stove detections on one appliance (by design, not measured) | Instances are per label and never merge across labels | Keep a class score per instance and merge across labels when centres coincide |

## If this were design-only

I would address this mapping problem using a filtering-based approach. In particular, I would extend the object representation to include not only its position but also its size. I would then maintain a Kalman filter for each object instance to probabilistically fuse subsequent observations. For more accurate object-size estimation from individual observations, I would use semantic segmentation, since 2D bounding boxes provide limited information about an object's actual spatial extent.

By maintaining probabilistic estimates of both object center and size, we could also project each object into the image space and use the resulting projection to improve data association with new observations

## Reproducing

```bash
uv run python fetch.py
uv run python scripts/fix_scene_a_sync.py   # builds data/scene_a_fixed
uv run python scripts/fix_scene_c_sync.py   # builds data/scene_c_fixed
uv run python run_part3.py --eval           # writes instances.json, scores the dev scenes
uv run python score.py
```
