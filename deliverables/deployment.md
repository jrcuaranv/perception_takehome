# Part 4 — Deploy and Measure

**Time spent:**
2 hours
**AI tools used:**
ChatGPT for grammar fixes
## 4a — On the robot

### What runs, at what rate, inside 150 ms
I would first benchmark the detector on the embedded GPU and optimize it with TensorRT, potentially using reduced precision (FP16/INT8). If it still cannot meet the 150 ms budget, I would use a lighter model and/or run detection at a lower rate while tracking objects between inference frames. At 0.5 m/s, the inference rate should be high enough that the robot does not move significantly between perception updates.
### What I give up
The main tradeoff would be some detection accuracy or responsiveness in exchange for meeting the real-time constraint. A lighter model, reduced input resolution, quantization, or lower inference rate could particularly affect small, distant, or partially occluded objects.
### 300 labelled frames in a new home

| Question | Answer |
|----------|--------|
| What I label | I would prioritize the object classes where the detector underperforms, including representative positive examples and difficult cases such as occlusion, poor illumination, unusual viewpoints, and long distances.  |
| How I choose the frames | I would select diverse and difficult frames rather than consecutive video frames, which are likely to be redundant. I would also include data from other homes when possible to reduce overfitting to a single environment. |
| How I know it worked before the robot goes back | I would keep a held-out subset from the target home and compare the adapted model against the original using per-class precision, recall, and task-relevant errors. I would only deploy the new model if it improves the target classes without significantly degrading performance on the others. |

## 4b — `robot_eval_results.json`

### What I see
The same grounding stack performs substantially worse on Rena-03 than Rena-02. Goal success also decreases sharply with distance, despite a much smaller decrease in detector confidence. Finally, although offline detector recall improved by 12 points, end-to-end goal success dropped by 6 points.
### Ranked explanations

**1.**
Rena-03 likely has a sensor calibration or geometric grounding issue.
- Reasoning: Despite running the same model and being evaluated in the same homes, Rena-03 has substantially lower goal success than Rena-02 (0.41 vs. 0.66) and much larger goal error (0.44 m vs. 0.21 m median; 1.10 m vs. 0.48 m P90). This suggests a unit-specific issue downstream of detection, such as incorrect RGB-depth extrinsics, depth scale, or camera-to-robot calibration.
- Confirms or eliminates it: Place a surveyed target at known 3D locations and compare the reconstructed world coordinates from both robots against ground truth.

**2.**
3D grounding accuracy degrades strongly with distance.
- Reasoning: Goal success drops from 0.88 below 1.5 m to only 0.12 beyond 3.5 m, while detector confidence decreases much more gradually (0.61 to 0.49). This suggests that depth uncertainty, RGB-depth misalignment, or other geometric errors increase with distance and are a major bottleneck beyond detector confidence alone.
- Confirms or eliminates it: Measure 3D object-position error against surveyed ground truth at different distances while using ground-truth 2D boxes, isolating geometric grounding from detection.

**3.**
Higher detector recall may have come at the cost of precision or instance discrimination.
- Reasoning: Offline detector recall improved by 12 points, while goal success dropped by 6 points. Higher recall alone does not guarantee better task performance, since additional false positives or incorrect class/instance assignments can produce incorrect goals. The large chair-stool confusions further support this possibility.
- Confirms or eliminates it: Compute the precision-recall curve and confusion matrix for the new and previous detectors on the same labeled evaluation set, and compare them at the operating confidence threshold.

### What I'd instrument next, in order

1. Per-unit geometric grounding error, including depth measurements, RGB-depth alignment, and reconstructed world coordinates.
2. Grounding error as a function of object distance.
3. Detector TP/FP/FN, confidence, predicted class, and selected instance for each grounding attempt.

## 4c — No ground truth

### How I'd know the map is right after a week
 Without ground truth, I would monitor the map through temporal and multi-view consistency. Assuming reliable robot localization, new observations can be projected into the world frame and compared against existing object instances. A healthy map should show stable object positions and identities across repeated observations from different viewpoints, with few unexplained new or disappearing instances. Goal success can provide an additional end-to-end indicator, although failures should be separated into detection, localization, mapping, and navigation errors rather than attributed to the map alone.

### How I'd know the goals are good

I would monitor whether the robot consistently reaches a position close enough to the target object to complete the intended task. Repeated failures or large corrections near the target would indicate inaccurate goal estimation, while consistent success across different objects, viewpoints, and distances would provide evidence that the generated goals are reliable.

### A second camera on the wrist

- Where its calibration enters: For a wrist camera, its extrinsic calibration is required to transform observations from the camera frame, through the end-effector and robot frames, into the world frame. 
- What breaks first if it's wrong: If this calibration is wrong, objects observed by the wrist camera will be incorrectly localized. For manipulation, this would first appear as inaccurate object positions relative to the end effector, potentially causing grasping or interaction failures. If the wrist camera is also used for mapping, incorrect calibration could produce shifted or duplicate object instances and progressively degrade the map.

## Feedback (optional)

### 1. What I would have asked you before starting

- I would have clarified whether the language-grounding task in Part 2 should use only information available in the specified frame or whether observations accumulated across frames can be used. This affects whether grounding should be treated as a per-frame problem or as a query over the persistent object map.
- Whether external pretrained models or libraries are allowed.

### 2. The biggest thing this exercise fails to test
- Real-time deployment on actual hardware, debugging with physical sensors, working with ROS, training/fine-tuning perception models, and designing experiments on a real robot.