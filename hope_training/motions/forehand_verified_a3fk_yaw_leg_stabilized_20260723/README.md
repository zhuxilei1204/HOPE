# Forehand Verified Motion Set 2026-07-23

This set is rebuilt from `forehand_user_corrected_a3fk_yaw_leg_stabilized_20260722` after the policy
preview showed a backhand-like motion.

Selection rule:
- keep only clips whose racket target center is on the A3 forehand side (`y < 0`);
- require the right wrist to stay on the A3 forehand side around strike;
- require strike-window right-wrist motion to move from the right side toward center/left (`v_y > 0`).

Kept clips:
- `nk_userhit_c02_f046_30fps_agibot_a3_a3fk_aligned.npz`
- `nk_userhit_c05_f123_30fps_agibot_a3_a3fk_aligned.npz`
- `nk_userhit_c06_f148_30fps_agibot_a3_a3fk_aligned.npz`
- `nk_userhit_c07_f171_30fps_agibot_a3_a3fk_aligned.npz`
- `nk_userhit_c09_f219_30fps_agibot_a3_a3fk_aligned.npz`

Files:
- `manifest_targetpos.tsv`: training manifest used by `scripts/train.py`.
- `forehand_qc_selection.tsv`: keep/exclude decision table for all 12 candidate clips.
- `forehand_qc_topdown.png`: top-down right-wrist and target-box QC plot.
- `forehand_reference_skeleton_preview.mp4`: low-cost reference-motion preview.

Rejected clips were not deleted from the old directory; they are only excluded from this verified
manifest.
