# Supplemental contact-aware A3 motions

This is the final Isaac-FK canonical output of the 2026-07-28 supplemental
motion conversion.

- `manifest.tsv`: all 11 converted single-hit clips, including review status.
- `manifest_train_ready.tsv`: the 5 clips that passed every automatic gate.
- `*.npz`: canonical 50 Hz, 31 joints, 14 tracked bodies.
- `*.yaml`: source frames, strike phase, retarget parameters, metrics, and
  MuJoCo-to-Isaac FK provenance.
- `isaac_fk_summary.yaml`: aggregate FK parity before body-array regeneration.

Training should consume `manifest_train_ready.tsv`, not every NPZ in this
directory. Review clips intentionally remain available for diagnosis but are
not approved motion priors.

Pipeline:

1. `contact_aware_retarget.py` solves foot contact, COM support, ground
   alignment, root/leg Jacobian IK, temporal filtering, and 50 Hz conversion.
2. `regenerate_motion_isaac_fk.py` keeps root/joints unchanged and regenerates
   tracked-body arrays using the Isaac training articulation.
3. `audit_motion_lower_body.py --contact-mode stored` independently checks the
   embedded support labels.

Detailed results:
`analysis/supplemental_motion_audit_20260728/CONTACT_AWARE_RETARGET_REPORT_ZH.md`.

