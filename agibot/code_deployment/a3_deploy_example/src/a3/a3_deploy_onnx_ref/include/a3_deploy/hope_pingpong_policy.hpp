// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#pragma once

// C++ deployment primitives for the HOPE ping-pong policy contract:
//
//   observation[1,111] -> raw_action[1,31]
//
// These are deliberately separate from the default A3 tokenizer policy path
// (obs_dict[1,1570] -> action[1,29]).  They are pure data transforms that can be
// called from A3PolicyDriver::CommandFn after RobotIOBackend and mocap/planner
// data have been synchronized.

#include "robot_io/robot_io_backend.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace a3_deploy {

inline constexpr std::size_t kHopeObsDim = 111;
inline constexpr std::size_t kHopeActionDim = 31;
inline constexpr double kHopeControlHz = 50.0;
inline constexpr int kHopeHeadYawIndex = 3;
inline constexpr int kHopeHeadPitchIndex = 4;

extern const std::array<const char*, kHopeActionDim> kHopeJointNames;

struct HopeRobotState {
  std::array<double, 3> base_pos_w{0.0, 0.0, 0.0};
  std::array<double, 4> base_quat_wxyz{1.0, 0.0, 0.0, 0.0};
  std::array<double, 3> base_ang_vel_b{0.0, 0.0, 0.0};
  std::array<double, kHopeActionDim> q{};
  std::array<double, kHopeActionDim> dq{};
};

struct HopeObsTarget {
  std::array<double, 3> pos_w{0.0, 0.0, 0.0};
  std::array<double, 3> vel_w{0.0, 0.0, 0.0};
  double time_to_strike = 1.0;
  double swing_side = 1.0;
};

// Copy the 31-DOF RobotIOBackend state into the HOPE policy state.  The base
// position comes from external localization/mocap (for OptiTrack, /P1/pose after
// the relay); orientation and gyro come from the primary robot IMU by default.
HopeRobotState MakeHopeRobotStateFromBackend(
    const robot_io::RobotState& state,
    const std::array<double, 3>& base_pos_w) noexcept;

std::array<float, kHopeObsDim> BuildHopeObservation(
    const HopeRobotState& state,
    const HopeObsTarget& target,
    const std::array<double, kHopeActionDim>& last_applied_action,
    const std::array<double, kHopeActionDim>& default_q,
    const std::array<double, 2>& station_xy) noexcept;

class HopeActionAdapter {
 public:
  static HopeActionAdapter FromYaml(const std::string& path);

  const std::array<double, kHopeActionDim>& default_q() const noexcept {
    return default_q_;
  }
  const std::array<double, kHopeActionDim>& action_scale() const noexcept {
    return action_scale_;
  }
  const std::array<double, kHopeActionDim>& clamp_lower() const noexcept {
    return clamp_lower_;
  }
  const std::array<double, kHopeActionDim>& clamp_upper() const noexcept {
    return clamp_upper_;
  }

  std::array<double, kHopeActionDim> AppliedAction(
      const std::array<float, kHopeActionDim>& raw_action,
      bool passive_neck = true) const noexcept;

  std::array<double, kHopeActionDim> DecodeAppliedAction(
      const std::array<double, kHopeActionDim>& applied_action,
      bool passive_neck = true) const noexcept;

  void DecodeRawAction(
      const std::array<float, kHopeActionDim>& raw_action,
      std::array<double, kHopeActionDim>& applied_action_out,
      std::array<double, kHopeActionDim>& q_des_out,
      bool passive_neck = true) const noexcept;

 private:
  std::array<double, kHopeActionDim> default_q_{};
  std::array<double, kHopeActionDim> action_scale_{};
  std::array<double, kHopeActionDim> clamp_lower_{};
  std::array<double, kHopeActionDim> clamp_upper_{};
};

// Fill a 31-DOF RobotCommand in backend layout order.  dq_des and tau_ff are
// zeroed; kp/kd are supplied by the caller so sim, probe, and real hardware can
// use different gains without changing policy math.
void FillHopeRobotCommand(
    const std::array<double, kHopeActionDim>& q_des,
    const std::array<double, kHopeActionDim>& kp,
    const std::array<double, kHopeActionDim>& kd,
    robot_io::RobotCommand& command_out);

bool HopeJointOrderMatchesA3Layout31();

}  // namespace a3_deploy
