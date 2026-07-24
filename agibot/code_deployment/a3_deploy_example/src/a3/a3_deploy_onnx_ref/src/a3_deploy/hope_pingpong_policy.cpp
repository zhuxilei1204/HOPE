// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#include "a3_deploy/hope_pingpong_policy.hpp"

#include "robot_io/a3_layout_extra.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace a3_deploy {

const std::array<const char*, kHopeActionDim> kHopeJointNames = {{
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
}};

namespace {

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;

Vec3 Cross(const Vec3& a, const Vec3& b) noexcept {
  return {
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
  };
}

double Dot(const Vec3& a, const Vec3& b) noexcept {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Quat NormalizeQuat(const Quat& q) noexcept {
  const double n = std::sqrt(
      q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  if (n < 1.0e-12 || !std::isfinite(n)) return {1.0, 0.0, 0.0, 0.0};
  return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

Vec3 QuatRotate(const Quat& q_raw, const Vec3& v) noexcept {
  const Quat q = NormalizeQuat(q_raw);
  const double w = q[0];
  const Vec3 xyz{q[1], q[2], q[3]};
  const Vec3 cross = Cross(xyz, v);
  const double dot = Dot(xyz, v);
  return {
      v[0] * (2.0 * w * w - 1.0) + cross[0] * (2.0 * w) + xyz[0] * (2.0 * dot),
      v[1] * (2.0 * w * w - 1.0) + cross[1] * (2.0 * w) + xyz[1] * (2.0 * dot),
      v[2] * (2.0 * w * w - 1.0) + cross[2] * (2.0 * w) + xyz[2] * (2.0 * dot),
  };
}

Vec3 QuatRotateInverse(const Quat& q_raw, const Vec3& v) noexcept {
  const Quat q = NormalizeQuat(q_raw);
  const double w = q[0];
  const Vec3 xyz{q[1], q[2], q[3]};
  const Vec3 cross = Cross(xyz, v);
  const double dot = Dot(xyz, v);
  return {
      v[0] * (2.0 * w * w - 1.0) - cross[0] * (2.0 * w) + xyz[0] * (2.0 * dot),
      v[1] * (2.0 * w * w - 1.0) - cross[1] * (2.0 * w) + xyz[1] * (2.0 * dot),
      v[2] * (2.0 * w * w - 1.0) - cross[2] * (2.0 * w) + xyz[2] * (2.0 * dot),
  };
}

Vec3 ProjectedGravityBody(const Quat& q) noexcept {
  return QuatRotateInverse(q, {0.0, 0.0, -1.0});
}

std::array<double, 2> BaseForwardXy(const Quat& q) noexcept {
  const Vec3 fwd = QuatRotate(q, {1.0, 0.0, 0.0});
  const double n = std::max(std::hypot(fwd[0], fwd[1]), 1.0e-6);
  return {fwd[0] / n, fwd[1] / n};
}

double ReadJointScalar(const YAML::Node& node,
                       const char* joint_name,
                       std::size_t index,
                       const char* field_name,
                       bool allow_scalar) {
  if (!node) {
    throw std::runtime_error(std::string("missing action_adapter field: ") + field_name);
  }
  if (node.IsMap()) {
    const YAML::Node value = node[joint_name];
    if (!value) {
      throw std::runtime_error(
          std::string(field_name) + " is missing joint '" + joint_name + "'");
    }
    return value.as<double>();
  }
  if (node.IsSequence()) {
    if (node.size() != kHopeActionDim) {
      throw std::runtime_error(
          std::string(field_name) + " must have 31 entries");
    }
    return node[index].as<double>();
  }
  if (allow_scalar && node.IsScalar()) {
    return node.as<double>();
  }
  throw std::runtime_error(
      std::string(field_name) + " must be a joint map or a 31-entry list");
}

std::array<double, kHopeActionDim> ResolvePerJoint(
    const YAML::Node& node, const char* field_name, bool allow_scalar = false) {
  std::array<double, kHopeActionDim> out{};
  for (std::size_t i = 0; i < kHopeActionDim; ++i) {
    out[i] = ReadJointScalar(node, kHopeJointNames[i], i, field_name, allow_scalar);
  }
  return out;
}

}  // namespace

HopeRobotState MakeHopeRobotStateFromBackend(
    const robot_io::RobotState& state,
    const std::array<double, 3>& base_pos_w) noexcept {
  HopeRobotState out;
  out.base_pos_w = base_pos_w;
  out.base_quat_wxyz = {
      state.imu_quat_wxyz[0],
      state.imu_quat_wxyz[1],
      state.imu_quat_wxyz[2],
      state.imu_quat_wxyz[3],
  };
  out.base_ang_vel_b = {
      state.imu_gyro[0],
      state.imu_gyro[1],
      state.imu_gyro[2],
  };

  const int q_n = static_cast<int>(state.q.size());
  const int dq_n = static_cast<int>(state.dq.size());
  for (int i = 0; i < static_cast<int>(kHopeActionDim); ++i) {
    out.q[static_cast<std::size_t>(i)] = (i < q_n) ? state.q[i] : 0.0;
    out.dq[static_cast<std::size_t>(i)] = (i < dq_n) ? state.dq[i] : 0.0;
  }
  return out;
}

std::array<float, kHopeObsDim> BuildHopeObservation(
    const HopeRobotState& state,
    const HopeObsTarget& target,
    const std::array<double, kHopeActionDim>& last_applied_action,
    const std::array<double, kHopeActionDim>& default_q,
    const std::array<double, 2>& station_xy) noexcept {
  std::array<float, kHopeObsDim> obs{};
  std::size_t o = 0;
  auto put = [&](double v) noexcept {
    obs[o++] = static_cast<float>(v);
  };

  for (double v : state.base_ang_vel_b) put(v);
  for (std::size_t i = 0; i < kHopeActionDim; ++i) put(state.q[i] - default_q[i]);
  for (double v : state.dq) put(v);
  for (double v : last_applied_action) put(v);

  const Vec3 gravity = ProjectedGravityBody(state.base_quat_wxyz);
  for (double v : gravity) put(v);

  const auto forward_xy = BaseForwardXy(state.base_quat_wxyz);
  put(forward_xy[0]);
  put(forward_xy[1]);

  put(station_xy[0] - state.base_pos_w[0]);
  put(station_xy[1] - state.base_pos_w[1]);

  for (int i = 0; i < 3; ++i) put(target.pos_w[i] - state.base_pos_w[i]);
  for (double v : target.vel_w) put(v);
  put(target.time_to_strike);
  put(target.swing_side);
  return obs;
}

HopeActionAdapter HopeActionAdapter::FromYaml(const std::string& path) {
  const YAML::Node doc = YAML::LoadFile(path);
  HopeActionAdapter adapter;
  adapter.default_q_ = ResolvePerJoint(doc["default_q"], "default_q");
  adapter.action_scale_ = ResolvePerJoint(doc["action_scale"], "action_scale", true);
  const YAML::Node clamp = doc["joint_position_clamp"];
  if (!clamp || !clamp.IsMap()) {
    throw std::runtime_error("missing joint_position_clamp mapping");
  }
  adapter.clamp_lower_ = ResolvePerJoint(
      clamp["lower"], "joint_position_clamp.lower");
  adapter.clamp_upper_ = ResolvePerJoint(
      clamp["upper"], "joint_position_clamp.upper");
  for (std::size_t i = 0; i < kHopeActionDim; ++i) {
    if (adapter.clamp_lower_[i] > adapter.clamp_upper_[i]) {
      throw std::runtime_error(
          std::string("joint_position_clamp lower > upper for ") +
          kHopeJointNames[i]);
    }
  }
  return adapter;
}

std::array<double, kHopeActionDim> HopeActionAdapter::AppliedAction(
    const std::array<float, kHopeActionDim>& raw_action,
    bool passive_neck) const noexcept {
  std::array<double, kHopeActionDim> applied{};
  for (std::size_t i = 0; i < kHopeActionDim; ++i) {
    const double v = static_cast<double>(raw_action[i]);
    applied[i] = std::isfinite(v) ? v : 0.0;
  }
  if (passive_neck) {
    applied[static_cast<std::size_t>(kHopeHeadYawIndex)] = 0.0;
    applied[static_cast<std::size_t>(kHopeHeadPitchIndex)] = 0.0;
  }
  return applied;
}

std::array<double, kHopeActionDim> HopeActionAdapter::DecodeAppliedAction(
    const std::array<double, kHopeActionDim>& applied_action,
    bool passive_neck) const noexcept {
  std::array<double, kHopeActionDim> q_des{};
  for (std::size_t i = 0; i < kHopeActionDim; ++i) {
    const double raw = std::isfinite(applied_action[i]) ? applied_action[i] : 0.0;
    const double unclamped = default_q_[i] + raw * action_scale_[i];
    q_des[i] = std::clamp(unclamped, clamp_lower_[i], clamp_upper_[i]);
  }
  if (passive_neck) {
    q_des[static_cast<std::size_t>(kHopeHeadYawIndex)] =
        default_q_[static_cast<std::size_t>(kHopeHeadYawIndex)];
    q_des[static_cast<std::size_t>(kHopeHeadPitchIndex)] =
        default_q_[static_cast<std::size_t>(kHopeHeadPitchIndex)];
  }
  return q_des;
}

void HopeActionAdapter::DecodeRawAction(
    const std::array<float, kHopeActionDim>& raw_action,
    std::array<double, kHopeActionDim>& applied_action_out,
    std::array<double, kHopeActionDim>& q_des_out,
    bool passive_neck) const noexcept {
  applied_action_out = AppliedAction(raw_action, passive_neck);
  q_des_out = DecodeAppliedAction(applied_action_out, passive_neck);
}

void FillHopeRobotCommand(
    const std::array<double, kHopeActionDim>& q_des,
    const std::array<double, kHopeActionDim>& kp,
    const std::array<double, kHopeActionDim>& kd,
    robot_io::RobotCommand& command_out) {
  command_out.q_des = Eigen::VectorXd::Zero(static_cast<int>(kHopeActionDim));
  command_out.dq_des = Eigen::VectorXd::Zero(static_cast<int>(kHopeActionDim));
  command_out.tau_ff = Eigen::VectorXd::Zero(static_cast<int>(kHopeActionDim));
  command_out.kp = Eigen::VectorXd::Zero(static_cast<int>(kHopeActionDim));
  command_out.kd = Eigen::VectorXd::Zero(static_cast<int>(kHopeActionDim));
  for (int i = 0; i < static_cast<int>(kHopeActionDim); ++i) {
    command_out.q_des[i] = q_des[static_cast<std::size_t>(i)];
    command_out.kp[i] = kp[static_cast<std::size_t>(i)];
    command_out.kd[i] = kd[static_cast<std::size_t>(i)];
  }
}

bool HopeJointOrderMatchesA3Layout31() {
  const auto& layout = robot_io::MakeA3Layout31();
  if (layout.names.size() != kHopeActionDim) return false;
  for (std::size_t i = 0; i < kHopeActionDim; ++i) {
    if (layout.names[i] != kHopeJointNames[i]) return false;
  }
  return true;
}

}  // namespace a3_deploy
