// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#include <gtest/gtest.h>

#include "a3_deploy/hope_pingpong_policy.hpp"
#include "robot_io/a3_layout_extra.hpp"

#include <Eigen/Core>

#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace {

class HopePingPongPolicyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tmp_dir_ = std::filesystem::temp_directory_path() /
               ("hope_pingpong_policy_test_" + std::to_string(::getpid()) +
                "_" +
                std::to_string(
                    ::testing::UnitTest::GetInstance()->random_seed()));
    std::filesystem::create_directories(tmp_dir_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tmp_dir_, ec);
  }

  std::filesystem::path WriteAdapterYaml() const {
    const auto path = tmp_dir_ / "action_adapter.yaml";
    std::ofstream f(path);
    f << "default_q: [";
    for (int i = 0; i < 31; ++i) {
      if (i) f << ", ";
      f << 0.01 * static_cast<double>(i);
    }
    f << "]\n";
    f << "action_scale: [";
    for (int i = 0; i < 31; ++i) {
      if (i) f << ", ";
      f << 0.1;
    }
    f << "]\n";
    f << "joint_position_clamp:\n";
    f << "  lower: [";
    for (int i = 0; i < 31; ++i) {
      if (i) f << ", ";
      f << -1.0;
    }
    f << "]\n";
    f << "  upper: [";
    for (int i = 0; i < 31; ++i) {
      if (i) f << ", ";
      f << 1.0;
    }
    f << "]\n";
    return path;
  }

  std::filesystem::path tmp_dir_;
};

}  // namespace

TEST_F(HopePingPongPolicyTest, ContractConstantsMatchCurrentModel) {
  EXPECT_EQ(a3_deploy::kHopeObsDim, 111u);
  EXPECT_EQ(a3_deploy::kHopeActionDim, 31u);
  EXPECT_DOUBLE_EQ(a3_deploy::kHopeControlHz, 50.0);
  EXPECT_EQ(a3_deploy::kHopeHeadYawIndex, 3);
  EXPECT_EQ(a3_deploy::kHopeHeadPitchIndex, 4);
}

TEST_F(HopePingPongPolicyTest, JointOrderMatchesA3Backend31DofLayout) {
  ASSERT_TRUE(a3_deploy::HopeJointOrderMatchesA3Layout31());
  const auto& layout = robot_io::MakeA3Layout31();
  ASSERT_EQ(layout.names.size(), a3_deploy::kHopeActionDim);
  for (std::size_t i = 0; i < a3_deploy::kHopeActionDim; ++i) {
    EXPECT_EQ(layout.names[i], a3_deploy::kHopeJointNames[i]) << "i=" << i;
  }
}

TEST_F(HopePingPongPolicyTest, BuildsObservationInPublic111DLayout) {
  a3_deploy::HopeRobotState state;
  state.base_pos_w = {1.0, 2.0, 0.5};
  state.base_quat_wxyz = {1.0, 0.0, 0.0, 0.0};
  state.base_ang_vel_b = {0.1, 0.2, 0.3};

  std::array<double, a3_deploy::kHopeActionDim> default_q{};
  std::array<double, a3_deploy::kHopeActionDim> last_action{};
  for (std::size_t i = 0; i < a3_deploy::kHopeActionDim; ++i) {
    default_q[i] = 0.01 * static_cast<double>(i);
    state.q[i] = default_q[i] + 0.001 * static_cast<double>(i + 1);
    state.dq[i] = -0.002 * static_cast<double>(i + 1);
    last_action[i] = 0.003 * static_cast<double>(i + 1);
  }

  a3_deploy::HopeObsTarget target;
  target.pos_w = {1.4, 1.5, 0.9};
  target.vel_w = {2.0, -0.7, 0.4};
  target.time_to_strike = 0.42;
  target.swing_side = -1.0;

  const auto obs = a3_deploy::BuildHopeObservation(
      state, target, last_action, default_q, {1.2, 2.3});

  EXPECT_FLOAT_EQ(obs[0], 0.1f);
  EXPECT_FLOAT_EQ(obs[1], 0.2f);
  EXPECT_FLOAT_EQ(obs[2], 0.3f);
  for (std::size_t i = 0; i < a3_deploy::kHopeActionDim; ++i) {
    EXPECT_FLOAT_EQ(obs[3 + i], static_cast<float>(0.001 * (i + 1))) << i;
    EXPECT_FLOAT_EQ(obs[34 + i], static_cast<float>(-0.002 * (i + 1))) << i;
    EXPECT_FLOAT_EQ(obs[65 + i], static_cast<float>(0.003 * (i + 1))) << i;
  }
  EXPECT_FLOAT_EQ(obs[96], 0.0f);
  EXPECT_FLOAT_EQ(obs[97], 0.0f);
  EXPECT_FLOAT_EQ(obs[98], -1.0f);
  EXPECT_FLOAT_EQ(obs[99], 1.0f);
  EXPECT_FLOAT_EQ(obs[100], 0.0f);
  EXPECT_FLOAT_EQ(obs[101], 0.2f);
  EXPECT_FLOAT_EQ(obs[102], 0.3f);
  EXPECT_FLOAT_EQ(obs[103], 0.4f);
  EXPECT_FLOAT_EQ(obs[104], -0.5f);
  EXPECT_FLOAT_EQ(obs[105], 0.4f);
  EXPECT_FLOAT_EQ(obs[106], 2.0f);
  EXPECT_FLOAT_EQ(obs[107], -0.7f);
  EXPECT_FLOAT_EQ(obs[108], 0.4f);
  EXPECT_FLOAT_EQ(obs[109], 0.42f);
  EXPECT_FLOAT_EQ(obs[110], -1.0f);
}

TEST_F(HopePingPongPolicyTest, AdapterZeroesPassiveHeadAndClamps) {
  const auto adapter =
      a3_deploy::HopeActionAdapter::FromYaml(WriteAdapterYaml().string());

  std::array<float, a3_deploy::kHopeActionDim> raw{};
  raw.fill(1.0f);
  raw[0] = 20.0f;   // default 0 + 20 * 0.1 -> clipped to +1
  raw[3] = 10.0f;   // passive head -> applied 0, q_des default
  raw[4] = -10.0f;  // passive head -> applied 0, q_des default

  std::array<double, a3_deploy::kHopeActionDim> applied{};
  std::array<double, a3_deploy::kHopeActionDim> q_des{};
  adapter.DecodeRawAction(raw, applied, q_des, /*passive_neck=*/true);

  EXPECT_DOUBLE_EQ(applied[0], 20.0);
  EXPECT_DOUBLE_EQ(q_des[0], 1.0);
  EXPECT_DOUBLE_EQ(applied[3], 0.0);
  EXPECT_DOUBLE_EQ(applied[4], 0.0);
  EXPECT_DOUBLE_EQ(q_des[3], adapter.default_q()[3]);
  EXPECT_DOUBLE_EQ(q_des[4], adapter.default_q()[4]);
  EXPECT_DOUBLE_EQ(q_des[5], adapter.default_q()[5] + 0.1);
}

TEST_F(HopePingPongPolicyTest, ConvertsBackendStateWithMocapBasePosition) {
  robot_io::RobotState backend;
  backend.q = Eigen::VectorXd::Zero(31);
  backend.dq = Eigen::VectorXd::Zero(31);
  backend.imu_quat_wxyz = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
  backend.imu_gyro = Eigen::Vector3d(0.4, 0.5, 0.6);
  for (int i = 0; i < 31; ++i) {
    backend.q[i] = 0.01 * i;
    backend.dq[i] = -0.02 * i;
  }

  const auto state = a3_deploy::MakeHopeRobotStateFromBackend(
      backend, {0.1, -0.2, 0.3});
  EXPECT_DOUBLE_EQ(state.base_pos_w[0], 0.1);
  EXPECT_DOUBLE_EQ(state.base_pos_w[1], -0.2);
  EXPECT_DOUBLE_EQ(state.base_pos_w[2], 0.3);
  EXPECT_DOUBLE_EQ(state.base_ang_vel_b[2], 0.6);
  EXPECT_DOUBLE_EQ(state.q[30], 0.30);
  EXPECT_DOUBLE_EQ(state.dq[30], -0.60);
}

TEST_F(HopePingPongPolicyTest, Fills31DofRobotCommand) {
  std::array<double, a3_deploy::kHopeActionDim> q_des{};
  std::array<double, a3_deploy::kHopeActionDim> kp{};
  std::array<double, a3_deploy::kHopeActionDim> kd{};
  for (int i = 0; i < 31; ++i) {
    q_des[static_cast<std::size_t>(i)] = 0.01 * i;
    kp[static_cast<std::size_t>(i)] = 10.0 + i;
    kd[static_cast<std::size_t>(i)] = 1.0 + 0.1 * i;
  }

  robot_io::RobotCommand cmd;
  a3_deploy::FillHopeRobotCommand(q_des, kp, kd, cmd);
  ASSERT_EQ(cmd.q_des.size(), 31);
  ASSERT_EQ(cmd.dq_des.size(), 31);
  ASSERT_EQ(cmd.tau_ff.size(), 31);
  EXPECT_DOUBLE_EQ(cmd.q_des[30], 0.30);
  EXPECT_DOUBLE_EQ(cmd.dq_des[30], 0.0);
  EXPECT_DOUBLE_EQ(cmd.tau_ff[30], 0.0);
  EXPECT_DOUBLE_EQ(cmd.kp[30], 40.0);
  EXPECT_DOUBLE_EQ(cmd.kd[30], 4.0);
}
