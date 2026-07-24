// Copyright (c) 2026, AgiBot Inc. All rights reserved.
//
// HOPE ping-pong body-drive runner:
//   /body_drive/*_joint_state + /body_drive/*_imu/data
//   -> observation[111] -> raw_action[31]
//   -> /body_drive/*_joint_command
//
// This intentionally does not reuse the legacy A3 tokenizer runner, whose
// contract is obs_dict[1570] -> action[29].

#include "a3_deploy/a3_policy_driver.hpp"
#include "a3_deploy/a3_policy_runtime.hpp"
#include "a3_deploy/hope_pingpong_policy.hpp"
#include "robot_io/a3_aimrt_backend.hpp"
#include "robot_io/robot_io_backend.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace {

constexpr std::int64_t kNsPerMs = 1'000'000LL;

std::atomic<bool> g_stop_requested{false};

void HandleSignal(int) {
  g_stop_requested.store(true, std::memory_order_release);
}

template <typename T>
T RequiredKey(const YAML::Node& node, const std::string& path) {
  if (!node || node.IsNull()) {
    throw std::runtime_error("required config key missing: " + path);
  }
  return node.as<T>();
}

template <typename T>
T OptionalKey(const YAML::Node& node, const T& fallback) {
  if (!node || node.IsNull()) return fallback;
  return node.as<T>();
}

std::string NormalizeToken(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   if (c == '-') return static_cast<char>('_');
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

std::filesystem::path ResolvePath(const std::string& raw,
                                  const std::filesystem::path& cfg_path) {
  if (raw.empty()) return {};
  const std::filesystem::path path(raw);
  if (path.is_absolute()) return path.lexically_normal();
  if (std::filesystem::exists(path)) {
    return std::filesystem::absolute(path).lexically_normal();
  }
  return (cfg_path.parent_path() / path).lexically_normal();
}

bool HasFlag(int argc, char** argv, const char* flag) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == flag) return true;
  }
  return false;
}

std::string ParseStringFlag(int argc, char** argv, const char* name,
                            const std::string& fallback) {
  const std::string prefix = std::string(name) + "=";
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
    if (arg == name && i + 1 < argc) return argv[i + 1];
  }
  return fallback;
}

double ParseDoubleFlag(int argc, char** argv, const char* name,
                       double fallback) {
  const std::string value = ParseStringFlag(argc, argv, name, {});
  if (value.empty()) return fallback;
  return std::stod(value);
}

std::uint64_t ParseUint64Flag(int argc, char** argv, const char* name,
                              std::uint64_t fallback) {
  const std::string value = ParseStringFlag(argc, argv, name, {});
  if (value.empty()) return fallback;
  return static_cast<std::uint64_t>(std::stoull(value));
}

std::array<double, a3_deploy::kHopeActionDim> LoadJointTargetMap31(
    const YAML::Node& node,
    const std::array<double, a3_deploy::kHopeActionDim>& fallback,
    const std::string& path) {
  if (!node || node.IsNull()) return fallback;

  YAML::Node values = node;
  if (node.IsMap() && node["values"]) {
    values = node["values"];
  }
  auto out = fallback;

  if (values.IsSequence()) {
    if (values.size() != a3_deploy::kHopeActionDim) {
      throw std::runtime_error(path + " sequence must contain 31 values");
    }
    for (std::size_t i = 0; i < a3_deploy::kHopeActionDim; ++i) {
      out[i] = values[i].as<double>();
    }
    return out;
  }

  if (!values.IsMap()) {
    throw std::runtime_error(path + " must be a map or a 31-value sequence");
  }

  for (std::size_t i = 0; i < a3_deploy::kHopeActionDim; ++i) {
    const char* name = a3_deploy::kHopeJointNames[i];
    if (!values[name]) {
      throw std::runtime_error(path + " missing joint: " + std::string(name));
    }
    out[i] = values[name].as<double>();
  }
  return out;
}

void PrintUsage(const char* progname) {
  std::cerr
      << "Usage: " << progname
      << " --runtime-cfg PATH [--onnx PATH] [--aimrt-cfg PATH]\n"
      << "       [--duration SEC] [--publish-commands] [--backend-only]\n"
      << "       [--status-every N] [--prepare-duration SEC]\n\n"
      << "Default is inference/probe mode with command publishers disabled.\n"
      << "Use --publish-commands only against MuJoCo body-drive sim or after\n"
      << "explicit real-robot bring-up checks.\n";
}

std::int64_t MsToNs(double ms) {
  return static_cast<std::int64_t>(std::llround(ms * 1'000'000.0));
}

struct RacketCommand {
  std::uint64_t task_id{0};
  std::uint32_t task_revision{0};
  int swing_side{1};
  std::array<double, 3> position{0.0, 0.0, 0.0};
  std::array<double, 3> velocity{0.0, 0.0, 0.0};
  double time_to_strike{1.0};
};

struct ExampleCommandFeed {
  double dt{0.02};
  double period_s{4.0};
  double lead_time_s{1.2};
  std::array<double, 3> forehand_pos{0.55, -0.35, 0.85};
  std::array<double, 3> backhand_pos{0.55, 0.25, 1.00};
  std::array<double, 3> forehand_vel{1.5, 1.3, 0.6};
  std::array<double, 3> backhand_vel{2.0, -0.7, 0.4};

  double t{0.0};
  std::uint64_t task_id{0};
  std::uint32_t revision{0};
  int issued_for_cycle{-1};
  int side{1};
  std::array<double, 3> pos{0.55, -0.35, 0.85};
  std::array<double, 3> vel{1.5, 1.3, 0.6};

  RacketCommand Poll() {
    const int cycle = static_cast<int>(std::floor(t / period_s));
    const double elapsed = t - static_cast<double>(cycle) * period_s;
    if (cycle != issued_for_cycle) {
      issued_for_cycle = cycle;
      ++task_id;
      revision = 0;
      side = (task_id % 2 == 1) ? 1 : -1;
      pos = (side >= 0) ? forehand_pos : backhand_pos;
      vel = (side >= 0) ? forehand_vel : backhand_vel;
    } else {
      ++revision;
    }
    RacketCommand cmd;
    cmd.task_id = task_id;
    cmd.task_revision = revision;
    cmd.swing_side = (side >= 0) ? 1 : -1;
    cmd.position = pos;
    cmd.velocity = vel;
    cmd.time_to_strike = std::max(lead_time_s - elapsed, 0.0);
    t += dt;
    return cmd;
  }
};

enum class Phase {
  Ready,
  Swing,
  FollowThrough,
  Recovery,
};

const char* PhaseName(Phase phase) {
  switch (phase) {
    case Phase::Ready:
      return "ready";
    case Phase::Swing:
      return "swing";
    case Phase::FollowThrough:
      return "follow_through";
    case Phase::Recovery:
      return "recovery";
  }
  return "unknown";
}

struct LifecycleConfig {
  double dt{0.02};
  double follow_through_s{0.6};
  double recovery_s{0.8};
  double ready_time_to_strike{1.0};
  double ready_reach_x{0.40};
  double ready_reach_y{0.20};
  double ready_reach_z{-0.05};
};

class SwingLifecycle {
 public:
  explicit SwingLifecycle(LifecycleConfig cfg) : cfg_(cfg) {}

  a3_deploy::HopeObsTarget Update(const RacketCommand& cmd,
                                  const a3_deploy::HopeRobotState& state) {
    if (cmd.task_id > last_engaged_task_id_ && CanEngage_()) {
      active_task_id_ = cmd.task_id;
      last_engaged_task_id_ = cmd.task_id;
      applied_revision_ = static_cast<int>(cmd.task_revision);
      swing_side_ = cmd.swing_side >= 0 ? 1 : -1;
      target_pos_w_ = cmd.position;
      target_vel_w_ = cmd.velocity;
      time_to_strike_ = cmd.time_to_strike;
      phase_ = Phase::Swing;
    } else if (active_task_id_ && cmd.task_id == *active_task_id_ &&
               phase_ == Phase::Swing && time_to_strike_ > 0.0 &&
               static_cast<int>(cmd.task_revision) > applied_revision_) {
      applied_revision_ = static_cast<int>(cmd.task_revision);
      target_pos_w_ = cmd.position;
      target_vel_w_ = cmd.velocity;
      time_to_strike_ = cmd.time_to_strike;
    }

    if (phase_ == Phase::Swing) {
      time_to_strike_ -= cfg_.dt;
      if (time_to_strike_ <= 0.0) {
        phase_ = Phase::FollowThrough;
        follow_t_ = 0.0;
      }
    } else if (phase_ == Phase::FollowThrough) {
      time_to_strike_ -= cfg_.dt;
      follow_t_ += cfg_.dt;
      if (follow_t_ >= cfg_.follow_through_s) {
        phase_ = Phase::Recovery;
        recover_t_ = 0.0;
      }
    } else if (phase_ == Phase::Recovery) {
      recover_t_ += cfg_.dt;
      if (recover_t_ >= cfg_.recovery_s) {
        phase_ = Phase::Ready;
        active_task_id_.reset();
      }
    }

    a3_deploy::HopeObsTarget target;
    target.swing_side = static_cast<double>(swing_side_);
    if (phase_ == Phase::Swing || phase_ == Phase::FollowThrough) {
      target.pos_w = target_pos_w_;
      target.vel_w = target_vel_w_;
      target.time_to_strike = time_to_strike_;
      return target;
    }

    const double side = swing_side_ >= 0 ? 1.0 : -1.0;
    target.pos_w = {state.base_pos_w[0] + cfg_.ready_reach_x,
                    state.base_pos_w[1] + side * cfg_.ready_reach_y,
                    state.base_pos_w[2] + cfg_.ready_reach_z};
    target.vel_w = {0.0, 0.0, 0.0};
    target.time_to_strike = cfg_.ready_time_to_strike;
    return target;
  }

  Phase phase() const noexcept { return phase_; }
  std::uint64_t active_task_id() const noexcept {
    return active_task_id_.value_or(0);
  }

 private:
  bool CanEngage_() const noexcept {
    return phase_ == Phase::Ready || phase_ == Phase::Recovery;
  }

  LifecycleConfig cfg_;
  Phase phase_{Phase::Ready};
  std::optional<std::uint64_t> active_task_id_;
  int swing_side_{1};
  std::array<double, 3> target_pos_w_{0.0, 0.0, 0.0};
  std::array<double, 3> target_vel_w_{0.0, 0.0, 0.0};
  double time_to_strike_{1.0};
  double follow_t_{0.0};
  double recover_t_{0.0};
  std::uint64_t last_engaged_task_id_{0};
  int applied_revision_{-1};
};

std::array<double, 3> LoadVec3(const YAML::Node& node,
                               const std::array<double, 3>& fallback,
                               const std::string& path) {
  if (!node || node.IsNull()) return fallback;
  if (!node.IsSequence() || node.size() != 3) {
    throw std::runtime_error(path + " must be a 3-entry list");
  }
  return {node[0].as<double>(), node[1].as<double>(), node[2].as<double>()};
}

std::array<double, 2> LoadVec2(const YAML::Node& node,
                               const std::array<double, 2>& fallback,
                               const std::string& path) {
  if (!node || node.IsNull()) return fallback;
  if (!node.IsSequence() || node.size() != 2) {
    throw std::runtime_error(path + " must be a 2-entry list");
  }
  return {node[0].as<double>(), node[1].as<double>()};
}

std::unordered_map<std::string, std::size_t> HopeJointIndexMap() {
  std::unordered_map<std::string, std::size_t> out;
  for (std::size_t i = 0; i < a3_deploy::kHopeActionDim; ++i) {
    out.emplace(a3_deploy::kHopeJointNames[i], i);
  }
  return out;
}

void FillGainRange(const YAML::Node& groups,
                   const std::string& group_name,
                   std::size_t first,
                   std::size_t last_exclusive,
                   std::array<double, a3_deploy::kHopeActionDim>& kp,
                   std::array<double, a3_deploy::kHopeActionDim>& kd) {
  const YAML::Node group = groups[group_name];
  if (!group) {
    throw std::runtime_error("simulation.pd_gains.groups missing " + group_name);
  }
  const double group_kp = RequiredKey<double>(group["kp"],
                                             std::string("simulation.pd_gains.groups.") +
                                                 group_name + ".kp");
  const double group_kd = RequiredKey<double>(group["kd"],
                                             std::string("simulation.pd_gains.groups.") +
                                                 group_name + ".kd");
  for (std::size_t i = first; i < last_exclusive; ++i) {
    kp[i] = group_kp;
    kd[i] = group_kd;
  }
}

void LoadPdGains(const YAML::Node& pd_gains,
                 std::array<double, a3_deploy::kHopeActionDim>& kp,
                 std::array<double, a3_deploy::kHopeActionDim>& kd) {
  if (!pd_gains) {
    throw std::runtime_error("simulation.pd_gains is required");
  }
  const YAML::Node groups = pd_gains["groups"];
  FillGainRange(groups, "waist", 0, 3, kp, kd);
  FillGainRange(groups, "neck", 3, 5, kp, kd);
  FillGainRange(groups, "arm", 5, 19, kp, kd);
  FillGainRange(groups, "leg", 19, 31, kp, kd);

  const auto name_to_idx = HopeJointIndexMap();
  const YAML::Node joints = pd_gains["joints"];
  if (!joints || !joints.IsMap()) return;
  for (const auto& item : joints) {
    const std::string name = item.first.as<std::string>();
    const auto it = name_to_idx.find(name);
    if (it == name_to_idx.end()) {
      throw std::runtime_error("simulation.pd_gains.joints unknown joint: " +
                               name);
    }
    const auto i = it->second;
    kp[i] = RequiredKey<double>(item.second["kp"],
                                "simulation.pd_gains.joints." + name + ".kp");
    kd[i] = RequiredKey<double>(item.second["kd"],
                                "simulation.pd_gains.joints." + name + ".kd");
  }
}

LifecycleConfig LoadLifecycle(const YAML::Node& node, double control_hz) {
  LifecycleConfig cfg;
  cfg.dt = 1.0 / control_hz;
  cfg.follow_through_s = OptionalKey<double>(node["follow_through_s"], 0.6);
  cfg.recovery_s = OptionalKey<double>(node["recovery_s"], 0.8);
  cfg.ready_time_to_strike =
      OptionalKey<double>(node["ready_time_to_strike"], 1.0);
  cfg.ready_reach_x = OptionalKey<double>(node["ready_reach_x"], 0.40);
  cfg.ready_reach_y = OptionalKey<double>(node["ready_reach_y"], 0.20);
  cfg.ready_reach_z = OptionalKey<double>(node["ready_reach_z"], -0.05);
  return cfg;
}

ExampleCommandFeed LoadExampleFeed(const YAML::Node& node, double control_hz) {
  ExampleCommandFeed feed;
  feed.dt = 1.0 / control_hz;
  feed.period_s = OptionalKey<double>(node["period_s"], feed.period_s);
  feed.lead_time_s = OptionalKey<double>(node["lead_time_s"], feed.lead_time_s);
  feed.forehand_pos =
      LoadVec3(node["forehand_pos"], feed.forehand_pos,
               "example_command_feed.forehand_pos");
  feed.backhand_pos =
      LoadVec3(node["backhand_pos"], feed.backhand_pos,
               "example_command_feed.backhand_pos");
  feed.forehand_vel =
      LoadVec3(node["forehand_vel"], feed.forehand_vel,
               "example_command_feed.forehand_vel");
  feed.backhand_vel =
      LoadVec3(node["backhand_vel"], feed.backhand_vel,
               "example_command_feed.backhand_vel");
  return feed;
}

std::string BuildBackendConfigString(const YAML::Node& backend,
                                     const std::string& aimrt_cfg_override,
                                     const std::filesystem::path& cfg_path,
                                     bool publish_commands,
                                     double policy_hz) {
  std::stringstream ss;
  bool first = true;
  auto add = [&](const std::string& kv) {
    if (!first) ss << ',';
    ss << kv;
    first = false;
  };

  const std::string raw_aimrt_cfg =
      aimrt_cfg_override.empty()
          ? RequiredKey<std::string>(backend["aimrt_cfg_path"],
                                     "backend.aimrt_cfg_path")
          : aimrt_cfg_override;
  add("cfg_file_path=" + ResolvePath(raw_aimrt_cfg, cfg_path).string());

  const std::string sync_mode =
      OptionalKey<std::string>(backend["sync_mode"], std::string{"min_skew_pair"});
  add("sync_mode=" + sync_mode);
  add("sync_hz=" +
      std::to_string(OptionalKey<double>(backend["sync_hz"], policy_hz * 2.0)));

  const std::array<const char*, 10> passthrough = {{
      "align_delay_ms",
      "phase_ms",
      "max_skew_ms",
      "max_sample_age_ms",
      "sync_ready_after_input_ms",
      "sync_release_margin_ms",
      "max_group_internal_skew_ms",
      "max_group_pair_skew_ms",
      "group_pair_search_depth",
      "max_backtrack",
  }};
  for (const char* key : passthrough) {
    if (backend[key]) add(std::string(key) + "=" + backend[key].as<std::string>());
  }
  if (backend["auto_phase"]) {
    add(std::string{"auto_phase="} +
        (backend["auto_phase"].as<bool>() ? "true" : "false"));
  }

  const bool cfg_publish = OptionalKey<bool>(backend["publish_enabled"], true);
  add(std::string{"publish_enabled="} +
      ((publish_commands && cfg_publish) ? "true" : "false"));
  return ss.str();
}

a3_deploy::A3PolicyRuntimeOptions LoadPolicyOptions(const YAML::Node& policy) {
  a3_deploy::A3PolicyRuntimeOptions options;
  options.backend = OptionalKey<std::string>(policy["backend"], options.backend);
  options.input_tensor_name =
      OptionalKey<std::string>(policy["input_tensor_name"], "observation");
  options.output_tensor_name =
      OptionalKey<std::string>(policy["output_tensor_name"], "raw_action");
  options.intra_op_num_threads =
      OptionalKey<int>(policy["intra_op_num_threads"], 1);
  options.inter_op_num_threads =
      OptionalKey<int>(policy["inter_op_num_threads"], 1);
  options.use_fp16 = OptionalKey<bool>(policy["fp16"], false);
  options.rknn_core_mask =
      OptionalKey<std::string>(policy["rknn_core_mask"], std::string{"auto"});
  return options;
}

a3_deploy::A3PolicyDriverOptions LoadDriverOptions(const YAML::Node& node,
                                                   double policy_hz) {
  a3_deploy::A3PolicyDriverOptions options;
  options.policy_hz = policy_hz;
  options.send_safe_halt_before_first_command = true;
  if (node["watchdog"]) {
    const auto watchdog = node["watchdog"];
    if (watchdog["max_frame_age_ms"]) {
      options.watchdog.max_frame_age_ns =
          MsToNs(watchdog["max_frame_age_ms"].as<double>());
    }
    if (watchdog["max_unaligned_frames"]) {
      options.watchdog.max_consecutive_unaligned =
          watchdog["max_unaligned_frames"].as<int>();
    }
  }
  if (node["rt"]) {
    const auto rt = node["rt"];
    options.sched.priority =
        OptionalKey<int>(rt["sched_fifo_priority"], options.sched.priority);
    options.sched.cpu =
        OptionalKey<int>(rt["cpu_affinity"], options.sched.cpu);
  }
  return options;
}

void SleepUntilStopOrDuration(double duration_s,
                              const a3_deploy::A3PolicyDriver* driver,
                              std::uint64_t status_every_ticks) {
  const auto start = std::chrono::steady_clock::now();
  std::uint64_t last_logged_bucket = 0;
  while (!g_stop_requested.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (duration_s > 0.0) {
      const auto elapsed =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
              .count();
      if (elapsed >= duration_s) break;
    }
    if (driver && status_every_ticks > 0) {
      const auto ticks = driver->PolicyTickCount();
      const auto bucket = ticks / status_every_ticks;
      if (bucket != last_logged_bucket) {
        last_logged_bucket = bucket;
        std::cerr << "[hope_body_drive] ticks=" << ticks
                  << " safe_halt=" << driver->SafeHaltCount() << "\n";
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (HasFlag(argc, argv, "--help") || HasFlag(argc, argv, "-h")) {
      PrintUsage(argv[0]);
      return 0;
    }

    const std::string cfg_arg = ParseStringFlag(
        argc, argv, "--runtime-cfg",
        "src/a3/a3_deploy_onnx_ref/config/hope_pingpong_body_drive.yaml");
    const auto cfg_path = std::filesystem::absolute(cfg_arg).lexically_normal();
    const YAML::Node cfg = YAML::LoadFile(cfg_path.string());
    if (OptionalKey<std::string>(cfg["contract"], "hope_pingpong") !=
        "hope_pingpong") {
      throw std::runtime_error("contract must be hope_pingpong");
    }
    if (NormalizeToken(OptionalKey<std::string>(
            cfg["observation_normalization"], "none")) != "none") {
      throw std::runtime_error("observation_normalization must be none");
    }
    if (!a3_deploy::HopeJointOrderMatchesA3Layout31()) {
      throw std::runtime_error(
          "HOPE joint order does not match A3 31-DOF body-drive layout");
    }

    const bool publish_commands = HasFlag(argc, argv, "--publish-commands");
    const bool backend_only = HasFlag(argc, argv, "--backend-only");
    const double duration_s =
        ParseDoubleFlag(argc, argv, "--duration",
                        OptionalKey<double>(cfg["duration_s"], 5.0));
    const std::uint64_t status_every =
        ParseUint64Flag(argc, argv, "--status-every",
                        OptionalKey<std::uint64_t>(
                            cfg["logging"]["status_every_ticks"], 100));
    const double prepare_s =
        ParseDoubleFlag(argc, argv, "--prepare-duration",
                        OptionalKey<double>(cfg["startup"]["prepare_s"], 0.0));
    const std::string aimrt_cfg_override =
        ParseStringFlag(argc, argv, "--aimrt-cfg", {});
    const double control_hz = OptionalKey<double>(cfg["control_hz"], 50.0);
    if (!std::isfinite(control_hz) || control_hz <= 0.0) {
      throw std::runtime_error("control_hz must be > 0");
    }
    if (!std::isfinite(prepare_s) || prepare_s < 0.0) {
      throw std::runtime_error("startup.prepare_s/--prepare-duration must be >= 0");
    }
    const std::uint64_t prepare_ticks =
        static_cast<std::uint64_t>(std::llround(prepare_s * control_hz));

    const YAML::Node policy_cfg = cfg["policy"];
    const std::string onnx_override = ParseStringFlag(argc, argv, "--onnx", {});
    const std::string model_raw =
        onnx_override.empty()
            ? RequiredKey<std::string>(policy_cfg["model_path"],
                                       "policy.model_path")
            : onnx_override;
    const auto model_path = ResolvePath(model_raw, cfg_path);
    const auto adapter_path = ResolvePath(
        RequiredKey<std::string>(cfg["action_adapter"]["config_path"],
                                 "action_adapter.config_path"),
        cfg_path);

    a3_deploy::A3PolicyRuntimeOptions runtime_options =
        LoadPolicyOptions(policy_cfg);
    runtime_options.input_tensor_name = "observation";
    runtime_options.output_tensor_name = "raw_action";

    std::unique_ptr<a3_deploy::A3PolicyRuntime> policy;
    if (!backend_only) {
      policy = a3_deploy::CreateA3PolicyRuntime(runtime_options);
      if (!policy) throw std::runtime_error("CreateA3PolicyRuntime failed");
      if (!policy->Initialize(model_path.string(), runtime_options)) {
        throw std::runtime_error("policy Initialize failed");
      }
      if (policy->GetInputDimension() != a3_deploy::kHopeObsDim ||
          policy->GetActionDimension() != a3_deploy::kHopeActionDim) {
        std::stringstream ss;
        ss << "policy dim mismatch: got input=" << policy->GetInputDimension()
           << " action=" << policy->GetActionDimension()
           << ", expected 111/31";
        throw std::runtime_error(ss.str());
      }
    }

    const auto adapter =
        a3_deploy::HopeActionAdapter::FromYaml(adapter_path.string());
    const auto prepare_q =
        LoadJointTargetMap31(cfg["startup"]["prepare_q"], adapter.default_q(),
                             "startup.prepare_q");
    std::array<double, a3_deploy::kHopeActionDim> kp{};
    std::array<double, a3_deploy::kHopeActionDim> kd{};
    LoadPdGains(cfg["simulation"]["pd_gains"], kp, kd);

    auto backend = robot_io::CreateBackend("a3");
    if (!backend) throw std::runtime_error("CreateBackend('a3') failed");
    const std::string backend_cfg = BuildBackendConfigString(
        cfg["backend"], aimrt_cfg_override, cfg_path, publish_commands,
        control_hz);
    if (!backend->Init(backend_cfg)) {
      throw std::runtime_error("backend Init failed");
    }

    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    std::cout << "HOPE body-drive runner\n"
              << "  model: " << model_path.string() << "\n"
              << "  adapter: " << adapter_path.string() << "\n"
              << "  policy_hz: " << control_hz << "\n"
              << "  prepare_s: " << prepare_s
              << " (" << prepare_ticks << " ticks)\n"
              << "  publish_commands: "
              << (publish_commands ? "true" : "false") << "\n";

    if (!backend->Start()) {
      throw std::runtime_error("backend Start failed");
    }
    std::cout << "  backend: started\n";

    if (backend_only) {
      std::cout << "  mode: backend-only\n";
      SleepUntilStopOrDuration(duration_s, nullptr, 0);
      backend->Stop();
      std::cout << "HOPE body-drive backend-only exit cleanly\n";
      return 0;
    }

    const bool passive_neck = OptionalKey<bool>(cfg["passive_neck"], true);
    const auto static_base_pos = LoadVec3(
        cfg["base_pose"]["position_w"], {0.0, 0.0, 0.0},
        "base_pose.position_w");
    const auto configured_station =
        LoadVec2(cfg["fixed_station_xy"], {static_base_pos[0], static_base_pos[1]},
                 "fixed_station_xy");

    ExampleCommandFeed command_feed =
        LoadExampleFeed(cfg["example_command_feed"], control_hz);
    SwingLifecycle lifecycle(LoadLifecycle(cfg["lifecycle"], control_hz));
    std::array<double, a3_deploy::kHopeActionDim> last_applied_action{};
    std::array<double, 2> station_xy = configured_station;
    const bool station_from_start =
        OptionalKey<bool>(cfg["fixed_station_from_start_base"], true);
    bool station_locked = !station_from_start;
    std::uint64_t debug_last_tick = 0;
    bool prepare_done_logged = (prepare_ticks == 0);

    auto command_fn =
        [&](std::uint64_t tick_idx, const robot_io::RobotState& state,
            robot_io::RobotCommand& command_out) -> bool {
      auto hope_state =
          a3_deploy::MakeHopeRobotStateFromBackend(state, static_base_pos);
      if (!station_locked) {
        station_xy = {hope_state.base_pos_w[0], hope_state.base_pos_w[1]};
        station_locked = true;
      }

      if (tick_idx < prepare_ticks) {
        a3_deploy::FillHopeRobotCommand(prepare_q, kp, kd,
                                        command_out);
        last_applied_action.fill(0.0);
        if (status_every > 0 &&
            (tick_idx == 0 || tick_idx % status_every == 0)) {
          std::cerr << "[hope_body_drive] prepare tick=" << tick_idx << "/"
                    << prepare_ticks << " q0=" << prepare_q[0]
                    << "\n";
        }
        return true;
      }
      if (!prepare_done_logged) {
        prepare_done_logged = true;
        std::cerr << "[hope_body_drive] prepare complete; starting policy\n";
      }
      const std::uint64_t policy_tick_idx = tick_idx - prepare_ticks;

      const RacketCommand cmd = command_feed.Poll();
      const auto target = lifecycle.Update(cmd, hope_state);
      const auto obs = a3_deploy::BuildHopeObservation(
          hope_state, target, last_applied_action, adapter.default_q(),
          station_xy);
      std::copy(obs.begin(), obs.end(), policy->MutableInputData());
      if (!policy->Infer()) return false;

      std::array<float, a3_deploy::kHopeActionDim> raw_action{};
      const float* act = policy->ActionData();
      for (std::size_t i = 0; i < raw_action.size(); ++i) {
        raw_action[i] = act[i];
      }
      std::array<double, a3_deploy::kHopeActionDim> q_des{};
      adapter.DecodeRawAction(raw_action, last_applied_action, q_des,
                              passive_neck);
      a3_deploy::FillHopeRobotCommand(q_des, kp, kd, command_out);

      if (status_every > 0 && policy_tick_idx % status_every == 0 &&
          policy_tick_idx != debug_last_tick) {
        debug_last_tick = policy_tick_idx;
        const char* side = target.swing_side >= 0.0 ? "forehand" : "backhand";
        std::cerr << "[hope_body_drive] tick=" << policy_tick_idx
                  << " phase=" << PhaseName(lifecycle.phase())
                  << " task=" << lifecycle.active_task_id()
                  << " side=" << side
                  << " tts=" << std::fixed << std::setprecision(2)
                  << target.time_to_strike
                  << " raw0=" << raw_action[0]
                  << " q0=" << q_des[0] << "\n";
      }
      return true;
    };

    a3_deploy::A3PolicyDriverOptions driver_options =
        LoadDriverOptions(cfg["policy_driver"], control_hz);
    auto driver = std::make_unique<a3_deploy::A3PolicyDriver>(
        *backend, command_fn, driver_options);
    if (!driver->StartDriver()) {
      throw std::runtime_error("A3PolicyDriver::StartDriver failed");
    }

    std::cout << "  mode: policy inference"
              << (publish_commands ? " + command publish" : " probe/no-publish")
              << "\n";
    SleepUntilStopOrDuration(duration_s, driver.get(), status_every);
    driver->StopDriver();
    backend->Stop();
    const std::uint64_t driver_ticks = driver->PolicyTickCount();
    const std::uint64_t onnx_ticks =
        (driver_ticks > prepare_ticks) ? (driver_ticks - prepare_ticks) : 0;
    std::cout << "HOPE body-drive exit cleanly: driver_ticks="
              << driver_ticks
              << " prepare_ticks=" << std::min(driver_ticks, prepare_ticks)
              << " onnx_ticks=" << onnx_ticks
              << " safe_halt=" << driver->SafeHaltCount() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "hope_pingpong_body_drive failed: " << e.what() << "\n";
    return 1;
  }
}
