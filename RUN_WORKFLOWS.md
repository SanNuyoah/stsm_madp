# STSM-MADP 固定四组运行流程
codex --sandbox danger-full-access --ask-for-approval never

docker：docker exec -it stsm_melodic_dev bash

本文档是当前工作区的主流程入口。固定只跑原有四组：

- `arm / baseline`
- `arm / stsm`
- `wheelchair / baseline`
- `wheelchair / stsm`

`stsm` 是当前完整 STSM-MADP 流程，包含拓扑通道、风险门控、MPC/轨迹修正以及 ADP critic。baseline 节点内部会强制关闭 ADP。

## 0. 每个新终端先加载环境

```bash
cd /home/sun/LLL/catkin_ws/src
source /opt/ros/melodic/setup.bash
source /home/sun/elfin_assist_ws/devel/setup.bash
source /home/sun/LLL/catkin_ws/devel/setup.bash
```

修改代码后重新编译：

```bash
catkin_make -C /home/sun/LLL/catkin_ws
```

如果 Gazebo/RViz 或 ROS 节点残留，先清理：

```bash
pkill -INT -f "roslaunch stsm_madp arm_view.launch" || true
pkill -INT -f "roslaunch stsm_madp wheelchair_view.launch" || true
pkill -INT -f "roslaunch stsm_madp arm_action.launch" || true
pkill -INT -f "roslaunch stsm_madp wheelchair_action.launch" || true
pkill -INT -f "stsm_handover|stsm_wheelchair|stsm_social_field_viz|stsm_metrics" || true
pkill -INT -f "gzserver.*eldercare_room.world|gzclient" || true
```

## 1. 一次性跑完四组实验

推荐用于正式出结果。默认不打开 Gazebo GUI 和 RViz，稳定、速度快，适合批量采集指标和轨迹：

```bash
cd /home/sun/LLL/catkin_ws/src
TARGET=all GUI=false RVIZ=false PLOT=true CLEAN_ENV=true \
  bash stsm_madp/scripts/run_experiments.sh
```

如果想边跑边看窗口：

```bash
TARGET=all GUI=true RVIZ=true PLOT=true CLEAN_ENV=true \
  bash stsm_madp/scripts/run_experiments.sh
```

脚本会自动执行：

1. 编译 `/home/sun/LLL/catkin_ws`
2. 启动机械臂场景 `arm_view.launch`
3. 运行 `arm / baseline`
4. 运行 `arm / stsm`
5. 关闭机械臂场景并切换到轮椅场景
6. 启动 `wheelchair_view.launch`
7. 运行 `wheelchair / baseline`
8. 运行 `wheelchair / stsm`
9. 收集汇总 CSV
10. 生成结果图

`RUN_ID` 默认自动生成，格式为 `YYYYMMDD_R###`，例如 `20260707_R001`。编号只统计当天已有目录；日期变化后会重新从 `R001` 开始。一般不要手动指定。确实需要指定时：

```bash
RUN_ID=20260707_R001 TARGET=all GUI=false RVIZ=false PLOT=true CLEAN_ENV=true \
  bash stsm_madp/scripts/run_experiments.sh
```

## 2. 只跑一类实验

只跑机械臂两组：

```bash
TARGET=arm GUI=false RVIZ=false PLOT=true CLEAN_ENV=true \
  bash stsm_madp/scripts/run_experiments.sh
```

只跑轮椅两组：

```bash
TARGET=wheelchair GUI=false RVIZ=false PLOT=true CLEAN_ENV=true \
  bash stsm_madp/scripts/run_experiments.sh
```

清空旧的扁平结果文件后再跑：

```bash
CLEAN=true TARGET=all GUI=false RVIZ=false PLOT=true CLEAN_ENV=true \
  bash stsm_madp/scripts/run_experiments.sh
```

## 3. 分开查看仿真

查看仿真时建议分两个终端：终端 A 只负责场景和 RViz，终端 B 只负责动作节点。这样 baseline 和 STSM 可以在同一个场景入口下分别运行。

### 3.1 机械臂仿真

终端 A 启动机械臂场景：

```bash
source /opt/ros/melodic/setup.bash
source /home/sun/elfin_assist_ws/devel/setup.bash
source /home/sun/LLL/catkin_ws/devel/setup.bash

roslaunch stsm_madp arm_view.launch gui:=true rviz:=true
```

终端 B 跑 baseline：

```bash
source /opt/ros/melodic/setup.bash
source /home/sun/elfin_assist_ws/devel/setup.bash
source /home/sun/LLL/catkin_ws/devel/setup.bash

roslaunch stsm_madp arm_action.launch baseline:=true
```

终端 B 跑 STSM：

```bash
roslaunch stsm_madp arm_action.launch baseline:=false
```

观察重点：

- Gazebo 中机械臂末端是否绕开老人头部/胸部高风险区。
- RViz 中 `/stsm/social_field_markers` 风险场是否显示正常。
- STSM 路径是否比 baseline 更少贴近头胸区域。
- 终端日志中是否出现 `selected corridor: morse_handover_...`；如果出现 `visible_front`，说明拓扑层失败后走了 fallback。

### 3.2 轮椅仿真

终端 A 启动轮椅场景：

```bash
source /opt/ros/melodic/setup.bash
source /home/sun/elfin_assist_ws/devel/setup.bash
source /home/sun/LLL/catkin_ws/devel/setup.bash

roslaunch stsm_madp wheelchair_view.launch gui:=true rviz:=true
```

终端 B 跑 baseline：

```bash
source /opt/ros/melodic/setup.bash
source /home/sun/elfin_assist_ws/devel/setup.bash
source /home/sun/LLL/catkin_ws/devel/setup.bash

roslaunch stsm_madp wheelchair_action.launch baseline:=true
```

终端 B 跑 STSM：

```bash
roslaunch stsm_madp wheelchair_action.launch baseline:=false
```

观察重点：

- baseline 是否直接向泊靠目标运动。
- STSM 是否选择 `graph_direct_*`、`morse_saddle_*` 或 `morse_minima_*` 通道。
- 是否避开床边转移区、老人附近区域和桌边高风险区域。
- 终端日志中是否记录 `num_critical_*`、`num_topology_edges`、`num_candidate_corridors` 对应的拓扑信息。

## 4. 分开查看结果

最近一次结果：

```bash
ls -l stsm_madp/results/latest
readlink -f stsm_madp/results/latest
```

某次完整运行目录：

```text
stsm_madp/results/runs/<RUN_ID>/
```

四组实验文件位置：

```text
stsm_madp/results/runs/<RUN_ID>/arm/baseline/metrics.csv
stsm_madp/results/runs/<RUN_ID>/arm/baseline/traj.csv
stsm_madp/results/runs/<RUN_ID>/arm/baseline/ros.log

stsm_madp/results/runs/<RUN_ID>/arm/stsm/metrics.csv
stsm_madp/results/runs/<RUN_ID>/arm/stsm/traj.csv
stsm_madp/results/runs/<RUN_ID>/arm/stsm/ros.log

stsm_madp/results/runs/<RUN_ID>/wheelchair/baseline/metrics.csv
stsm_madp/results/runs/<RUN_ID>/wheelchair/baseline/traj.csv
stsm_madp/results/runs/<RUN_ID>/wheelchair/baseline/ros.log

stsm_madp/results/runs/<RUN_ID>/wheelchair/stsm/metrics.csv
stsm_madp/results/runs/<RUN_ID>/wheelchair/stsm/traj.csv
stsm_madp/results/runs/<RUN_ID>/wheelchair/stsm/ros.log
```

查看四组是否成功：

```bash
cat stsm_madp/results/runs/<RUN_ID>/manifest.csv
```

查看汇总指标：

```bash
column -s, -t < stsm_madp/results/summary/all_metrics.csv
column -s, -t < stsm_madp/results/summary/ablation_table.csv
column -s, -t < stsm_madp/results/summary/best_runs.csv
```

虽然文件名仍叫 `ablation_table.csv`，当前内容只汇总固定四组基础实验，不再表示消融实验。

## 5. 重新生成结果图

生成精选论文图：

```bash
python3 stsm_madp/scripts/visualization/plot.py paper --run-id <RUN_ID>
```

生成全量调试图：

```bash
python3 stsm_madp/scripts/visualization/plot.py all --run-id <RUN_ID>
```

只生成拓扑通道图：

```bash
python3 stsm_madp/scripts/visualization/plot.py topology
```

常用输出：

```text
stsm_madp/results/paper_figures/final/arm_compare.png
stsm_madp/results/paper_figures/final/wheelchair_compare.png
stsm_madp/results/paper_figures/final/arm_risk_field_path.png
stsm_madp/results/paper_figures/final/wheelchair_risk_field_path.png
stsm_madp/results/paper_figures/final/arm_topology.png
stsm_madp/results/paper_figures/final/wheelchair_topology.png
```

## 6. 关键指标怎么看

基础成功指标：

```text
success_goal
success_safe
stop_triggered
stop_reason
duration_s
J_social
risk_exceed_pct
final_dist_to_goal
```

机械臂的 `success_goal` 不再只表示末端到达手部目标。现在必须同时满足：

```text
arm_reached_hand = 1
arm_hold_completed = 1
arm_retreat_started = 1
arm_wait_reached = 1
arm_home_return_started = 1
arm_home_returned = 1
arm_task_complete = 1
```

也就是完成“递到手部目标 -> 保持递物 -> 回退到 wait_pose -> 回到 home”的完整流程后，机械臂 `success_goal/success_safe` 才会记为成功。只到达手部目标但没有回退和回 home 时，`arm_reached_hand=1`，但 `arm_task_complete=0`、`success_goal=0`。

轮椅拓扑通道指标：

```text
selected_corridor_label
topology_used
topology_fallback_used
num_critical_minima
num_critical_saddles
num_topology_edges
num_candidate_corridors
corridor_base_cost
corridor_total_cost
final_approach_used
```

轮椅 STSM 默认只在启动时计算一次 Morse 拓扑通道。控制循环中不会按固定周期重复计算拓扑图；只有明显脱离通道或长时间无进展，并且距离上次拓扑重规划超过 `topology/replan_min_interval=30.0s`，才会尝试重规划。这样可以避免拓扑图/A* 计算阻塞 `/cmd_vel` 发布。

机械臂安全和递物指标：

```text
selected_corridor_label
min_head_dist
min_chest_dist
mean_phi_arm_max_point
max_phi_arm_max_point
path_adp_mean
path_adp_max
arm_solver_success_rate
```

ADP 相关指标：

```text
adp_enabled
critic_version
mean_adp_value
max_adp_value
terminal_adp_cost
path_adp_mean
path_adp_max
```

baseline 中 `adp_enabled` 应为 0；STSM 中是否启用 ADP 由 `ADP_ENABLED` 控制，但实验目录仍固定叫 `stsm`。

## 7. 常用参数

批量脚本：

```text
TARGET=all|arm|wheelchair
GUI=true|false
RVIZ=true|false
PLOT=true|false
CLEAN=true|false
CLEAN_ENV=true|false
RUN_ID=YYYYMMDD_R###        # 可选；默认自动生成
KEEP_ON_FAIL=true|false
```

ADP 参数：

```text
ADP_ENABLED=true|false      # 只影响 STSM；baseline 内部强制关闭
ADP_MODEL=<path>
LAMBDA_ADP=0.005
LAMBDA_ADP_CORRIDOR=0.05
LAMBDA_ADP_TERMINAL=0.0015
LAMBDA_ADP_PATH=0.005
LAMBDA_ADP_ARM=0.008
ADP_SOLVER_MODE=dls_adp
USE_CVXPY=false
ADP_DEBUG=false
```

轮椅完成和重规划参数：

```text
WC_COMPLETION_TOLERANCE=0.25
WC_COMPLETION_HOLD_S=1.5
WC_MAX_RUNTIME_S=180.0
WC_NO_PROGRESS_TIMEOUT_S=45.0
WC_REPLAN_PERIOD=5.0
WC_NO_PROGRESS_REPLAN_TIME=5.0
WC_REPLAN_TUBE_MARGIN=0.08
WC_NEAR_GOAL_RADIUS=0.50
WC_FINAL_HEADING_THRESHOLD=0.75
WC_FINAL_HEADING_GAIN=1.6
WC_FINAL_CREEP_V=0.10
WC_FINAL_MIN_V=0.16
```

轮椅拓扑规划参数：

```text
topology/grid_resolution=0.15
topology/k_paths=2
topology/max_graph_nodes=24
topology/periodic_replan=false
topology/replan_min_interval=30.0
topology/replan_on_tube_exit=true
topology/replan_on_no_progress=true
```

兴趣点/门控参数：

```text
ARM_INTEREST_ENABLED=true
ARM_INTEREST_GATE_ENABLED=true
ARM_INTEREST_RHO_WARN=3.5
ARM_INTEREST_RHO_STOP=6.0
WC_INTEREST_ENABLED=true
WC_INTEREST_GATE_ENABLED=true
WC_FOOTPRINT_FORBIDDEN_STOP_ENABLED=true
```

机械臂递物必须接近手部目标，接近头胸风险场是任务几何上不可避免的。因此默认机械臂中心 EE 门控为 `rho_warn=3.5`、`rho_stop=6.0`，风险较高时主要减速和记录指标，只有极端风险才 STOP。腕部、肘部、夹持物等兴趣点门控默认启用，实时取最坏兴趣点风险参与 SLOW/STOP。

轮椅前缘、脚踏、后角等 footprint 兴趣点门控默认启用，实时取最坏 footprint 风险参与 SLOW/STOP；中心点门控和 footprint 门控会组合，最终使用更保守的控制约束。

## 8. 故障定位

环境刚启动就退出：

```bash
tail -120 stsm_madp/results/runs/<RUN_ID>/config/arm_view_env.log
tail -120 stsm_madp/results/runs/<RUN_ID>/config/wheelchair_view_env.log
```

某一组动作失败：

```bash
tail -160 stsm_madp/results/runs/<RUN_ID>/<robot>/<variant>/ros.log
```

检查某次运行所有文件：

```bash
find stsm_madp/results/runs/<RUN_ID> -maxdepth 3 -type f | sort
```

检查 metrics/traj 是否缺失：

```bash
cat stsm_madp/results/runs/<RUN_ID>/manifest.csv
```

出现 `entity already exists`：

- 旧 Gazebo 没关干净。
- 先执行第 0 节清理命令，再重新启动。

出现 `OverflowError: int exceeds XML-RPC limits`：

- 不要用纯数字 `RUN_ID`。
- 使用 `YYYYMMDD_R###`，例如 `20260707_R001`。

轮椅长时间慢速转向或原地：

- 看 `ros.log` 中的 `dist` 和 `cmd=(v,w)`。
- 看 `metrics.csv` 的 `stop_triggered`、`stop_reason`、`final_dist_to_goal`。
- 默认 STSM 进入 `WC_COMPLETION_TOLERANCE=0.25` 并保持 `WC_COMPLETION_HOLD_S=1.5` 秒后结束。

拓扑层没有生效：

- 看 `selected_corridor_label` 是否以 `morse_` 开头。
- 看 `topology_used` 是否为 1。
- 看 `topology_fallback_used` 是否为 1；为 1 时表示拓扑生成失败，系统走了旧 fallback。
