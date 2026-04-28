# Prompt for GLM5.1 — 撰写 MPhil 论文 Chapter 3: Proposed Method

## 任务
撰写chapter3.tex，约50页LaTeX内容，介绍Proposed Method。

## 绝对禁止
1. **不得出现任何"Instinct"相关字样**（Instinct, instinct_rl, instinctlab, instinct_onboard等）
2. **不得在本章介绍消融实验的具体工作**，但需介绍与消融实验不同的关键设计细节
3. 数据准备/采集不在本章

## 论文规范
- 学术英语，风格参考chapter2.tex（正式、严谨、引用充分）
- HKUST(GZ) MPhil Thesis格式
- 用\section/\subsection/\subsubsection, \begin{figure}, \begin{equation}, \cite{...}

## 章节结构
```
\chapter{Proposed Method}
\section{Overview}
\section{Policy Input Representation}
\section{Memory Module}
\section{Attention Module}
\section{Mixture-of-Experts Module}
\section{Adversarial Motion Prior Module}
\section{Implementation Details}
```

---

## 3.1 Overview
- 整体框架图：depth_image + proprio_obs → Memory-Attention Encoder → [memory_latent(128), attention_latent(8), depth_latent(128), non_depth_raw] → MoE Actor-Critic → joint position action
- 与现有方法核心区别：现有用height scanner或单帧depth CNN，本方法用temporal-subsampled multi-frame depth + GRU memory + cross-attention + MoE
- 强调target-driven parkour挑战和针对性设计

## 3.2 Policy Input Representation

**Proprioceptive observations** (history_length=8, flatten):
- base_ang_vel(3)+noise U(-0.2,0.2), scale=0.25
- projected_gravity(3)+noise U(-0.05,0.05)
- velocity_commands(3): PoseVelocityCommand基于target位置
- joint_pos_rel(29)+noise U(-0.01,0.01)
- joint_vel_rel(29)+noise U(-0.5,0.5), scale=0.05
- last_action(29)
- Total proprio: 8×(3+3+3+29+29+29)=768

**Depth image observations (关键创新)**:
- 传感器: RayCaster Camera on torso_link, 64×36, FOV 89.51°×58.29°
- **Temporal subsampling**: 每隔history_skip_frames=5帧采样一次，共num_output_frames=8帧堆叠
- 论文中称"temporal-subsampled depth image stack"而非"sparse depth"
- Noise pipeline: CropAndResize(crop=(18,0,16,16))→GaussianBlur(k=3,σ=1)→DepthNormalization(0-2.5m→0-1)
- Delayed frame: delayed_frame_ranges=(0,1)模拟传感器延迟
- Shape: [8, 36, 64]
- **对比**: 现有用height scanner(1D)或单帧depth CNN；本方法8帧间隔5步覆盖~0.8s时序深度信息

**Critic观测**: 额外base_lin_vel(3)×8=24维, 无noise

**Target-driven command**: PoseVelocityCommand在地形上采样target，计算robot→target方向向量，velocity_control_stiffness=2.0, heading_control_stiffness=2.0, 不同地形不同速度范围(perlin_rough: vx∈[0.45,1.0]), target_dis_threshold=0.4m

## 3.3 Memory Module

**设计理念**: parkour需长期地形记忆，现有RNN hidden state容量有限。

**网络结构**:
- non_depth_raw → Linear(→128) → state_proj
- [state_proj(128) | depth_latent.detach(128)] = 256维 → **GRUCell**(input=256, hidden=128) → memory_t(128)

**Memory Consistency Loss (关键创新)**:
- memory_target_head: 128→128→ELU→64→ELU→128
- Loss = MSE(normalize(memory_target_head(memory_t)), normalize(depth_latent.detach()))
- **动机**: 曾尝试预测未来1s的depth_latent(失败：学习信号不足，1s精确时间点难学，身体位姿变化剧烈)。最终方案：当前时刻MSE loss——在memory中写入当前地形信息/解释，memory持续生效一段时间
- GRU hidden state在episode done时清零，跨rollout存储

**对比**: DreamWaQ等RNN隐式记忆容量有限；本方法显式GRU+consistency loss确保memory编码有意义地形表示

## 3.4 Attention Module

**设计理念**: 深度图不同区域重要性不同，脚下障碍比远处更关键。

**Cross-Attention结构**:
- K/V: depth_obs [N,8,36,64] → permute→reshape [N,2304,8]
- Q: non_depth_raw → Linear(→8) → unsqueeze(1) → [N,1,8]
- MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True) → attn_out [N,8]
- embed_dim必须=depth frame count(8)
- 2个head学习不同空间关注模式(近处脚下/远处路径)
- attention weights可reshape回[N,2,36,64]可视化

**对比**: He et al. attention map encoding处理单帧heightmap；本方法用cross-attention让本体感知查询多帧深度图关键区域，更适配parkour场景

## 3.5 Mixture-of-Experts Module

**设计理念**: parkour需要多样技能(走/爬/跳/跨)，单一网络难以同时精通。

**MoE Layer结构**:
- Gate: Linear(input_dim, num_experts) → softmax → gate_scores [batch, 8]
- 8个Expert: 各为MLP(hidden_dims=[256,256,256]+output_dim)
- Output: einsum(gate_scores, expert_outputs) → 加权混合
- Actor和Critic各用独立MoE

**对比**: 现有skill-chaining方法(Extreme Parkour)需显式技能分解+高层选择器；本方法MoE实现隐式技能路由，端到端训练，无需显式技能定义

## 3.6 Adversarial Motion Prior Module

**AMP框架**:
- Discriminator区分policy运动(amp_policy)与参考运动(amp_reference)
- amp_policy观测: projected_gravity(3)+joint_pos_rel(29)+joint_vel_rel(29,scale=0.05)+base_lin_vel(3)+base_ang_vel(3), history_length=10
- amp_reference观测: 同结构但来自motion reference(重定向后的人类动捕数据)
- AMP reward与task reward加权组合训练
- 参考数据: 自采动捕数据(走路/跑步/上楼梯等)，经retargeting到G1机器人

**对比**: 现有AMP工作多用公开数据集或简单行走；本方法用自采parkour相关动捕+retargeting，覆盖更丰富的运动模式

## 3.7 Implementation Details

**奖励函数** (分3类):
- Task rewards: track_lin_vel_xy_exp(w=2.0,std=0.5), track_ang_vel_z_exp(w=2.0,std=0.5), heading_error(w=-1.0), dont_wait(w=-0.5), is_alive(w=3.0), stand_still(w=-0.3,offset=4.0)
- Regularization: volume_points_penetration(w=-4.0), feet_air_time(w=0.5), feet_slide(w=-0.4), joint_deviation_hip(w=-0.5), ang_vel_xy_l2(w=-0.05), dof_torques_l2(w=-1.5e-7), dof_acc_l2(w=-1.25e-7), dof_vel_l2(w=-1e-4), action_rate_l2(w=-0.005), flat_orientation_l2(w=-3.0), pelvis_orientation_l2(w=-3.0), feet_flat_ori(w=-0.4), feet_at_plane(w=-0.1,height_offset=0.035/0.058withshoe), feet_close_xy(w=0.4,threshold=0.12), energy(w=-5e-5), freeze_upper_body(w=-0.004)
- Safety: dof_pos_limits(w=-1.0), dof_vel_limits(w=-1.0,soft=0.9), torque_limits(w=-0.01,limit_ratio=0.8), undesired_contacts(w=-1.0,threshold=1.0,exclude ankle)

**Memory Consistency Loss训练**: 作为额外loss加入PPO更新，total_loss = PPO_loss + λ * memory_consistency_loss

**环境设置**:
- 机器人: Unitree G1 29-DOF (torso base), with shoe URDF
- Actuators: delayed PD actuators
- 4096 parallel envs, decimation=4, dt=0.005s, episode=20s
- 地形: 10类(perlin_rough, square_gaps, pyramid_stairs, boxes, slopes等), 10rows×20cols, 每块8m×8m, 带wall_prob=0.3的虚拟墙
- Curriculum: tracking_exp_vel基于速度跟踪表现调整地形难度
- Virtual obstacles: GreedyconcatEdgeCylinder检测地形尖锐边缘，用于penetration penalty

**Sensor Randomization (关键，与A4消融不同)**:
- Camera offset randomization: 位置(x,y,z)和姿态(roll,pitch,yaw)随机偏移，模拟安装误差
- Full noise pipeline: CropAndResize + GaussianBlur + DepthNormalization + RandomGaussianNoise + DepthArtifactNoise
- Delayed frames: (0,1)帧随机延迟

**Domain Randomization**:
- Physics material: friction(0.3-1.6), restitution(0.05-0.5), 64 buckets
- Reset base: position±0.1m, velocity±0.2
- Reset joints: position±0.15

**Termination**: time_out, terrain_out_bound(2m), base_contact(torso>1.0), bad_orientation(>1.0rad), root_height(<0.5m), dataset_exhausted

**Symmetric Augmentation**: 对motion reference做左右对称增强，joint_mapping=[0,1,3,2,5,4,7,6,9,8,11,10,13,12]

**Volume Points Penetration**: 在脚踝周围生成3D网格点(10×5×2)，检测穿入虚拟障碍的深度×速度，惩罚w=-4.0

**Sim-to-Real**: Policy导出为ONNX，encoder和actor分离，memory hidden state外部循环反馈

## 各模块定性效果与行为分析（在对应section末尾阐述）

### Attention Module 的效果表现
- **学会了避让行为**：通过可视化attention weight矩阵，发现策略对可能影响姿态/导致绊倒的障碍物有额外的注意力聚焦
- **墙体避让**：当深度图显示前方有墙体时，attention权重集中在墙体区域，策略学会停止前行而非盲目冲撞
- **鸿沟跨越的精细行为**：attention引导策略学会先走到沟边（而非直接跳入），再调整重心，最后一大步跨过——这是一个两阶段决策过程，attention让策略能区分"接近沟边"和"跨越鸿沟"两种不同状态
- **与无attention方案的对比**：仅有AMP的方案虽然通过AMP学到了自然姿态、通过depth camera学会初步避让，但无法关注脚下的stone等小障碍，更不会谨慎地预先抬脚——这正是当前AMP-based locomotion工作的痛点：近处盲区障碍物处理不足

### Memory Module 的效果表现
- **初期高抬脚行为**：episode开始时，memory尚未积累足够的地形信息，但velocity command又强制机器人前行。策略学到了一种鲁棒的应对方式——前几步大幅抬脚（这并非训练bug，而是策略在信息不足时的保守策略：宁可多抬脚也不愿被未知障碍绊倒）
- **记忆积累后的泰然行走**：当memory积累了足够的GRU hidden state后（约几步之后），策略对地形已有充分理解，可以平稳高效地行走，不再过度抬脚
- **这一行为模式的意义**：说明memory模块确实在编码和保持地形信息，且策略学会根据memory的"置信度"调整行为——memory空时保守，memory满时高效
- **与无memory方案的对比**：无memory方案无法维持对地形的长期理解，在复杂地形中反复"遗忘"已走过的地形特征，导致在远处障碍处理上摔跤

### 综合效果
- Memory提供"地形是什么"的长期理解，Attention提供"当前该关注哪里"的即时聚焦，二者互补
- Memory解决"远处障碍遗忘"问题（时间维度），Attention解决"近处障碍忽视"问题（空间维度）
- AMP提供自然运动风格底座，MoE提供多技能路由能力
- 完整方案 = AMP(风格) + Memory(时间理解) + Attention(空间聚焦) + MoE(技能路由) + Temporal-subsampled depth(丰富感知)

## 写作要求
1. 每个模块先讲motivation(为什么需要)→设计→实现细节→定性效果分析→与现有方法对比
2. 对比时引用chapter2中的相关工作(DreamWaQ, RPL, HumanoidParkour, DeepWholeBodyParkour, AMP, He et al. attention map等)
3. 图表：至少包含整体架构图、memory-attention encoder详细结构图、深度图temporal subsampling示意图、MoE结构图、attention weight可视化示例
4. 数学公式：PPO objective, AMP discriminator/generator objective, MoE gate softmax, memory consistency loss, reward function各项
5. 语气客观，避免"we propose"过度使用，用"the proposed framework employs"等
