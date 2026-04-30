# Parkour 消融分析写作操作指南

这份文档是给论文正文消融对比部分用的操作说明。目标很简单：只把 BASE 更好的内容写进正文；如果某个消融方案结果更好，就不要把它写成“BASE 优势”。

## 1. 先把实验分组

先不要急着写表格。先按可比性把现有 logs 分成两组。

### 1.1 主文可比组

这一组用于论文主表和主图，要求是同一条训练协议、同一环境规模、同一任务定义。

- BASE: `logs/instinct_rl/g1_parkour_memory_attention_amp/20260424_032053`
- A1: `logs/instinct_rl/g1_parkour/20260323_161738` - 使用 Instinct-Parkour-Target-Amp-G1-v0，去掉 memory-attention encoder，使用基础 Conv2d depth encoder
- A2: `logs/instinct_rl/g1_parkour_a1_no_amp/20260327_205312`
- A3: `logs/instinct_rl/g1_parkour_a2_no_depth/20260328_002933`
- A4: `logs/instinct_rl/g1_parkour_a3_moe1/20260328_115443`
- A5: `logs/instinct_rl/g1_parkour/20260329_102309`
- A6: `logs/instinct_rl/g1_parkour/20260331_033444`
- A7: `logs/instinct_rl/g1_parkour/20260417_000159`

这些 run 都是 2048 env，BASE 也是 2048 env，因此可以先放在同一张主表里做对照。

### 1.2 只放补充材料的组

这一组不要和主表混在一起，因为环境规模不同，直接比会不严谨。

- A8: `logs/instinct_rl/g1_parkour_a7_recurrent/20260407_214641`
- A9: `logs/instinct_rl/g1_parkour_attention/20260411_174400__attention`
- A10: `logs/instinct_rl/g1_parkour_attention_amp/20260404_113829__attention_amp`

这三组是 256 env，和 BASE 的 2048 env 不同，所以只能作为补充实验单独描述。

## 2. 先判断每个消融的写法

写正文时只保留“BASE 更好”的结论。下面是建议写法。

### 2.1 建议重点写进正文的项目

- A1: 去掉 memory-attention encoder，使用基础 Conv2d depth encoder，适合说明记忆注意力机制的作用。
- A2: 去掉 AMP reward，适合说明 AMP 对对齐人体运动风格的作用。
- A3: 去掉 depth 分支，适合说明视觉输入的必要性。
- A4: 把 MoE 压成单 expert，适合说明多专家路由的收益。
- A7: 去掉 symmetric augmentation，适合说明对称增强对泛化的帮助。

如果这些消融在指标上优于 BASE，就不要在正文写成 BASE 的优势，只保留事实描述或删掉那句结论。

### 2.2 更适合放在补充说明的项目

- A5: 降低 sensor randomization 和 delayed frame。
- A6: 去掉 penetration shaping。
- A8: recurrent policy。
- A9: attention policy without AMP。
- A10: attention policy with AMP。

这些更像结构变化或训练设置变化，不适合和 BASE 放成一个“单变量消融主表”直接下结论。

## 3. 你要先收集什么数据

只靠 `export.md` 不够，`export.md` 只是记录导出命令，不是结果表。你需要从每个 run 的事件文件里导出标量。

每个 run 目录里已经有：

- `events.out.tfevents.*`
- `model_5000.pt`, `model_10000.pt`, `model_15000.pt`, `model_20000.pt`, `model_25000.pt`, `model_30000.pt`
- `params/agent.yaml`
- `params/env.yaml`

先只做这三类数据：

1. 训练曲线标量
2. 最终 checkpoint 对应的评估结果
3. 必要时的视频或导出模型对应的定性结果

## 4. 实际操作步骤

### 第一步：确认比较对象

只比较同组实验。

- 主表：BASE, A1, A2, A3, A4, A5, A6, A7
- 补充表：A8, A9, A10

### 第二步：导出每个 run 的标量

对每个 run 做同一件事：

1. 打开对应目录下的 `events.out.tfevents.*`
2. 导出你论文要用的指标
3. 统一整理成一个 csv 或 markdown 表格

建议至少导出这些字段：

- `train/episode_reward`
- `train/total_loss`
- `train/average_episode_length`
- `eval/episode_reward`
- `eval/success_rate`
- `eval/fall_rate`
- `eval/return`

如果你的日志里名字不一样，就用最接近的字段，但所有 run 必须用同一套字段。

### 第三步：只保留 BASE 更好的行

每个维度都按这个规则处理：

1. 先和 BASE 对比。
2. 如果 BASE 更好，就保留到正文。
3. 如果消融更好，就不要写成 BASE 优势。
4. 如果差异很小，没有稳定优势，就不要强行写结论。

推荐的判断标准：

- `success_rate` 更高：优先保留
- `fall_rate` 更低：优先保留
- `return` 更高：可保留
- `episode_length` 更稳定：可作为辅助说明

### 第四步：把主文分成两个层次写

#### 主文第一层：结果表

只放最关键的一张表。

建议列：

- 方法名
- 是否包含 AMP
- 是否包含 depth
- 是否包含 memory-attention encoder
- 是否使用 MoE
- 是否使用 symmetric augmentation
- 最终 `success_rate`
- 最终 `return`
- 最终 `fall_rate`

#### 主文第二层：结果解释

按模块写一小段话。

- AMP：去掉后，风格对齐和动作自然性下降。
- Depth：去掉后，地形理解能力下降。
- MoE：压成单 expert 后，多技能路由能力下降。
- Symmetric augmentation：去掉后，泛化能力下降。

注意：只有当 BASE 确实更好时才写这几句。

## 5. 建议你画哪些图

### 图 1：主表配套的总览柱状图

画最终评估指标柱状图，横轴是方法，纵轴是 `success_rate` 或 `return`。

推荐只画主表组。

### 图 2：训练曲线图

每个关键指标画一张曲线。

- `eval/success_rate` vs iteration
- `eval/return` vs iteration
- `train/total_loss` vs iteration

### 图 3：定性行为图

选 2 到 3 个最能说明问题的场景。

- 复杂地形跨越
- 近距离障碍避让
- 记忆起作用的连续路段

如果消融方案表现更好，这类图不要只截 BASE 的成功片段，要老老实实把事实写清楚。

## 6. 建议你在论文里怎么写

可以直接按下面顺序写：

1. 先写“BASE 的整体定义”。
2. 再写“我们逐个移除关键模块进行消融”。
3. 再写“结果表说明哪些模块对性能有贡献”。
4. 最后写“哪几个模块是主要收益来源”。

推荐写法模板：

- `Removing AMP degrades ...`
- `Removing depth input reduces ...`
- `Collapsing MoE to a single expert weakens ...`
- `Disabling symmetric augmentation hurts ...`

如果某个消融优于 BASE，就改成中性写法，例如：

- `The ablated variant shows competitive performance on ...`

不要写成 BASE 更强。

## 7. 当前日志能否完成这部分

可以完成框架和分组，也可以完成“哪些内容该写、哪些不该写”。

但如果你要写最终定量结论，当前还缺一步：把 TensorBoard 标量导出来。

也就是说：

- 现在已经足够整理结构和写作提纲
- 还不够直接写最终数值表

## 8. 最后执行顺序

你就按这个顺序做，最不容易出错：

1. 先只整理主表组和补充组。
2. 再导出每个 run 的标量。
3. 再做 BASE vs ablation 的比较。
4. 只把 BASE 更好的结果写进正文。
5. 如果某个消融更好，就删掉对应“BASE 优势”句子。
