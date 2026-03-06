# VR 多步态人形遥操作（A+B）鲁棒训练计划说明

> 项目目标：在 VR 遥操作场景下，实现 **三步态（stand / walk / run + transitions）** 的稳定下肢行走，同时支持 **复杂上肢/手部动作**；鲁棒性体现在 **sim2sim** 与 **sim2real** 下都不容易摔、不容易在切换时崩。

---

## 1. 背景与动机

### 1.1 任务特点（VR 场景）
- VR 遥操作中：**手/臂动作更复杂、更高频、更不规则**；会显著改变躯干姿态与角动量。
- 腿部相对“职责单纯”：主要承担 **可靠行走、步态切换、扰动恢复**。
- 如果用一个全身策略同时学“上肢表达 + 下肢接触行走”，训练难度高且鲁棒性差（上肢目标变化会频繁把腿带崩）。

### 1.2 相关工作启发
- **HugWBC**：强调“上半身外部干预/遥操作下 locomotion 仍需稳定”，并通过 intervention training 提升鲁棒性。
- **FALCON**：强调“上下肢解耦”能减少策略冲突、提升训练稳定性与鲁棒性。
- **H2O**：通过“预训练特权模仿器/可行性筛选”从海量人类动作中筛出机器人可执行片段，构建大规模可行动作库，并训练实时模仿策略以实现零样本 sim2real。

---

## 2. 核心思路：A + B（责任解耦 + 可执行先验 + 干预鲁棒训练）

### A：上肢可执行动作先验（Feasible Upper-Body Motion Prior，H2O-style）
目标：保证上肢动作“丰富但可执行”，避免 VR 输入产生不可达/自碰撞/关节饱和/扭腰过猛等不良行为，降低对腿部的灾难性扰动。

**A 的产物：**
- 一个“可执行上肢动作库/先验”（来自人类动作数据或 VR 录制轨迹，经重定向+筛选得到）。
- 或一个可行性判别器（feasibility filter），用于在线判断/投影上肢目标是否可执行。

### B：腿部抗干预鲁棒多步态策略（Intervention-Robust Multi-Gait Locomotion）
目标：训练一个 gait-conditioned 下肢策略 π_leg，使其在上肢干预持续存在时仍能：
- 稳定行走 / 站立；
- 平滑切换（walk↔run, walk↔stand 等）；
- 抗扰恢复（push/扭腰/手抖带来的扰动）。

**B 的训练关键：**
- 训练时“上肢被接管”不是例外，而是常态（intervention-aware training）。
- 使用 **Phase 0–4 课程**从易到难引入干预，并在 Phase 4 对齐真实人体/VR 干预分布。
- 可选：用 **ADR / LP-Teacher** 将课程从“手工 phase”升级为“自动调度”。

---

## 3. 系统接口设计（上肢干预信号：手/末端目标接口）

### 3.1 上肢控制接口（推荐）
- 干预输出为 **手目标位姿**：
  \[
  x^{des}_{hand}(t) = [p^{des}(t), R^{des}(t)]
  \]
- VR 输入 → 重定向（坐标对齐/尺度归一/可达投影）→ 滤波限幅（低通 + rate/jerk limit）→ IK/WBC 映射到上肢关节目标/力矩。

### 3.2 腿优先协调（必须）
- 上肢追踪必须受限于“腿优先”安全约束：
  - torso roll/pitch 与角速度阈值限制；
  - 手目标变化率限制；
  - IK/WBC 解不稳时回退到最近可行目标或 neutral pose。

---

## 4. 训练流程总览

### 4.1 训练对象
- 主要训练对象：**腿部策略 B（π_leg）**。
- 上肢干预通过 A 生成（训练时上肢被外部信号接管），用于构造“真实扰动分布”。

### 4.2 干预难度参数化（用于课程/自动调度）
每个 episode 采样干预参数向量：
\[
d = [amp, freq, duty, transition\_stress, hold\_T]
\]
- `amp`：上肢目标幅值（手目标偏移尺度/关节目标偏移）
- `freq`：干预变化速度/频谱（低频连贯 + 可选高频抖动）
- `duty`：干预占空比（I(t)=1 的比例）
- `transition_stress`：步态切换窗口内干预加压倍率（或强制 I(t)=1）
- `hold_T`：目标保持时长（越短越难）

---

## 5. 课程学习（Phase 0–4）

> 目的：让腿部策略先建立稳定基本功，再逐步适应更强、更真实的上肢干预分布；重点强化“切换窗口”这一最脆弱点。

### Phase 0：无干预（基础）
- I(t)=0
- 学会：稳站、稳走、基本切换（低速）

### Phase 1：低频小幅 + 低占空比
- `amp` 小、`duty` 低、`hold_T` 大（慢变化）
- 让策略开始适应“上肢会动”但不至于崩溃

### Phase 2：加入转身/摆臂模式，中幅中占空比
- 增加结构化干预（turning / arm swing）
- `amp`、`duty` 增大；加入少量高频扰动

### Phase 3：强干预 + 切换窗口持续干预（重点）
- 干预更强、更频繁
- 切换窗口（例如前后 0.3–0.5s）：
  - `amp *= transition_stress` 或强制 I(t)=1
- 目标：切换时也不崩（transition success）

### Phase 4：真实人体/VR 干预对齐（分布对齐）
- 干预来自 VR 回放或 A 的可行动作库
- 采用混合课程：真实干预 mixing ratio 从 20% → 50% → 80% 逐步提高
- 仍保持可行性筛选与滤波限幅，避免“不可执行目标”破坏训练

---

## 6. A：训练集与可行性筛选（H2O-style）

### 6.1 数据来源
- VR 录制：手柄位姿、躯干朝向、时间戳（最贴目标分布）
- 人体动作数据：上半身相关关节/手轨迹（规模化）

### 6.2 处理流程
1) 重定向（retarget）：坐标对齐、尺度归一、可达投影
2) 信号整形：低通滤波 + rate/jerk 限幅
3) 可行性筛选（feasibility filter）：
   - IK 可解、关节限位不超、速度不超
   - 自碰撞检查
   - torso roll/pitch 与角速度不超阈值
4) 输出：
   - 可执行上肢动作库（可用于回放干预/训练上肢策略）
   - 或在线判别器（输入目标→输出可行/不可行 + 投影建议）

---

## 7. B：腿部策略（π_leg）训练细节

### 7.1 输入（建议最小可用）
- proprioception：IMU 姿态/角速度、腿部关节角/角速、足接触
- command：期望速度 (vx, vy, wz) + gait_id（站/走/跑）
- 可选增强：
  - history encoder（更抗延迟/摩擦/接触差异）
  - intervention indicator I(t)（让腿知道上肢是否接管）

### 7.2 输出
- 腿部关节目标（PD tracking）或残差（若有底层 WBC/PD）

### 7.3 奖励（最小闭环组合）
- 速度跟踪（walk/run）
- 姿态稳定（torso roll/pitch、角速度）
- 足滑移/拖脚惩罚
- 能耗与动作平滑
- 切换窗额外稳定项（或动作突变惩罚增强）

---

## 8. 自动调度（可选升级）：ADR / LP-Teacher

### ADR（最容易落地）
- 根据评估成功率 SR：
  - SR > τ_high → 扩大 amp_max、duty_max，减小 hold_T（更难）
  - SR < τ_low → 缩小范围或保持（防崩）
- 优点：实现简单、稳定、工程可复现

### LP-Teacher（更“CL论文味”）
- 将 d 空间离散成 bins，维护每个 bin 的学习进展：
  \[
  LP = |R_{recent} - R_{past}|
  \]
- 优先采样 LP 高的 bins（“最该练的前沿”）+ 少量随机探索防偏科

---

## 9. 评测与消融（必须做）

### 9.1 核心指标
- fall rate（跌倒率）
- transition failure（切换失败率）
- time-to-fall / survival time
- recovery time（扰动后恢复时间）
- torso roll/pitch peak（躯干峰值偏转）
- 手跟踪误差（上肢目标 vs 实际）与 torque saturation（关节饱和率）

### 9.2 sim2sim / sim2real
- sim2sim：改接触/solver/timestep/摩擦模型
- sim2real：真实延迟/摩擦差异/传感噪声下测试

### 9.3 关键消融
- 无课程（直接大范围强干预） vs Phase 0–4
- Phase 0–3（合成） vs 加 Phase 4（真实对齐）
- 手工 Phase vs ADR / LP-Teacher（若做自动调度）
- 有无 feasibility filter（A）的对比（证明 A 的必要性）

---

## 10. 里程碑（建议）

- M1：腿部 π_leg 在 Phase 0 稳定三步态（无干预）
- M2：完成 Phase 1–3 课程，切换窗口抗干预显著提升
- M3：完成 A 的上肢可行性筛选与动作库构建（VR回放/人体动作）
- M4：Phase 4 真实分布对齐，完成 sim2sim 报告
- M5：若有硬件：完成 sim2real 零样本/小校准测试与故障模式分析

---

## 11. 风险与对策

- **学成站桩（为稳不走）**：保持速度跟踪权重；Phase 1 duty 低起步；对“静止但不完成命令”降低回报
- **切换必崩**：transition_stress 单独 curriculum；切换窗加强动作平滑惩罚；切换任务采样比例提高
- **上肢干预太假（对腿影响小）**：监控 torso 峰值与角速度 RMS；必要时提高 amp/放宽上肢限制（但保持可行）
- **IK/WBC 抖动污染训练**：增加滤波/限幅；IK 不收敛回退策略；可行性筛选更严格

---

## 12. 交付物

- 训练代码：Phase 0–4 课程 + 上肢干预生成器（手目标接口）+ 可选 ADR/LP-Teacher
- 数据与工具：VR回放管线 / 人体动作重定向 / feasibility filter
- 报告与图：训练曲线、消融结果、失败案例分类、sim2sim/sim2real 对比
- 最终演示：VR 控制上肢动作 + 腿部三步态稳定行走与切换

---

## 13. 训练体系（代码已同步）

当前代码保留两套训练体系，并由 `scripts/train.py` 统一切换：

- `phase`：手工阶段课程（Phase 0-4），按阈值推进难度。
- `lp_teacher`：LP-Teacher 自动采样高学习进展区间。

训练入口：

```bash
python scripts/train.py --curriculum-system phase --use-intervention
python scripts/train.py --curriculum-system lp_teacher --use-intervention
```

对应配置：

- `configs/curriculum/phase.yaml`
- `configs/curriculum/lp_teacher.yaml`

说明：

- 两套体系共用同一套 PPO、环境和奖励实现。
- `phase` 更适合可控、可解释、好复现实验。
- `lp_teacher` 更适合自动探索任务难度前沿。
