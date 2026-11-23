# G1 双臂收拾毛毯（脚本叠毯）

宇树 G1 双臂把桌面上平铺的柔性毛毯抓起、前移、落到原布后半幅上，完成一次对折。
毛毯是真实柔体仿真，不是刚体摆拍。

## 快速运行

依赖平台核心（MuJoCo / NumPy / PyYAML）。录视频额外需要 `Pillow` 与系统 `ffmpeg`。

```bash
# 跑一遍并打印折叠指标
python -m demo.cloth_folding.run

# 同时录制视频（固定 showcase 机位，1280x720）
python -m demo.cloth_folding.run --video out.mp4 --fps 30
```

运行结束打印：仿真时长、布料顶点数、布料 x/y/z 范围、抬离桌面顶点数、上下层 x 重叠量。

## GUI 预览

文件 → 打开世界 → `worlds/g1_cloth_fold.yaml` → 工具栏 Run。

该世界的 `g1` 挂了 `args.shared_world_controller: true`，内核加载时在**内核进程内**
执行 [controller.py](controller.py)（包一层 `ClothFold`）。布料顶点钉扎、基座锁定
这类「逐物理步写状态」的操作只能在内核里做——普通控制器子进程只能写 `ctrl`。
世界要求 `ctrl_period: 1`，一个控制 tick 跨多个物理步会漏掉钉扎窗口。

## 物理实现

- **布料**：`type: cloth` 的 flexcomp 二维网格（每个网格点是带 3 个世界轴滑块的质点），
  膜/弯刚度维持形状；重力、桌面接触、自碰撞（`self-collide: narrow`）都是真实仿真，
  对折后上下两层不会互相穿透。
- **双臂运动**：启动时把每个关键帧的 TCP 目标与姿态用阻尼最小二乘解成 7 个手臂关节角，
  运行中用余弦插值；七自由度冗余臂在贴近身体处容易陷局部极小，右臂用左臂解镜像当种子。
  肩 roll 只放开外展方向，否则肩壳会插进躯干，IK 顶到关节限位也到不了位。
- **抓取**：G1 三指功率手的拇指尖与食指尖竖直错开约 9cm，捏不住 1cm 级薄布
  （实测捏空或顶穿）；硬钉布顶点又会撕布。因此改用**合规抓持力**：合指后锁定手指间的
  布顶点，用有限刚度的弹簧-阻尼力让它们跟随腕部位姿，松手即解除。布料本体仍是完整
  柔体仿真。
- **关键帧**：悬停 → 下探到垂边 → 提起 → 前移过中线 → 落到远半幅 → 沿桌面同高度后撤。
  后撤必须平着走：抬高会勾住布料把折边带起，横向张开会把布挤成窄条（两种都实测失败）。
  受手形限制，折峰无法压平。

## 三套世界

| 世界 | 用途 | 关键配置 |
|---|---|---|
| `worlds/g1_cloth_fold.yaml` | 脚本叠毯 / 录视频 | `ctrl_period: 1`，内核内共享控制器 |
| `worlds/g1_cloth_teleop.yaml` | PICO 遥操作采集 | `ctrl_period: 8` → 31.25 Hz；LeRobot v2.1 |
| `worlds/g1_cloth_augment.yaml` | 数据增强回放 | 读 action 重放并追加回数据集 |

- **遥操作**（[controllers/g1_cloth_teleop.py](../../controllers/g1_cloth_teleop.py)）：
  PICO 头显浏览器打开 `http://<PC 的 IP>:8090`，左右手柄增量映射到双臂 TCP，扳机控制
  2F85 夹爪，A 键开关录制，B 重对中，X 换随机场景，Y 丢弃当前集。数据集默认写到
  `~/桌面/g1_cloth_teleop_dataset`（世界 `args.dataset` 可改），16 维 state/action，
  openpi π0.5 的 LeRobot loader 可直接读。
- **数据增强**（[controllers/g1_cloth_augment.py](../../controllers/g1_cloth_augment.py)）：
  加载即自动开始，无需 PICO。读取已有 episode 的 action，在重新随机化（布料颜色/摆放、
  桌面颜色、灯光、第三人称机位、图像后处理）的场景里重放同一段关节轨迹，并把新帧
  追加回数据集。`replays_per_episode` 默认 3。**输出目录就是源数据集，先复制一份再跑。**

## 文件

```
run.py           ClothFold 核心 + 命令行入口（自建世界、可录视频）
controller.py    GUI 预览用的内核内协调器（包装 ClothFold）
```
