from collections import defaultdict
import math
from random import normalvariate
from matplotlib import pyplot as plt
from env_cuda import Env
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
from model import Model


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default=None)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_iters', type=int, default=50000)
parser.add_argument('--coef_v', type=float, default=1.0, help='smooth l1 of norm(v_set - v_real)')
parser.add_argument('--coef_speed', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_v_pred', type=float, default=2.0, help='mse loss for velocity estimation (no odom)')
parser.add_argument('--coef_collide', type=float, default=2.0, help='softplus loss for collision (large if close to obstacle, zero otherwise)')
parser.add_argument('--coef_obj_avoidance', type=float, default=1.5, help='quadratic clearance loss')
parser.add_argument('--coef_d_acc', type=float, default=0.01, help='control acceleration regularization')
parser.add_argument('--coef_d_jerk', type=float, default=0.001, help='control jerk regularizatinon')
parser.add_argument('--coef_d_snap', type=float, default=0.0, help='legacy')
parser.add_argument('--coef_ground_affinity', type=float, default=0., help='legacy')
parser.add_argument('--coef_bias', type=float, default=0.0, help='legacy')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--grad_decay', type=float, default=0.4)
parser.add_argument('--speed_mtp', type=float, default=1.0)
parser.add_argument('--fov_x_half_tan', type=float, default=0.53)
parser.add_argument('--timesteps', type=int, default=150)
parser.add_argument('--cam_angle', type=int, default=10)
parser.add_argument('--single', default=False, action='store_true')
parser.add_argument('--gate', default=False, action='store_true')
parser.add_argument('--ground_voxels', default=False, action='store_true')
parser.add_argument('--scaffold', default=False, action='store_true')
parser.add_argument('--random_rotation', default=False, action='store_true')
parser.add_argument('--yaw_drift', default=False, action='store_true')
parser.add_argument('--no_odom', default=False, action='store_true')
args = parser.parse_args()
writer = SummaryWriter()
print(args)

device = torch.device('cuda')

env = Env(args.batch_size, 64, 48, args.grad_decay, device,
          fov_x_half_tan=args.fov_x_half_tan, single=args.single,
          gate=args.gate, ground_voxels=args.ground_voxels,
          scaffold=args.scaffold, speed_mtp=args.speed_mtp,
          random_rotation=args.random_rotation, cam_angle=args.cam_angle)
if args.no_odom:
    model = Model(7, 6)
else:
    model = Model(7+3, 6)
model = model.to(device)

if args.resume:
    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, False)
    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)
optim = AdamW(model.parameters(), args.lr)
sched = CosineAnnealingLR(optim, args.num_iters, args.lr * 0.01)

ctl_dt = 1 / 15


scaler_q = defaultdict(list)
def smooth_dict(ori_dict):
    for k, v in ori_dict.items():
        scaler_q[k].append(float(v))

def barrier(x: torch.Tensor, v_to_pt):
    return (v_to_pt * (1 - x).relu().pow(2)).mean()

def is_save_iter(i):
    if i < 2000:
        return (i + 1) % 250 == 0
    return (i + 1) % 1000 == 0

pbar = tqdm(range(args.num_iters), ncols=80)
# depths = []
# states = []
B = args.batch_size
for i in pbar:
    env.reset()
    model.reset()
    p_history = []
    v_history = []
    target_v_history = []
    vec_to_pt_history = []
    act_diff_history = []
    v_preds = []
    vid = []
    v_net_feats = []
    h = None

    act_lag = 1
    act_buffer = [env.act] * (act_lag + 1)
    target_v_raw = env.p_target - env.p
    # yaw的噪声
    if args.yaw_drift:
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack([
            torch.cos(drift_av), -torch.sin(drift_av), zeros,
            torch.sin(drift_av), torch.cos(drift_av), zeros,
            zeros, zeros, ones,
        ], -1).reshape(B, 3, 3)

    # 先跑完整段轨迹，再循环计算loss
    for t in range(args.timesteps): 
        # 模拟飞行频率不稳定
        ctl_dt = normalvariate(1 / 15, 0.1 / 15) 
        depth, flow = env.render(ctl_dt)
        p_history.append(env.p)
        
        # 根据不同的几何体遍历每个体素找到最接近的
        vec_to_pt_history.append(env.find_vec_to_nearest_pt())

        if is_save_iter(i):
            vid.append(depth[4])

        if args.yaw_drift:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()
        env.run(act_buffer[t], ctl_dt, target_v_raw)

        # 得到旋转矩阵
        R = env.R
        fwd = env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)
        # 只关心机头的朝向，z直接为0
        fwd[:, 2] = 0
        # z朝上
        up[:, 2] = 1
        fwd = F.normalize(fwd, 2, -1)
        # 相当于正则化一下旋转矩阵
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

        # 目标速度归一化
        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)
        state = [
            torch.squeeze(target_v[:, None] @ R, 1), # body系下的速度方向
            env.R[:, 2],  # 真实的推力方向
            env.margin[:, None]]
        local_v = torch.squeeze(env.v[:, None] @ R, 1)
        if not args.no_odom:
            state.insert(0, local_v)
        state = torch.cat(state, -1)

        #  # 转化为逆深度，因为比较近的深度会影响比较大
        x = 3 / depth.clamp_(0.3, 24) - 0.6 + torch.randn_like(depth) * 0.02
        x = F.max_pool2d(x[:, None], 4, 4)
        act, values, h = model(x, state, h)

        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        v_preds.append(v_pred)
        # 先把期望加速度换算为推力（内部坐标系）， 然后再补偿重力，使 hover 时 thrust 正好维持在 g
        # v_pred是阻尼，相当于drag
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std
        act_buffer.append(act)
        v_net_feats.append(torch.cat([act, local_v, h], -1))

        v_history.append(env.v)
        target_v_history.append(target_v)
 
    # ==================== 第一部分：整理历史数据 ====================
    # 将整个 episode 中记录的位置历史堆叠成张量，形状: (T, B, 3)
    p_history = torch.stack(p_history)
    # 地面亲和力损失：惩罚无人机飞到地面以下（z < 0）
    # relu() 只保留 z < 0 的部分，然后平方求平均
    loss_ground_affinity = p_history[..., 2].relu().pow(2).mean()
    # 将动作历史堆叠，形状: (T+act_lag+1, B, 3)
    act_buffer = torch.stack(act_buffer)

    # ==================== 第二部分：速度跟踪损失 (loss_v) ====================
    # 将速度历史堆叠，形状: (T, B, 3)
    v_history = torch.stack(v_history)
    # 计算速度的累积和，用于后续计算滑动平均
    v_history_cum = v_history.cumsum(0)
    # 计算 30 步的滑动平均速度（平滑处理，减少噪声影响）
    # v_history_avg[t] = mean(v_history[t:t+30])
    v_history_avg = (v_history_cum[30:] - v_history_cum[:-30]) / 30
    # 将目标速度历史堆叠
    target_v_history = torch.stack(target_v_history)
    T, B, _ = v_history.shape
    # 计算平均速度与目标速度的差异（注意索引对齐：target_v_history[1:1-30] 对应 v_history_avg）
    delta_v = torch.norm(v_history_avg - target_v_history[1:1-30], 2, -1)
    # 使用 smooth_l1_loss 计算速度跟踪误差（比 MSE 更鲁棒）
    loss_v = F.smooth_l1_loss(delta_v, torch.zeros_like(delta_v))

    # ==================== 第三部分：速度预测损失 (loss_v_pred) ====================
    # 将网络预测的速度堆叠
    v_preds = torch.stack(v_preds)
    # 预测速度 vs 真实速度的 MSE 损失（用于无里程计模式下的速度估计）
    loss_v_pred = F.mse_loss(v_preds, v_history.detach())

    # ==================== 第四部分：速度偏差损失 (loss_bias) ====================
    # 偏差损失：惩罚速度偏离目标方向的分量
    # 理想情况下，v_history 应该完全沿着 target_v_history_normalized 方向
    target_v_history_norm = torch.norm(target_v_history, 2, -1)
    target_v_history_normalized = target_v_history / target_v_history_norm[..., None]
    fwd_v = torch.sum(v_history * target_v_history_normalized, -1)
    loss_bias = F.mse_loss(v_history, fwd_v[..., None] * target_v_history_normalized) * 3

    # ==================== 第五部分：控制平滑性损失 ====================
    # Jerk（加速度的变化率）：act_buffer.diff(1, 0) 计算相邻时间步的差值
    # 乘以 15 是因为控制频率是 15Hz，将单位转换为 1/s²
    jerk_history = act_buffer.diff(1, 0).mul(15)
    # Snap（Jerk 的变化率）：先归一化推力（减去重力），然后两次差分
    # 乘以 15² 转换为正确的单位
    snap_history = F.normalize(act_buffer - env.g_std).diff(1, 0).diff(1, 0).mul(15**2)
    # 加速度正则化：惩罚过大的控制输入（节省能量，提高稳定性）
    loss_d_acc = act_buffer.pow(2).sum(-1).mean()
    # Jerk 正则化：惩罚控制输入的快速变化（提高平滑性）
    loss_d_jerk = jerk_history.pow(2).sum(-1).mean()
    # Snap 正则化：进一步惩罚控制输入的剧烈变化
    loss_d_snap = snap_history.pow(2).sum(-1).mean()

    # ==================== 第六部分：避障损失 ====================
    # 将"到最近障碍点的向量"历史堆叠
    vec_to_pt_history = torch.stack(vec_to_pt_history)
    # 计算到最近障碍点的距离
    distance = torch.norm(vec_to_pt_history, 2, -1)
    # 减去安全裕度，得到"净距离"（负值表示碰撞）
    distance = distance - env.margin
    with torch.no_grad():
        # 计算距离的变化率（负值表示正在接近障碍物）
        # 乘以 135（≈ 15Hz * 9）转换为合适的单位，clamp_min(1) 确保最小权重为 1
        # v_to_pt 越大，说明越接近障碍物，避障损失权重越大
        # 后面一个量减去前一个量，如果为负，说明正在接近障碍物，权重越大
        v_to_pt = (-torch.diff(distance, 1, 1) * 135).clamp_min(1)
    # 障碍物回避损失：使用 barrier 函数
    # barrier(x, v) = v * (1-x).relu()²，当 x < 1（距离小于安全裕度）时产生损失
    # 距离越小，损失越大；v_to_pt 越大（越接近），损失权重越大
    loss_obj_avoidance = barrier(distance[:, 1:], v_to_pt)
    # 碰撞损失：使用 softplus 函数，当距离为负（已碰撞）时产生大损失
    # softplus(-32 * distance) 在 distance < 0 时快速增长
    loss_collide = F.softplus(distance[:, 1:].mul(-32)).mul(v_to_pt).mean()

    # ==================== 第七部分：速度大小损失 (loss_speed) ====================
    # 计算实际速度的模长（速度大小）
    speed_history = v_history.norm(2, -1)
    # 比较"速度在目标方向上的投影"与"目标速度大小"
    # 理想情况下两者应该相等
    loss_speed = F.smooth_l1_loss(fwd_v, target_v_history_norm)

    # ==================== 第八部分：总损失计算 ====================
    # 将所有损失项按权重加权求和
    loss = args.coef_v * loss_v + \
        args.coef_obj_avoidance * loss_obj_avoidance + \
        args.coef_bias * loss_bias + \
        args.coef_d_acc * loss_d_acc + \
        args.coef_d_jerk * loss_d_jerk + \
        args.coef_d_snap * loss_d_snap + \
        args.coef_speed * loss_speed + \
        args.coef_v_pred * loss_v_pred + \
        args.coef_collide * loss_collide + \
        args.coef_ground_affinity * loss_ground_affinity

    # ==================== 第九部分：异常检测和优化 ====================
    # 检查损失是否为 NaN（通常表示训练不稳定或数值溢出）
    if torch.isnan(loss):
        print("loss is nan, exiting...")
        exit(1)

    # 更新进度条显示
    pbar.set_description_str(f'loss: {loss:.3f}')
    # 清零梯度
    optim.zero_grad()
    # 反向传播：梯度从损失函数穿过整个可微分物理仿真回传到网络参数
    loss.backward()
    # 更新网络参数
    optim.step()
    # 更新学习率（余弦退火调度器）
    sched.step()


    # ==================== 第十部分：记录和可视化 ====================
    with torch.no_grad():
        # 计算平均速度（每个 batch 的平均）
        avg_speed = speed_history.mean(0)
        # 判断是否成功：所有时间步的距离都 > 0（没有碰撞）
        # flatten(0, 1) 将 (T, B) 展平为 (T*B,)，然后检查每个样本是否全程无碰撞
        success = torch.all(distance.flatten(0, 1) > 0, 0)
        # 成功率：成功样本数 / 总样本数
        _success = success.sum() / B
        # 将各项指标存入平滑队列（用于后续计算移动平均）
        smooth_dict({
            'loss': loss,
            'loss_v': loss_v,
            'loss_v_pred': loss_v_pred,
            'loss_obj_avoidance': loss_obj_avoidance,
            'loss_d_acc': loss_d_acc,
            'loss_d_jerk': loss_d_jerk,
            'loss_d_snap': loss_d_snap,
            'loss_bias': loss_bias,
            'loss_speed': loss_speed,
            'loss_collide': loss_collide,
            'loss_ground_affinity': loss_ground_affinity,
            'success': _success,
            'max_speed': speed_history.max(0).values.mean(),
            'avg_speed': avg_speed.mean(),
            'ar': (success * avg_speed).mean()})  # 平均成功率 × 平均速度
        log_dict = {}
        # 在特定迭代时保存可视化图表
        if is_save_iter(i):
            # vid = torch.stack(vid).cpu().div(10).clamp(0, 1)[None, :, None]
            # 绘制位置历史（选择 batch 中的第 4 个样本）
            fig_p, ax = plt.subplots()
            p_history = p_history[:, 4].cpu()
            ax.plot(p_history[:, 0], label='x')
            ax.plot(p_history[:, 1], label='y')
            ax.plot(p_history[:, 2], label='z')
            ax.legend()
            # 绘制速度历史
            fig_v, ax = plt.subplots()
            v_history = v_history[:, 4].cpu()
            ax.plot(v_history[:, 0], label='x')
            ax.plot(v_history[:, 1], label='y')
            ax.plot(v_history[:, 2], label='z')
            ax.legend()
            # 绘制加速度历史
            fig_a, ax = plt.subplots()
            act_buffer = act_buffer[:, 4].cpu()
            ax.plot(act_buffer[:, 0], label='x')
            ax.plot(act_buffer[:, 1], label='y')
            ax.plot(act_buffer[:, 2], label='z')
            ax.legend()
            # writer.add_video('demo', vid, i + 1, 15)
            # 将图表添加到 TensorBoard
            writer.add_figure('p_history', fig_p, i + 1)
            writer.add_figure('v_history', fig_v, i + 1)
            writer.add_figure('a_reals', fig_a, i + 1)
        # 每 10000 次迭代保存一次模型检查点
        if (i + 1) % 10000 == 0:
            torch.save(model.state_dict(), f'single/checkpoint{i//10000:04d}.pth')
        # 每 25 次迭代记录一次指标到 TensorBoard
        if (i + 1) % 25 == 0:
            for k, v in scaler_q.items():
                # 计算移动平均并记录
                writer.add_scalar(k, sum(v) / len(v), i + 1)
            # 清空队列，准备下一轮记录
            scaler_q.clear()
