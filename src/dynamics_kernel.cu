#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename scalar_t>
__global__ void update_state_vec_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R_new,
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a_thr,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_pred,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> alpha,
    float yaw_inertia) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (b >= B) return;
    // a_thr = a_thr - self.g_std;
    scalar_t ax = a_thr[b][0];
    scalar_t ay = a_thr[b][1];
    scalar_t az = a_thr[b][2] + 9.80665;
    // thrust = torch.norm(a_thr, 2, -1, True);
    scalar_t thrust = sqrt(ax*ax+ay*ay+az*az);
    // self.up_vec = a_thr / thrust;
    scalar_t ux = ax / thrust;
    scalar_t uy = ay / thrust;
    scalar_t uz = az / thrust;
    // forward_vec = self.forward_vec * yaw_inertia + v_pred;
    scalar_t fx = R[b][0][0] * yaw_inertia + v_pred[b][0];
    scalar_t fy = R[b][1][0] * yaw_inertia + v_pred[b][1];
    scalar_t fz = R[b][2][0] * yaw_inertia + v_pred[b][2];
    // forward_vec = F.normalize(forward_vec, 2, -1);
    // forward_vec = (1-alpha) * forward_vec + alpha * self.forward_vec
    scalar_t t = sqrt(fx * fx + fy * fy + fz * fz);
    fx = (1 - alpha[b][0]) * (fx / t) + alpha[b][0] * R[b][0][0];
    fy = (1 - alpha[b][0]) * (fy / t) + alpha[b][0] * R[b][1][0];
    fz = (1 - alpha[b][0]) * (fz / t) + alpha[b][0] * R[b][2][0];
    // forward_vec[2] = (forward_vec[0] * self_up_vec[0] + forward_vec[1] * self_up_vec[1]) / -self_up_vec[2]
    fz = (fx * ux + fy * uy) / -uz;
    // self.forward_vec = F.normalize(forward_vec, 2, -1);
    t = sqrt(fx * fx + fy * fy + fz * fz);
    fx /= t;
    fy /= t;
    fz /= t;
    // self.left_vec = torch.cross(self.up_vec, self.forward_vec);
    R_new[b][0][0] = fx;
    R_new[b][0][1] = uy * fz - uz * fy;
    R_new[b][0][2] = ux;
    R_new[b][1][0] = fy;
    R_new[b][1][1] = uz * fx - ux * fz;
    R_new[b][1][2] = uy;
    R_new[b][2][0] = fz;
    R_new[b][2][1] = ux * fy - uy * fx;
    R_new[b][2][2] = uz;
}


template <typename scalar_t>
__global__ void run_forward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dg,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z_drag_coef,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> drag_2,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pitch_ctl_delay,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_pred,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_wind,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> p_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> a_next,
    float ctl_dt, float airmode_av2a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (i >= B) return;
    // alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
    scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);
    // self.act = act_pred * (1 - alpha) + self.act * alpha
    for (int j=0; j<3; j++)
        act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;
    // self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
    // v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
    scalar_t v_rel_wind_x = v[i][0] - v_wind[i][0];
    scalar_t v_rel_wind_y = v[i][1] - v_wind[i][1];
    scalar_t v_rel_wind_z = v[i][2] - v_wind[i][2];
    scalar_t v_up_s = v_rel_wind_x * R[i][0][2] + v_rel_wind_y * R[i][1][2] + v_rel_wind_z * R[i][2][2];
    // scalar_t v_up[3]
    // for (int j=0; j<3; j++){
    //     v_up[j] = v_up_s * R[i][j][2];
    // }
    scalar_t v_fwd_s = v_rel_wind_x * R[i][0][0] + v_rel_wind_y * R[i][1][0] + v_rel_wind_z * R[i][2][0];
    scalar_t v_left_s = v_rel_wind_x * R[i][0][1] + v_rel_wind_y * R[i][1][1] + v_rel_wind_z * R[i][2][1];
    scalar_t v_up_2 = v_up_s * abs(v_up_s);
    scalar_t v_fwd_2 = v_fwd_s * abs(v_fwd_s);
    scalar_t v_left_2 = v_left_s * abs(v_left_s);

    scalar_t a_drag_2[3], a_drag_1[3];
    for (int j=0; j<3; j++){
        a_drag_2[j] = v_up_2 * R[i][j][2] * z_drag_coef[i][0] + v_left_2 * R[i][j][1] + v_fwd_2 * R[i][j][0];
        a_drag_1[j] = v_up_s * R[i][j][2] * z_drag_coef[i][0] + v_left_s * R[i][j][1] + v_fwd_s * R[i][j][0];
    }
    // v_prep = self.v - v_up
    // scalar_t v_prep[3];
    // for (int j=0; j<3; j++)
    //     v_prep[j] = v[i][j] - v_up[j];
    // motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
    scalar_t dot = act[i][0] * act_next[i][0] + act[i][1] * act_next[i][1] + (act[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
    scalar_t n1 = act[i][0] * act[i][0] + act[i][1] * act[i][1] + (act[i][2] + 9.80665) * (act[i][2] + 9.80665);
    scalar_t n2 = act_next[i][0] * act_next[i][0] + act_next[i][1] * act_next[i][1] + (act_next[i][2] + 9.80665) * (act_next[i][2] + 9.80665);
    scalar_t av = acos(max(-1., min(1., dot / max(1e-8, sqrt(n1) * sqrt(n2))))) / ctl_dt;

    scalar_t ax = act[i][0];
    scalar_t ay = act[i][1];
    scalar_t az = act[i][2] + 9.80665;
    scalar_t thrust = sqrt(ax*ax+ay*ay+az*az);
    scalar_t airmode_a[3] = {
        ax / thrust * av * airmode_av2a,
        ay / thrust * av * airmode_av2a,
        az / thrust * av * airmode_av2a};
    // scalar_t motor_velocity = sqrt(sqrt(act_x * act_x + act_y * act_y + act_z * act_z));
    // z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
    // a_next = self.act + self.dg - z_drag
    // scalar_t v_scalar = sqrt(v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]);
    for (int j=0; j<3; j++)
        a_next[i][j] = act_next[i][j] + dg[i][j] - a_drag_2[j] * drag_2[i][0] - a_drag_1[j] * drag_2[i][1] + airmode_a[j];
    // self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
    for (int j=0; j<3; j++)
        p_next[i][j] = p[i][j] + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;
    // self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
    for (int j=0; j<3; j++)
        v_next[i][j] = v[i][j] + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;
}


template <typename scalar_t>
__global__ void run_backward_cuda_kernel(
    torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> R,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> dg,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> z_drag_coef,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> drag_2,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> pitch_ctl_delay,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> v_wind,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> act_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_act_pred,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_act,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_a,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_act_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_p_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> d_v_next,
    torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> _d_a_next,
    float grad_decay,
    float ctl_dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B = R.size(0);
    if (i >= B) return;
    // alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
    scalar_t alpha = exp(-pitch_ctl_delay[i][0] * ctl_dt);
    // // self.act = act_pred * (1 - alpha) + self.act * alpha
    // for (int j=0; j<3; j++)
    //     act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;
    // // self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
    // // v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
    // scalar_t v_up_s = v[i][0] * R[i][0][2] + v[i][1] * R[i][1][2] + v[i][2] * R[i][2][2];
    // scalar_t v_up[3];
    // for (int j=0; j<3; j++)
    //     v_up[j] = v_up_s * R[i][j][2];
    // // v_prep = self.v - v_up
    // scalar_t v_prep[3];
    // for (int j=0; j<3; j++)
    //     v_prep[j] = v[i][j] - v_up[j];
    // motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
    scalar_t act_x = act_next[i][0];
    scalar_t act_y = act_next[i][1];
    scalar_t act_z = act_next[i][2] + 9.80665;
    // scalar_t motor_velocity = sqrt(sqrt(act_x * act_x + act_y * act_y + act_z * act_z));
    // // z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
    // // a_next = self.act + self.dg - z_drag
    // y = ax; dy/dx = a; dy = a dx
    // scalar_t v_scalar = sqrt(v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]);
    // for (int j=0; j<3; j++)
    //     a_next[i][j] = act_next[i][j] + dg[i][j] - z_drag_coef[i][0] * v_prep[j] * motor_velocity * 0.07 - drag_2[i][0] * v_scalar * v[i][j];
    // // self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
    // for (int j=0; j<3; j++)
    //     p_next[i][j] = p[i][j] + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;
    // // self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
    // for (int j=0; j<3; j++)
    //     v_next[i][j] = v[i][j] + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;

    scalar_t d_act_next[3] = {_d_act_next[i][0], _d_act_next[i][1], _d_act_next[i][2]};
    scalar_t d_a_next[3] = {_d_a_next[i][0], _d_a_next[i][1], _d_a_next[i][2]};
    // backward starts here
    for (int j=0; j<3; j++){
        // v_next[i][j] = v[i][j] + 0.5 * (a[i][j] + a_next[i][j]) * ctl_dt;
        d_v[i][j] = d_v_next[i][j] * pow(grad_decay, ctl_dt);
        d_a[i][j] = 0.5 * ctl_dt * d_v_next[i][j];
        d_a_next[j] += 0.5 * ctl_dt * d_v_next[i][j];
    }
    for (int j=0; j<3; j++){
        // p_next[i][j] = p[i][j] + v[i][j] * ctl_dt + 0.5 * a[i][j] * ctl_dt * ctl_dt;
        d_p[i][j] = d_p_next[i][j] * pow(grad_decay, ctl_dt);
        d_v[i][j] += ctl_dt * d_p_next[i][j];
        d_a[i][j] += 0.5 * ctl_dt * ctl_dt * d_p_next[i][j];
    }
    // scalar_t d_v_prep[3] = {0, 0, 0};
    scalar_t d_a_drag_2[3];
    scalar_t d_a_drag_1[3];
    // scalar_t d_v_scalar = 0;
    for (int j=0; j<3; j++){
        // a_next[i][j] = act_next[i][j] + dg[i][j] - z_drag_coef[i][0] * v_prep[j] * motor_velocity * 0.07 - a_drag_2 - a_drag_1;
        d_act_next[j] += d_a_next[j];
        // d_v_prep[j] -= z_drag_coef[i][0] * d_a_next[j] * motor_velocity * 0.07;
        // d_v_scalar -= d_a_next[j] * drag_2[i][0] * v[i][j];
        // d_v[i][j] -= d_a_next[j] * drag_2[i][0] * v_scalar;
        d_a_drag_2[j] = -d_a_next[j] * drag_2[i][0];
        d_a_drag_1[j] = -d_a_next[j] * drag_2[i][1];
    }
    // for (int j=0; j<3; j++)
    //     d_v[i][j] += d_v_scalar * v[i][j] / v_scalar;

    scalar_t v_rel_wind_x = v[i][0] - v_wind[i][0];
    scalar_t v_rel_wind_y = v[i][1] - v_wind[i][1];
    scalar_t v_rel_wind_z = v[i][2] - v_wind[i][2];
    scalar_t v_fwd_s = v_rel_wind_x * R[i][0][0] + v_rel_wind_y * R[i][1][0] + v_rel_wind_z * R[i][2][0];
    scalar_t v_left_s = v_rel_wind_x * R[i][0][1] + v_rel_wind_y * R[i][1][1] + v_rel_wind_z * R[i][2][1];
    scalar_t v_up_s = v_rel_wind_x * R[i][0][2] + v_rel_wind_y * R[i][1][2] + v_rel_wind_z * R[i][2][2];
    scalar_t d_v_fwd_s = 0;
    scalar_t d_v_left_s = 0;
    scalar_t d_v_up_s = 0;
    for (int j=0; j<3; j++){
        // a_drag_2[j] = v_up_s * v_up_s * R[i][j][2] * z_drag_coef[i][0] + v_left_s * v_left_s * R[i][j][1] + v_fwd_s * v_fwd_s * R[i][j][0];
        d_v_fwd_s += d_a_drag_2[j] * 2 * abs(v_fwd_s) * R[i][j][0];
        d_v_left_s += d_a_drag_2[j] * 2 * abs(v_left_s) * R[i][j][1];
        d_v_up_s += d_a_drag_2[j] * 2 * abs(v_up_s) * R[i][j][2] * z_drag_coef[i][0];
        d_v_fwd_s += d_a_drag_1[j] * R[i][j][0];
        d_v_left_s += d_a_drag_1[j] * R[i][j][1];
        d_v_up_s += d_a_drag_1[j] * R[i][j][2] * z_drag_coef[i][0];
    }

    // scalar_t d_v_up[3] = {0, 0, 0};
    // for (int j=0; j<3; j++){
    //     // v_prep[j] = v[i][j] - v_up[j];
    //     d_v[i][j] += d_v_prep[j];
    //     d_v_up[j] -= d_v_prep[j];
    // }
    // for (int j=0; j<3; j++){
    //     // v_up[j] = v_up_s * R[i][j][2];
    //     d_v_up_s += d_v_up[j] * R[i][j][2];
    // }
    // scalar_t v_up_s = v[i][0] * R[i][0][2] + v[i][1] * R[i][1][2] + v[i][2] * R[i][2][2];
    for (int j=0; j<3; j++){
        d_v[i][j] += R[i][j][0] * d_v_fwd_s;
        d_v[i][j] += R[i][j][1] * d_v_left_s;
        d_v[i][j] += R[i][j][2] * d_v_up_s;
    }
    for (int j=0; j<3; j++){
        // act_next[i][j] = act_pred[i][j] * (1 - alpha) + act[i][j] * alpha;
        d_act_pred[i][j] = (1 - alpha) * d_act_next[j];
        d_act[i][j] = alpha * d_act_next[j];
    }
}

} // namespace

std::vector<torch::Tensor> run_forward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor act_pred,
    torch::Tensor act,
    torch::Tensor p,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor a,
    float ctl_dt,
    float airmode_av2a){

    torch::Tensor act_next = torch::empty_like(act);
    torch::Tensor p_next = torch::empty_like(p);
    torch::Tensor v_next = torch::empty_like(v);
    torch::Tensor a_next = torch::empty_like(a);

    const int threads = R.size(0);
    const dim3 blocks(1);
    AT_DISPATCH_FLOATING_TYPES(R.scalar_type(), "run_forward_cuda", ([&] {
        run_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            dg.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z_drag_coef.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drag_2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pitch_ctl_delay.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_wind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            a.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            a_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            ctl_dt, airmode_av2a);
    }));
    return {act_next, p_next, v_next, a_next};
}

std::vector<torch::Tensor> run_backward_cuda(
    torch::Tensor R,
    torch::Tensor dg,
    torch::Tensor z_drag_coef,
    torch::Tensor drag_2,
    torch::Tensor pitch_ctl_delay,
    torch::Tensor v,
    torch::Tensor v_wind,
    torch::Tensor act_next,
    torch::Tensor _d_act_next,
    torch::Tensor d_p_next,
    torch::Tensor d_v_next,
    torch::Tensor _d_a_next,
    float grad_decay,
    float ctl_dt){

    torch::Tensor d_act_pred = torch::empty_like(dg);
    torch::Tensor d_act = torch::empty_like(dg);
    torch::Tensor d_p = torch::empty_like(dg);
    torch::Tensor d_v = torch::empty_like(dg);
    torch::Tensor d_a = torch::empty_like(dg);

    const int threads = R.size(0);
    const dim3 blocks(1);
    AT_DISPATCH_FLOATING_TYPES(R.scalar_type(), "run_backward_cuda", ([&] {
        run_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            dg.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            z_drag_coef.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            drag_2.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            pitch_ctl_delay.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_wind.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_act_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_act.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_a.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_act_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_p_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            d_v_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            _d_a_next.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            grad_decay, ctl_dt);
    }));
    return {d_act_pred, d_act, d_p, d_v, d_a};
}

torch::Tensor update_state_vec_cuda(
    torch::Tensor R,
    torch::Tensor a_thr,
    torch::Tensor v_pred,
    torch::Tensor alpha,
    float yaw_inertia) {
    const int threads = a_thr.size(0);
    const dim3 blocks(1);
    torch::Tensor R_new = torch::empty_like(R);
    AT_DISPATCH_FLOATING_TYPES(a_thr.scalar_type(), "update_state_vec", ([&] {
        update_state_vec_cuda_kernel<scalar_t><<<blocks, threads>>>(
            R_new.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            R.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
            a_thr.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            v_pred.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            alpha.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
            yaw_inertia);
    }));
    return R_new;
}
