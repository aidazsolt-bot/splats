// OpenCL 1.2: project Gaussians + soft splat with integer atomics (fixed-point).

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

__kernel void project_gaussians(
    __global const float* means,
    __global const float* scales_log,
    __global const float* quats,
    __global const float* viewmat,
    __global const float* K,
    __global float* means2d,
    __global float* cov2d,
    __global float* depths,
    __global float* radii,
    __global uchar* valid,
    const int N,
    const int width,
    const int height,
    const float near_plane
) {
    int i = get_global_id(0);
    if (i >= N) return;

    float Rcw00 = viewmat[0], Rcw01 = viewmat[1], Rcw02 = viewmat[2];
    float Rcw10 = viewmat[4], Rcw11 = viewmat[5], Rcw12 = viewmat[6];
    float Rcw20 = viewmat[8], Rcw21 = viewmat[9], Rcw22 = viewmat[10];
    float tcw0 = viewmat[3], tcw1 = viewmat[7], tcw2 = viewmat[11];

    float mx = means[i*3+0], my = means[i*3+1], mz = means[i*3+2];
    float cx = Rcw00*mx + Rcw01*my + Rcw02*mz + tcw0;
    float cy = Rcw10*mx + Rcw11*my + Rcw12*mz + tcw1;
    float cz = Rcw20*mx + Rcw21*my + Rcw22*mz + tcw2;
    depths[i] = cz;
    if (cz <= near_plane) { valid[i] = 0; radii[i] = 0; return; }

    float qw = quats[i*4+0], qx = quats[i*4+1], qy = quats[i*4+2], qz = quats[i*4+3];
    float nrm = native_sqrt(qw*qw + qx*qx + qy*qy + qz*qz) + 1e-12f;
    qw /= nrm; qx /= nrm; qy /= nrm; qz /= nrm;
    float r00 = 1 - 2*(qy*qy + qz*qz);
    float r01 = 2*(qx*qy - qz*qw);
    float r02 = 2*(qx*qz + qy*qw);
    float r10 = 2*(qx*qy + qz*qw);
    float r11 = 1 - 2*(qx*qx + qz*qz);
    float r12 = 2*(qy*qz - qx*qw);
    float r20 = 2*(qx*qz - qy*qw);
    float r21 = 2*(qy*qz + qx*qw);
    float r22 = 1 - 2*(qx*qx + qy*qy);

    float sx = native_exp(scales_log[i*3+0]);
    float sy = native_exp(scales_log[i*3+1]);
    float sz = native_exp(scales_log[i*3+2]);
    float m00 = r00*sx, m01 = r01*sy, m02 = r02*sz;
    float m10 = r10*sx, m11 = r11*sy, m12 = r12*sz;
    float m20 = r20*sx, m21 = r21*sy, m22 = r22*sz;
    float cw00 = m00*m00 + m01*m01 + m02*m02;
    float cw01 = m00*m10 + m01*m11 + m02*m12;
    float cw02 = m00*m20 + m01*m21 + m02*m22;
    float cw11 = m10*m10 + m11*m11 + m12*m12;
    float cw12 = m10*m20 + m11*m21 + m12*m22;
    float cw22 = m20*m20 + m21*m21 + m22*m22;

    float t00 = Rcw00*cw00 + Rcw01*cw01 + Rcw02*cw02;
    float t01 = Rcw00*cw01 + Rcw01*cw11 + Rcw02*cw12;
    float t02 = Rcw00*cw02 + Rcw01*cw12 + Rcw02*cw22;
    float t10 = Rcw10*cw00 + Rcw11*cw01 + Rcw12*cw02;
    float t11 = Rcw10*cw01 + Rcw11*cw11 + Rcw12*cw12;
    float t12 = Rcw10*cw02 + Rcw11*cw12 + Rcw12*cw22;
    float t20 = Rcw20*cw00 + Rcw21*cw01 + Rcw22*cw02;
    float t21 = Rcw20*cw01 + Rcw21*cw11 + Rcw22*cw12;
    float t22 = Rcw20*cw02 + Rcw21*cw12 + Rcw22*cw22;

    float cc00 = t00*Rcw00 + t01*Rcw10 + t02*Rcw20;
    float cc01 = t00*Rcw01 + t01*Rcw11 + t02*Rcw21;
    float cc02 = t00*Rcw02 + t01*Rcw12 + t02*Rcw22;
    float cc11 = t10*Rcw01 + t11*Rcw11 + t12*Rcw21;
    float cc12 = t10*Rcw02 + t11*Rcw12 + t12*Rcw22;
    float cc22 = t20*Rcw02 + t21*Rcw12 + t22*Rcw22;

    float fx = K[0], fy = K[4], px = K[2], py = K[5];
    float z = cz;
    float z2 = z*z + 1e-8f;
    float u = fx * cx / z + px;
    float v = fy * cy / z + py;
    means2d[i*2+0] = u;
    means2d[i*2+1] = v;

    float j00 = fx / z, j02 = -fx * cx / z2;
    float j11 = fy / z, j12 = -fy * cy / z2;
    float W00 = j00*cc00 + j02*cc02;
    float W01 = j00*cc01 + j02*cc12;
    float W02 = j00*cc02 + j02*cc22;
    float W10 = j11*cc01 + j12*cc02;
    float W11 = j11*cc11 + j12*cc12;
    float W12 = j11*cc12 + j12*cc22;
    float a = W00*j00 + W02*j02 + 0.3f;
    float b = W01*j11 + W02*j12;
    float c = W11*j11 + W12*j12 + 0.3f;
    cov2d[i*4+0] = a; cov2d[i*4+1] = b; cov2d[i*4+2] = b; cov2d[i*4+3] = c;

    float mid = 0.5f * (a + c);
    float disc = native_sqrt(fmax((0.5f*(a-c))*(0.5f*(a-c)) + b*b, 0.0f) + 1e-12f);
    float rad = ceil(3.0f * native_sqrt(fmax(mid + disc, 1e-8f)));
    radii[i] = rad;
    int lim = width > height ? width : height;
    valid[i] = (u + rad >= 0) && (u - rad < width) && (v + rad >= 0) && (v - rad < height)
               && (rad > 0.5f) && (rad < (float)lim) ? (uchar)1 : (uchar)0;
}

__kernel void clear_accum(
    __global int* acc_rgb_i, // H*W*3 fixed-point
    __global int* acc_a_i,
    const int npix
) {
    int i = get_global_id(0);
    if (i >= npix) return;
    acc_a_i[i] = 0;
    acc_rgb_i[i*3+0] = 0;
    acc_rgb_i[i*3+1] = 0;
    acc_rgb_i[i*3+2] = 0;
}

__kernel void splat_gaussians(
    __global const float* means2d,
    __global const float* cov2d,
    __global const float* depths,
    __global const float* radii,
    __global const uchar* valid,
    __global const float* opacities,
    __global const float* rgbs,
    __global int* acc_rgb_i,
    __global int* acc_a_i,
    const int N,
    const int width,
    const int height,
    const float depth_gamma,
    const float d_min,
    const float d_range,
    const int max_radius,
    const float fp_scale
) {
    int i = get_global_id(0);
    if (i >= N || !valid[i]) return;

    float radf = radii[i];
    if (radf > (float)max_radius) radf = (float)max_radius;
    int r = (int)radf;
    if (r < 1) return;

    float cx = means2d[i*2+0], cy = means2d[i*2+1];
    float a = cov2d[i*4+0], b = cov2d[i*4+1], c = cov2d[i*4+3];
    float det = fmax(a*c - b*b, 1e-8f);
    float inv00 = c/det, inv01 = -b/det, inv11 = a/det;
    float dw = native_exp(-depth_gamma * (depths[i] - d_min) / (d_range + 1e-6f));
    float opac = opacities[i] * dw;
    float cr = rgbs[i*3+0], cg = rgbs[i*3+1], cb = rgbs[i*3+2];

    int x0 = max(0, (int)floor(cx - radf));
    int x1 = min(width, (int)ceil(cx + radf) + 1);
    int y0 = max(0, (int)floor(cy - radf));
    int y1 = min(height, (int)ceil(cy + radf) + 1);

    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            float dx = (float)x - cx;
            float dy = (float)y - cy;
            float power = -0.5f * (inv00*dx*dx + 2.0f*inv01*dx*dy + inv11*dy*dy);
            if (power < -40.0f) continue;
            float alpha = opac * native_exp(power);
            if (alpha > 0.99f) alpha = 0.99f;
            int ai = (int)(alpha * fp_scale + 0.5f);
            if (ai <= 0) continue;
            int pix = y * width + x;
            atomic_add(acc_a_i + pix, ai);
            atomic_add(acc_rgb_i + pix*3 + 0, (int)(alpha * cr * fp_scale + 0.5f));
            atomic_add(acc_rgb_i + pix*3 + 1, (int)(alpha * cg * fp_scale + 0.5f));
            atomic_add(acc_rgb_i + pix*3 + 2, (int)(alpha * cb * fp_scale + 0.5f));
        }
    }
}

__kernel void finalize_image(
    __global const int* acc_rgb_i,
    __global const int* acc_a_i,
    __global float* out_rgb,
    __global float* out_a,
    const int npix,
    const float fp_scale
) {
    int i = get_global_id(0);
    if (i >= npix) return;
    float aa = (float)acc_a_i[i] / fp_scale;
    float denom = fmax(aa, 1e-6f);
    out_rgb[i*3+0] = ((float)acc_rgb_i[i*3+0] / fp_scale) / denom;
    out_rgb[i*3+1] = ((float)acc_rgb_i[i*3+1] / fp_scale) / denom;
    out_rgb[i*3+2] = ((float)acc_rgb_i[i*3+2] / fp_scale) / denom;
    out_a[i] = 1.0f - native_exp(-aa);
}
