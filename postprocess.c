/**
 * postprocess.c — CenterNet 目标检测解码器（纯 C 实现）
 *
 * ═══════════════════════════════════════════════════════════════
 * 功能：将 NPU/ONNX 模型输出的张量解码为边界框检测结果。
 *       实现了 CenterNet 论文 "Objects as Points" (Zhou et al., 2019)
 *       的官方解码算法（ctdet_decode），无需 NMS。
 * ═══════════════════════════════════════════════════════════════
 *
 * 核心思想：CenterNet 将每个物体建模为一个“中心点”，
 *           回归该点的偏移（offset）和物体宽高（size）。
 *           这与基于锚框的检测器（YOLO、SSD）有本质区别。
 *
 * 输入张量布局（ONNX 输出，按通道拼接）：
 *   [hm_class0 (128×128) | hm_class1 (128×128) |
 *    reg_x (128×128)     | reg_y (128×128)     |
 *    wh_w (128×128)      | wh_h (128×128)]
 *
 * 解码算法步骤：
 *   1. 对热力图通道应用 sigmoid，得到概率图 Ŷ_c
 *   2. 3×3 最大池化（替代 IoU 非极大值抑制）
 *   3. 峰值检测：若 Ŷ_c[y,x] == 池化结果 且 > 阈值 τ，则为有效检测
 *   4. 解码边界框：中心 = (x + reg_x) × 步长，宽高 = exp(wh_log) × 输入尺寸
 *   5. 缩放到相机分辨率，按分数排序，保留 Top-K
 *
 * ═══════════════════════════════════════════════════════════════
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "postprocess.h"
#include "centernet.h"

/**
 * sigmoid - 计算 sigmoid 激活函数
 * @x: 输入值
 * 返回：σ(x) = 1 / (1 + e^{-x})
 */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * maxpool_3x3_per_channel - 对每个通道独立进行 3×3 最大池化
 * @src: 输入数据（逐通道连续存储）
 * @dst: 输出池化结果
 * @h:   特征图高度
 * @w:   特征图宽度
 * @ch:  通道数
 * 说明：该操作用于替代传统 NMS，是 CenterNet 的重要创新之一。
 *       输出中每个位置取 3×3 邻域内的最大值。
 */
static void maxpool_3x3_per_channel(const float *src, float *dst,
                                     int h, int w, int ch)
{
    int plane = h * w;
    for (int c = 0; c < ch; c++) {
        const float *sp = src + c * plane;
        float       *dp = dst + c * plane;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float maxv = -1e9f;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int ny = y + dy, nx = x + dx;
                        if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
                            float v = sp[ny * w + nx];
                            if (v > maxv) maxv = v;
                        }
                    }
                }
                dp[y * w + x] = maxv;
            }
        }
    }
}

/**
 * det_cmp_desc - 比较两个检测结果的分数，用于降序排序
 * @a, @b: 指向 Detection 的指针
 * 返回：负数表示 a > b，正数表示 a < b，0 表示相等
 */
static int det_cmp_desc(const void *a, const void *b) {
    float sa = ((const Detection*)a)->score;
    float sb = ((const Detection*)b)->score;
    return (sa < sb) ? 1 : ((sa > sb) ? -1 : 0);
}

/* ================================================================
 * 主解码函数
 * ================================================================ */

/**
 * centernet_decode - 将模型输出解码为检测框列表
 * @output:   模型输出张量（一维 float 数组，按通道连续存储）
 * @dets:     输出检测结果数组（调用者分配）
 * @max_dets: 最多返回的检测框数量
 * @cam_w:    原始摄像头图像宽度（用于坐标缩放）
 * @cam_h:    原始摄像头图像高度（用于坐标缩放）
 * 返回：实际检测到的物体数量
 *
 * 算法细节：
 *   - 热力图通道数量由 NUM_CLASSES 定义（本实现为 1，仅人脸）
 *   - 特征图尺寸为 HEATMAP_H × HEATMAP_W = 128×128（步长 4）
 *   - 偏移量用于修正因下采样导致的量化误差
 *   - 宽高以对数形式回归，解码时需要 exp() 和缩放
 */
int centernet_decode(const float *output, Detection *dets, int max_dets,
                     int cam_w, int cam_h)
{
    int plane   = HEATMAP_H * HEATMAP_W;   /* 128*128 = 16384 */
    int out_sz  = CN_CH * plane;           /* 6 * 16384 = 98304 */
    static int debug_cnt = 0;
    debug_cnt++;

    /* ---- 1. 拷贝输出并对热力图通道进行 sigmoid ---- */
    float *hm = malloc(out_sz * sizeof(float));
    if (!hm) return 0;
    memcpy(hm, output, out_sz * sizeof(float));

    for (int c = 0; c < NUM_CLASSES; c++) {
        float *hp = hm + c * plane;
        for (int i = 0; i < plane; i++)
            hp[i] = sigmoid(hp[i]);
    }

    /* ---- 2. 3×3 最大池化（用于局部峰值检测，替代 NMS） ---- */
    float *pooled = malloc(out_sz * sizeof(float));
    if (!pooled) { free(hm); return 0; }
    maxpool_3x3_per_channel(hm, pooled, HEATMAP_H, HEATMAP_W, NUM_CLASSES);

    /* ---- 3. 峰值检测并解码边界框 ---- */
    Detection raw[TOPK_MAX * 2];  /* 临时存储，用于排序 */
    int ndet = 0;

    for (int c = 0; c < NUM_CLASSES && ndet < max_dets * 2; c++) {
        const float *hp = hm + c * plane;
        const float *pp = pooled + c * plane;

        for (int y = 0; y < HEATMAP_H && ndet < max_dets * 2; y++) {
            for (int x = 0; x < HEATMAP_W && ndet < max_dets * 2; x++) {
                int   idx   = y * HEATMAP_W + x;
                float score = hp[idx];

                /* 置信度阈值过滤 */
                if (score <= CONF_THRESH) continue;

                /* 峰值检查：必须等于池化后的值（即局部最大） */
                if (score < pp[idx] - 1e-6f) continue;

                /* ============================================================
                 * 解码边界框（匹配官方 ctdet_decode）
                 *
                 * 注意：训练时将 wh 编码为 log(w / INPUT_SIZE)，
                 * 因此解码时必须使用 expf() 并乘以 MODEL_W。
                 * 参见 train_catsdogs.py 第 100-101 行。
                 * ============================================================ */

                /* 读取偏移量（亚像素修正） */
                float off_x = output[(CN_OFF_START + 0) * plane + idx];
                float off_y = output[(CN_OFF_START + 1) * plane + idx];

                /* 读取宽高对数，解码得到实际尺寸 */
                float wh_log_w = output[(CN_WH_START + 0) * plane + idx];
                float wh_log_h = output[(CN_WH_START + 1) * plane + idx];
                float wh_w = expf(wh_log_w) * MODEL_W;
                float wh_h = expf(wh_log_h) * MODEL_H;

                /* 计算模型输入空间（512×512）中的中心点坐标 */
                float cx_in = (x + off_x) * MODEL_STRIDE;
                float cy_in = (y + off_y) * MODEL_STRIDE;

                /* 计算边界框（输入空间） */
                float x1_in = cx_in - wh_w * 0.5f;
                float y1_in = cy_in - wh_h * 0.5f;
                float x2_in = cx_in + wh_w * 0.5f;
                float y2_in = cy_in + wh_h * 0.5f;

                /* 缩放到原始摄像头分辨率 */
                float scale_x = (float)cam_w / (float)MODEL_W;
                float scale_y = (float)cam_h / (float)MODEL_H;

                float x1 = x1_in * scale_x;
                float y1 = y1_in * scale_y;
                float x2 = x2_in * scale_x;
                float y2 = y2_in * scale_y;

                /* 裁剪到图像边界 */
                if (x1 < 0) x1 = 0;
                if (y1 < 0) y1 = 0;
                if (x2 >= cam_w) x2 = (float)(cam_w - 1);
                if (y2 >= cam_h) y2 = (float)(cam_h - 1);

                /* 过滤过小或无效框 */
                if (x2 - x1 < 4.0f || y2 - y1 < 4.0f) continue;
                if (x2 <= x1 || y2 <= y1) continue;

                /* 存储检测结果 */
                raw[ndet].x1    = x1;
                raw[ndet].y1    = y1;
                raw[ndet].x2    = x2;
                raw[ndet].y2    = y2;
                raw[ndet].score = score;
                raw[ndet].cls   = c;
                ndet++;
            }
        }
    }

    /* ---- 5. 按分数降序排序，保留 Top-K ---- */
    qsort(raw, ndet, sizeof(Detection), det_cmp_desc);
    if (ndet > max_dets) ndet = max_dets;
    memcpy(dets, raw, ndet * sizeof(Detection));

    free(hm);
    free(pooled);
    return ndet;
}