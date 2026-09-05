/**
 * preprocess.c - CenterNet NPU 推理图像预处理
 *
 * 功能：将摄像头采集的 BGR 图像（640x480 uint8）转换为 NPU 模型需要的
 *       RGB float32 NCHW 张量，尺寸为 [1, 3, 512, 512]，数值归一化到 [0,1]。
 *
 * 预处理流程：
 *   1. 双线性插值缩放：640x480 → 512x512
 *   2. BGR → RGB 通道重排
 *   3. uint8 [0,255] → float32 [0,1]
 *   4. 标准化：(值 - 均值) / 标准差（匹配训练配置）
 *   5. HWC 布局 → NCHW 布局
 *
 * 均值和标准差采用官方 CenterNet COCO 配置（RGB 顺序）：
 *   mean = [0.408, 0.447, 0.470]
 *   std  = [0.289, 0.274, 0.278]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "preprocess.h"
#include "centernet.h"

/**
 * preprocess_frame - 将摄像头 BGR 帧预处理为 NPU 推理输入张量
 * @bgr_cam: 输入 BGR 图像数据（uint8，连续存储，宽度为 cam_w，高度为 cam_h）
 * @cam_w:   输入图像宽度（像素）
 * @cam_h:   输入图像高度（像素）
 * @tensor:  输出张量（float32，调用者分配，大小至少为 3 * MODEL_W * MODEL_H）
 * 说明：执行缩放、颜色空间转换、归一化、布局转换，结果可直接拷贝到 NPU 输入缓冲区。
 *       不进行任何边界检查，调用者需确保缓冲区足够大。
 */
void preprocess_frame(const unsigned char *bgr_cam, int cam_w, int cam_h,
                      float *tensor)
{
    int     in_w  = cam_w;
    int     in_h  = cam_h;
    int     out_w = MODEL_W;
    int     out_h = MODEL_H;
    int     ch    = 3;

    /* 步骤 1 & 2：双线性插值缩放 + BGR→RGB 通道重排 */
    unsigned char *resized = malloc(out_w * out_h * ch);
    if (!resized) return;

    float scale_x = (float)in_w / out_w;
    float scale_y = (float)in_h / out_h;

    for (int y = 0; y < out_h; y++) {
        float src_y = y * scale_y;
        int   y0    = (int)src_y;
        int   y1    = (y0 + 1 < in_h) ? y0 + 1 : y0;
        float dy    = src_y - y0;

        for (int x = 0; x < out_w; x++) {
            float src_x = x * scale_x;
            int   x0    = (int)src_x;
            int   x1    = (x0 + 1 < in_w) ? x0 + 1 : x0;
            float dx    = src_x - x0;

            for (int c = 0; c < ch; c++) {
                /* 双线性插值计算目标像素的每个通道值 */
                float v00 = bgr_cam[(y0 * in_w + x0) * ch + c];
                float v01 = bgr_cam[(y0 * in_w + x1) * ch + c];
                float v10 = bgr_cam[(y1 * in_w + x0) * ch + c];
                float v11 = bgr_cam[(y1 * in_w + x1) * ch + c];

                float v = v00 * (1 - dx) * (1 - dy)
                        + v01 * dx * (1 - dy)
                        + v10 * (1 - dx) * dy
                        + v11 * dx * dy;

                /* BGR → RGB：将 BGR 顺序转换为 RGB 顺序 */
                int src_c = 2 - c;  /* c=0(B)→src_c=2(R), c=1(G)→1(G), c=2(R)→0(B) */
                resized[(y * out_w + x) * ch + src_c] = (unsigned char)(v + 0.5f);
            }
        }
    }

    /* 步骤 3、4、5：归一化 + 均值/标准差标准化 + HWC→NCHW 布局 */
    int plane = out_w * out_h;
    static const float mean[3] = {0.408f, 0.447f, 0.470f};  /* RGB 顺序 */
    static const float std[3]  = {0.289f, 0.274f, 0.278f};

    for (int c = 0; c < ch; c++) {
        float *tp = tensor + c * plane;
        for (int i = 0; i < plane; i++) {
            float val = resized[i * ch + c] / 255.0f;      /* 归一化到 [0,1] */
            tp[i] = (val - mean[c]) / std[c];              /* 标准化 */
        }
    }

    free(resized);
}