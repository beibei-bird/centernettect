/**
 * centernet.h — CenterNet Object Detection System Configuration
 *
 * ═══════════════════════════════════════════════════════════════
 * Objects as Points (Zhou, Wang, Krähenbühl, 2019)
 * https://arxiv.org/abs/1904.07850
 * ═══════════════════════════════════════════════════════════════
 *
 * Based on official xingyizhou/CenterNet implementation.
 * Architecture: ResNet-18 backbone + 3× deconv neck (output stride=4)
 *
 * Key CenterNet characteristics in this project:
 * - Objects represented as CENTER POINTS (not anchor boxes)
 * - Heatmap + offset + size output heads
 * - 3×3 maxpool replaces IoU-NMS
 * - No anchor boxes, no region proposals, single forward pass
 *
 * Pipeline:
 *   Camera(V4L2) → Preprocess → NPU(ACL) → CenterNet Decode → Draw → HTTP MJPEG → VLC
 *
 * Board: Atlas 200I DK A2, Ascend 310B4, CANN 7.0.RC1, gcc 11.3
 */
#ifndef CENTERNET_H
#define CENTERNET_H

/* ================================================================
 * Camera
 * ================================================================ */
#define CAM_DEVICE        "/dev/video0"
#define CAM_W             640
#define CAM_H             480
#define CAM_FPS           25
#define CAM_BUF_COUNT     4

/* ================================================================
 * Model Input (CenterNet ResNet-18)
 * ================================================================ */
#define MODEL_PATH        "/root/centernet_detection/models/centernet.om"
#define MODEL_W           512
#define MODEL_H           512
#define MODEL_C           3
#define MODEL_STRIDE      4           /* ResNet conv1+maxpool = stride 4 */
#define HEATMAP_W         (MODEL_W / MODEL_STRIDE)   /* 128 */
#define HEATMAP_H         (MODEL_H / MODEL_STRIDE)   /* 128 */

/* ================================================================
 * CenterNet ONNX Output Layout
 * Concatenated: [heatmap(NUM_CLASSES) | reg(2) | wh(2)]
 * Order matches: torch.cat([hm, reg, wh], dim=1)
 * ================================================================ */
#define NUM_CLASSES       1           /* face */
#define CN_CH             (NUM_CLASSES + 4)  /* total output channels: 6 */
#define CN_HM_START       0
#define CN_OFF_START      NUM_CLASSES        /* = 2 */
#define CN_WH_START       (NUM_CLASSES + 2)  /* = 4 */

/* ================================================================
 * Detection Parameters
 * ================================================================ */
#define CONF_THRESH       0.4f        /* score threshold */
#define TOPK_MAX          100         /* max detections per frame */

/* ================================================================
 * HTTP MJPEG Server
 * ================================================================ */
#define HTTP_PORT         8555
#define JPEG_QUALITY      15
#define MAX_CLIENTS       8

/* ================================================================
 * Colors (BGR packed uint32)
 * ================================================================ */
#define RGB_BLUE          0x00FF0000u
#define RGB_GREEN         0x0000FF00u
#define RGB_RED           0x000000FFu
#define RGB_YELLOW        0x0000FFFFu
#define RGB_CYAN          0x00FFFF00u
#define RGB_ORANGE        0x0000A5FFu

/* ================================================================
 * Tensor Sizes
 * ================================================================ */
#define INPUT_BYTES        (MODEL_C * MODEL_W * MODEL_H * sizeof(float))
#define OUTPUT_BYTES       (CN_CH * HEATMAP_H * HEATMAP_W * sizeof(float))
#define CAM_FRAME_BYTES    (CAM_W * CAM_H * 3)

/* ================================================================
 * Data Structures
 * ================================================================ */
typedef struct {
    float x1, y1, x2, y2;  /* bounding box (scaled to camera frame) */
    float score;            /* confidence score */
    int   cls;              /* class index */
} Detection;

typedef struct {
    unsigned char *data;    /* JPEG encoded frame data */
    int            size;    /* JPEG byte size */
    int            seq;     /* frame sequence number */
} SharedFrame;

#endif /* CENTERNET_H */
