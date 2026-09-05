/* postprocess.h - CenterNet Heatmap Decoder */
#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include "centernet.h"

/* Decode CenterNet output tensor into bounding box detections.
 *
 * Algorithm (matching xingyizhou/CenterNet official decode.py ctdet_decode):
 *   1. Apply sigmoid to heatmap channels (raw logits → probabilities)
 *   2. Apply 3x3 max-pool NMS on heatmap (per class channel)
 *   3. Find peaks: heatmap == pooled AND score > CONF_THRESH
 *   4. For each peak:
 *      - Read offset (sub-pixel correction in output space)
 *      - Read width/height (in output space units)
 *      - Compute center: cx = (x + off_x) * MODEL_STRIDE (→ input space)
 *      - Compute box: x1 = (cx - w*MODEL_STRIDE/2) * cam_w / MODEL_W
 *   5. Sort by score descending, keep top-K
 *
 * output:  NPU output tensor [CN_CH * HEATMAP_H * HEATMAP_W] float32
 *          Layout: [hm_class0(128*128) | hm_class1(128*128) |
 *                   off_x(128*128) | off_y(128*128) |
 *                   wh_w(128*128)  | wh_h(128*128)]
 * dets:    output array for detections
 * max_dets: maximum number of detections to return
 * cam_w, cam_h: camera frame dimensions
 * Returns: number of detections found.
 */
int centernet_decode(const float *output, Detection *dets, int max_dets,
                     int cam_w, int cam_h);

#endif
