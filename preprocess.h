/* preprocess.h - Image Preprocessing for CenterNet */
#ifndef PREPROCESS_H
#define PREPROCESS_H

/* Preprocess a BGR camera frame for CenterNet NPU inference.
 * bgr_cam: input BGR image (cam_w * cam_h * 3, uint8)
 * cam_w, cam_h: camera frame dimensions
 * output:  pre-allocated float32 buffer [3 * MODEL_W * MODEL_H] in NCHW layout.
 *          Must be DMA-capable (allocated with aclrtMallocHost).
 *          RGB order, normalized to [0, 1].
 */
void preprocess_frame(const unsigned char *bgr_cam, int cam_w, int cam_h,
                      float *output);

#endif
