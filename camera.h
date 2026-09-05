/* camera.h - V4L2 Camera Interface */
#ifndef CAMERA_H
#define CAMERA_H

/* Initialize V4L2 camera device.
 * dev: device path (e.g. "/dev/video0")
 * w, h: desired capture resolution
 * Returns opaque camera context, NULL on failure.
 */
void* camera_init(const char *dev, int w, int h);

/* Grab one BGR frame from camera.
 * ctx: camera context from camera_init()
 * bgr: output buffer (must be at least w*h*3 bytes)
 * Returns 0 on success, -1 on failure.
 */
int   camera_grab(void *ctx, unsigned char *bgr);

/* Close camera and release resources. */
void  camera_close(void *ctx);

#endif
