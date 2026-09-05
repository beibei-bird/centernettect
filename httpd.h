/* httpd.h - HTTP MJPEG Streaming Server */
#ifndef HTTPD_H
#define HTTPD_H

/* Start HTTP MJPEG server.
 * port: TCP port to listen on
 * Returns 0 on success, -1 on failure.
 * VLC URL: http://192.168.137.100:<port>/video
 */
int httpd_start(int port);

/* Send a BGR frame to all connected HTTP clients as MJPEG.
 * bgr: BGR image data (w * h * 3)
 * w, h: image dimensions
 */
void httpd_send_frame(const unsigned char *bgr, int w, int h);

/* Encode a BGR frame to JPEG. Returns size (0 on error).
 * Caller should NOT free *out_jpg - ownership stays with httpd.
 * Next call to this function reuses the same buffer.
 */
int httpd_encode_frame(const unsigned char *bgr, int w, int h,
                       unsigned char **out_jpg);

/* Send a pre-encoded JPEG frame to all clients.
 * Use this with httpd_encode_frame to avoid re-encoding.
 */
void httpd_send_jpg(const unsigned char *jpg, int size);

/* Stop HTTP server and close all connections. */
void httpd_stop(void);

#endif
