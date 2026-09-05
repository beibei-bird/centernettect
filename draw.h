/* draw.h - Bounding Box Visualization */
#ifndef DRAW_H
#define DRAW_H

#include "centernet.h"

/* Draw detection boxes and labels on a BGR frame (in-place).
 * bgr:      BGR image buffer (w * h * 3, modified in-place)
 * w, h:     image dimensions
 * dets:     array of detections
 * ndet:     number of detections
 * names:    class name strings (array of ndet_class strings)
 */
void draw_detections(unsigned char *bgr, int w, int h,
                     const Detection *dets, int ndet, const char **names);

#endif
