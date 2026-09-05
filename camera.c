/**
 * camera.c - V4L2 Camera Capture (pure C, YUYV -> BGR)
 *
 * 功能：使用 V4L2 框架从摄像头捕获 YUYV 格式图像，并转换为 BGR 格式。
 *       提供初始化、抓帧、关闭等接口，供上层目标检测等模块调用。
 *       内部使用内存映射（MMAP）方式管理缓冲区，支持非阻塞抓取。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include "camera.h"
#include "centernet.h"

/**
 * Camera 结构体 - 保存摄像头设备句柄、参数及缓冲区信息
 */
struct Camera {
    int fd;                 // V4L2 设备文件描述符
    int w, h;               // 图像宽高（实际生效的分辨率）
    unsigned char *bufs[8]; // MMAP 映射的缓冲区指针数组
    unsigned int  len[8];   // 每个缓冲区的长度
    unsigned int  nbufs;    // 缓冲区数量
};

/**
 * xioctl - 封装 ioctl，自动处理 EINTR 重试
 */
static int xioctl(int fd, int req, void *arg)
{
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}

/**
 * clamp - 将整数值限制在 0~255 范围内，用于 RGB 分量
 */
static inline unsigned char clamp(int v) {
    return (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

/**
 * yuyv_to_bgr - 将 YUYV 格式（4:2:2）转换为 BGR24 格式（每个像素 3 字节）
 * @yuv: 输入 YUYV 数据（每 4 字节包含 2 个像素）
 * @bgr: 输出 BGR 数据（每 3 字节一个像素，按 BGR 顺序存储）
 * @npixels: 像素总数（必须为偶数）
 */
static void yuyv_to_bgr(const unsigned char *yuv, unsigned char *bgr, int npixels)
{
    for (int i = 0; i < npixels; i += 2) {
        int y0 = yuv[0], u = yuv[1], y1 = yuv[2], v = yuv[3];
        yuv += 4;
        int c0 = y0 - 16, c1 = y1 - 16;
        int d = u - 128, e = v - 128;
        // 将 YUV 转换为 BGR（使用标准 BT.601 系数）
        bgr[0] = clamp((298 * c0 + 516 * d + 128) >> 8);
        bgr[1] = clamp((298 * c0 - 100 * d - 208 * e + 128) >> 8);
        bgr[2] = clamp((298 * c0 + 409 * e + 128) >> 8);
        bgr[3] = clamp((298 * c1 + 516 * d + 128) >> 8);
        bgr[4] = clamp((298 * c1 - 100 * d - 208 * e + 128) >> 8);
        bgr[5] = clamp((298 * c1 + 409 * e + 128) >> 8);
        bgr += 6; // 两个像素共 6 字节
    }
}

/**
 * camera_init - 初始化摄像头设备
 * @dev: 设备路径，如 "/dev/video0"
 * @w:   期望宽度
 * @h:   期望高度
 * 返回：Camera 句柄（成功），NULL（失败）
 * 执行：打开设备，设置格式、帧率，申请并映射缓冲区，开始流
 */
void* camera_init(const char *dev, int w, int h)
{
    struct Camera *cam = calloc(1, sizeof(struct Camera));
    if (!cam) return NULL;
    cam->fd = -1;
    cam->w = w; cam->h = h;

    cam->fd = open(dev, O_RDWR | O_NONBLOCK);
    if (cam->fd < 0) { perror("[CAM] open"); goto fail; }

    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = w;
    fmt.fmt.pix.height = h;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (xioctl(cam->fd, VIDIOC_S_FMT, &fmt) < 0) {
        perror("[CAM] fmt"); goto fail;
    }
    cam->w = fmt.fmt.pix.width;
    cam->h = fmt.fmt.pix.height;
    printf("[CAM] %s %dx%d YUYV\n", dev, cam->w, cam->h);

    // 设置帧率（使用 CAM_FPS 宏）
    struct v4l2_streamparm sp = {0};
    sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    sp.parm.capture.timeperframe.numerator = 1;
    sp.parm.capture.timeperframe.denominator = CAM_FPS;
    xioctl(cam->fd, VIDIOC_S_PARM, &sp);

    // 申请缓冲区（MMAP 方式）
    struct v4l2_requestbuffers req = {0};
    req.count = CAM_BUF_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(cam->fd, VIDIOC_REQBUFS, &req) < 0) { perror("[CAM] reqbufs"); goto fail; }
    cam->nbufs = req.count;

    // 逐个映射缓冲区
    for (unsigned int i = 0; i < req.count; i++) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        xioctl(cam->fd, VIDIOC_QUERYBUF, &buf);
        cam->bufs[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, cam->fd, buf.m.offset);
        cam->len[i] = buf.length;
    }

    // 将所有缓冲区入队
    for (unsigned int i = 0; i < cam->nbufs; i++) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        xioctl(cam->fd, VIDIOC_QBUF, &buf);
    }
    // 开始视频流
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(cam->fd, VIDIOC_STREAMON, &type);

    printf("[CAM] Ready, %u buffers\n", cam->nbufs);
    return cam;

fail:
    if (cam->fd >= 0) close(cam->fd);
    free(cam);
    return NULL;
}

/**
 * camera_grab - 捕获一帧图像并转换为 BGR 格式
 * @ctx: camera_init 返回的句柄
 * @bgr: 输出缓冲区（至少需 w*h*3 字节）
 * 返回：0 成功，-1 失败
 * 执行：从驱动取出一个已填充的缓冲区，转换数据，然后重新入队
 */
int camera_grab(void *ctx, unsigned char *bgr)
{
    struct Camera *cam = (struct Camera *)ctx;
    if (!cam) return -1;
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    // 非阻塞等待，直到有帧可用
    while (1) {
        if (xioctl(cam->fd, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) { usleep(5000); continue; }
            return -1;
        }
        break;
    }

    // 将 YUYV 转为 BGR，存放在外部提供的 bgr 缓冲区
    yuyv_to_bgr(cam->bufs[buf.index], bgr, cam->w * cam->h);
    // 将缓冲区重新入队以供驱动继续使用
    xioctl(cam->fd, VIDIOC_QBUF, &buf);
    return 0;
}

/**
 * camera_close - 关闭摄像头，释放所有资源
 * @ctx: camera_init 返回的句柄
 * 执行：停止流，解除映射，关闭设备，释放结构体内存
 */
void camera_close(void *ctx)
{
    struct Camera *cam = (struct Camera *)ctx;
    if (!cam) return;
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(cam->fd, VIDIOC_STREAMOFF, &type);
    for (unsigned int i = 0; i < cam->nbufs; i++)
        munmap(cam->bufs[i], cam->len[i]);
    close(cam->fd);
    free(cam);
}