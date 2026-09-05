/**
 * httpd.c - HTTP MJPEG 流服务器（使用 libjpeg-turbo 硬件加速编码）
 *
 * 功能：启动一个 HTTP 服务器，以 multipart/x-mixed-replace 格式提供 MJPEG 视频流。
 *       使用 libjpeg-turbo 将 BGR 图像编码为 JPEG（支持 ARM NEON 加速），
 *       同时管理多个客户端连接（最多 8 个），通过线程非阻塞接收新连接。
 *       编码结果缓存在内存中，供所有客户端共享，提高性能。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <errno.h>
#include <jpeglib.h>
#include "httpd.h"
#include "centernet.h"

/* ── libjpeg-turbo 编码（利用 NEON 硬件加速） ── */
static unsigned char *jpg_cache = NULL;   // 缓存的 JPEG 数据
static unsigned long jpg_cache_sz = 0;    // 缓存数据大小
static unsigned char *rgb_buf = NULL;     // 临时 RGB 缓冲区（BGR→RGB 转换用）

/**
 * httpd_encode_frame - 将 BGR 图像编码为 JPEG
 * @bgr: 输入 BGR24 图像数据
 * @w:   图像宽度
 * @h:   图像高度
 * @out: 输出参数，返回 JPEG 数据指针（指向内部缓存，调用者不应释放）
 * 返回：JPEG 数据大小（字节），负值表示失败
 * 说明：内部进行 BGR→RGB 转换，使用 libjpeg-turbo 压缩，质量由 JPEG_QUALITY 宏控制。
 *       编码结果存入静态缓存，下次调用会释放旧缓存并更新。
 */
int httpd_encode_frame(const unsigned char *bgr, int w, int h, unsigned char **out)
{
    if (!rgb_buf) rgb_buf = malloc(w * h * 3);

    /* BGR → RGB 单次遍历转换 */
    int npix = w * h;
    for (int i = 0; i < npix; i++) {
        rgb_buf[i*3+0] = bgr[i*3+2];
        rgb_buf[i*3+1] = bgr[i*3+1];
        rgb_buf[i*3+2] = bgr[i*3+0];
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    unsigned char *tmp = NULL;
    unsigned long tmp_sz = 0;
    jpeg_mem_dest(&cinfo, &tmp, &tmp_sz);   // 输出到内存

    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, JPEG_QUALITY, TRUE);  // 质量由外部宏定义

    jpeg_start_compress(&cinfo, TRUE);
    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = rgb_buf + cinfo.next_scanline * w * 3;
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    /* 替换旧缓存 */
    free(jpg_cache);
    jpg_cache = tmp;
    jpg_cache_sz = tmp_sz;
    *out = jpg_cache;
    return (int)tmp_sz;
}

/**
 * httpd_send_frame - 便捷函数：编码并发送当前帧给所有客户端
 * @bgr: BGR 图像数据
 * @w, @h: 图像尺寸
 * 内部调用 httpd_encode_frame 编码，再通过 httpd_send_jpg 分发。
 */
void httpd_send_frame(const unsigned char *bgr, int w, int h) {
    unsigned char *j = NULL;
    int s = httpd_encode_frame(bgr, w, h, &j);
    if (s > 0) httpd_send_jpg(j, s);
}

/* ── HTTP 服务器状态 ── */
static int sfd = -1, cfds[8], nc = 0;          // 监听 socket、客户端 socket 数组及数量
static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER; // 保护客户端列表
static volatile int run = 1;                   // 运行标志

/**
 * aloop - 后台线程：持续接受新客户端连接
 * @_: 未使用
 * 说明：循环调用 accept，将新客户端加入 cfds 数组（最多 8 个），
 *       立即发送 MJPEG 流所需的 HTTP 头。
 *       若数组满则直接关闭连接。线程在 run=0 时退出。
 */
static void *aloop(void *_) {
    (void)_;
    while (run) {
        struct sockaddr_in a; socklen_t al = sizeof(a);
        int fd = accept(sfd, (struct sockaddr*)&a, &al);
        if (fd < 0) { if (run) usleep(100000); continue; }
        // 禁用 Nagle 算法，减少延迟
        { int f = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &f, sizeof(f)); }
        // 发送 MJPEG 流头部（multipart 边界）
        const char *hdr = "HTTP/1.0 200 OK\r\n"
                          "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                          "Cache-Control: no-cache\r\n"
                          "\r\n"
                          "--frame\r\n";
        send(fd, hdr, strlen(hdr), MSG_NOSIGNAL);
        pthread_mutex_lock(&lk);
        if (nc < 8) cfds[nc++] = fd; else close(fd);
        pthread_mutex_unlock(&lk);
    }
    return NULL;
}

/**
 * httpd_start - 启动 HTTP 服务器
 * @port: 监听端口
 * 返回：0 成功，-1 失败
 * 说明：创建 TCP socket，绑定端口，开始监听，并启动接受线程。
 */
int httpd_start(int port) {
    sfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sfd < 0) return -1;
    { int o = 1; setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &o, sizeof(o)); }
    struct sockaddr_in a = {AF_INET, htons(port), {INADDR_ANY}, {0}};
    if (bind(sfd, (struct sockaddr*)&a, sizeof(a)) < 0) { close(sfd); return -1; }
    if (listen(sfd, 5) < 0) { close(sfd); return -1; }
    pthread_t t; pthread_create(&t, NULL, aloop, NULL); pthread_detach(t);
    printf("[HTTP] :%d\n", port);
    return 0;
}

/**
 * httpd_send_jpg - 将 JPEG 数据发送给所有已连接的客户端
 * @j: JPEG 数据指针
 * @s: 数据长度
 * 说明：为每个客户端构造 multipart 块（包含边界和 Content-Length），
 *       使用 sendmsg 一次发送头 + 数据 + 尾。
 *       若发送失败则关闭该客户端并从数组中移除。
 *       加锁保证线程安全。
 */
void httpd_send_jpg(const unsigned char *j, int s) {
    if (!j || s <= 0) return;
    char h[80]; int hs = snprintf(h, sizeof(h), "Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n", s);
    pthread_mutex_lock(&lk);
    for (int i = 0; i < nc; ) {
        struct iovec iv[4] = {{"--frame\r\n",9},{h,hs},{(void*)j,s},{"\r\n",2}};
        struct msghdr m = {0}; m.msg_iov = iv; m.msg_iovlen = 4;
        if (sendmsg(cfds[i], &m, MSG_NOSIGNAL) < 0) {
            close(cfds[i]);
            cfds[i] = cfds[--nc];  // 用最后一个替换当前
        } else i++;
    }
    pthread_mutex_unlock(&lk);
}

/**
 * httpd_stop - 停止 HTTP 服务器，释放所有资源
 * 说明：设置运行标志为 0，关闭所有客户端 socket 和监听 socket，
 *       释放 JPEG 缓存和 RGB 缓冲区。
 */
void httpd_stop(void) {
    run = 0;
    pthread_mutex_lock(&lk);
    for (int i = 0; i < nc; i++) close(cfds[i]); nc = 0;
    pthread_mutex_unlock(&lk);
    if (sfd >= 0) { close(sfd); sfd = -1; }
    free(jpg_cache); jpg_cache = NULL;
    free(rgb_buf); rgb_buf = NULL;
}