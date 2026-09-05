/**
 * main.c - CenterNet 目标检测系统主程序（纯 C）
 *
 * 功能：整合 V4L2 摄像头采集、NPU（Ascend 310B4）推理、CenterNet 解码、
 *       检测框绘制和 HTTP MJPEG 视频流服务，构建完整的端到端目标检测系统。
 * 流程：摄像头 → 预处理 → NPU 推理 → 后处理解码 → 绘制检测框 → HTTP 推流
 *
 * 基于 xingyizhou/CenterNet 官方实现，使用 ResNet-18 作为骨干网络。
 * 编译和运行详见代码头部注释。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <time.h>

#include "acl/acl.h"
#include "centernet.h"
#include "camera.h"
#include "preprocess.h"
#include "postprocess.h"
#include "draw.h"
#include "httpd.h"

/* ================================================================
 * 全局状态
 * ================================================================ */
static volatile int running = 1;                          // 程序运行标志
static unsigned char *shared_bgr = NULL;                // 共享 BGR 帧缓冲区（供主线程发送）
static unsigned char *shared_jpg = NULL;                // 共享 JPEG 数据缓存
static int            shared_jpg_size = 0;              // 当前 JPEG 数据大小
static int            shared_jpg_seq  = 0;              // 帧序号（用于检测新帧）
static pthread_mutex_t frame_lock = PTHREAD_MUTEX_INITIALIZER; // 保护共享缓冲区

/* 类别名称（仅检测人脸，可根据需要扩展） */
static const char *class_names[] = {"face"};

/* ================================================================
 * NPU / ACL 上下文结构体及操作函数
 * ================================================================ */
typedef struct {
    uint32_t       model_id;      // 已加载模型的 ID
    aclmdlDesc    *desc;          // 模型描述符
    aclmdlDataset *in_ds;         // 输入数据集
    aclmdlDataset *out_ds;        // 输出数据集
    void          *dev_in;        // 设备端输入缓冲区
    void          *dev_out;       // 设备端输出缓冲区
    float         *host_in;       // 主机端输入缓冲区（支持 DMA）
    float         *host_out;      // 主机端输出缓冲区（支持 DMA）
    int            in_sz;         // 输入数据大小（字节）
    int            out_sz;        // 输出数据大小（字节）
} NPUCtx;

/**
 * npu_init - 初始化 NPU 环境并加载模型
 * @n:          NPU 上下文指针
 * @model_path: 模型文件路径（.om 格式）
 * 返回：0 成功，-1 失败
 * 说明：初始化 ACL 运行时，设置设备，加载模型，分配主机和设备内存，
 *       创建数据集，并准备好 DMA 缓冲区。
 */
static int npu_init(NPUCtx *n, const char *model_path)
{
    aclError ret;

    /* 初始化 ACL */
    ret = aclInit(NULL);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU] aclInit failed: %d\n", ret);
        return -1;
    }

    ret = aclrtSetDevice(0);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU] aclrtSetDevice failed: %d\n", ret);
        return -1;
    }

    /* 加载模型 */
    ret = aclmdlLoadFromFile(model_path, &n->model_id);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU] aclmdlLoadFromFile failed: %d\n", ret);
        return -1;
    }

    /* 获取模型描述 */
    n->desc = aclmdlCreateDesc();
    ret = aclmdlGetDesc(n->desc, n->model_id);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU] aclmdlGetDesc failed: %d\n", ret);
        return -1;
    }

    n->in_sz  = aclmdlGetInputSizeByIndex(n->desc, 0);
    n->out_sz = aclmdlGetOutputSizeByIndex(n->desc, 0);

    printf("[NPU] Model loaded: input=%d bytes, output=%d bytes\n",
           n->in_sz, n->out_sz);
    printf("[NPU] Input tensor:  [1, 3, 512, 512] float32\n");
    printf("[NPU] Output tensor: [1, %d, 128, 128] float32\n", CN_CH);

    /* 分配设备内存 */
    ret = aclrtMalloc(&n->dev_in,  n->in_sz,  ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU] malloc dev_in failed: %d\n", ret);
        return -1;
    }

    ret = aclrtMalloc(&n->dev_out, n->out_sz, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU] malloc dev_out failed: %d\n", ret);
        return -1;
    }

    /* 创建数据集并绑定缓冲区 */
    n->in_ds = aclmdlCreateDataset();
    ret = aclmdlAddDatasetBuffer(n->in_ds,
          aclCreateDataBuffer(n->dev_in, n->in_sz));
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU] create input dataset failed: %d\n", ret);
        return -1;
    }

    n->out_ds = aclmdlCreateDataset();
    ret = aclmdlAddDatasetBuffer(n->out_ds,
          aclCreateDataBuffer(n->dev_out, n->out_sz));
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU] create output dataset failed: %d\n", ret);
        return -1;
    }

    /* 分配主机端 DMA 缓冲区（用于零拷贝传输） */
    {
        void *tmp = NULL;
        ret = aclrtMallocHost(&tmp, n->in_sz);
        n->host_in = (float *)tmp;
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[NPU] malloc host_in failed: %d\n", ret);
            return -1;
        }
        ret = aclrtMallocHost(&tmp, n->out_sz);
        n->host_out = (float *)tmp;
        if (ret != ACL_SUCCESS) {
            fprintf(stderr, "[NPU] malloc host_out failed: %d\n", ret);
            return -1;
        }
    }

    printf("[NPU] Ready\n");
    return 0;
}

/**
 * npu_run - 执行一次 NPU 推理
 * @n: NPU 上下文指针
 * 返回：0 成功，-1 失败
 * 说明：将主机输入缓冲区数据拷贝到设备，执行推理，再将输出拷贝回主机输出缓冲区。
 *       调用前需保证 host_in 已填充数据，推理后结果在 host_out 中。
 */
static int npu_run(NPUCtx *n)
{
    aclError ret;

    /* 拷贝输入到设备 */
    ret = aclrtMemcpy(n->dev_in, n->in_sz, n->host_in, n->in_sz,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU] memcpy H2D failed: %d\n", ret);
        return -1;
    }

    /* 执行推理 */
    ret = aclmdlExecute(n->model_id, n->in_ds, n->out_ds);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU] execute failed: %d\n", ret);
        return -1;
    }

    /* 拷贝输出到主机 */
    ret = aclrtMemcpy(n->host_out, n->out_sz, n->dev_out, n->out_sz,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        fprintf(stderr, "[NPU] memcpy D2H failed: %d\n", ret);
        return -1;
    }

    return 0;
}

/**
 * npu_cleanup - 释放 NPU 资源
 * @n: NPU 上下文指针
 * 说明：释放所有分配的内存、数据集、模型描述符，卸载模型，重置设备并 finalize ACL。
 */
static void npu_cleanup(NPUCtx *n)
{
    if (n->host_in)  aclrtFreeHost(n->host_in);
    if (n->host_out) aclrtFreeHost(n->host_out);
    if (n->dev_in)   aclrtFree(n->dev_in);
    if (n->dev_out)  aclrtFree(n->dev_out);
    if (n->in_ds)    aclmdlDestroyDataset(n->in_ds);
    if (n->out_ds)   aclmdlDestroyDataset(n->out_ds);
    if (n->desc)     aclmdlDestroyDesc(n->desc);
    if (n->model_id) aclmdlUnload(n->model_id);
    aclrtResetDevice(0);
    aclFinalize();
}

/* ================================================================
 * 检测线程函数
 * ================================================================ */
/**
 * detect_thread_proc - 检测线程主循环
 * @arg: 包含 NPU 上下文和相机句柄的指针数组
 * 返回：NULL
 * 说明：独立线程中循环执行：抓帧 → 预处理 → NPU 推理 → 解码 → 绘制 → 更新共享帧。
 *       该线程需单独设置 ACL 设备上下文。
 */
static void* detect_thread_proc(void *arg)
{
    NPUCtx *npu = ((void**)arg)[0];
    void   *cam = ((void**)arg)[1];

    /* ACL 设备上下文为线程本地，必须在线程内设置 */
    aclrtSetDevice(0);

    /* 分配工作帧缓冲区 */
    unsigned char *frame_bgr = malloc(CAM_FRAME_BYTES);
    if (!frame_bgr) {
        fprintf(stderr, "[DETECT] OOM for frame buffer\n");
        return NULL;
    }

    Detection dets[TOPK_MAX];
    int       frame_count = 0;
    time_t    t_start     = time(NULL);

    while (running) {
        /* ---- 1. 从摄像头抓取一帧 ---- */
        if (camera_grab(cam, frame_bgr) < 0) {
            usleep(5000);
            continue;
        }

        /* ---- 2. 预处理（放入 DMA 缓冲区 host_in） ---- */
        preprocess_frame(frame_bgr, CAM_W, CAM_H, npu->host_in);

        /* ---- 3. NPU 推理 ---- */
        if (npu_run(npu) < 0) continue;

        /* ---- 4. CenterNet 解码（后处理） ---- */
        int ndet = centernet_decode(npu->host_out, dets, TOPK_MAX,
                                    CAM_W, CAM_H);

        /* ---- 5. 在帧上绘制检测框 ---- */
        draw_detections(frame_bgr, CAM_W, CAM_H, dets, ndet, class_names);

        /* ---- 6. 将 BGR 拷贝到共享缓冲区，并编码为 JPEG ---- */
        pthread_mutex_lock(&frame_lock);
        memcpy(shared_bgr, frame_bgr, CAM_FRAME_BYTES);
        shared_jpg_size = httpd_encode_frame(frame_bgr, CAM_W, CAM_H, &shared_jpg);
        shared_jpg_seq++;
        pthread_mutex_unlock(&frame_lock);

        /* ---- 7. 统计信息 ---- */
        frame_count++;

        /* 每 30 帧输出一次性能数据 */
        if (frame_count % 30 == 0) {
            float elapsed = (float)(time(NULL) - t_start + 1);
            float fps     = frame_count / elapsed;
            printf("[DETECT] %d frames, %.1f FPS, %d objects\n",
                   frame_count, fps, ndet);
        }
    }

    free(frame_bgr);
    return NULL;
}

/* ================================================================
 * 信号处理函数
 * ================================================================ */
/**
 * on_signal - 信号处理函数（SIGINT/SIGTERM）
 * @sig: 信号编号（未使用）
 * 说明：将 running 置 0，通知所有循环退出，实现优雅停止。
 */
static void on_signal(int sig)
{
    (void)sig;
    running = 0;
    printf("\n[STOP] Shutting down...\n");
}

/* ================================================================
 * 主函数
 * ================================================================ */
/**
 * main - 程序入口
 * @argc: 参数个数
 * @argv: 参数列表（可选：模型路径、HTTP端口）
 * 返回：0 正常退出，非0 错误
 * 说明：初始化各模块，启动检测线程，主循环负责将 JPEG 帧发送给 HTTP 客户端。
 *       可按 Ctrl+C 停止。
 */
int main(int argc, char **argv)
{
    const char *model_path = (argc > 1) ? argv[1] : MODEL_PATH;
    int         http_port  = (argc > 2) ? atoi(argv[2]) : HTTP_PORT;

    printf("\n");
    printf("================================================\n");
    printf("  CenterNet Object Detection (Pure C)\n");
    printf("  Based on xingyizhou/CenterNet\n");
    printf("  Backend: ResNet-18 | NPU: Ascend 310B4\n");
    printf("================================================\n");
    printf("  Model:  %s\n", model_path);
    printf("  Stream: http://192.168.137.100:%d/video\n", http_port);
    printf("================================================\n\n");

    /* 注册信号处理 */
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    /* ---- 初始化摄像头（自动尝试多个设备） ---- */
    const char *cam_devices[] = {"/dev/video0", "/dev/video1", "/dev/video2", NULL};
    void *cam = NULL;
    for (int i = 0; cam_devices[i]; i++) {
        cam = camera_init(cam_devices[i], CAM_W, CAM_H);
        if (cam) break;
        printf("[INIT] Trying next device...\n");
    }
    if (!cam) {
        fprintf(stderr, "[FATAL] No camera found\n");
        return 1;
    }

    /* ---- 初始化 NPU ---- */
    NPUCtx npu;
    memset(&npu, 0, sizeof(npu));
    if (npu_init(&npu, model_path) < 0) {
        fprintf(stderr, "[FATAL] NPU init failed\n");
        camera_close(cam);
        return 1;
    }

    /* ---- 分配共享帧缓冲区 ---- */
    shared_bgr = malloc(CAM_FRAME_BYTES);
    if (!shared_bgr) {
        fprintf(stderr, "[FATAL] OOM for shared buffer\n");
        npu_cleanup(&npu);
        camera_close(cam);
        return 1;
    }
    memset(shared_bgr, 128, CAM_FRAME_BYTES);  /* 初始灰色帧 */

    /* ---- 启动 HTTP 服务器 ---- */
    if (httpd_start(http_port) < 0) {
        fprintf(stderr, "[FATAL] HTTP server start failed\n");
        free(shared_bgr);
        npu_cleanup(&npu);
        camera_close(cam);
        return 1;
    }

    /* ---- 启动检测线程 ---- */
    void *targs[2] = { &npu, cam };
    pthread_t detect_tid;
    if (pthread_create(&detect_tid, NULL, detect_thread_proc, targs) != 0) {
        fprintf(stderr, "[FATAL] Thread create failed\n");
        httpd_stop();
        free(shared_bgr);
        npu_cleanup(&npu);
        camera_close(cam);
        return 1;
    }

    /* 检测线程负责编码 JPEG，主线程负责将新帧发送给所有 HTTP 客户端 */
    printf("[MAIN] Running... Press Ctrl+C to stop\n\n");
    int last_seq = -1;
    while (running) {
        int cur_seq, js;
        unsigned char *j;

        pthread_mutex_lock(&frame_lock);
        cur_seq = shared_jpg_seq;
        j  = shared_jpg;
        js = shared_jpg_size;
        pthread_mutex_unlock(&frame_lock);

        if (cur_seq != last_seq && j && js > 0) {
            httpd_send_jpg(j, js);
            last_seq = cur_seq;
        }
        usleep(100000);  /* 约 10fps 发送，避免 TCP 拥塞 */
    }

    /* ---- 清理资源 ---- */
    printf("[STOP] Waiting for detection thread...\n");
    pthread_join(detect_tid, NULL);
    httpd_stop();
    free(shared_bgr);
    free(shared_jpg);
    npu_cleanup(&npu);
    camera_close(cam);

    printf("[STOP] Done.\n");
    return 0;
}