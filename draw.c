/**
 * draw.c - 目标检测结果可视化（纯 C 像素操作）
 * 
 * 功能：在 BGR 图像上绘制检测框、类别标签和置信度。
 *       采用鲜艳的颜色、3 像素粗边框、大号文字（5x7 字体 2 倍缩放），
 *       并在标签下方添加半透明背景以提高可读性。
 *       依赖 camera.h 中的 CAM_W/CAM_H 宏定义图像尺寸。
 */

#include <stdio.h>
#include <string.h>
#include "draw.h"
#include "centernet.h"

/* 预定义类别颜色（BGR 格式） */
static const unsigned int class_colors[] = {
    0x00FF4040u,  /* person: 蓝色 */
    0x000040FFu,  /* car:    红色 */
    0x0040FF40u,  /* dog:    绿色 */
    0x00FFFF00u,  /* cat:    青色 */
    0x0040FFFFu,  /* bird:   黄色 */
};
#define N_COLORS (sizeof(class_colors) / sizeof(class_colors[0]))

/**
 * draw_rect - 绘制填充矩形
 * @img:    BGR 图像数据指针
 * @img_w:  图像宽度（像素）
 * @x, @y:  矩形左上角坐标
 * @w, @h:  矩形宽高
 * @color:  BGR 颜色（0x00BBGGRR）
 * 说明：自动裁剪到图像边界，逐像素设置颜色。
 */
static void draw_rect(unsigned char *img, int img_w,
                      int x, int y, int w, int h, unsigned int color)
{
    unsigned char b = (color >> 16) & 0xFF;
    unsigned char g = (color >> 8)  & 0xFF;
    unsigned char r = (color >> 0)  & 0xFF;

    int x0 = x < 0 ? 0 : x;
    int y0 = y < 0 ? 0 : y;
    int x1 = x + w; if (x1 > CAM_W) x1 = CAM_W;
    int y1 = y + h; if (y1 > CAM_H) y1 = CAM_H;

    for (int row = y0; row < y1; row++) {
        for (int col = x0; col < x1; col++) {
            int idx = (row * img_w + col) * 3;
            img[idx + 0] = b;
            img[idx + 1] = g;
            img[idx + 2] = r;
        }
    }
}

/**
 * draw_rect_border - 绘制空心矩形边框（指定粗细）
 * @img:    BGR 图像数据
 * @img_w:  图像宽度
 * @x, @y:  左上角坐标
 * @w, @h:  宽高
 * @color:  边框颜色
 * @thick:  边框像素厚度（>=1）
 * 说明：通过多次调用 draw_rect 绘制顶部、底部、左侧和右侧的横/竖条。
 */
static void draw_rect_border(unsigned char *img, int img_w,
                             int x, int y, int w, int h,
                             unsigned int color, int thick)
{
    for (int t = 0; t < thick; t++) {
        draw_rect(img, img_w, x+t, y+t, w-t*2, 1, color);
        draw_rect(img, img_w, x+t, y+h-1-t, w-t*2, 1, color);
        draw_rect(img, img_w, x+t, y+t, 1, h-t*2, color);
        draw_rect(img, img_w, x+w-1-t, y+t, 1, h-t*2, color);
    }
}

/* ---- 5x7 点阵字体（ASCII 可见字符） ---- */
static const unsigned char font5x7[128][7] = {
    ['0'] = {0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1F},
    ['1'] = {0x04, 0x06, 0x04, 0x04, 0x04, 0x04, 0x0E},
    ['2'] = {0x1F, 0x10, 0x10, 0x1F, 0x01, 0x01, 0x1F},
    ['3'] = {0x1F, 0x10, 0x10, 0x1F, 0x10, 0x10, 0x1F},
    ['4'] = {0x11, 0x11, 0x11, 0x1F, 0x10, 0x10, 0x10},
    ['5'] = {0x1F, 0x01, 0x01, 0x1F, 0x10, 0x10, 0x1F},
    ['6'] = {0x1F, 0x01, 0x01, 0x1F, 0x11, 0x11, 0x1F},
    ['7'] = {0x1F, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10},
    ['8'] = {0x1F, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x1F},
    ['9'] = {0x1F, 0x11, 0x11, 0x1F, 0x10, 0x10, 0x1F},
    ['%'] = {0x13, 0x13, 0x08, 0x04, 0x02, 0x19, 0x19},
    ['.'] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06},
    [' '] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    ['a'] = {0x00, 0x0F, 0x10, 0x1F, 0x11, 0x11, 0x1F},
    ['b'] = {0x01, 0x01, 0x0F, 0x11, 0x11, 0x11, 0x0F},
    ['c'] = {0x00, 0x0F, 0x01, 0x01, 0x01, 0x01, 0x0F},
    ['d'] = {0x10, 0x10, 0x1E, 0x11, 0x11, 0x11, 0x1E},
    ['e'] = {0x00, 0x0F, 0x11, 0x1F, 0x01, 0x01, 0x0F},
    ['f'] = {0x0C, 0x02, 0x07, 0x02, 0x02, 0x02, 0x02},
    ['g'] = {0x00, 0x0E, 0x11, 0x11, 0x1E, 0x10, 0x0F},
    ['h'] = {0x01, 0x01, 0x0D, 0x13, 0x11, 0x11, 0x11},
    ['i'] = {0x02, 0x00, 0x03, 0x02, 0x02, 0x02, 0x02},
    ['j'] = {0x04, 0x00, 0x06, 0x04, 0x04, 0x04, 0x03},
    ['k'] = {0x01, 0x01, 0x09, 0x05, 0x03, 0x05, 0x09},
    ['l'] = {0x03, 0x02, 0x02, 0x02, 0x02, 0x02, 0x06},
    ['m'] = {0x00, 0x1B, 0x15, 0x15, 0x15, 0x15, 0x15},
    ['n'] = {0x00, 0x0D, 0x13, 0x11, 0x11, 0x11, 0x11},
    ['o'] = {0x00, 0x0E, 0x11, 0x11, 0x11, 0x11, 0x0E},
    ['p'] = {0x00, 0x0D, 0x13, 0x11, 0x11, 0x0F, 0x01},
    ['q'] = {0x00, 0x16, 0x19, 0x11, 0x11, 0x1E, 0x10},
    ['r'] = {0x00, 0x0D, 0x13, 0x01, 0x01, 0x01, 0x01},
    ['s'] = {0x00, 0x0E, 0x01, 0x0E, 0x10, 0x10, 0x0F},
    ['t'] = {0x02, 0x07, 0x02, 0x02, 0x02, 0x12, 0x0C},
    ['u'] = {0x00, 0x11, 0x11, 0x11, 0x11, 0x19, 0x16},
    ['v'] = {0x00, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04},
    ['w'] = {0x00, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A},
    ['x'] = {0x00, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11},
    ['y'] = {0x00, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x03},
    ['z'] = {0x00, 0x1F, 0x08, 0x04, 0x02, 0x11, 0x1F},
};

/**
 * plot2x - 绘制一个 2×2 放大的像素点（用于字体缩放）
 * @img:    BGR 图像
 * @img_w:  图像宽度
 * @px,@py: 原始坐标（该点将被放大为 2x2 方块）
 * @color:  BGR 颜色
 * 说明：在 (px,py) 处绘制一个 2x2 的色块，边界自动裁剪。
 */
static void plot2x(unsigned char *img, int img_w, int px, int py, unsigned int color)
{
    unsigned char b = (color >> 16) & 0xFF;
    unsigned char g = (color >> 8)  & 0xFF;
    unsigned char r = (color >> 0)  & 0xFF;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int fx = px + dx, fy = py + dy;
            if (fx >= 0 && fx < img_w && fy >= 0 && fy < CAM_H) {
                int idx = (fy * img_w + fx) * 3;
                img[idx+0]=b; img[idx+1]=g; img[idx+2]=r;
            }
        }
    }
}

/**
 * draw_text_scaled - 绘制 2 倍放大的文本（使用 5x7 字体）
 * @img:    BGR 图像
 * @img_w:  图像宽度
 * @x, @y:  文本起始坐标（左上角）
 * @text:   要绘制的字符串（仅 ASCII 可见字符）
 * @color:  文本颜色（BGR）
 * 说明：每个字符实际占用 10x14 像素（5x7 * 2），字符间距 2 像素。
 */
static void draw_text_scaled(unsigned char *img, int img_w, int x, int y,
                              const char *text, unsigned int color)
{
    int cx = x;
    for (const char *c = text; *c; c++) {
        const unsigned char *glyph = font5x7[(int)(unsigned char)*c];
        for (int row = 0; row < 7; row++) {
            for (int col = 0; col < 5; col++) {
                if (glyph[row] & (1 << col)) {
                    plot2x(img, img_w, cx + col * 2, y + row * 2, color);
                }
            }
        }
        cx += 12;  /* 5*2 + 2 gap */
    }
}

/**
 * draw_detections - 主绘制函数：在图像上绘制所有检测结果
 * @bgr:   BGR 图像数据（会被直接修改）
 * @w:     图像宽度（应与 CAM_W 一致）
 * @h:     图像高度（未使用，但保留接口）
 * @dets:  检测结果数组
 * @ndet:  检测数量
 * @names: 类别名称数组（可为 NULL，此时显示 "???"）
 * 说明：为每个检测绘制 3px 粗边框，并在框上方绘制半透明背景标签，
 *       标签内容为“类别名 置信度%”，文字为白色。
 */
void draw_detections(unsigned char *bgr, int w, int h,
                     const Detection *dets, int ndet, const char **names)
{
    (void)h;

    for (int i = 0; i < ndet; i++) {
        const Detection *d = &dets[i];
        int x1 = (int)d->x1, y1 = (int)d->y1;
        int x2 = (int)d->x2, y2 = (int)d->y2;
        int bw = x2 - x1, bh = y2 - y1;

        if (bw < 1 || bh < 1) continue;

        unsigned int color = class_colors[d->cls % N_COLORS];

        /* 绘制 3 像素粗边框 */
        draw_rect_border(bgr, w, x1, y1, bw, bh, color, 3);

        /* 构造标签字符串：类别名 + 置信度百分数 */
        char label[64];
        int pct = (int)(d->score * 100);
        if (pct > 99) pct = 99;
        snprintf(label, sizeof(label), "%s %d%%",
                 names ? names[d->cls] : "???", pct);

        int lw = (int)strlen(label) * 12 + 4;  /* 每个字符宽 12px + 左右内边距 2px */
        int lh = 16;                             /* 字符高 14px + 上下内边距 1px */
        int ly = y1 - lh;
        if (ly < 0) ly = 0;

        /* 半透明黑色背景（实际为纯黑，因无 alpha 混合） */
        draw_rect(bgr, w, x1, ly, lw, lh, 0x00202020u);

        /* 标签边框（使用类别颜色） */
        draw_rect(bgr, w, x1, ly, lw, 1, color);
        draw_rect(bgr, w, x1, ly + lh - 1, lw, 1, color);
        draw_rect(bgr, w, x1, ly, 1, lh, color);
        draw_rect(bgr, w, x1 + lw - 1, ly, 1, lh, color);

        /* 白色文本 */
        draw_text_scaled(bgr, w, x1 + 2, ly + 1, label, 0x00FFFFFFu);
    }
}