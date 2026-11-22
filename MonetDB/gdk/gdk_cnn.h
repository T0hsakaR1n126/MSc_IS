/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#ifndef GDK_CNN_H
#define GDK_CNN_H

#include "gdk.h"

/* 固定 3×3、stride=1、padding=1，因此输出 H、W 与输入相同。 */
typedef struct {
	size_t batch;
	size_t in_channels;
	size_t out_channels;
	size_t height;
	size_t width;
} CnnConv2dShape;

/* MaxPool 固定 2×2、stride=2、padding=0，输出 H/2、W/2（向下取整）。 */
typedef struct {
	size_t batch;
	size_t channels;
	size_t height;
	size_t width;
} CnnPool2dShape;

typedef enum {
	CNN_OPS_OK = 0,
	CNN_OPS_INVALID_INPUT,
	CNN_OPS_SIZE_OVERFLOW
} CnnOpsResult;

/*
 * float32 卷积前向，不翻转卷积核，采用深度学习常用的互相关约定。
 *
 * input:  [batch,in_channels,height,width]，连续 NCHW。
 * weight: [out_channels,in_channels,3,3]，连续 OIHW。
 * bias:   [out_channels]，当前固定网络要求提供偏置。
 * output: [batch,out_channels,height,width]，连续 NCHW。
 *
 * 所有维度必须大于零；四个 count 都是 float 元素数，不是字节数，
 * 且必须与形状精确匹配。各指针必须指向相应长度的有效数组。
 * 调用者提供输出内存，并保证输出与 input / weight / bias 均不重叠。
 * 输入和参数应为有限值；本函数不额外扫描 NaN/Inf 或检测浮点溢出。
 *
 * 维度、长度、整数溢出检查全部通过后才读写数组；检查失败时输出不变。
 * 本函数不分配内存、不执行 SQL、不修改输入、参数或模型状态。
 */
gdk_export CnnOpsResult cnn_conv2d_forward(const CnnConv2dShape *shape, const float *input, size_t input_count, const float *weight, size_t weight_count, const float *bias, size_t bias_count, float *output, size_t output_count);

/*
 * 逐元素 ReLU：正数原样输出，其余有限值输出 0.0f。
 * input_count 必须大于零，output_count 必须与其相等，均为 float 元素数。
 * 调用者提供有效数组；允许 input == output 完全原地计算，禁止部分重叠。
 * 输入必须为有限值，本函数不额外扫描 NaN/Inf。
 * 不分配内存；参数或整数溢出检查失败时输出不变。
 */
gdk_export CnnOpsResult cnn_relu_forward(const float *input, size_t input_count,
				       float *output, size_t output_count);

/*
 * float32 最大池化前向，输入为连续 [batch,channels,height,width] NCHW。
 * batch、channels 必须大于零，height、width 至少为 2。
 * 输出为 [batch,channels,height/2,width/2]；奇数边长的末行/末列不参与。
 * input_count、output_count 均为 float 元素数，必须与形状精确匹配。
 *
 * argmax 是可选的临时反向缓存：纯前向可传 NULL、0；启用时其长度必须
 * 等于输出元素数，记录每个最大值在对应输入 H×W 平面内的索引 h*W+w。
 * 同值时保留按行遍历遇到的第一个最大值；该缓存不属于持久化模型参数。
 *
 * 调用者提供全部数组，input、output、argmax（如有）三者不得重叠。
 * 输入必须为有限值，本函数不额外扫描 NaN/Inf。
 * 不分配内存；全部检查通过后才写 output 和 argmax，检查失败时两者不变。
 */
gdk_export CnnOpsResult cnn_maxpool2d_forward(const CnnPool2dShape *shape,
					    const float *input, size_t input_count,
					    float *output, size_t output_count,
					    size_t *argmax, size_t argmax_count);

#endif /* GDK_CNN_H */
