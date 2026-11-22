/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#ifndef GDK_CNN_FORWARD_H
#define GDK_CNN_FORWARD_H

#include "gdk_cnn.h"

/* 借用连续 float32 数组；count 为元素数，不拥有参数内存。 */
typedef struct {
	const float *values;
	size_t count;
} CnnTensorView;

/*
 * 固定网络的六个参数视图；不依赖数据库中的 ModelId 或张量存储格式。
 * 卷积权重为 OIHW，全连接权重为 OI，各偏置为一维数组。
 */
typedef struct {
	CnnTensorView conv1_weight; /* [4,3,3,3]，108 个元素。 */
	CnnTensorView conv1_bias;   /* [4]。 */
	CnnTensorView conv2_weight; /* [8,4,3,3]，288 个元素。 */
	CnnTensorView conv2_bias;   /* [8]。 */
	CnnTensorView fc_weight;    /* [2,8]，16 个元素。 */
	CnnTensorView fc_bias;      /* [2]。 */
} CnnFixedParameters;

/*
 * 拥有固定 batch size 的前向工作区，可反复用于同样大小的 batch。
 * 调用者先置零，再调用 init；除读取计算结果外，不得手动修改这些字段。
 * Conv 直接写 relu1/2 后原地激活，不另存重复的卷积输出。
 * 池化 argmax 是输入通道平面内索引，供后续反向使用，不属于模型参数。
 * ready 仅在最近一次完整前向成功后为 true；失败后的缓存不得使用。
 */
typedef struct {
	size_t batch;
	bool ready;
	float *relu1; /* [B,4,64,64]。 */
	float *pool1; /* [B,4,32,32]。 */
	float *relu2; /* [B,8,32,32]。 */
	float *pool2; /* [B,8,16,16]。 */
	float *gap;   /* [B,8]。 */
	float *logits; /* [B,2]，尚未经过 Softmax。 */
	size_t *pool1_argmax; /* [B,4,32,32]。 */
	size_t *pool2_argmax; /* [B,8,16,16]。 */
} CnnForwardWorkspace;

/* workspace 必须完全为空；检查/分配失败时保持原样，不发布部分工作区。 */
gdk_export CnnOpsResult cnn_forward_workspace_init(CnnForwardWorkspace *workspace, size_t batch);

/* 允许 NULL 以及重复调用；释放后恢复为空工作区。 */
gdk_export void cnn_forward_workspace_destroy(CnnForwardWorkspace *workspace);

/*
 * 执行固定的 64×64 RGB 网络，input 为连续 [workspace->batch,3,64,64]。
 * input_count 必须精确匹配；六个参数的元素数必须与上述形状相符。
 * 所有输入/参数必须为有限值，并且不与任何工作区数组重叠。
 * 输入和参数由调用者持有；本函数不修改它们，也不记录借用指针。
 * 后续若用缓存做反向，调用者必须保留对应输入和参数，并先反向再更新。
 *
 * 非空工作区在每次调用开始时先置 ready=false。
 * 参数检查失败或中间计算产生非有限值时返回错误；工作区可能已部分
 * 改写，不能继续使用其缓存。工作区仍可重新前向或销毁。
 * 成功后 ready=true，logits 为两个类别的原始分数，损失需另行计算。
 * 不分配内存，不执行 SQL，不更新参数、优化器状态或训练进度。
 */
gdk_export CnnOpsResult cnn_fixed_forward(const CnnFixedParameters *parameters, const float *input, size_t input_count, CnnForwardWorkspace *workspace);

#endif /* GDK_CNN_FORWARD_H */
