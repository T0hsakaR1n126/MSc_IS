/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#ifndef GDK_CNN_BACKWARD_H
#define GDK_CNN_BACKWARD_H

#include "gdk_cnn_forward.h"

/*
 * 固定网络分类末端的反向工作区；调用者先用 {0} 初始化，再调用 init。
 * init 后 batch 固定；除读取结果外，不得手动修改工作区字段。
 * 梯度为临时数据，与参数、Adam 状态及持久化模型分别管理。
 * ready 仅在最近一次末端反向完整成功后为 true。
 */
typedef struct {
	size_t batch;
	bool ready;
	double mean_loss;
	float *d_logits; /* [B,2]，平均交叉熵对 logits 的梯度。 */
	float *d_gap;    /* [B,8]。 */
	float *d_pool2;  /* [B,8,16,16]。 */
	float d_fc_weight[16]; /* [2,8]，全连接权重梯度。 */
	float d_fc_bias[2];    /* [2]，全连接偏置梯度。 */
} CnnHeadBackwardWorkspace;

/*
 * workspace 必须为空（初始 {0}）；batch 必须能用于固定网络前向工作区。
 * 先检查全部元素数、字节数，再分配三个数组；失败时 workspace 保持原样。
 */
gdk_export CnnOpsResult cnn_head_backward_workspace_init(CnnHeadBackwardWorkspace *workspace, size_t batch);

/* 允许 NULL 以及重复调用；释放三个数组并将整个对象清零。 */
gdk_export void cnn_head_backward_workspace_destroy(CnnHeadBackwardWorkspace *workspace);

/*
 * 只执行分类末端的 CE -> Linear -> GAP 反向，不是完整 CNN 反向。
 * forward 必须 ready=true，且与 backward 的 batch 相同；读取其中的
 * gap 和 logits。labels 为 [B]，仅允许 0/1。parameters 只使用 fc_weight
 * 和 fc_bias，元素数须分别为 16、2；无关卷积参数视图不参与检查。
 * 上述输入及参数均必须为有限值，并与 backward 的内存互不重叠。
 *
 * 调用者必须保证 forward 对应相同参数，前向后未更新模型、未被其他
 * batch 覆盖；本函数只读 forward、parameters 和 labels，不分配内存。
 * CE 使用 batch 平均损失，其梯度只在 CE 处除以 batch 一次。
 *
 * 非空 backward 每次调用开始先置 ready=false。失败时可能已有部分
 * 梯度，不能使用；仍可重新计算或销毁。完整成功后保存 mean_loss 并置
 * ready=true。暂不计算卷积参数梯度，不更新参数、优化器或持久化记录。
 */
gdk_export CnnOpsResult cnn_fixed_head_backward(const CnnFixedParameters *parameters, const CnnForwardWorkspace *forward, const int32_t *labels, size_t label_count, CnnHeadBackwardWorkspace *backward);

/*
 * 固定 CNN 的完整反向工作区，参数梯度与模型参数分别保存。
 * d_activation 容量为 [B,4,64,64]，先使用前 B*8*32*32 个元素处理
 * Pool2/ReLU2 的梯度，Conv2 完成后复用全部容量处理 Pool1/ReLU1。
 * d_pool1 独立保存 Conv2 的输入梯度；FC 梯度及损失仍在 head 中。
 * 调用者先用 {0} 初始化；除读取结果外，不得手动修改字段。
 */
typedef struct {
	size_t batch;
	bool ready;
	CnnHeadBackwardWorkspace head;
	float *d_activation; /* 容量 [B,4,64,64]，复用的激活层梯度。 */
	float *d_pool1;      /* [B,4,32,32]。 */
	float d_conv1_weight[108]; /* [4,3,3,3]。 */
	float d_conv1_bias[4];     /* [4]。 */
	float d_conv2_weight[288]; /* [8,4,3,3]。 */
	float d_conv2_bias[8];     /* [8]。 */
} CnnBackwardWorkspace;

/* 空工作区按固定 batch size 分配；任何检查或分配失败时原对象不变。 */
gdk_export CnnOpsResult cnn_backward_workspace_init(CnnBackwardWorkspace *workspace,
						   size_t batch);

/* 释放嵌套 head 和两个临时数组，整体清零；允许 NULL 及重复调用。 */
gdk_export void cnn_backward_workspace_destroy(CnnBackwardWorkspace *workspace);

/*
 * 完整执行 CE -> FC -> GAP -> Pool2 -> ReLU2 -> Conv2 -> Pool1 -> ReLU1
 * -> Conv1 反向，计算全部六个参数张量的梯度，不更新模型。
 * input 为原前向的 [B,3,64,64]；全部参数、标签及 forward 缓存必须对应
 * 同一次前向，期间不得更新模型或用其他 batch 覆盖缓存。数量及状态
 * 检查不能代替该对应关系，由调用者保证输入、参数和缓存的生命周期。
 *
 * forward 必须 ready=true；各工作区 batch 相同，输入及全部缓存、参数
 * 必须为有限值，ReLU 缓存不能为负。label_count=B，标签只允许 0/1。
 * d_input 可传 NULL、0，仅计算参数梯度；启用时必须有 B*3*64*64 个 float。
 * 反向工作区不得与任何输入、参数、标签或 forward 缓存重叠；可选的
 * d_input 也须独立于上述全部输入及反向工作区中的数组。
 *
 * 本函数不分配内存，所有梯度均覆写，batch 平均只在 CE 处执行一次。
 * 每次调用先使 ready 和 head.ready 失效；错误后中间梯度、损失及可选
 * d_input 可能已改写，均不可使用。成功后二者均为 true；参数梯度分布于
 * 本工作区和 head，不进入持久化模型，不修改输入、前向缓存或参数。
 */
gdk_export CnnOpsResult cnn_fixed_backward(const CnnFixedParameters *parameters, const float *input, size_t input_count, const CnnForwardWorkspace *forward, const int32_t *labels, size_t label_count, CnnBackwardWorkspace *backward, float *d_input, size_t d_input_count);

#endif /* GDK_CNN_BACKWARD_H */
