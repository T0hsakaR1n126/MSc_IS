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

#endif /* GDK_CNN_BACKWARD_H */
