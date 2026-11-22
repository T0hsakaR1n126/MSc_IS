/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#include "monetdb_config.h"
#include "sql_cnn_optimizer.h"

#include <stdint.h>
#include <string.h>

CnnOpsResult
cnn_model_sgd_step(CnnModel *model, CnnForwardWorkspace *forward,
	CnnBackwardWorkspace *backward, double learning_rate)
{
	if (model == NULL || forward == NULL || backward == NULL ||
	    !forward->ready || !backward->ready || !backward->head.ready ||
	    forward->batch == 0 || forward->batch != backward->batch ||
	    backward->batch != backward->head.batch)
		return CNN_OPS_INVALID_INPUT;
	if (model->step >= (uint64_t) GDK_lng_max)
		return CNN_OPS_SIZE_OVERFLOW;

	/* 六组梯度在工作区中分开保存，布局含义由 SQL 模型模块解释。 */
	const float *sources[CNN_TENSOR_COUNT] = {
		backward->d_conv1_weight, backward->d_conv1_bias,
		backward->d_conv2_weight, backward->d_conv2_bias,
		backward->head.d_fc_weight, backward->head.d_fc_bias
	};
	float gradients[CNN_PARAMETER_COUNT];
	for (size_t i = 0; i < CNN_TENSOR_COUNT; i++) {
		const CnnTensorSpec *spec = &cnn_tensor_specs[i];
		memcpy(gradients + spec->offset, sources[i], spec->count * sizeof(float));
	}
	CnnOpsResult result = cnn_sgd_step(model->parameter, CNN_PARAMETER_COUNT,
		gradients, CNN_PARAMETER_COUNT, learning_rate);
	if (result != CNN_OPS_OK)
		return result;

	/* 只有整组参数成功更新，才能推进进度并消费本次计算结果。 */
	model->step++;
	forward->ready = false;
	backward->ready = false;
	backward->head.ready = false;
	return CNN_OPS_OK;
}
