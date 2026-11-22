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
#include "sql_cnn_forward.h"

#include <stdint.h>

static CnnTensorView
cnn_model_view(const CnnModel *model, size_t tensor_index)
{
	const CnnTensorSpec *spec = &cnn_tensor_specs[tensor_index];
	return (CnnTensorView) {
		.values = model->parameter + spec->offset,
		.count = spec->count
	};
}

CnnOpsResult
cnn_model_forward(const CnnModel *model, const CnnBatch *batch,
		  CnnForwardWorkspace *workspace)
{
	if (workspace == NULL)
		return CNN_OPS_INVALID_INPUT;
	workspace->ready = false;
	if (model == NULL || batch == NULL || batch->input == NULL || batch->count == 0)
		return CNN_OPS_INVALID_INPUT;
	if (batch->count > SIZE_MAX / CNN_IMAGE_VALUES)
		return CNN_OPS_SIZE_OVERFLOW;
	if (workspace->batch != batch->count)
		return CNN_OPS_INVALID_INPUT;

	/* 参数存储布局由 SQL 模型模块解释，GDK 只接收六个数组视图。 */
	const CnnFixedParameters parameters = {
		.conv1_weight = cnn_model_view(model, 0),
		.conv1_bias = cnn_model_view(model, 1),
		.conv2_weight = cnn_model_view(model, 2),
		.conv2_bias = cnn_model_view(model, 3),
		.fc_weight = cnn_model_view(model, 4),
		.fc_bias = cnn_model_view(model, 5)
	};
	return cnn_fixed_forward(&parameters, batch->input,
		batch->count * CNN_IMAGE_VALUES, workspace);
}
