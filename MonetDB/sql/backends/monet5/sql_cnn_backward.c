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
#include "sql_cnn_backward.h"

CnnOpsResult
cnn_model_head_backward(const CnnModel *model, const CnnBatch *batch,
	const CnnForwardWorkspace *forward, CnnHeadBackwardWorkspace *backward)
{
	if (backward == NULL)
		return CNN_OPS_INVALID_INPUT;
	backward->ready = false;
	if (model == NULL || batch == NULL || forward == NULL ||
	    batch->labels == NULL || batch->count == 0 ||
	    batch->count != forward->batch || batch->count != backward->batch)
		return CNN_OPS_INVALID_INPUT;

	const CnnTensorSpec *weight = &cnn_tensor_specs[4];
	const CnnTensorSpec *bias = &cnn_tensor_specs[5];
	/* 这段反向只需要分类层参数；不访问或计算卷积层梯度。 */
	const CnnFixedParameters parameters = {
		.fc_weight = {
			.values = model->parameter + weight->offset,
			.count = weight->count
		},
		.fc_bias = {
			.values = model->parameter + bias->offset,
			.count = bias->count
		}
	};
	return cnn_fixed_head_backward(&parameters, forward, batch->labels,
		batch->count, backward);
}
