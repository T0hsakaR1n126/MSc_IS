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
#include "gdk_cnn_backward.h"

#include <math.h>
#include <stdint.h>

typedef struct {
	size_t logits;
	size_t gap;
	size_t pool2;
} CnnHeadBackwardSizes;

static bool
cnn_head_backward_count(size_t batch, size_t per_image, size_t *count)
{
	if (batch > SIZE_MAX / per_image)
		return false;
	*count = batch * per_image;
	return *count <= SIZE_MAX / sizeof(float);
}

/* 末端梯度必须能与同样 batch 的固定网络前向工作区配套使用。 */
static CnnOpsResult
cnn_head_backward_sizes(size_t batch, CnnHeadBackwardSizes *sizes)
{
	if (batch == 0)
		return CNN_OPS_INVALID_INPUT;
	if (!cnn_head_backward_count(batch, 2, &sizes->logits) ||
	    !cnn_head_backward_count(batch, 8, &sizes->gap) ||
	    !cnn_head_backward_count(batch, 8 * 16 * 16, &sizes->pool2) ||
	    batch > SIZE_MAX / (4 * 64 * 64 * sizeof(float)) ||
	    batch > SIZE_MAX / (4 * 32 * 32 * sizeof(size_t)))
		return CNN_OPS_SIZE_OVERFLOW;
	return CNN_OPS_OK;
}

static bool
cnn_head_backward_empty(const CnnHeadBackwardWorkspace *workspace)
{
	if (workspace->batch != 0 || workspace->ready || workspace->mean_loss != 0.0 ||
	    workspace->d_logits != NULL || workspace->d_gap != NULL || workspace->d_pool2 != NULL)
		return false;
	for (size_t i = 0; i < 16; i++) {
		if (workspace->d_fc_weight[i] != 0.0f)
			return false;
	}
	for (size_t i = 0; i < 2; i++) {
		if (workspace->d_fc_bias[i] != 0.0f)
			return false;
	}
	return true;
}

static bool
cnn_head_backward_finite(const float *values, size_t count)
{
	for (size_t i = 0; i < count; i++) {
		if (!isfinite(values[i]))
			return false;
	}
	return true;
}

void
cnn_head_backward_workspace_destroy(CnnHeadBackwardWorkspace *workspace)
{
	if (workspace == NULL)
		return;
	GDKfree(workspace->d_logits);
	GDKfree(workspace->d_gap);
	GDKfree(workspace->d_pool2);
	*workspace = (CnnHeadBackwardWorkspace) {0};
}

CnnOpsResult
cnn_head_backward_workspace_init(CnnHeadBackwardWorkspace *workspace, size_t batch)
{
	if (workspace == NULL || !cnn_head_backward_empty(workspace))
		return CNN_OPS_INVALID_INPUT;
	CnnHeadBackwardSizes sizes;
	CnnOpsResult result = cnn_head_backward_sizes(batch, &sizes);
	if (result != CNN_OPS_OK)
		return result;

	CnnHeadBackwardWorkspace temporary = {.batch = batch};
	temporary.d_logits = GDKmalloc(sizes.logits * sizeof(float));
	if (temporary.d_logits == NULL)
		goto allocation_failed;
	temporary.d_gap = GDKmalloc(sizes.gap * sizeof(float));
	if (temporary.d_gap == NULL)
		goto allocation_failed;
	temporary.d_pool2 = GDKmalloc(sizes.pool2 * sizeof(float));
	if (temporary.d_pool2 == NULL)
		goto allocation_failed;
	*workspace = temporary;
	return CNN_OPS_OK;

allocation_failed:
	cnn_head_backward_workspace_destroy(&temporary);
	return CNN_OPS_OUT_OF_MEMORY;
}

CnnOpsResult
cnn_fixed_head_backward(const CnnFixedParameters *parameters,
			const CnnForwardWorkspace *forward,
			const int32_t *labels, size_t label_count,
			CnnHeadBackwardWorkspace *backward)
{
	if (backward == NULL)
		return CNN_OPS_INVALID_INPUT;
	backward->ready = false;
	if (parameters == NULL || forward == NULL || !forward->ready || labels == NULL ||
	    forward->batch != backward->batch || label_count != backward->batch ||
	    forward->gap == NULL || forward->logits == NULL ||
	    backward->d_logits == NULL || backward->d_gap == NULL || backward->d_pool2 == NULL)
		return CNN_OPS_INVALID_INPUT;
	CnnHeadBackwardSizes sizes;
	CnnOpsResult result = cnn_head_backward_sizes(backward->batch, &sizes);
	if (result != CNN_OPS_OK)
		return result;
	if (parameters->fc_weight.values == NULL || parameters->fc_weight.count != 16 ||
	    parameters->fc_bias.values == NULL || parameters->fc_bias.count != 2)
		return CNN_OPS_INVALID_INPUT;
	for (size_t n = 0; n < label_count; n++) {
		if (labels[n] != 0 && labels[n] != 1)
			return CNN_OPS_INVALID_INPUT;
	}
	if (!cnn_head_backward_finite(forward->gap, sizes.gap) ||
	    !cnn_head_backward_finite(forward->logits, sizes.logits) ||
	    !cnn_head_backward_finite(parameters->fc_weight.values, 16) ||
	    !cnn_head_backward_finite(parameters->fc_bias.values, 2))
		return CNN_OPS_INVALID_INPUT;

	/* 损失暂存局部变量，只有整个末端反向成功才发布给调用者。 */
	double mean_loss;
	result = cnn_softmax_cross_entropy(forward->logits, sizes.logits,
		labels, label_count, backward->batch, 2, NULL, 0, &mean_loss);
	if (result != CNN_OPS_OK)
		return result;
	if (!isfinite(mean_loss))
		return CNN_OPS_INVALID_INPUT;
	result = cnn_softmax_cross_entropy_backward(forward->logits, sizes.logits,
		labels, label_count, backward->batch, 2, backward->d_logits, sizes.logits);
	if (result != CNN_OPS_OK)
		return result;
	if (!cnn_head_backward_finite(backward->d_logits, sizes.logits))
		return CNN_OPS_INVALID_INPUT;

	const CnnLinearShape linear = {backward->batch, 8, 2};
	result = cnn_linear_backward(&linear, forward->gap, sizes.gap,
		parameters->fc_weight.values, parameters->fc_weight.count,
		backward->d_logits, sizes.logits, backward->d_gap, sizes.gap,
		backward->d_fc_weight, 16, backward->d_fc_bias, 2);
	if (result != CNN_OPS_OK)
		return result;
	if (!cnn_head_backward_finite(backward->d_gap, sizes.gap) ||
	    !cnn_head_backward_finite(backward->d_fc_weight, 16) ||
	    !cnn_head_backward_finite(backward->d_fc_bias, 2))
		return CNN_OPS_INVALID_INPUT;

	const CnnGapShape gap = {backward->batch, 8, 16, 16};
	result = cnn_global_average_pool_backward(&gap, backward->d_gap, sizes.gap,
		backward->d_pool2, sizes.pool2);
	if (result != CNN_OPS_OK)
		return result;
	if (!cnn_head_backward_finite(backward->d_pool2, sizes.pool2))
		return CNN_OPS_INVALID_INPUT;
	backward->mean_loss = mean_loss;
	backward->ready = true;
	return CNN_OPS_OK;
}
