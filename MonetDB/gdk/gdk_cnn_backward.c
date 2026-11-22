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

typedef struct {
	CnnHeadBackwardSizes head;
	size_t input;
	size_t activation;
	size_t pool1;
	size_t relu2;
} CnnBackwardSizes;

static CnnOpsResult
cnn_backward_sizes(size_t batch, CnnBackwardSizes *sizes)
{
	CnnOpsResult result = cnn_head_backward_sizes(batch, &sizes->head);
	if (result != CNN_OPS_OK)
		return result;
	if (!cnn_head_backward_count(batch, 3 * 64 * 64, &sizes->input) ||
	    !cnn_head_backward_count(batch, 4 * 64 * 64, &sizes->activation) ||
	    !cnn_head_backward_count(batch, 4 * 32 * 32, &sizes->pool1) ||
	    !cnn_head_backward_count(batch, 8 * 32 * 32, &sizes->relu2))
		return CNN_OPS_SIZE_OVERFLOW;
	return CNN_OPS_OK;
}

static bool
cnn_backward_zero(const float *values, size_t count)
{
	for (size_t i = 0; i < count; i++) {
		if (values[i] != 0.0f)
			return false;
	}
	return true;
}

static bool
cnn_backward_empty(const CnnBackwardWorkspace *workspace)
{
	return workspace->batch == 0 && !workspace->ready &&
	       cnn_head_backward_empty(&workspace->head) &&
	       workspace->d_activation == NULL && workspace->d_pool1 == NULL &&
	       cnn_backward_zero(workspace->d_conv1_weight, 108) &&
	       cnn_backward_zero(workspace->d_conv1_bias, 4) &&
	       cnn_backward_zero(workspace->d_conv2_weight, 288) &&
	       cnn_backward_zero(workspace->d_conv2_bias, 8);
}

static bool
cnn_backward_activation_valid(const float *values, size_t count)
{
	for (size_t i = 0; i < count; i++) {
		if (!isfinite(values[i]) || values[i] < 0.0f)
			return false;
	}
	return true;
}

void
cnn_backward_workspace_destroy(CnnBackwardWorkspace *workspace)
{
	if (workspace == NULL)
		return;
	cnn_head_backward_workspace_destroy(&workspace->head);
	GDKfree(workspace->d_activation);
	GDKfree(workspace->d_pool1);
	*workspace = (CnnBackwardWorkspace) {0};
}

CnnOpsResult
cnn_backward_workspace_init(CnnBackwardWorkspace *workspace, size_t batch)
{
	if (workspace == NULL || !cnn_backward_empty(workspace))
		return CNN_OPS_INVALID_INPUT;
	CnnBackwardSizes sizes;
	CnnOpsResult result = cnn_backward_sizes(batch, &sizes);
	if (result != CNN_OPS_OK)
		return result;

	CnnBackwardWorkspace temporary = {.batch = batch};
	result = cnn_head_backward_workspace_init(&temporary.head, batch);
	if (result != CNN_OPS_OK)
		goto allocation_failed;
	temporary.d_activation = GDKmalloc(sizes.activation * sizeof(float));
	if (temporary.d_activation == NULL) {
		result = CNN_OPS_OUT_OF_MEMORY;
		goto allocation_failed;
	}
	temporary.d_pool1 = GDKmalloc(sizes.pool1 * sizeof(float));
	if (temporary.d_pool1 == NULL) {
		result = CNN_OPS_OUT_OF_MEMORY;
		goto allocation_failed;
	}
	*workspace = temporary;
	return CNN_OPS_OK;

allocation_failed:
	cnn_backward_workspace_destroy(&temporary);
	return result;
}

CnnOpsResult
cnn_fixed_backward(const CnnFixedParameters *parameters,
		   const float *input, size_t input_count,
		   const CnnForwardWorkspace *forward,
		   const int32_t *labels, size_t label_count,
		   CnnBackwardWorkspace *backward,
		   float *d_input, size_t d_input_count)
{
	if (backward == NULL)
		return CNN_OPS_INVALID_INPUT;
	backward->ready = false;
	backward->head.ready = false;
	if (parameters == NULL || input == NULL || forward == NULL || !forward->ready || labels == NULL ||
	    forward->batch != backward->batch || backward->head.batch != backward->batch ||
	    label_count != backward->batch || backward->d_activation == NULL || backward->d_pool1 == NULL ||
	    backward->head.d_logits == NULL || backward->head.d_gap == NULL || backward->head.d_pool2 == NULL ||
	    forward->relu1 == NULL || forward->pool1 == NULL || forward->relu2 == NULL || forward->pool2 == NULL ||
	    forward->gap == NULL || forward->logits == NULL ||
	    forward->pool1_argmax == NULL || forward->pool2_argmax == NULL ||
	    (d_input == NULL && d_input_count != 0))
		return CNN_OPS_INVALID_INPUT;
	CnnBackwardSizes sizes;
	CnnOpsResult result = cnn_backward_sizes(backward->batch, &sizes);
	if (result != CNN_OPS_OK)
		return result;
	if (input_count != sizes.input || (d_input != NULL && d_input_count != sizes.input))
		return CNN_OPS_INVALID_INPUT;

	const CnnTensorView *views[] = {
		&parameters->conv1_weight, &parameters->conv1_bias,
		&parameters->conv2_weight, &parameters->conv2_bias,
		&parameters->fc_weight, &parameters->fc_bias
	};
	const size_t counts[] = {108, 4, 288, 8, 16, 2};
	for (size_t i = 0; i < sizeof(views) / sizeof(views[0]); i++) {
		if (views[i]->values == NULL || views[i]->count != counts[i])
			return CNN_OPS_INVALID_INPUT;
	}
	for (size_t n = 0; n < label_count; n++) {
		if (labels[n] != 0 && labels[n] != 1)
			return CNN_OPS_INVALID_INPUT;
	}
	if (!cnn_head_backward_finite(input, input_count) ||
	    !cnn_backward_activation_valid(forward->relu1, sizes.activation) ||
	    !cnn_backward_activation_valid(forward->relu2, sizes.relu2) ||
	    !cnn_head_backward_finite(forward->pool1, sizes.pool1) ||
	    !cnn_head_backward_finite(forward->pool2, sizes.head.pool2) ||
	    !cnn_head_backward_finite(forward->gap, sizes.head.gap) ||
	    !cnn_head_backward_finite(forward->logits, sizes.head.logits))
		return CNN_OPS_INVALID_INPUT;
	for (size_t i = 0; i < sizeof(views) / sizeof(views[0]); i++) {
		if (!cnn_head_backward_finite(views[i]->values, views[i]->count))
			return CNN_OPS_INVALID_INPUT;
	}

	result = cnn_fixed_head_backward(parameters, forward, labels, label_count, &backward->head);
	if (result != CNN_OPS_OK)
		goto computation_failed;

	const size_t batch = backward->batch;
	const CnnPool2dShape pool2 = {batch, 8, 32, 32};
	result = cnn_maxpool2d_backward(&pool2, backward->head.d_pool2, sizes.head.pool2,
		forward->pool2_argmax, sizes.head.pool2, backward->d_activation, sizes.relu2);
	if (result != CNN_OPS_OK)
		goto computation_failed;
	if (!cnn_head_backward_finite(backward->d_activation, sizes.relu2)) {
		result = CNN_OPS_INVALID_INPUT;
		goto computation_failed;
	}
	result = cnn_relu_backward(forward->relu2, sizes.relu2,
		backward->d_activation, sizes.relu2, backward->d_activation, sizes.relu2);
	if (result != CNN_OPS_OK)
		goto computation_failed;
	if (!cnn_head_backward_finite(backward->d_activation, sizes.relu2)) {
		result = CNN_OPS_INVALID_INPUT;
		goto computation_failed;
	}

	const CnnConv2dShape conv2 = {batch, 4, 8, 32, 32};
	result = cnn_conv2d_backward(&conv2, forward->pool1, sizes.pool1,
		parameters->conv2_weight.values, parameters->conv2_weight.count,
		backward->d_activation, sizes.relu2, backward->d_pool1, sizes.pool1,
		backward->d_conv2_weight, 288, backward->d_conv2_bias, 8);
	if (result != CNN_OPS_OK)
		goto computation_failed;
	if (!cnn_head_backward_finite(backward->d_pool1, sizes.pool1) ||
	    !cnn_head_backward_finite(backward->d_conv2_weight, 288) ||
	    !cnn_head_backward_finite(backward->d_conv2_bias, 8)) {
		result = CNN_OPS_INVALID_INPUT;
		goto computation_failed;
	}

	/* Conv2 已消耗第二层的激活梯度，接下来复用同一大数组。 */
	const CnnPool2dShape pool1 = {batch, 4, 64, 64};
	result = cnn_maxpool2d_backward(&pool1, backward->d_pool1, sizes.pool1,
		forward->pool1_argmax, sizes.pool1, backward->d_activation, sizes.activation);
	if (result != CNN_OPS_OK)
		goto computation_failed;
	if (!cnn_head_backward_finite(backward->d_activation, sizes.activation)) {
		result = CNN_OPS_INVALID_INPUT;
		goto computation_failed;
	}
	result = cnn_relu_backward(forward->relu1, sizes.activation,
		backward->d_activation, sizes.activation, backward->d_activation, sizes.activation);
	if (result != CNN_OPS_OK)
		goto computation_failed;
	if (!cnn_head_backward_finite(backward->d_activation, sizes.activation)) {
		result = CNN_OPS_INVALID_INPUT;
		goto computation_failed;
	}

	const CnnConv2dShape conv1 = {batch, 3, 4, 64, 64};
	result = cnn_conv2d_backward(&conv1, input, input_count,
		parameters->conv1_weight.values, parameters->conv1_weight.count,
		backward->d_activation, sizes.activation, d_input, d_input_count,
		backward->d_conv1_weight, 108, backward->d_conv1_bias, 4);
	if (result != CNN_OPS_OK)
		goto computation_failed;
	if (!cnn_head_backward_finite(backward->d_conv1_weight, 108) ||
	    !cnn_head_backward_finite(backward->d_conv1_bias, 4) ||
	    (d_input != NULL && !cnn_head_backward_finite(d_input, d_input_count))) {
		result = CNN_OPS_INVALID_INPUT;
		goto computation_failed;
	}
	backward->ready = true;
	backward->head.ready = true;
	return CNN_OPS_OK;

computation_failed:
	backward->ready = false;
	backward->head.ready = false;
	return result;
}
