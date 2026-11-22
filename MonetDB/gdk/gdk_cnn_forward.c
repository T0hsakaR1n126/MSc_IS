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
#include "gdk_cnn_forward.h"

#include <math.h>
#include <stdint.h>

typedef struct {
	size_t input;
	size_t relu1;
	size_t pool1;
	size_t relu2;
	size_t pool2;
	size_t gap;
	size_t logits;
} CnnForwardSizes;

static bool
cnn_forward_count(size_t batch, size_t per_image, size_t *count)
{
	if (batch > SIZE_MAX / per_image)
		return false;
	*count = batch * per_image;
	return *count <= SIZE_MAX / sizeof(float);
}

/* 初始化和执行共用同一尺寸计算，所有乘法先于分配及数组访问检查。 */
static CnnOpsResult
cnn_forward_sizes(size_t batch, CnnForwardSizes *sizes)
{
	if (batch == 0)
		return CNN_OPS_INVALID_INPUT;
	if (!cnn_forward_count(batch, 3 * 64 * 64, &sizes->input) ||
	    !cnn_forward_count(batch, 4 * 64 * 64, &sizes->relu1) ||
	    !cnn_forward_count(batch, 4 * 32 * 32, &sizes->pool1) ||
	    !cnn_forward_count(batch, 8 * 32 * 32, &sizes->relu2) ||
	    !cnn_forward_count(batch, 8 * 16 * 16, &sizes->pool2) ||
	    !cnn_forward_count(batch, 8, &sizes->gap) ||
	    !cnn_forward_count(batch, 2, &sizes->logits) ||
	    sizes->pool1 > SIZE_MAX / sizeof(size_t) ||
	    sizes->pool2 > SIZE_MAX / sizeof(size_t))
		return CNN_OPS_SIZE_OVERFLOW;
	return CNN_OPS_OK;
}

static bool
cnn_forward_empty(const CnnForwardWorkspace *workspace)
{
	return workspace->batch == 0 && !workspace->ready &&
	       workspace->relu1 == NULL && workspace->pool1 == NULL &&
	       workspace->relu2 == NULL && workspace->pool2 == NULL &&
	       workspace->gap == NULL && workspace->logits == NULL &&
	       workspace->pool1_argmax == NULL && workspace->pool2_argmax == NULL;
}

static bool
cnn_forward_finite(const float *values, size_t count)
{
	for (size_t i = 0; i < count; i++) {
		if (!isfinite(values[i]))
			return false;
	}
	return true;
}

void
cnn_forward_workspace_destroy(CnnForwardWorkspace *workspace)
{
	if (workspace == NULL)
		return;
	GDKfree(workspace->relu1);
	GDKfree(workspace->pool1);
	GDKfree(workspace->relu2);
	GDKfree(workspace->pool2);
	GDKfree(workspace->gap);
	GDKfree(workspace->logits);
	GDKfree(workspace->pool1_argmax);
	GDKfree(workspace->pool2_argmax);
	*workspace = (CnnForwardWorkspace) {0};
}

CnnOpsResult
cnn_forward_workspace_init(CnnForwardWorkspace *workspace, size_t batch)
{
	if (workspace == NULL || !cnn_forward_empty(workspace))
		return CNN_OPS_INVALID_INPUT;
	CnnForwardSizes sizes;
	CnnOpsResult result = cnn_forward_sizes(batch, &sizes);
	if (result != CNN_OPS_OK)
		return result;

	CnnForwardWorkspace temporary = {.batch = batch};
	temporary.relu1 = GDKmalloc(sizes.relu1 * sizeof(float));
	if (temporary.relu1 == NULL)
		goto allocation_failed;
	temporary.pool1 = GDKmalloc(sizes.pool1 * sizeof(float));
	if (temporary.pool1 == NULL)
		goto allocation_failed;
	temporary.relu2 = GDKmalloc(sizes.relu2 * sizeof(float));
	if (temporary.relu2 == NULL)
		goto allocation_failed;
	temporary.pool2 = GDKmalloc(sizes.pool2 * sizeof(float));
	if (temporary.pool2 == NULL)
		goto allocation_failed;
	temporary.gap = GDKmalloc(sizes.gap * sizeof(float));
	if (temporary.gap == NULL)
		goto allocation_failed;
	temporary.logits = GDKmalloc(sizes.logits * sizeof(float));
	if (temporary.logits == NULL)
		goto allocation_failed;
	temporary.pool1_argmax = GDKmalloc(sizes.pool1 * sizeof(size_t));
	if (temporary.pool1_argmax == NULL)
		goto allocation_failed;
	temporary.pool2_argmax = GDKmalloc(sizes.pool2 * sizeof(size_t));
	if (temporary.pool2_argmax == NULL)
		goto allocation_failed;

	*workspace = temporary;
	return CNN_OPS_OK;

allocation_failed:
	cnn_forward_workspace_destroy(&temporary);
	return CNN_OPS_OUT_OF_MEMORY;
}

CnnOpsResult
cnn_fixed_forward(const CnnFixedParameters *parameters,
		  const float *input, size_t input_count,
		  CnnForwardWorkspace *workspace)
{
	if (workspace == NULL)
		return CNN_OPS_INVALID_INPUT;
	workspace->ready = false;
	if (parameters == NULL || input == NULL ||
	    workspace->relu1 == NULL || workspace->pool1 == NULL ||
	    workspace->relu2 == NULL || workspace->pool2 == NULL ||
	    workspace->gap == NULL || workspace->logits == NULL ||
	    workspace->pool1_argmax == NULL || workspace->pool2_argmax == NULL)
		return CNN_OPS_INVALID_INPUT;
	CnnForwardSizes sizes;
	CnnOpsResult result = cnn_forward_sizes(workspace->batch, &sizes);
	if (result != CNN_OPS_OK)
		return result;
	if (input_count != sizes.input)
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
	if (!cnn_forward_finite(input, input_count))
		return CNN_OPS_INVALID_INPUT;
	for (size_t i = 0; i < sizeof(views) / sizeof(views[0]); i++) {
		if (!cnn_forward_finite(views[i]->values, views[i]->count))
			return CNN_OPS_INVALID_INPUT;
	}

	const size_t batch = workspace->batch;
	const CnnConv2dShape conv1 = {batch, 3, 4, 64, 64};
	const CnnConv2dShape conv2 = {batch, 4, 8, 32, 32};
	const CnnPool2dShape pool1 = {batch, 4, 64, 64};
	const CnnPool2dShape pool2 = {batch, 8, 32, 32};
	const CnnGapShape gap = {batch, 8, 16, 16};
	const CnnLinearShape linear = {batch, 8, 2};

	result = cnn_conv2d_forward(&conv1, input, input_count,
		parameters->conv1_weight.values, parameters->conv1_weight.count,
		parameters->conv1_bias.values, parameters->conv1_bias.count,
		workspace->relu1, sizes.relu1);
	if (result != CNN_OPS_OK)
		return result;
	if (!cnn_forward_finite(workspace->relu1, sizes.relu1))
		return CNN_OPS_INVALID_INPUT;
	result = cnn_relu_forward(workspace->relu1, sizes.relu1,
		workspace->relu1, sizes.relu1);
	if (result != CNN_OPS_OK)
		return result;
	result = cnn_maxpool2d_forward(&pool1, workspace->relu1, sizes.relu1,
		workspace->pool1, sizes.pool1, workspace->pool1_argmax, sizes.pool1);
	if (result != CNN_OPS_OK)
		return result;

	result = cnn_conv2d_forward(&conv2, workspace->pool1, sizes.pool1,
		parameters->conv2_weight.values, parameters->conv2_weight.count,
		parameters->conv2_bias.values, parameters->conv2_bias.count,
		workspace->relu2, sizes.relu2);
	if (result != CNN_OPS_OK)
		return result;
	if (!cnn_forward_finite(workspace->relu2, sizes.relu2))
		return CNN_OPS_INVALID_INPUT;
	result = cnn_relu_forward(workspace->relu2, sizes.relu2,
		workspace->relu2, sizes.relu2);
	if (result != CNN_OPS_OK)
		return result;
	result = cnn_maxpool2d_forward(&pool2, workspace->relu2, sizes.relu2,
		workspace->pool2, sizes.pool2, workspace->pool2_argmax, sizes.pool2);
	if (result != CNN_OPS_OK)
		return result;

	result = cnn_global_average_pool_forward(&gap, workspace->pool2, sizes.pool2,
		workspace->gap, sizes.gap);
	if (result != CNN_OPS_OK)
		return result;
	result = cnn_linear_forward(&linear, workspace->gap, sizes.gap,
		parameters->fc_weight.values, parameters->fc_weight.count,
		parameters->fc_bias.values, parameters->fc_bias.count,
		workspace->logits, sizes.logits);
	if (result != CNN_OPS_OK)
		return result;
	if (!cnn_forward_finite(workspace->logits, sizes.logits))
		return CNN_OPS_INVALID_INPUT;
	workspace->ready = true;
	return CNN_OPS_OK;
}
