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
#include "gdk_cnn_optimizer.h"

#include <float.h>
#include <math.h>
#include <stdint.h>

CnnOpsResult
cnn_sgd_step(float *parameters, size_t parameter_count,
	const float *gradients, size_t gradient_count, double learning_rate)
{
	if (parameters == NULL || gradients == NULL || parameter_count == 0 ||
	    !isfinite(learning_rate) || learning_rate <= 0.0)
		return CNN_OPS_INVALID_INPUT;
	if (parameter_count > SIZE_MAX / sizeof(float))
		return CNN_OPS_SIZE_OVERFLOW;
	if (gradient_count != parameter_count)
		return CNN_OPS_INVALID_INPUT;

	/* 只检查不写入，后面的坏梯度也不能留下部分更新。 */
	for (size_t i = 0; i < parameter_count; i++) {
		if (!isfinite(parameters[i]) || !isfinite(gradients[i]))
			return CNN_OPS_INVALID_INPUT;
		double candidate = (double) parameters[i] - learning_rate * (double) gradients[i];
		if (!isfinite(candidate) || candidate > (double) FLT_MAX || candidate < -(double) FLT_MAX)
			return CNN_OPS_INVALID_INPUT;
	}
	for (size_t i = 0; i < parameter_count; i++)
		parameters[i] = (float) ((double) parameters[i] - learning_rate * (double) gradients[i]);
	return CNN_OPS_OK;
}
