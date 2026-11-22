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
#include "sql_cnn_ops.h"

#include <stdbool.h>
#include <stdint.h>

/* 任何尺寸乘法都先检查，避免回绕后错误分配或越界访问。 */
static bool
cnn_ops_multiply(size_t left, size_t right, size_t *result)
{
	if (left != 0 && right > SIZE_MAX / left)
		return false;
	*result = left * right;
	return true;
}

CnnOpsResult
cnn_conv2d_forward(const CnnConv2dShape *shape, const float *input, size_t input_count, const float *weight, size_t weight_count, const float *bias, size_t bias_count, float *output, size_t output_count)
{
	if (shape == NULL || input == NULL || weight == NULL ||
	    bias == NULL || output == NULL)
		return CNN_OPS_INVALID_INPUT;
	if (shape->batch == 0 || shape->in_channels == 0 ||
	    shape->out_channels == 0 || shape->height == 0 || shape->width == 0)
		return CNN_OPS_INVALID_INPUT;

	size_t plane, input_planes, output_planes, kernel_planes;
	size_t expected_input, expected_output, expected_weight;
	if (!cnn_ops_multiply(shape->height, shape->width, &plane) ||
	    !cnn_ops_multiply(shape->batch, shape->in_channels, &input_planes) ||
	    !cnn_ops_multiply(input_planes, plane, &expected_input) ||
	    !cnn_ops_multiply(shape->batch, shape->out_channels, &output_planes) ||
	    !cnn_ops_multiply(output_planes, plane, &expected_output) ||
	    !cnn_ops_multiply(shape->out_channels, shape->in_channels, &kernel_planes) ||
	    !cnn_ops_multiply(kernel_planes, 9, &expected_weight))
		return CNN_OPS_SIZE_OVERFLOW;
	if (expected_input > SIZE_MAX / sizeof(float) ||
	    expected_output > SIZE_MAX / sizeof(float) ||
	    expected_weight > SIZE_MAX / sizeof(float) ||
	    shape->out_channels > SIZE_MAX / sizeof(float))
		return CNN_OPS_SIZE_OVERFLOW;
	if (input_count != expected_input || weight_count != expected_weight ||
	    bias_count != shape->out_channels || output_count != expected_output)
		return CNN_OPS_INVALID_INPUT;

	/* 先完成全部校验，下面才开始写输出。 */
	for (size_t n = 0; n < shape->batch; n++) {
		for (size_t oc = 0; oc < shape->out_channels; oc++) {
			size_t output_base = (n * shape->out_channels + oc) * plane;
			for (size_t h = 0; h < shape->height; h++) {
				for (size_t w = 0; w < shape->width; w++) {
					float sum = 0.0f;
					for (size_t ic = 0; ic < shape->in_channels; ic++) {
						size_t input_base = (n * shape->in_channels + ic) * plane;
						for (size_t kh = 0; kh < 3; kh++) {
							/* 跳过补零位置；先检查边界，避免 size_t 下溢。 */
							if ((kh == 0 && h == 0) ||
							    (kh == 2 && h == shape->height - 1))
								continue;
							size_t ih = h;
							if (kh == 0)
								ih--;
							else if (kh == 2)
								ih++;
							size_t weight_row = ((oc * shape->in_channels + ic) * 3 + kh) * 3;
							for (size_t kw = 0; kw < 3; kw++) {
								if ((kw == 0 && w == 0) ||
								    (kw == 2 && w == shape->width - 1))
									continue;
								size_t iw = w;
								if (kw == 0)
									iw--;
								else if (kw == 2)
									iw++;
								sum += input[input_base + ih * shape->width + iw] *
								       weight[weight_row + kw];
							}
						}
					}
					output[output_base + h * shape->width + w] = sum + bias[oc];
				}
			}
		}
	}
	return CNN_OPS_OK;
}
