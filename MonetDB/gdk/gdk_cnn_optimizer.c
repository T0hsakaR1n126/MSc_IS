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

typedef struct {
	double learning_rate;
	double beta1;
	double beta2;
	double epsilon;
	double first_correction;
	double second_correction;
} CnnAdamCoefficients;

typedef struct {
	double parameter;
	double first_moment;
	double second_moment;
} CnnAdamCandidate;

static double
cnn_adam_bias_correction(double beta, uint64_t step)
{
	if (beta == 0.0)
		return 1.0;
	/* 计算 1 - beta^step，避免 beta^step 接近 1 时相减丢失精度。 */
	return -expm1((double) step * log(beta));
}

static double
cnn_adam_delta(double rate, double corrected_m, double denominator)
{
	double direction = corrected_m / denominator;
	if (isnormal(direction) || corrected_m == 0.0)
		return rate * direction;

	/* 极端学习率或 epsilon 可能让中间比值溢出或下溢，但最终步长可表示。 */
	int rate_exponent, moment_exponent, denominator_exponent;
	double rate_fraction = frexp(rate, &rate_exponent);
	double moment_fraction = frexp(corrected_m, &moment_exponent);
	double denominator_fraction = frexp(denominator, &denominator_exponent);
	return scalbn((rate_fraction * moment_fraction) / denominator_fraction,
		rate_exponent + moment_exponent - denominator_exponent);
}

static CnnAdamCandidate
cnn_adam_candidate(const CnnAdamCoefficients *coefficients, double parameter,
	double gradient, double first_moment, double second_moment)
{
	double m = coefficients->beta1 * first_moment + (1.0 - coefficients->beta1) * gradient;
	double v = coefficients->beta2 * second_moment + (1.0 - coefficients->beta2) * (gradient * gradient);
	double corrected_m = m / coefficients->first_correction;
	double corrected_v = v / coefficients->second_correction;
	double denominator = sqrt(corrected_v) + coefficients->epsilon;
	double delta = cnn_adam_delta(coefficients->learning_rate, corrected_m, denominator);
	return (CnnAdamCandidate) {parameter - delta, m, v};
}

static bool
cnn_adam_fits_float(double value)
{
	return isfinite(value) && value >= -(double) FLT_MAX && value <= (double) FLT_MAX;
}

CnnOpsResult
cnn_adam_step(float *parameters, size_t parameter_count,
	const float *gradients, size_t gradient_count,
	float *first_moment, size_t first_count, float *second_moment, size_t second_count,
	uint64_t step, double learning_rate, double beta1, double beta2, double epsilon)
{
	if (parameters == NULL || gradients == NULL || first_moment == NULL ||
	    second_moment == NULL || parameter_count == 0 || step == 0 ||
	    !isfinite(learning_rate) || learning_rate <= 0.0 ||
	    !(beta1 >= 0.0 && beta1 < 1.0) || !(beta2 >= 0.0 && beta2 < 1.0) ||
	    !isfinite(epsilon) || epsilon <= 0.0)
		return CNN_OPS_INVALID_INPUT;
	if (parameter_count > SIZE_MAX / sizeof(float))
		return CNN_OPS_SIZE_OVERFLOW;
	if (gradient_count != parameter_count || first_count != parameter_count || second_count != parameter_count)
		return CNN_OPS_INVALID_INPUT;

	const CnnAdamCoefficients coefficients = {
		.learning_rate = learning_rate,
		.beta1 = beta1,
		.beta2 = beta2,
		.epsilon = epsilon,
		.first_correction = cnn_adam_bias_correction(beta1, step),
		.second_correction = cnn_adam_bias_correction(beta2, step)
	};

	/* 先验证整组更新，保证后部错误不会留下部分参数或部分矩。 */
	for (size_t i = 0; i < parameter_count; i++) {
		if (!isfinite(parameters[i]) || !isfinite(gradients[i]) ||
		    !isfinite(first_moment[i]) || !isfinite(second_moment[i]) || second_moment[i] < 0.0f)
			return CNN_OPS_INVALID_INPUT;
		CnnAdamCandidate candidate = cnn_adam_candidate(&coefficients, parameters[i],
			gradients[i], first_moment[i], second_moment[i]);
		if (!cnn_adam_fits_float(candidate.parameter) || !cnn_adam_fits_float(candidate.first_moment) ||
		    !cnn_adam_fits_float(candidate.second_moment))
			return CNN_OPS_INVALID_INPUT;
	}
	for (size_t i = 0; i < parameter_count; i++) {
		CnnAdamCandidate candidate = cnn_adam_candidate(&coefficients, parameters[i],
			gradients[i], first_moment[i], second_moment[i]);
		parameters[i] = (float) candidate.parameter;
		first_moment[i] = (float) candidate.first_moment;
		second_moment[i] = (float) candidate.second_moment;
	}
	return CNN_OPS_OK;
}
