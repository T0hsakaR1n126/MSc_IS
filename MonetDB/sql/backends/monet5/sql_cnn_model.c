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
#include "sql_cnn_model.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

/* C99 编译期检查；目标平台采用 IEEE-754 binary32 浮点表示。 */
typedef char cnn_binary32_required[
	(CHAR_BIT == 8 && sizeof(float) == 4 && FLT_RADIX == 2 &&
	 FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128) ? 1 : -1
];

//每行描述一个完整张量；排列顺序同时决定初始化随机数的使用顺序。
const CnnTensorSpec cnn_tensor_specs[CNN_TENSOR_COUNT] = {
	{"conv1", "weight", "weight", "[4,3,3,3]", "OIHW", 0, 108},
	{"conv1", "bias",   "bias",   "[4]",       "O", 108, 4},
	{"conv2", "weight", "weight", "[8,4,3,3]", "OIHW", 112, 288},
	{"conv2", "bias",   "bias",   "[8]",       "O", 400, 8},
	{"fc",    "weight", "weight", "[2,8]",     "OI", 408, 16},
	{"fc",    "bias",   "bias",   "[2]",       "O", 424, 2}
};

/* SplitMix64：状态仅属于当前初始化过程，不使用全局 rand() 状态。 */
static uint64_t
cnn_random_u64(uint64_t *state)
{
	uint64_t z;

	*state += UINT64_C(0x9e3779b97f4a7c15);
	z = *state;
	z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
	return z ^ (z >> 31);
}

//取 24 位整数并除以 2^24，得到 [0, 1) 范围内的 float。
static float
cnn_random_unit(uint64_t *state)
{
	uint32_t value = (uint32_t) (cnn_random_u64(state) >> 40);

	return (float) value * (1.0f / 16777216.0f);
}

static void
cnn_fill_uniform(CnnModel *model, size_t tensor_index,
		 float limit, uint64_t *rng)
{
	const CnnTensorSpec *spec = &cnn_tensor_specs[tensor_index];
	float *value = model->parameter + spec->offset;

	for (size_t i = 0; i < spec->count; i++) {
		float u = cnn_random_unit(rng);
		value[i] = (2.0f * u - 1.0f) * limit;
	}
}

void
cnn_model_init(CnnModel *model, uint64_t seed)
{
	//全部参数初始置零，在目标浮点表示下，全零位模式对应 0.0f
	memset(model, 0, sizeof(*model));
	model->seed = seed;
	uint64_t rng = seed;

	//调整三个W的初始值
	//Conv1：He uniform，fan_in = 3 * 3 * 3（i * h * w）
	cnn_fill_uniform(model, 0, sqrtf(6.0f / 27.0f), &rng);

	//Conv2：He uniform，fan_in = 4 * 3 * 3
	cnn_fill_uniform(model, 2, sqrtf(6.0f / 36.0f), &rng);

	//FC：Xavier uniform，fan_in + fan_out = 8 + 2
	cnn_fill_uniform(model, 4, sqrtf(6.0f / 10.0f), &rng);

	// 三个 bias 和 step 保持为零；初始化算法后续记录到 TrainingSpec
}

void
cnn_adam_state_init(CnnAdamState *state)
{
	// 尚无历史梯度，所有参数的一阶矩 m、二阶矩 v 均从零开始
	memset(state, 0, sizeof(*state));
}

bool
cnn_f32_encode_le(const float *source, size_t count,
		  uint8_t *target, size_t target_size)
{
	if (count > SIZE_MAX / 4 || target_size != count * 4)
		return false;
	if (count != 0 && (source == NULL || target == NULL))
		return false;

	for (size_t i = 0; i < count; i++) {
		uint32_t bits;

		if (!isfinite(source[i]))
			return false;

		// 复制位模式，不进行 float 到整数的数值转换
		memcpy(&bits, &source[i], sizeof(bits));
		target[4 * i + 0] = (uint8_t) (bits >> 0);
		target[4 * i + 1] = (uint8_t) (bits >> 8);
		target[4 * i + 2] = (uint8_t) (bits >> 16);
		target[4 * i + 3] = (uint8_t) (bits >> 24);
	}
	return true;
}

bool
cnn_f32_decode_le(const uint8_t *source, size_t source_size,
		  float *target, size_t count)
{
	if (count > SIZE_MAX / 4 || source_size != count * 4)
		return false;
	if (count != 0 && (source == NULL || target == NULL))
		return false;

	for (size_t i = 0; i < count; i++) {
		uint32_t bits =
			((uint32_t) source[4 * i + 0] << 0) |
			((uint32_t) source[4 * i + 1] << 8) |
			((uint32_t) source[4 * i + 2] << 16) |
			((uint32_t) source[4 * i + 3] << 24);
		float value;

		memcpy(&value, &bits, sizeof(value));
		if (!isfinite(value))
			return false;
		target[i] = value;
	}
	return true;
}
