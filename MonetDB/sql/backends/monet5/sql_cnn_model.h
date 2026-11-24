/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#ifndef SQL_CNN_MODEL_H
#define SQL_CNN_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
	CNN_TENSOR_COUNT = 6,
	CNN_PARAMETER_COUNT = 426
};

//描述一个张量结构
typedef struct {
	const char *layer_id;
	const char *tensor_name;
	const char *tensor_role;
	const char *shape;
	const char *layout;

	size_t offset; 
	size_t count;
} CnnTensorSpec;

//固定网络的内存工作状态
typedef struct {
	float parameter[CNN_PARAMETER_COUNT];
	uint64_t seed;
	uint64_t step;
} CnnModel;

//普通 SGD 不需要此对象；Adam 的状态与模型参数分开管理。
typedef struct {
	float m[CNN_PARAMETER_COUNT];
	float v[CNN_PARAMETER_COUNT];
} CnnAdamState;

extern const CnnTensorSpec cnn_tensor_specs[CNN_TENSOR_COUNT];

void cnn_model_init(CnnModel *model, uint64_t seed);
void cnn_adam_state_init(CnnAdamState *state);
uint64_t cnn_splitmix64_next(uint64_t *state);

/*
 * 编码约定：IEEE-754 binary32、小端字节序，每个值占 4 字节。
 * 缓冲区大小必须恰好等于 count * 4，且数值必须有限。
 * 输入和输出缓冲区不能重叠。
 * 返回 false 时，调用者应丢弃本次输出，不能使用部分结果。
 */
bool cnn_f32_encode_le(const float *source, size_t count, uint8_t *target, size_t target_size);
bool cnn_f32_decode_le(const uint8_t *source, size_t source_size, float *target, size_t count);

#endif /* SQL_CNN_MODEL_H */
