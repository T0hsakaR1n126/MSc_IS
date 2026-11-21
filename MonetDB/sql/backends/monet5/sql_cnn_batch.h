/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#ifndef SQL_CNN_BATCH_H
#define SQL_CNN_BATCH_H

#include <stddef.h>
#include <stdint.h>

enum {
	CNN_IMAGE_HEIGHT = 64,
	CNN_IMAGE_WIDTH = 64,
	CNN_IMAGE_CHANNELS = 3,
	CNN_PLANE_VALUES = 64 * 64,
	CNN_IMAGE_VALUES = 3 * 64 * 64
};

/*
 * 借用图片数据，不拥有通道内存。
 * plane[0]、plane[1]、plane[2] 分别是按行展开的 R、G、B uint8 通道。
 * 打包结束前，三个指针及其各自的 4096 字节数据必须保持有效。
 * image_id 是持久化逻辑标识，不是进程指针或内部 BAT 标识。
 */
typedef struct {
	int64_t image_id;
	int32_t label;
	size_t height;
	size_t width;
	size_t channels;

	const uint8_t *plane[CNN_IMAGE_CHANNELS];
	size_t plane_bytes[CNN_IMAGE_CHANNELS];
} CnnImageView;

//拥有三个动态数组；打包完成后不再依赖原通道内存，使用完必须释放。
typedef struct {
	size_t count;
	float *input;       /* 连续 [count,3,64,64]，NCHW。 */
	int32_t *labels;    /* [count]，顺序与输入图片一致。 */
	int64_t *image_ids; /* [count]，顺序与输入图片一致。 */
} CnnBatch;

typedef enum {
	CNN_BATCH_OK = 0,
	CNN_BATCH_INVALID_INPUT,
	CNN_BATCH_SIZE_OVERFLOW,
	CNN_BATCH_OUT_OF_MEMORY
} CnnBatchResult;

/*
 * out 必须先初始化为空：CnnBatch batch = {0};
 * 本函数不执行 SQL、不抽样、不洗牌，严格保持 images 的输入顺序。
 * count 是实际图片数，最后不足整批时直接传入实际数量即可。
 * 失败时 out 保持原样；成功后 out 拥有独立的数据副本。
 */
CnnBatchResult cnn_batch_build(CnnBatch *out, const CnnImageView *images, size_t count);

//允许传入 NULL；释放后对象重新变为空 batch。
void cnn_batch_destroy(CnnBatch *batch);

#endif /* SQL_CNN_BATCH_H */
