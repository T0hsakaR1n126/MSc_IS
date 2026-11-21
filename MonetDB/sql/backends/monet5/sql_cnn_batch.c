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
#include "sql_cnn_batch.h"
#include "gdk.h"

void
cnn_batch_destroy(CnnBatch *batch)
{
	if (batch == NULL)
		return;

	GDKfree(batch->input);
	GDKfree(batch->labels);
	GDKfree(batch->image_ids);
	*batch = (CnnBatch) {0};
}

CnnBatchResult
cnn_batch_build(CnnBatch *out, const CnnImageView *images, size_t count)
{
	if (out == NULL || images == NULL || count == 0)
		return CNN_BATCH_INVALID_INPUT;

	/* 避免覆盖尚未释放的旧 batch。 */
	if (out->count != 0 || out->input != NULL ||
	    out->labels != NULL || out->image_ids != NULL)
		return CNN_BATCH_INVALID_INPUT;

	/* 所有分配大小的乘法之前先检查溢出。 */
	if (count > SIZE_MAX / (CNN_IMAGE_VALUES * sizeof(float)) ||
	    count > SIZE_MAX / sizeof(int32_t) ||
	    count > SIZE_MAX / sizeof(int64_t))
		return CNN_BATCH_SIZE_OVERFLOW;

	/* 分配和写入之前，先检查所有图片和通道。 */
	for (size_t n = 0; n < count; n++) {
		const CnnImageView *image = &images[n];

		if (image->image_id < 0 ||
		    image->height != CNN_IMAGE_HEIGHT ||
		    image->width != CNN_IMAGE_WIDTH ||
		    image->channels != CNN_IMAGE_CHANNELS ||
		    (image->label != 0 && image->label != 1))
			return CNN_BATCH_INVALID_INPUT;

		for (size_t c = 0; c < CNN_IMAGE_CHANNELS; c++) {
			if (image->plane[c] == NULL ||
			    image->plane_bytes[c] != CNN_PLANE_VALUES)
				return CNN_BATCH_INVALID_INPUT;
		}
	}

	CnnBatch temporary = {0};
	temporary.input = GDKmalloc(count * CNN_IMAGE_VALUES * sizeof(float));
	temporary.labels = GDKmalloc(count * sizeof(*temporary.labels));
	temporary.image_ids = GDKmalloc(count * sizeof(*temporary.image_ids));

	if (temporary.input == NULL || temporary.labels == NULL ||
	    temporary.image_ids == NULL) {
		cnn_batch_destroy(&temporary);
		return CNN_BATCH_OUT_OF_MEMORY;
	}

	for (size_t n = 0; n < count; n++) {
		const CnnImageView *image = &images[n];
		temporary.labels[n] = image->label;
		temporary.image_ids[n] = image->image_id;

		for (size_t c = 0; c < CNN_IMAGE_CHANNELS; c++) {
			/* k = h * 64 + w，整段起点对应 input[n][c][0][0]。 */
			size_t base = (n * CNN_IMAGE_CHANNELS + c) * CNN_PLANE_VALUES;
			for (size_t k = 0; k < CNN_PLANE_VALUES; k++) {
				temporary.input[base + k] =
					(float) image->plane[c][k] / 255.0f;
			}
		}
	}

	/* 完整构建成功后才交给调用者；失败不会发布部分 batch。 */
	temporary.count = count;
	*out = temporary;
	return CNN_BATCH_OK;
}
