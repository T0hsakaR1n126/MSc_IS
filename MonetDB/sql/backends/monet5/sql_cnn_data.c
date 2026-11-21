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
#include "sql_cnn_data.h"

#include "sql_execute.h"
#include "sql_scenario.h"
#include "sql_privileges.h"
#include "sql_storage.h"

#include <stdio.h>
#include <string.h>

enum {
	CNN_DATA_POSITION,
	CNN_DATA_IMAGE_ID,
	CNN_DATA_LABEL,
	CNN_DATA_HEIGHT,
	CNN_DATA_WIDTH,
	CNN_DATA_CHANNELS,
	CNN_DATA_RED,
	CNN_DATA_GREEN,
	CNN_DATA_BLUE,
	CNN_DATA_COLUMN_COUNT
};

/* 与其他原生 SQL 操作一致：错误使外层活动事务失败，不自行提交。 */
static str
cnn_data_result(mvc *m, str msg)
{
	if (msg != MAL_SUCCEED && m != NULL && m->session != NULL &&
	    m->session->tr != NULL && m->session->tr->active &&
	    m->session->status == 0)
		m->session->status = -1;
	return msg;
}

/*
 * 内部 SQL 执行器会临时采用管理员身份，因此先检查原调用者权限。
 * 本文件只执行固定的图片表 SQL 模板，不接受用户提供的 SQL 文本。
 */
static str
cnn_data_prepare(Client cntxt, mvc *m)
{
	if (cntxt == NULL || cntxt->sqlcontext == NULL || m == NULL ||
	    m->session == NULL || m->session->tr == NULL ||
	    ((backend *) cntxt->sqlcontext)->mvc != m)
		throw(SQL, "cnn.data", SQLSTATE(HY002) "No matching SQL context available");
	str msg = SQLtrans(m);
	if (msg != MAL_SUCCEED)
		return msg;
	if (m->session->status != 0)
		throw(SQL, "cnn.data", SQLSTATE(25000) "Current transaction has failed");

	sql_schema *schema = mvc_bind_schema(m, "cnn");
	const char *names[] = {"trainingimage", "imagechannel"};
	for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
		sql_table *table = schema ? mvc_bind_table(m, schema, names[i]) : NULL;
		if (table == NULL)
			throw(SQL, "cnn.data", SQLSTATE(42S02) "Required image table is missing: cnn.%s", names[i]);
		if (!table_privs(m, table, PRIV_SELECT))
			throw(SQL, "cnn.data", SQLSTATE(42501) "SELECT privilege required on cnn.%s", names[i]);
	}
	return MAL_SUCCEED;
}

/*
 * 每个请求都携带独立的批次位置。重复 ID 不会被去重，缺失图片也不会
 * 被 INNER JOIN 静默丢弃；后续检查 LEFT JOIN 产生的 SQL NULL。
 */
static str
cnn_data_query(const int64_t *image_ids, size_t count, str *query)
{
	static const char prefix[] =
		"WITH requested(batch_pos, image_id) AS (VALUES ";
	static const char suffix[] =
		") SELECT q.batch_pos, i.ImageId, i.Label, i.Height, i.Width, i.Channels, "
		"r.ChannelValue, g.ChannelValue, b.ChannelValue "
		"FROM requested q "
		"LEFT JOIN cnn.TrainingImage i ON i.ImageId = q.image_id "
		"LEFT JOIN cnn.ImageChannel r ON r.ChannelRef = i.RRef "
		"LEFT JOIN cnn.ImageChannel g ON g.ChannelRef = i.GRef "
		"LEFT JOIN cnn.ImageChannel b ON b.ChannelRef = i.BRef "
		"ORDER BY q.batch_pos;";
	/* 两个非负 64 位整数和固定 CAST 文本，每行最多需要这些字节。 */
	const size_t row_capacity = 96;
	size_t fixed = sizeof(prefix) - 1 + sizeof(suffix);
	*query = NULL;
	if (count > (SIZE_MAX - fixed) / row_capacity)
		throw(SQL, "cnn.data", SQLSTATE(22003) "Batch SQL size would overflow");
	size_t capacity = fixed + count * row_capacity;
	str buffer = GDKmalloc(capacity);
	if (buffer == NULL)
		throw(SQL, "cnn.data", SQLSTATE(HY013) MAL_MALLOC_FAIL);

	size_t used = sizeof(prefix) - 1;
	memcpy(buffer, prefix, used);
	for (size_t n = 0; n < count; n++) {
		int written = snprintf(buffer + used, capacity - used,
			"%s(CAST(%zu AS BIGINT),CAST(%" PRId64 " AS BIGINT))",
			n == 0 ? "" : ",", n, image_ids[n]);
		if (written < 0 || (size_t) written >= capacity - used) {
			GDKfree(buffer);
			throw(SQL, "cnn.data", SQLSTATE(22003) "Cannot format batch SQL");
		}
		used += (size_t) written;
	}
	if (sizeof(suffix) > capacity - used) {
		GDKfree(buffer);
		throw(SQL, "cnn.data", SQLSTATE(22003) "Batch SQL buffer is too small");
	}
	memcpy(buffer + used, suffix, sizeof(suffix));
	*query = buffer;
	return MAL_SUCCEED;
}

/* 所有结果列都固定到 batch 打包结束，避免借用的 BLOB 数据提前失效。 */
static str
cnn_data_columns(res_table *result, size_t count, BAT **columns,
		 BATiter *iterators, size_t *opened)
{
	const int types[CNN_DATA_COLUMN_COUNT] = {
		TYPE_lng, TYPE_lng, TYPE_int, TYPE_int, TYPE_int, TYPE_int,
		TYPE_blob, TYPE_blob, TYPE_blob
	};
	if (result == NULL || result->next != NULL ||
	    result->nr_cols != CNN_DATA_COLUMN_COUNT || result->cols == NULL)
		throw(SQL, "cnn.data", SQLSTATE(22000) "Unexpected image query result");

	for (size_t i = 0; i < CNN_DATA_COLUMN_COUNT; i++) {
		columns[i] = BATdescriptor(result->cols[i].b);
		if (columns[i] == NULL)
			throw(SQL, "cnn.data", SQLSTATE(HY013) MAL_MALLOC_FAIL);
		if (columns[i]->ttype != types[i] || BATcount(columns[i]) != (BUN) count)
			throw(SQL, "cnn.data", SQLSTATE(22000) "Invalid image query column type or row count");
		iterators[i] = bat_iterator(columns[i]);
		(*opened)++;
	}
	return MAL_SUCCEED;
}

/* 定长列使用 BUNtloc，原始 uint8 通道使用变长 BLOB 的 data 和 nitems。 */
static str
cnn_data_views(const int64_t *image_ids, size_t count,
	       BATiter *iterators, CnnImageView *views)
{
	for (size_t n = 0; n < count; n++) {
		BUN row = (BUN) n;
		lng position = *(const lng *) BUNtloc(&iterators[CNN_DATA_POSITION], row);
		lng id = *(const lng *) BUNtloc(&iterators[CNN_DATA_IMAGE_ID], row);
		int label = *(const int *) BUNtloc(&iterators[CNN_DATA_LABEL], row);
		int height = *(const int *) BUNtloc(&iterators[CNN_DATA_HEIGHT], row);
		int width = *(const int *) BUNtloc(&iterators[CNN_DATA_WIDTH], row);
		int channels = *(const int *) BUNtloc(&iterators[CNN_DATA_CHANNELS], row);
		if (is_lng_nil(position) || position != (lng) n ||
		    is_lng_nil(id) || id != (lng) image_ids[n] ||
		    is_int_nil(label) || (label != 0 && label != 1) ||
		    is_int_nil(height) || height != CNN_IMAGE_HEIGHT ||
		    is_int_nil(width) || width != CNN_IMAGE_WIDTH ||
		    is_int_nil(channels) || channels != CNN_IMAGE_CHANNELS)
			throw(SQL, "cnn.data", SQLSTATE(22000) "Requested image is missing or invalid: %" PRId64, image_ids[n]);

		views[n] = (CnnImageView) {
			.image_id = (int64_t) id,
			.label = (int32_t) label,
			.height = (size_t) height,
			.width = (size_t) width,
			.channels = (size_t) channels
		};
		for (size_t c = 0; c < CNN_IMAGE_CHANNELS; c++) {
			const blob *payload = BUNtvar(&iterators[CNN_DATA_RED + c], row);
			if (payload == NULL || is_blob_nil(payload) ||
			    payload->nitems != CNN_PLANE_VALUES)
				throw(SQL, "cnn.data", SQLSTATE(22000) "Image channel is missing or invalid: %" PRId64, image_ids[n]);
			views[n].plane[c] = payload->data;
			views[n].plane_bytes[c] = payload->nitems;
		}
	}
	return MAL_SUCCEED;
}

str
cnn_batch_load_ids(Client cntxt, mvc *m, const int64_t *image_ids,
		   size_t count, CnnBatch *out)
{
	str msg = MAL_SUCCEED, query = NULL;
	res_table *result = NULL;
	BAT *columns[CNN_DATA_COLUMN_COUNT] = {NULL};
	BATiter iterators[CNN_DATA_COLUMN_COUNT];
	size_t opened = 0;
	CnnImageView *views = NULL;

	if (out == NULL || image_ids == NULL || count == 0) {
		msg = createException(SQL, "cnn.data", SQLSTATE(22023) "A non-empty image ID list and output batch are required");
		goto cleanup;
	}
	if (out->count != 0 || out->input != NULL ||
	    out->labels != NULL || out->image_ids != NULL) {
		msg = createException(SQL, "cnn.data", SQLSTATE(22023) "Output batch must be empty");
		goto cleanup;
	}
	/* 大小检查先于读取 ID 数组，以及任何查询构建和数据分配。 */
	if (count > (uint64_t) GDK_lng_max || count > (uint64_t) BUN_MAX ||
	    count > SIZE_MAX / sizeof(CnnImageView) ||
	    count > SIZE_MAX / (CNN_IMAGE_VALUES * sizeof(float)) ||
	    count > SIZE_MAX / sizeof(int32_t) || count > SIZE_MAX / sizeof(int64_t)) {
		msg = createException(SQL, "cnn.data", SQLSTATE(22003) "Batch allocation size would overflow");
		goto cleanup;
	}
	for (size_t n = 0; n < count; n++) {
		if (image_ids[n] < 0) {
			msg = createException(SQL, "cnn.data", SQLSTATE(22023) "Image IDs must be non-negative BIGINT values");
			goto cleanup;
		}
	}
	msg = cnn_data_prepare(cntxt, m);
	if (msg == MAL_SUCCEED)
		msg = cnn_data_query(image_ids, count, &query);
	if (msg == MAL_SUCCEED)
		msg = SQLstatementIntern(cntxt, query, "cnn.load_images", true, false, &result);
	if (msg == MAL_SUCCEED)
		msg = cnn_data_columns(result, count, columns, iterators, &opened);
	if (msg != MAL_SUCCEED)
		goto cleanup;
	views = GDKmalloc(count * sizeof(*views));
	if (views == NULL) {
		msg = createException(SQL, "cnn.data", SQLSTATE(HY013) MAL_MALLOC_FAIL);
		goto cleanup;
	}
	msg = cnn_data_views(image_ids, count, iterators, views);
	if (msg == MAL_SUCCEED) {
		CnnBatchResult built = cnn_batch_build(out, views, count);
		if (built == CNN_BATCH_OUT_OF_MEMORY)
			msg = createException(SQL, "cnn.data", SQLSTATE(HY013) MAL_MALLOC_FAIL);
		else if (built == CNN_BATCH_SIZE_OVERFLOW)
			msg = createException(SQL, "cnn.data", SQLSTATE(22003) "Batch allocation size would overflow");
		else if (built != CNN_BATCH_OK)
			msg = createException(SQL, "cnn.data", SQLSTATE(22000) "Cannot pack image batch");
	}

cleanup:
	/* 必须先完成复制，再解除迭代器、BAT 和查询结果的引用。 */
	for (size_t i = 0; i < opened; i++)
		bat_iterator_end(&iterators[i]);
	for (size_t i = 0; i < CNN_DATA_COLUMN_COUNT; i++) {
		if (columns[i] != NULL)
			BBPunfix(columns[i]->batCacheid);
	}
	res_tables_destroy(result);
	GDKfree(views);
	GDKfree(query);
	return cnn_data_result(m, msg);
}

/* 只接收已经解码并整理好的三个 uint8 平面，不负责图片文件解码。 */
static str
cnn_data_validate_rgb(const CnnImageView *image, const char *split)
{
	if (image == NULL || image->image_id < 0 ||
	    image->height != CNN_IMAGE_HEIGHT || image->width != CNN_IMAGE_WIDTH ||
	    image->channels != CNN_IMAGE_CHANNELS ||
	    (image->label != 0 && image->label != 1))
		throw(SQL, "cnn.data", SQLSTATE(22023) "Invalid RGB image metadata");
	if (split == NULL || strNil(split) ||
	    (strcmp(split, "train") != 0 && strcmp(split, "validation") != 0 &&
	     strcmp(split, "test") != 0))
		throw(SQL, "cnn.data", SQLSTATE(22023) "Image split must be 'train', 'validation' or 'test'");
	for (size_t c = 0; c < CNN_IMAGE_CHANNELS; c++) {
		if (image->plane[c] == NULL || image->plane_bytes[c] != CNN_PLANE_VALUES)
			throw(SQL, "cnn.data", SQLSTATE(22023) "Each RGB channel must contain 4096 uint8 bytes");
	}
	return MAL_SUCCEED;
}

/* 写入前在原用户上下文检查两表的 INSERT 权限，避免内部 SQL 提权。 */
static str
cnn_data_prepare_write(Client cntxt, mvc *m)
{
	str msg = cnn_data_prepare(cntxt, m);
	if (msg != MAL_SUCCEED)
		return msg;
	sql_schema *schema = mvc_bind_schema(m, "cnn");
	const char *names[] = {"trainingimage", "imagechannel"};
	for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
		sql_table *table = schema ? mvc_bind_table(m, schema, names[i]) : NULL;
		if (table == NULL)
			throw(SQL, "cnn.data", SQLSTATE(42S02) "Required image table is missing: cnn.%s", names[i]);
		if (!table_privs(m, table, PRIV_INSERT))
			throw(SQL, "cnn.data", SQLSTATE(42501) "INSERT privilege required on cnn.%s", names[i]);
	}
	/* NEXT VALUE FOR 在本分支要求序列所属 schema 的所有者或管理员权限。 */
	if (!mvc_schema_privs(m, schema))
		throw(SQL, "cnn.data", SQLSTATE(42501) "Schema privilege required to use cnn.image_channel_ref_seq");
	return MAL_SUCCEED;
}

/* COUNT 和 NEXT VALUE 查询都返回恰好一个非负 BIGINT。 */
static str
cnn_data_read_scalar(Client cntxt, const char *query, lng *value)
{
	res_table *result = NULL;
	BAT *column = NULL;
	str msg = SQLstatementIntern(cntxt, query, "cnn.image_integer", true, false, &result);
	if (msg != MAL_SUCCEED)
		goto cleanup;
	if (result == NULL || result->next != NULL || result->nr_cols != 1 ||
	    result->cols == NULL) {
		msg = createException(SQL, "cnn.data", SQLSTATE(22000) "Unexpected image integer query result");
		goto cleanup;
	}
	column = BATdescriptor(result->cols[0].b);
	if (column == NULL) {
		msg = createException(SQL, "cnn.data", SQLSTATE(HY013) MAL_MALLOC_FAIL);
		goto cleanup;
	}
	if (column->ttype != TYPE_lng || BATcount(column) != 1) {
		msg = createException(SQL, "cnn.data", SQLSTATE(22000) "Invalid image integer query column");
		goto cleanup;
	}
	BATiter iterator = bat_iterator(column);
	lng found = *(const lng *) BUNtloc(&iterator, 0);
	bat_iterator_end(&iterator);
	if (is_lng_nil(found) || found < 0)
		msg = createException(SQL, "cnn.data", SQLSTATE(22000) "Invalid image integer value");
	else
		*value = found;
cleanup:
	if (column != NULL)
		BBPunfix(column->batCacheid);
	res_tables_destroy(result);
	return msg;
}

/* 原始字节直接转成 BLOB 字面量；这里不做 float32 编码或除以 255。 */
static str
cnn_data_write_channel(Client cntxt, lng reference, const uint8_t *values)
{
	static const char digits[] = "0123456789abcdef";
	static const char suffix[] = "');";
	const size_t capacity = CNN_PLANE_VALUES * 2 + 192;
	str query = GDKmalloc(capacity);
	if (query == NULL)
		throw(SQL, "cnn.data", SQLSTATE(HY013) MAL_MALLOC_FAIL);
	int length = snprintf(query, capacity,
		"INSERT INTO cnn.ImageChannel (ChannelRef, ChannelValue) VALUES (%" PRId64 ", X'",
		reference);
	if (length < 0 || (size_t) length > capacity - CNN_PLANE_VALUES * 2 - sizeof(suffix)) {
		GDKfree(query);
		throw(SQL, "cnn.data", SQLSTATE(22003) "Cannot format image channel SQL");
	}
	size_t offset = (size_t) length;
	for (size_t k = 0; k < CNN_PLANE_VALUES; k++) {
		query[offset + 2 * k] = digits[values[k] >> 4];
		query[offset + 2 * k + 1] = digits[values[k] & 15];
	}
	memcpy(query + offset + CNN_PLANE_VALUES * 2, suffix, sizeof(suffix));
	str msg = SQLstatementIntern(cntxt, query, "cnn.write_channel", true, false, NULL);
	GDKfree(query);
	return msg;
}

/* split 已通过固定白名单，其他值均为校验后的整数。 */
static str
cnn_data_write_image(Client cntxt, const CnnImageView *image,
		     const char *split, const lng *references)
{
	char query[512];
	int length = snprintf(query, sizeof(query),
		"INSERT INTO cnn.TrainingImage "
		"(ImageId, Split, Label, Height, Width, Channels, RRef, GRef, BRef) "
		"VALUES (%" PRId64 ", '%s', %" PRId32 ", 64, 64, 3, %" PRId64 ", %" PRId64 ", %" PRId64 ");",
		image->image_id, split, image->label, references[0], references[1], references[2]);
	if (length < 0 || (size_t) length >= sizeof(query))
		throw(SQL, "cnn.data", SQLSTATE(22003) "Cannot format image metadata SQL");
	return SQLstatementIntern(cntxt, query, "cnn.write_image", true, false, NULL);
}

str
cnn_image_store_rgb(Client cntxt, mvc *m, const CnnImageView *image,
		    const char *split)
{
	str msg = cnn_data_validate_rgb(image, split);
	if (msg != MAL_SUCCEED)
		goto cleanup;
	msg = cnn_data_prepare_write(cntxt, m);
	if (msg != MAL_SUCCEED)
		goto cleanup;

	/* 拒绝同名图片，不能覆盖后再留下原通道或新生成的孤立通道。 */
	char query[256];
	int length = snprintf(query, sizeof(query),
		"SELECT CAST(COUNT(*) AS BIGINT) FROM cnn.TrainingImage WHERE ImageId = %" PRId64 ";",
		image->image_id);
	if (length < 0 || (size_t) length >= sizeof(query)) {
		msg = createException(SQL, "cnn.data", SQLSTATE(22003) "Cannot format image lookup SQL");
		goto cleanup;
	}
	lng existing = 0;
	msg = cnn_data_read_scalar(cntxt, query, &existing);
	if (msg != MAL_SUCCEED)
		goto cleanup;
	if (existing != 0) {
		msg = createException(SQL, "cnn.data", SQLSTATE(23505) "ImageId already exists; RGB image is not overwritten");
		goto cleanup;
	}

	/* 序列可在回滚后留空洞；引用不能用 MAX(id) 或进程地址推算。 */
	lng references[CNN_IMAGE_CHANNELS] = {0};
	for (size_t c = 0; c < CNN_IMAGE_CHANNELS; c++) {
		msg = cnn_data_read_scalar(cntxt,
			"SELECT NEXT VALUE FOR cnn.image_channel_ref_seq;", &references[c]);
		if (msg != MAL_SUCCEED)
			goto cleanup;
		if (references[c] <= 0) {
			msg = createException(SQL, "cnn.data", SQLSTATE(22000) "Image channel sequence must produce positive values");
			goto cleanup;
		}
		for (size_t previous = 0; previous < c; previous++) {
			if (references[c] == references[previous]) {
				msg = createException(SQL, "cnn.data", SQLSTATE(22000) "Image channel references must be distinct");
				goto cleanup;
			}
		}
	}
	/* 三个通道与图片行共用外层事务；任何失败都让整次写入回滚。 */
	for (size_t c = 0; c < CNN_IMAGE_CHANNELS; c++) {
		msg = cnn_data_write_channel(cntxt, references[c], image->plane[c]);
		if (msg != MAL_SUCCEED)
			goto cleanup;
	}
	msg = cnn_data_write_image(cntxt, image, split, references);
cleanup:
	return cnn_data_result(m, msg);
}
