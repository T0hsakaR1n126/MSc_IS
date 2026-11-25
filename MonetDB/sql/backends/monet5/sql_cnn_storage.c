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
#include "sql_cnn_storage.h"

#include "sql_execute.h"
#include "sql_scenario.h"
#include "sql_privileges.h"
#include "sql_storage.h"

#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
	CNN_QUOTED_ID,
	CNN_QUOTED_ARCHITECTURE,
	CNN_QUOTED_PREPROCESS,
	CNN_QUOTED_LABELS,
	CNN_QUOTED_MODEL_COUNT,
	CNN_QUOTED_TRAINING = CNN_QUOTED_MODEL_COUNT,
	CNN_QUOTED_COUNT
};

typedef struct {
	const char *name;
	bool writing;
} CnnStorageTable;

// 本模块只执行固定存储操作，由调用者提交事务。
static str
cnn_storage_result(mvc *m, str msg)
{
	if (msg != MAL_SUCCEED && m != NULL && m->session != NULL &&
	    m->session->tr != NULL && m->session->tr->active &&
	    m->session->status == 0)
		m->session->status = -1;
	return msg;
}

static bool
cnn_blank(const char *value)
{
	if (value == NULL || strNil(value))
		return true;
	while (*value && isspace((unsigned char) *value))
		value++;
	return *value == '\0';
}

static void
cnn_free_strings(str *values, size_t count)
{
	for (size_t i = 0; i < count; i++)
		GDKfree(values[i]);
}

//动态分配 SQL，避免 CLOB 元数据被固定缓冲区截断
static str cnn_queryf(str *query, const char *format, ...)
	__attribute__((format(printf, 2, 3)));

static str
cnn_queryf(str *query, const char *format, ...)
{
	va_list args, copy;
	*query = NULL;
	va_start(args, format);
	va_copy(copy, args);
	int length = vsnprintf(NULL, 0, format, copy);
	va_end(copy);
	if (length < 0) {
		va_end(args);
		throw(SQL, "cnn.storage", SQLSTATE(22000) "Cannot format internal SQL");
	}
	str buffer = GDKmalloc((size_t) length + 1);
	if (buffer == NULL) {
		va_end(args);
		throw(SQL, "cnn.storage", SQLSTATE(HY013) MAL_MALLOC_FAIL);
	}
	int written = vsnprintf(buffer, (size_t) length + 1, format, args);
	va_end(args);
	if (written != length) {
		GDKfree(buffer);
		throw(SQL, "cnn.storage", SQLSTATE(22000) "Cannot format internal SQL");
	}
	*query = buffer;
	return MAL_SUCCEED;
}

//返回包含外层单引号的 SQL 字面量；JSON 内容本身由调用者生成。
static str
cnn_quote(str *quoted, const char *value)
{
	*quoted = NULL;
	if (value == NULL || strNil(value))
		throw(SQL, "cnn.storage", SQLSTATE(22023) "Metadata must not be NULL");
	if (strlen(value) > (SIZE_MAX - 3) / 2)
		throw(SQL, "cnn.storage", SQLSTATE(22000) "Metadata is too large");
	str escaped = SQLescapeString((str) value);
	if (escaped == NULL)
		throw(SQL, "cnn.storage", SQLSTATE(HY013) MAL_MALLOC_FAIL);
	str msg = cnn_queryf(quoted, "'%s'", escaped);
	GDKfree(escaped);
	return msg;
}

static str
cnn_quote_model_info(const CnnInitialModelInfo *info, str *quoted)
{
	const char *values[CNN_QUOTED_MODEL_COUNT] = {
		info->model_id, info->architecture, info->preprocess_spec,
		info->class_labels
	};
	for (size_t i = 0; i < CNN_QUOTED_MODEL_COUNT; i++) {
		str msg = cnn_quote(&quoted[i], values[i]);
		if (msg != MAL_SUCCEED)
			return msg;
	}
	return MAL_SUCCEED;
}

static str
cnn_quote_info(const CnnInitialModelInfo *info, str *quoted)
{
	str msg = cnn_quote_model_info(info, quoted);
	if (msg == MAL_SUCCEED)
		msg = cnn_quote(&quoted[CNN_QUOTED_TRAINING], info->training_spec);
	return msg;
}

static str
cnn_validate_model_info(const CnnInitialModelInfo *info)
{
	if (info == NULL || cnn_blank(info->model_id) ||
	    cnn_blank(info->architecture) || cnn_blank(info->preprocess_spec) ||
	    cnn_blank(info->class_labels))
		throw(SQL, "cnn.storage", SQLSTATE(22023) "Model metadata must not be NULL or blank");
	for (size_t i = 0; i < CNN_TENSOR_COUNT; i++) {
		const CnnTensorSpec *spec = &cnn_tensor_specs[i];
		if (spec->count == 0 || spec->offset > CNN_PARAMETER_COUNT ||
		    spec->count > CNN_PARAMETER_COUNT - spec->offset)
			throw(SQL, "cnn.storage", SQLSTATE(22000) "Invalid fixed-network tensor description");
	}
	return MAL_SUCCEED;
}

static str
cnn_validate_info(const CnnInitialModelInfo *info, bool initial)
{
	str msg = cnn_validate_model_info(info);
	if (msg != MAL_SUCCEED)
		return msg;
	if (cnn_blank(info->training_spec))
		throw(SQL, "cnn.storage", SQLSTATE(22023) "Model metadata must not be NULL or blank");
	if (initial && info->version != 1)
		throw(SQL, "cnn.storage", SQLSTATE(22023) "Initial model version must be 1");
	return MAL_SUCCEED;
}

/*
 * SQLstatementIntern 会临时使用管理员身份。必须在调用它之前，
 * 在原用户上下文里检查权限；所有 SQL 都是本模块固定的表访问模板。
 */
static str
cnn_prepare_tables(Client cntxt, mvc *m, const CnnStorageTable *tables, size_t count)
{
	if (cntxt == NULL || cntxt->sqlcontext == NULL || m == NULL ||
	    m->session == NULL || ((backend *) cntxt->sqlcontext)->mvc != m)
		throw(SQL, "cnn.storage", SQLSTATE(HY002) "No matching SQL context available");
	str msg = SQLtrans(m);
	if (msg != MAL_SUCCEED)
		return msg;
	if (m->session->status != 0)
		throw(SQL, "cnn.storage", SQLSTATE(25000) "Current transaction has failed");
	sql_schema *schema = mvc_bind_schema(m, "cnn");
	for (size_t i = 0; i < count; i++) {
		const char *name = tables[i].name;
		sql_table *table = schema ? mvc_bind_table(m, schema, name) : NULL;
		if (table == NULL)
			throw(SQL, "cnn.storage", SQLSTATE(42S02) "Required CNN table is missing: %s", name);
		if (!table_privs(m, table, PRIV_SELECT))
			throw(SQL, "cnn.storage", SQLSTATE(42501) "SELECT privilege required on cnn.%s", name);
		if (tables[i].writing && !table_privs(m, table, PRIV_INSERT))
			throw(SQL, "cnn.storage", SQLSTATE(42501) "INSERT privilege required on cnn.%s", name);
	}
	return MAL_SUCCEED;
}

static str
cnn_prepare(Client cntxt, mvc *m, bool writing, bool use_adam, bool write_metrics)
{
	const CnnStorageTable tables[] = {
		{"model", writing},
		{"modeltensor", writing},
		{"optimizertensor", writing && use_adam},
		{"trainingmetric", writing && write_metrics},
	};
	return cnn_prepare_tables(cntxt, m, tables, sizeof(tables) / sizeof(tables[0]));
}

// SELECT 的结果，包括标量聚合，在当前 MonetDB 中均包装为 BAT。
static str
cnn_single_bat(res_table *result, int type, BAT **output)
{
	*output = NULL;
	if (result == NULL || result->next != NULL || result->nr_cols != 1)
		throw(SQL, "cnn.storage", SQLSTATE(22000) "Unexpected storage query result");
	BAT *b = BATdescriptor(result->cols[0].b);
	if (b == NULL)
		throw(SQL, "cnn.storage", SQLSTATE(HY013) MAL_MALLOC_FAIL);
	if (b->ttype != type || BATcount(b) != 1) {
		BBPunfix(b->batCacheid);
		throw(SQL, "cnn.storage", SQLSTATE(22000) "Missing or incompatible model data");
	}
	*output = b;
	return MAL_SUCCEED;
}

static str
cnn_read_lng(Client cntxt, const char *query, lng *value)
{
	res_table *result = NULL;
	BAT *b = NULL;
	str msg = SQLstatementIntern(cntxt, query, "cnn.read_integer", true, false, &result);
	if (msg == MAL_SUCCEED)
		msg = cnn_single_bat(result, TYPE_lng, &b);
	if (msg == MAL_SUCCEED) {
		BATiter iterator = bat_iterator(b);
		lng found = *(const lng *) BUNtloc(&iterator, 0);
		bat_iterator_end(&iterator);
		if (is_lng_nil(found) || found < 0)
			msg = createException(SQL, "cnn.storage", SQLSTATE(22000) "Invalid model integer value");
		else
			*value = found;
	}
	if (b != NULL)
		BBPunfix(b->batCacheid);
	res_tables_destroy(result);
	return msg;
}

static str
cnn_count(Client cntxt, const char *table, const str id, int version, lng *value)
{
	str query = NULL;
	// table 只来自本文件中的固定表名，不接受用户提供的标识符。
	str msg = version == 0
		? cnn_queryf(&query, "SELECT CAST(COUNT(*) AS BIGINT) FROM cnn.%s WHERE ModelId = %s;", table, id)
		: cnn_queryf(&query, "SELECT CAST(COUNT(*) AS BIGINT) FROM cnn.%s WHERE ModelId = %s AND Version = %d;", table, id, version);
	if (msg == MAL_SUCCEED)
		msg = cnn_read_lng(cntxt, query, value);
	GDKfree(query);
	return msg;
}

static str
cnn_check_counts(Client cntxt, const str id, int version, bool use_adam, size_t metric_count)
{
	const char *tables[] = {"ModelTensor", "OptimizerTensor", "TrainingMetric"};
	lng expected[] = {CNN_TENSOR_COUNT, use_adam ? 2 * CNN_TENSOR_COUNT : 0, (lng) metric_count};
	for (size_t i = 0; i < sizeof(tables) / sizeof(tables[0]); i++) {
		lng found = 0;
		str msg = cnn_count(cntxt, tables[i], id, version, &found);
		if (msg != MAL_SUCCEED)
			return msg;
		if (found != expected[i])
			throw(SQL, "cnn.storage", SQLSTATE(22000) "Unexpected row count in cnn.%s", tables[i]);
	}
	return MAL_SUCCEED;
}

static str
cnn_write_model(Client cntxt, const CnnInitialModelInfo *info, int epoch,
		const CnnModel *model, const str *quoted)
{
	str query = NULL;
	str msg = cnn_queryf(&query,
		"INSERT INTO cnn.Model (ModelId, Version, Architecture, Status, "
		"InputShape, InputLayout, DType, PreprocessSpec, ClassLabels, TrainingSpec, "
		"Epoch, Step, \"seed\") VALUES (%s, %d, %s, '%s', "
		"'[B,3,64,64]', 'NCHW', 'float32', %s, %s, %s, %d, %" PRIu64 ", %" PRIu64 ");",
		quoted[CNN_QUOTED_ID], info->version, quoted[CNN_QUOTED_ARCHITECTURE],
		epoch == 0 ? "initialized" : "trained",
		quoted[CNN_QUOTED_PREPROCESS], quoted[CNN_QUOTED_LABELS],
		quoted[CNN_QUOTED_TRAINING], epoch, model->step, model->seed);
	if (msg == MAL_SUCCEED)
		msg = SQLstatementIntern(cntxt, query, "cnn.write_model", true, false, NULL);
	GDKfree(query);
	return msg;
}

static str
cnn_read_model(Client cntxt, const CnnInitialModelInfo *info,
	       const str *quoted, uint64_t *seed)
{
	str query = NULL;
	str msg = cnn_queryf(&query,
		"SELECT \"seed\" FROM cnn.Model WHERE ModelId = %s AND Version = %d "
		"AND Architecture = %s AND Status = 'initialized' "
		"AND InputShape = '[B,3,64,64]' AND InputLayout = 'NCHW' AND DType = 'float32' "
		"AND PreprocessSpec = %s AND ClassLabels = %s AND TrainingSpec = %s "
		"AND Epoch = 0 AND Step = 0;",
		quoted[CNN_QUOTED_ID], info->version, quoted[CNN_QUOTED_ARCHITECTURE],
		quoted[CNN_QUOTED_PREPROCESS], quoted[CNN_QUOTED_LABELS], quoted[CNN_QUOTED_TRAINING]);
	lng value = 0;
	if (msg == MAL_SUCCEED)
		msg = cnn_read_lng(cntxt, query, &value);
	if (msg == MAL_SUCCEED)
		*seed = (uint64_t) value;
	GDKfree(query);
	return msg;
}

static str
cnn_find_trained_version(Client cntxt, const str *quoted, int *version)
{
	str query = NULL;
	lng latest = 0;
	str msg = cnn_queryf(&query,
		"SELECT CAST(COALESCE(MAX(Version), 0) AS BIGINT) FROM cnn.Model "
		"WHERE ModelId = %s AND Status = 'trained' AND Epoch > 0;",
		quoted[CNN_QUOTED_ID]);
	if (msg == MAL_SUCCEED)
		msg = cnn_read_lng(cntxt, query, &latest);
	GDKfree(query);
	if (msg != MAL_SUCCEED)
		return msg;
	if (latest == 0)
		throw(SQL, "cnn.storage", SQLSTATE(22023) "No trained model version found for the requested ModelId");
	if (latest > GDK_int_max)
		throw(SQL, "cnn.storage", SQLSTATE(22000) "Invalid trained model version");

	/* Select the newest trained version before checking compatibility so an
	 * incompatible latest snapshot cannot cause a fallback to older weights. */
	lng compatible = 0;
	msg = cnn_queryf(&query,
		"SELECT CAST(COUNT(*) AS BIGINT) FROM cnn.Model WHERE ModelId = %s AND Version = %d "
		"AND Architecture = %s AND PreprocessSpec = %s AND ClassLabels = %s "
		"AND InputShape = '[B,3,64,64]' AND InputLayout = 'NCHW' AND DType = 'float32';",
		quoted[CNN_QUOTED_ID], (int) latest, quoted[CNN_QUOTED_ARCHITECTURE],
		quoted[CNN_QUOTED_PREPROCESS], quoted[CNN_QUOTED_LABELS]);
	if (msg == MAL_SUCCEED)
		msg = cnn_read_lng(cntxt, query, &compatible);
	GDKfree(query);
	if (msg != MAL_SUCCEED)
		return msg;
	if (compatible != 1)
		throw(SQL, "cnn.storage", SQLSTATE(22000) "Latest trained model version has incompatible metadata");
	*version = (int) latest;
	return MAL_SUCCEED;
}

//每个张量分别保存；十六进制只是 SQL 的字节传输表示。
static str
cnn_tensor_hex(const float *values, size_t count, str *hex)
{
	*hex = NULL;
	uint8_t bytes[CNN_PARAMETER_COUNT * 4];
	if (count > CNN_PARAMETER_COUNT ||
	    !cnn_f32_encode_le(values, count, bytes, count * 4))
		throw(SQL, "cnn.storage", SQLSTATE(22000) "Cannot encode tensor values");
	str encoded = GDKmalloc(count * 8 + 1);
	if (encoded == NULL)
		throw(SQL, "cnn.storage", SQLSTATE(HY013) MAL_MALLOC_FAIL);
	static const char digits[] = "0123456789abcdef";
	for (size_t i = 0; i < count * 4; i++) {
		encoded[2 * i] = digits[bytes[i] >> 4];
		encoded[2 * i + 1] = digits[bytes[i] & 15];
	}
	encoded[count * 8] = '\0';
	*hex = encoded;
	return MAL_SUCCEED;
}

static str
cnn_quote_tensor(const CnnTensorSpec *spec, str *quoted)
{
	const char *fields[] = {
		spec->layer_id, spec->tensor_name, spec->tensor_role,
		spec->shape, spec->layout
	};
	for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++) {
		str msg = cnn_quote(&quoted[i], fields[i]);
		if (msg != MAL_SUCCEED)
			return msg;
	}
	return MAL_SUCCEED;
}

//state == NULL 表示参数，否则仅允许内部固定状态名 m 或 v。
static str
cnn_write_tensor(Client cntxt, const str id, int version,
		 const CnnTensorSpec *spec, const char *state, const float *values)
{
	str quoted[5] = {NULL}, hex = NULL, query = NULL;
	str msg = cnn_quote_tensor(spec, quoted);
	if (msg == MAL_SUCCEED)
		msg = cnn_tensor_hex(values, spec->count, &hex);
	if (msg == MAL_SUCCEED && state == NULL)
		msg = cnn_queryf(&query,
			"INSERT INTO cnn.ModelTensor (ModelId, Version, LayerId, TensorName, "
			"TensorRole, Shape, Layout, DType, TensorValue) "
			"VALUES (%s, %d, %s, %s, %s, %s, %s, 'float32', X'%s');",
			id, version, quoted[0], quoted[1], quoted[2], quoted[3], quoted[4], hex);
	else if (msg == MAL_SUCCEED)
		msg = cnn_queryf(&query,
			"INSERT INTO cnn.OptimizerTensor (ModelId, Version, LayerId, TensorName, "
			"StateName, Shape, DType, TensorValue) "
			"VALUES (%s, %d, %s, %s, '%s', %s, 'float32', X'%s');",
			id, version, quoted[0], quoted[1], state, quoted[3], hex);
	if (msg == MAL_SUCCEED)
		msg = SQLstatementIntern(cntxt, query, "cnn.write_tensor", true, false, NULL);
	cnn_free_strings(quoted, 5);
	GDKfree(hex);
	GDKfree(query);
	return msg;
}

static str
cnn_read_blob(Client cntxt, const char *query, float *values, size_t count)
{
	res_table *result = NULL;
	BAT *b = NULL;
	str msg = SQLstatementIntern(cntxt, query, "cnn.read_tensor", true, false, &result);
	if (msg == MAL_SUCCEED)
		msg = cnn_single_bat(result, TYPE_blob, &b);
	if (msg == MAL_SUCCEED) {
		BATiter iterator = bat_iterator(b);
		const blob *payload = BUNtvar(&iterator, 0);
		if (payload == NULL || is_blob_nil(payload) ||
		    payload->nitems != count * 4 ||
		    !cnn_f32_decode_le(payload->data, payload->nitems, values, count))
			msg = createException(SQL, "cnn.storage", SQLSTATE(22000) "Invalid tensor payload");
		bat_iterator_end(&iterator);
	}
	if (b != NULL)
		BBPunfix(b->batCacheid);
	res_tables_destroy(result);
	return msg;
}

static str
cnn_read_tensor(Client cntxt, const str id, int version,
		const CnnTensorSpec *spec, const char *state, float *values)
{
	str quoted[5] = {NULL}, query = NULL;
	str msg = cnn_quote_tensor(spec, quoted);
	if (msg == MAL_SUCCEED && state == NULL)
		msg = cnn_queryf(&query,
			"SELECT TensorValue FROM cnn.ModelTensor WHERE ModelId = %s AND Version = %d "
			"AND LayerId = %s AND TensorName = %s AND TensorRole = %s "
			"AND Shape = %s AND Layout = %s AND DType = 'float32';",
			id, version, quoted[0], quoted[1], quoted[2], quoted[3], quoted[4]);
	else if (msg == MAL_SUCCEED)
		msg = cnn_queryf(&query,
			"SELECT TensorValue FROM cnn.OptimizerTensor WHERE ModelId = %s AND Version = %d "
			"AND LayerId = %s AND TensorName = %s AND StateName = '%s' "
			"AND Shape = %s AND DType = 'float32';",
			id, version, quoted[0], quoted[1], state, quoted[3]);
	if (msg == MAL_SUCCEED)
		msg = cnn_read_blob(cntxt, query, values, spec->count);
	cnn_free_strings(quoted, 5);
	GDKfree(query);
	return msg;
}

static str
cnn_validate_adam(const CnnAdamState *adam, bool initial)
{
	for (size_t i = 0; i < CNN_PARAMETER_COUNT; i++) {
		if (initial) {
			if (adam->m[i] != 0.0f || adam->v[i] != 0.0f)
				throw(SQL, "cnn.storage", SQLSTATE(22000) "Initial Adam states must be finite and zero");
		} else if (!isfinite(adam->m[i]) || !isfinite(adam->v[i]) || adam->v[i] < 0.0f) {
			throw(SQL, "cnn.storage", SQLSTATE(22000) "Adam states must be finite and second moments non-negative");
		}
	}
	return MAL_SUCCEED;
}

static str
cnn_validate_parameters(const CnnModel *model)
{
	for (size_t i = 0; i < CNN_PARAMETER_COUNT; i++) {
		if (!isfinite(model->parameter[i]))
			throw(SQL, "cnn.storage", SQLSTATE(22000) "Model parameters must be finite");
	}
	return MAL_SUCCEED;
}

static str
cnn_write_snapshot(Client cntxt, const CnnInitialModelInfo *info, int epoch,
			const CnnModel *model, const CnnAdamState *adam, const str *quoted)
{
	str msg = cnn_write_model(cntxt, info, epoch, model, quoted);
	// 外键要求先有父模型，再有参数，最后才有对应优化器状态。
	for (size_t i = 0; msg == MAL_SUCCEED && i < CNN_TENSOR_COUNT; i++) {
		const CnnTensorSpec *spec = &cnn_tensor_specs[i];
		msg = cnn_write_tensor(cntxt, quoted[CNN_QUOTED_ID], info->version,
				       spec, NULL, model->parameter + spec->offset);
		if (msg == MAL_SUCCEED && info->use_adam)
			msg = cnn_write_tensor(cntxt, quoted[CNN_QUOTED_ID], info->version,
					       spec, "m", adam->m + spec->offset);
		if (msg == MAL_SUCCEED && info->use_adam)
			msg = cnn_write_tensor(cntxt, quoted[CNN_QUOTED_ID], info->version,
					       spec, "v", adam->v + spec->offset);
	}
	return msg;
}

str
cnn_model_save_initial(Client cntxt, mvc *m, const CnnInitialModelInfo *info,
		       const CnnModel *model, const CnnAdamState *adam)
{
	str quoted[CNN_QUOTED_COUNT] = {NULL};
	str msg = cnn_validate_info(info, true);
	if (msg != MAL_SUCCEED)
		goto cleanup;
	if (model == NULL || (info->use_adam ? adam == NULL : adam != NULL)) {
		msg = createException(SQL, "cnn.storage", SQLSTATE(22023) "Invalid model or optimizer state argument");
		goto cleanup;
	}
	if (model->step != 0 || model->seed > (uint64_t) GDK_lng_max) {
		msg = createException(SQL, "cnn.storage", SQLSTATE(22023) "Initial step must be zero and seed must fit SQL BIGINT");
		goto cleanup;
	}
	msg = cnn_validate_parameters(model);
	if (msg == MAL_SUCCEED && info->use_adam)
		msg = cnn_validate_adam(adam, true);
	if (msg != MAL_SUCCEED)
		goto cleanup;
	msg = cnn_prepare(cntxt, m, true, info->use_adam, false);
	if (msg == MAL_SUCCEED)
		msg = cnn_quote_info(info, quoted);
	if (msg != MAL_SUCCEED)
		goto cleanup;
	lng existing = 0;
	msg = cnn_count(cntxt, "Model", quoted[CNN_QUOTED_ID], 0, &existing);
	if (msg != MAL_SUCCEED)
		goto cleanup;
	if (existing != 0) {
		msg = createException(SQL, "cnn.storage", SQLSTATE(23505) "ModelId already exists; initial state is not overwritten");
		goto cleanup;
	}
	msg = cnn_write_snapshot(cntxt, info, 0, model, adam, quoted);
	if (msg == MAL_SUCCEED)
		msg = cnn_check_counts(cntxt, quoted[CNN_QUOTED_ID], info->version, info->use_adam, 0);
cleanup:
	cnn_free_strings(quoted, CNN_QUOTED_COUNT);
	return cnn_storage_result(m, msg);
}

str
cnn_model_load_initial(Client cntxt, mvc *m, const CnnInitialModelInfo *info,
		       CnnModel *model, CnnAdamState *adam)
{
	str quoted[CNN_QUOTED_COUNT] = {NULL};
	CnnModel temporary = {0};
	CnnAdamState temporary_adam = {0};
	str msg = cnn_validate_info(info, true);
	if (msg != MAL_SUCCEED)
		goto cleanup;
	if (model == NULL || (info->use_adam ? adam == NULL : adam != NULL)) {
		msg = createException(SQL, "cnn.storage", SQLSTATE(22023) "Invalid model or optimizer output argument");
		goto cleanup;
	}
	msg = cnn_prepare(cntxt, m, false, info->use_adam, false);
	if (msg == MAL_SUCCEED)
		msg = cnn_quote_info(info, quoted);
	if (msg == MAL_SUCCEED)
		msg = cnn_read_model(cntxt, info, quoted, &temporary.seed);
	if (msg == MAL_SUCCEED)
		msg = cnn_check_counts(cntxt, quoted[CNN_QUOTED_ID], info->version, info->use_adam, 0);
	for (size_t i = 0; msg == MAL_SUCCEED && i < CNN_TENSOR_COUNT; i++) {
		const CnnTensorSpec *spec = &cnn_tensor_specs[i];
		msg = cnn_read_tensor(cntxt, quoted[CNN_QUOTED_ID], info->version,
				      spec, NULL, temporary.parameter + spec->offset);
		if (msg == MAL_SUCCEED && info->use_adam)
			msg = cnn_read_tensor(cntxt, quoted[CNN_QUOTED_ID], info->version,
					      spec, "m", temporary_adam.m + spec->offset);
		if (msg == MAL_SUCCEED && info->use_adam)
			msg = cnn_read_tensor(cntxt, quoted[CNN_QUOTED_ID], info->version,
					      spec, "v", temporary_adam.v + spec->offset);
	}
	if (msg == MAL_SUCCEED && info->use_adam)
		msg = cnn_validate_adam(&temporary_adam, true);
	// 完整恢复成功后才发布；失败不会污染调用者现有的工作参数。
	if (msg == MAL_SUCCEED) {
		*model = temporary;
		if (info->use_adam)
			*adam = temporary_adam;
	}
cleanup:
	cnn_free_strings(quoted, CNN_QUOTED_COUNT);
	return cnn_storage_result(m, msg);
}

str
cnn_model_load_trained(Client cntxt, mvc *m, const CnnInitialModelInfo *expected,
		       CnnModel *model, int *version)
{
	static const CnnStorageTable tables[] = {
		{"model", false},
		{"modeltensor", false},
	};
	str quoted[CNN_QUOTED_MODEL_COUNT] = {NULL};
	CnnModel temporary = {0};
	int trained_version = 0;
	lng tensor_count = 0;
	str msg = cnn_validate_model_info(expected);
	if (msg != MAL_SUCCEED)
		goto cleanup;
	if (model == NULL || version == NULL) {
		msg = createException(SQL, "cnn.storage", SQLSTATE(22023) "Model and version outputs are required");
		goto cleanup;
	}
	msg = cnn_prepare_tables(cntxt, m, tables, sizeof(tables) / sizeof(tables[0]));
	if (msg == MAL_SUCCEED)
		msg = cnn_quote_model_info(expected, quoted);
	if (msg == MAL_SUCCEED)
		msg = cnn_find_trained_version(cntxt, quoted, &trained_version);
	if (msg == MAL_SUCCEED)
		msg = cnn_count(cntxt, "ModelTensor", quoted[CNN_QUOTED_ID], trained_version, &tensor_count);
	if (msg == MAL_SUCCEED && tensor_count != CNN_TENSOR_COUNT)
		msg = createException(SQL, "cnn.storage", SQLSTATE(22000) "Latest trained model version has an unexpected parameter tensor count");
	for (size_t i = 0; msg == MAL_SUCCEED && i < CNN_TENSOR_COUNT; i++) {
		const CnnTensorSpec *spec = &cnn_tensor_specs[i];
		msg = cnn_read_tensor(cntxt, quoted[CNN_QUOTED_ID], trained_version,
				      spec, NULL, temporary.parameter + spec->offset);
	}
	if (msg == MAL_SUCCEED) {
		*model = temporary;
		*version = trained_version;
	}
cleanup:
	cnn_free_strings(quoted, CNN_QUOTED_MODEL_COUNT);
	return cnn_storage_result(m, msg);
}

str
cnn_model_check_trainable(Client cntxt, mvc *m, const CnnInitialModelInfo *info)
{
	str id = NULL;
	lng versions = 0;
	str msg = cnn_validate_info(info, true);
	if (msg == MAL_SUCCEED)
		msg = cnn_prepare(cntxt, m, true, info->use_adam, true);
	if (msg == MAL_SUCCEED)
		msg = cnn_quote(&id, info->model_id);
	if (msg == MAL_SUCCEED)
		msg = cnn_count(cntxt, "Model", id, 0, &versions);
	if (msg == MAL_SUCCEED && versions > 1)
		msg = createException(SQL, "cnn.storage", SQLSTATE(23505) "ModelId already has trained versions");
	GDKfree(id);
	return cnn_storage_result(m, msg);
}

static str
cnn_validate_metric(const CnnEpochMetric *metric)
{
	if (metric == NULL)
		throw(SQL, "cnn.storage", SQLSTATE(22023) "Expected one training metric for the epoch");
	if (!isfinite(metric->loss) || metric->loss < 0.0 ||
	    !isfinite(metric->accuracy) || metric->accuracy < 0.0 || metric->accuracy > 1.0 ||
	    !isfinite(metric->f1) || metric->f1 < 0.0 || metric->f1 > 1.0 ||
	    !isfinite(metric->runtime) || metric->runtime < 0.0)
		throw(SQL, "cnn.storage", SQLSTATE(22023) "Invalid epoch metric values");
	return MAL_SUCCEED;
}

enum { CNN_METRIC_NUMBER_SIZE = 32 };

/* 用整数尾数和指数构造 DOUBLE 字面量，省略本地小数分隔符。 */
static bool
cnn_format_metric_double(char out[CNN_METRIC_NUMBER_SIZE], double value)
{
	char scientific[64];
	int length = snprintf(scientific, sizeof(scientific), "%.16e", value == 0.0 ? 0.0 : value);
	if (length < 0 || (size_t) length >= sizeof(scientific))
		return false;
	char *exponent = strrchr(scientific, 'e');
	if (exponent == NULL || exponent - scientific < 18)
		return false;
	length = snprintf(out, CNN_METRIC_NUMBER_SIZE, "%c%.16se%d",
			  scientific[0], exponent - 16, atoi(exponent + 1) - 16);
	return length > 0 && length < CNN_METRIC_NUMBER_SIZE;
}

static str
cnn_write_metric(Client cntxt, const str id, int version, int epoch, const CnnEpochMetric *metric)
{
	char loss[CNN_METRIC_NUMBER_SIZE], accuracy[CNN_METRIC_NUMBER_SIZE];
	char f1[CNN_METRIC_NUMBER_SIZE], runtime[CNN_METRIC_NUMBER_SIZE];
	if (!cnn_format_metric_double(loss, metric->loss) ||
	    !cnn_format_metric_double(accuracy, metric->accuracy) ||
	    !cnn_format_metric_double(f1, metric->f1) ||
	    !cnn_format_metric_double(runtime, metric->runtime))
		throw(SQL, "cnn.storage", SQLSTATE(22000) "Cannot format epoch metric");

	str query = NULL;
	str msg = cnn_queryf(&query,
		"INSERT INTO cnn.TrainingMetric (ModelId, Version, Epoch, Split, Loss, Accuracy, F1, Runtime) "
			"VALUES (%s, %d, %d, 'train', %s, %s, %s, %s);",
			id, version, epoch, loss, accuracy, f1, runtime);
	if (msg == MAL_SUCCEED)
		msg = SQLstatementIntern(cntxt, query, "cnn.write_metric", true, false, NULL);
	GDKfree(query);
	return msg;
}

str
cnn_model_save_epoch(Client cntxt, mvc *m, const CnnInitialModelInfo *info, int epoch,
			     const CnnModel *model, const CnnAdamState *adam, const CnnEpochMetric *metric)
{
	str quoted[CNN_QUOTED_COUNT] = {NULL};
	str msg = cnn_validate_info(info, false);
	if (msg != MAL_SUCCEED)
		goto cleanup;
	if (epoch < 1 || epoch >= GDK_int_max || info->version != epoch + 1 ||
	    model == NULL || model->step < (uint64_t) epoch ||
	    model->step > (uint64_t) GDK_lng_max || model->seed > (uint64_t) GDK_lng_max ||
	    (info->use_adam ? adam == NULL : adam != NULL)) {
		msg = createException(SQL, "cnn.storage", SQLSTATE(22023) "Invalid trained snapshot metadata or state");
		goto cleanup;
	}
	msg = cnn_validate_metric(metric);
	if (msg == MAL_SUCCEED)
		msg = cnn_validate_parameters(model);
	if (msg == MAL_SUCCEED && info->use_adam)
		msg = cnn_validate_adam(adam, false);
	if (msg == MAL_SUCCEED)
		msg = cnn_prepare(cntxt, m, true, info->use_adam, true);
	if (msg == MAL_SUCCEED)
		msg = cnn_quote_info(info, quoted);
	if (msg == MAL_SUCCEED)
		msg = cnn_write_snapshot(cntxt, info, epoch, model, adam, quoted);
	if (msg == MAL_SUCCEED)
		msg = cnn_write_metric(cntxt, quoted[CNN_QUOTED_ID], info->version, epoch, metric);
	if (msg == MAL_SUCCEED)
		msg = cnn_check_counts(cntxt, quoted[CNN_QUOTED_ID], info->version, info->use_adam, 1);

cleanup:
	cnn_free_strings(quoted, CNN_QUOTED_COUNT);
	return cnn_storage_result(m, msg);
}
