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
#include "sql_cnn.h"
#include "sql_cnn_data.h"
#include "sql_cnn_forward.h"
#include "sql_cnn_storage.h"
#include "sql_cnn_train.h"
#include "sql_execute.h"

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <string.h>

//判断字符串是否为 NULL、空字符串或纯空白
static bool
sql_cnn_blank(const char *value)
{
	if (value == NULL || strNil(value)) return true;

	while (*value && isspace((unsigned char) *value)) value++;

	return *value == '\0';
}

/*
 * 固定网络的模型描述。创建、训练和推理使用相同描述校验已保存模型。
 */
static void
sql_cnn_initial_info(CnnInitialModelInfo *info, const char *model_id, bool use_adam)
{
	static const char architecture[] =
		"{"
		"\"format_version\":1,"
		"\"name\":\"cat_dog_small_cnn\","
		"\"layers\":["
		"{\"id\":\"conv1\",\"op\":\"conv2d\","
		"\"in_channels\":3,\"out_channels\":4,"
		"\"kernel_size\":[3,3],\"stride\":[1,1],"
		"\"padding\":[1,1],\"dilation\":[1,1],"
		"\"groups\":1,\"bias\":true},"
		"{\"id\":\"relu1\",\"op\":\"relu\"},"
		"{\"id\":\"pool1\",\"op\":\"maxpool2d\","
		"\"kernel_size\":[2,2],\"stride\":[2,2],"
		"\"padding\":[0,0],\"dilation\":[1,1],"
		"\"ceil_mode\":false},"
		"{\"id\":\"conv2\",\"op\":\"conv2d\","
		"\"in_channels\":4,\"out_channels\":8,"
		"\"kernel_size\":[3,3],\"stride\":[1,1],"
		"\"padding\":[1,1],\"dilation\":[1,1],"
		"\"groups\":1,\"bias\":true},"
		"{\"id\":\"relu2\",\"op\":\"relu\"},"
		"{\"id\":\"pool2\",\"op\":\"maxpool2d\","
		"\"kernel_size\":[2,2],\"stride\":[2,2],"
		"\"padding\":[0,0],\"dilation\":[1,1],"
		"\"ceil_mode\":false},"
		"{\"id\":\"gap\",\"op\":\"global_average_pool\","
		"\"axes\":[2,3],\"keepdims\":false},"
		"{\"id\":\"fc\",\"op\":\"linear\","
		"\"in_features\":8,\"out_features\":2,\"bias\":true}"
		"],\"output\":\"logits\""
		"}";

	static const char preprocess[] =
		"{"
		"\"format_version\":1,"
		"\"required_image_shape\":[64,64,3],"
		"\"channel_order\":[\"R\",\"G\",\"B\"],"
		"\"channel_storage\":\"separate_planes\","
		"\"plane_layout\":\"row_major\","
		"\"source_dtype\":\"uint8\","
		"\"batch_layout\":\"NCHW\","
		"\"target_dtype\":\"float32\","
		"\"normalization\":\"divide_by_255\""
		"}";

	static const char labels[] = "[\"cat\",\"dog\"]";

	//尚未绑定训练配置；JSON null 不代表整列是 SQL NULL。
	static const char sgd_training[] =
		"{"
		"\"format_version\":1,\"stage\":\"initialization\","
		"\"initialization\":{\"version\":1,\"rng\":\"splitmix64\","
		"\"conv\":\"he_uniform\",\"fc\":\"xavier_uniform\",\"bias\":\"zeros\"},"
		"\"tensor_encoding\":{\"version\":1,\"dtype\":\"float32\","
		"\"byte_order\":\"little\"},"
		"\"optimizer\":{\"name\":\"sgd\",\"momentum\":0,\"learning_rate\":null},"
		"\"epochs\":null,\"batch_size\":null,\"dataset_manifest\":null,"
		"\"split_spec\":null,\"shuffle\":null,\"execution_settings\":null"
		"}";

	static const char adam_training[] =
		"{"
		"\"format_version\":1,\"stage\":\"initialization\","
		"\"initialization\":{\"version\":1,\"rng\":\"splitmix64\","
		"\"conv\":\"he_uniform\",\"fc\":\"xavier_uniform\",\"bias\":\"zeros\"},"
		"\"tensor_encoding\":{\"version\":1,\"dtype\":\"float32\","
		"\"byte_order\":\"little\"},"
		"\"optimizer\":{\"name\":\"adam\",\"learning_rate\":null,"
		"\"beta1\":0.9,\"beta2\":0.999,\"epsilon\":1e-8},"
		"\"epochs\":null,\"batch_size\":null,\"dataset_manifest\":null,"
		"\"split_spec\":null,\"shuffle\":null,\"execution_settings\":null"
		"}";

	*info = (CnnInitialModelInfo) {
		.model_id = model_id,
		.version = 1,
		.architecture = architecture,
		.preprocess_spec = preprocess,
		.class_labels = labels,
		.training_spec = use_adam ? adam_training : sgd_training,
		.use_adam = use_adam
	};
}

/*
 * 共用的内部创建逻辑。当前由 cnn_create 调用，未来 cnn_train 可复用。
 * 存储模块处理同名模型、权限和失败事务；外层 SQL 调用负责提交。
 */
static str
sql_cnn_create_initial(Client cntxt, mvc *m, const char *model_id, const char *optimizer, lng init_seed)
{
	if (sql_cnn_blank(model_id))
		throw(SQL, "sql.cnn_create", SQLSTATE(22023) "model_id must not be NULL or blank");
	if (sql_cnn_blank(optimizer) ||
	    (strcmp(optimizer, "sgd") != 0 && strcmp(optimizer, "adam") != 0))
		throw(SQL, "sql.cnn_create", SQLSTATE(22023) "optimizer must be 'sgd' or 'adam'");
	if (is_lng_nil(init_seed) || init_seed < 0)
		throw(SQL, "sql.cnn_create", SQLSTATE(22023) "init_seed must be a non-negative BIGINT");

	bool use_adam = strcmp(optimizer, "adam") == 0;
	CnnInitialModelInfo info;
	sql_cnn_initial_info(&info, model_id, use_adam);

	CnnModel model;
	cnn_model_init(&model, (uint64_t) init_seed);

	CnnAdamState adam;
	CnnAdamState *adam_state = NULL;
	if (use_adam) {
		cnn_adam_state_init(&adam);
		adam_state = &adam;
	}

	return cnn_model_save_initial(cntxt, m, &info, &model, adam_state);
}

str
sql_cnn_create(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	mvc *m = NULL;
	str msg = getSQLContext(cntxt, mb, &m, NULL);
	if (msg != MAL_SUCCEED)
		return msg;

	// SQL 的三个输入对应位置 1、2、3。
	const char *model_id = *getArgReference_str(stk, pci, 1);
	const char *optimizer = *getArgReference_str(stk, pci, 2);
	lng init_seed = *getArgReference_lng(stk, pci, 3);

	return sql_cnn_create_initial(cntxt, m, model_id, optimizer, init_seed);
}

str
sql_cnn_train(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	mvc *m = NULL;
	str msg = MAL_SUCCEED;

	msg = getSQLContext(cntxt, mb, &m, NULL);
	if (msg != MAL_SUCCEED)
		return msg;

	//读取参数
	const char *model_id = *getArgReference_str(stk, pci, 1);
	const char *data_query = *getArgReference_str(stk, pci, 2);
	int epochs = *getArgReference_int(stk, pci, 3);
	int batch_size = *getArgReference_int(stk, pci, 4);
	const char *optimizer = *getArgReference_str(stk, pci, 5);
	dbl learning_rate = *getArgReference_dbl(stk, pci, 6);

	//检查输入参数
	if (sql_cnn_blank(model_id))
		throw(SQL, "sql.cnn_train", SQLSTATE(22023) "model_id must not be NULL or blank");

	if (sql_cnn_blank(data_query))
		throw(SQL, "sql.cnn_train",SQLSTATE(22023) "data_query must not be NULL or blank");

	if (is_int_nil(epochs) || epochs <= 0)
		throw(SQL, "sql.cnn_train", SQLSTATE(22023) "epochs must be a positive integer");
	if (epochs == INT_MAX)
		throw(SQL, "sql.cnn_train", SQLSTATE(22003) "epochs exceeds model version range");

	if (is_int_nil(batch_size) || batch_size <= 0)
		throw(SQL, "sql.cnn_train", SQLSTATE(22023) "batch_size must be a positive integer");

	if (sql_cnn_blank(optimizer) || (strcmp(optimizer, "sgd") != 0 && strcmp(optimizer, "adam") != 0))
		throw(SQL, "sql.cnn_train", SQLSTATE(22023) "optimizer must be 'sgd' or 'adam'");

	if (is_dbl_nil(learning_rate) || !isfinite(learning_rate) || learning_rate <= 0)
		throw(SQL, "sql.cnn_train", SQLSTATE(22023) "learning_rate must be finite and positive");

	/* The internal SQL executor elevates privileges. Use data_query only to
	 * select the supported fixed query, never as executable SQL text. */
	while (isspace((unsigned char) *data_query))
		data_query++;
	size_t query_length = strlen(data_query);
	while (query_length && isspace((unsigned char) data_query[query_length - 1]))
		query_length--;
	if (query_length && data_query[query_length - 1] == ';')
		query_length--;
	if (query_length != sizeof(CNN_TRAIN_DATA_QUERY) - 1 ||
	    memcmp(data_query, CNN_TRAIN_DATA_QUERY, query_length) != 0)
		throw(SQL, "sql.cnn_train", SQLSTATE(22023) "Supported data_query for training: " CNN_TRAIN_DATA_QUERY);
	CnnInitialModelInfo info;
	sql_cnn_initial_info(&info, model_id, strcmp(optimizer, "adam") == 0);
	return cnn_train_initialized_model(cntxt, m, &info, epochs, (size_t) batch_size, learning_rate);
}

/* split 已校验为三个固定划分之一，查询只读取排序后的图片 ID。 */
static str
cnn_prediction_ids(Client cntxt, mvc *m, const char *split, int limit, BAT **ids)
{
	sql_schema *schema = mvc_bind_schema(m, "cnn");
	sql_table *table = schema ? mvc_bind_table(m, schema, "trainingimage") : NULL;
	if (table == NULL)
		throw(SQL, "cnn.predict", SQLSTATE(42S02) "cnn.TrainingImage is missing");
	if (!table_privs(m, table, PRIV_SELECT))
		throw(SQL, "cnn.predict", SQLSTATE(42501) "SELECT privilege required on cnn.TrainingImage");

	char query[192];
	int length = snprintf(query, sizeof(query),
		"SELECT ImageId FROM cnn.TrainingImage WHERE Split = '%s' ORDER BY ImageId LIMIT %d;",
		split, limit);
	if (length < 0 || (size_t) length >= sizeof(query))
		throw(SQL, "cnn.predict", SQLSTATE(22000) "Cannot format image selection");

	res_table *result = NULL;
	BAT *column = NULL;
	str msg = SQLstatementIntern(cntxt, query, "cnn.select_prediction_images", true, false, &result);
	if (msg == MAL_SUCCEED &&
	    (result == NULL || result->next != NULL || result->nr_cols != 1 ||
	     result->cols == NULL || (column = BATdescriptor(result->cols[0].b)) == NULL ||
	     column->ttype != TYPE_lng))
		msg = createException(SQL, "cnn.predict", SQLSTATE(22000) "Expected one BIGINT ImageId column");
	res_tables_destroy(result);
	if (msg == MAL_SUCCEED)
		*ids = column;
	else if (column != NULL)
		BBPunfix(column->batCacheid);
	return msg;
}

enum {
	CNN_PREDICTION_COLUMNS = 6,
	CNN_PREDICTION_BATCH_SIZE = 64
};

static str
cnn_predict_batch(Client cntxt, mvc *m, const CnnModel *model, int version,
		  const int64_t *ids, size_t count, BAT **columns)
{
	CnnBatch batch = {0};
	CnnForwardWorkspace workspace = {0};
	str msg = cnn_batch_load_ids(cntxt, m, ids, count, &batch);
	if (msg != MAL_SUCCEED)
		goto cleanup;

	CnnOpsResult rc = cnn_forward_workspace_init(&workspace, batch.count);
	if (rc != CNN_OPS_OK) {
		msg = createException(SQL, "cnn.predict", SQLSTATE(HY000) "Cannot allocate CNN workspace (status %d)", (int) rc);
		goto cleanup;
	}
	rc = cnn_model_forward(model, &batch, &workspace);
	if (rc != CNN_OPS_OK) {
		msg = createException(SQL, "cnn.predict", SQLSTATE(22000) "CNN forward failed (status %d)", (int) rc);
		goto cleanup;
	}

	for (size_t row = 0; row < batch.count && msg == MAL_SUCCEED; row++) {
		const lng id = batch.image_ids[row];
		const int label = batch.labels[row];
		const double cat_logit = workspace.logits[2 * row];
		const double dog_logit = workspace.logits[2 * row + 1];
		const int prediction = dog_logit > cat_logit;
		const double max_logit = fmax(cat_logit, dog_logit);
		const double cat_exp = exp(cat_logit - max_logit);
		const double dog_exp = exp(dog_logit - max_logit);
		const double total = cat_exp + dog_exp;
		const double cat_probability = cat_exp / total;
		const double dog_probability = dog_exp / total;
		const void *values[] = {&version, &id, &label, &prediction, &cat_probability, &dog_probability};
		for (size_t c = 0; c < CNN_PREDICTION_COLUMNS; c++) {
			if (BUNappend(columns[c], values[c], false) != GDK_SUCCEED) {
				msg = createException(SQL, "cnn.predict", SQLSTATE(HY013) MAL_MALLOC_FAIL);
				break;
			}
		}
	}

cleanup:
	cnn_forward_workspace_destroy(&workspace);
	cnn_batch_destroy(&batch);
	return msg;
}

str
sql_cnn_predict(Client cntxt, MalBlkPtr mb, MalStkPtr stk, InstrPtr pci)
{
	mvc *m = NULL;
	str msg = getSQLContext(cntxt, mb, &m, NULL);
	if (msg != MAL_SUCCEED)
		return msg;

	const char *model_id = *getArgReference_str(stk, pci, CNN_PREDICTION_COLUMNS);
	const char *split = *getArgReference_str(stk, pci, CNN_PREDICTION_COLUMNS + 1);
	int limit = *getArgReference_int(stk, pci, CNN_PREDICTION_COLUMNS + 2);
	if (sql_cnn_blank(model_id))
		throw(SQL, "cnn.predict", SQLSTATE(22023) "model_id must not be NULL or blank");
	if (sql_cnn_blank(split) ||
	    (strcmp(split, "train") != 0 && strcmp(split, "validation") != 0 && strcmp(split, "test") != 0))
		throw(SQL, "cnn.predict", SQLSTATE(22023) "image_split must be 'train', 'validation' or 'test'");
	if (limit <= 0)
		throw(SQL, "cnn.predict", SQLSTATE(22023) "max_images must be a positive integer");

	CnnInitialModelInfo info;
	CnnModel model;
	int version;
	sql_cnn_initial_info(&info, model_id, false);
	msg = cnn_model_load_trained(cntxt, m, &info, &model, &version);
	if (msg != MAL_SUCCEED)
		return msg;

	BAT *ids = NULL;
	msg = cnn_prediction_ids(cntxt, m, split, limit, &ids);
	if (msg != MAL_SUCCEED)
		return msg;
	BAT *columns[CNN_PREDICTION_COLUMNS] = {0};
	const int types[CNN_PREDICTION_COLUMNS] = {TYPE_int, TYPE_lng, TYPE_int, TYPE_int, TYPE_dbl, TYPE_dbl};
	const BUN count = BATcount(ids);
	for (size_t c = 0; c < CNN_PREDICTION_COLUMNS; c++) {
		columns[c] = COLnew(0, types[c], count, TRANSIENT);
		if (columns[c] == NULL) {
			msg = createException(SQL, "cnn.predict", SQLSTATE(HY013) MAL_MALLOC_FAIL);
			goto cleanup;
		}
	}
	for (BUN offset = 0; offset < count;) {
		if (GDKexiting() || TIMEOUT_TEST(&cntxt->qryctx)) {
			msg = createException(SQL, "cnn.predict", SQLSTATE(57014) "CNN prediction cancelled");
			goto cleanup;
		}
		int64_t batch_ids[CNN_PREDICTION_BATCH_SIZE];
		const size_t batch_count = MIN(count - offset, CNN_PREDICTION_BATCH_SIZE);
		BATiter it = bat_iterator(ids);
		for (size_t row = 0; row < batch_count; row++)
			batch_ids[row] = *(const lng *) BUNtloc(&it, offset + row);
		bat_iterator_end(&it);

		msg = cnn_predict_batch(cntxt, m, &model, version, batch_ids, batch_count, columns);
		if (msg != MAL_SUCCEED)
			goto cleanup;
		offset += batch_count;
	}
	for (size_t c = 0; c < CNN_PREDICTION_COLUMNS; c++) {
		*getArgReference_bat(stk, pci, c) = columns[c]->batCacheid;
		BBPkeepref(columns[c]);
		columns[c] = NULL;
	}
cleanup:
	BBPunfix(ids->batCacheid);
	for (size_t c = 0; c < CNN_PREDICTION_COLUMNS; c++)
		BBPreclaim(columns[c]);
	return msg;
}
