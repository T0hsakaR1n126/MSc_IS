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
#include "sql_cnn_storage.h"

#include <ctype.h>
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
 * 固定网络的初始化描述。未来加载初始化模型时应复用同一生成函数，
 * 因为当前存储接口按原始描述文本检查兼容性，不解析 JSON。
 * 这里记录输入约定，实际图片预处理和训练仍待实现。
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
sql_cnn_create_initial(Client cntxt, mvc *m, const char *model_id,
		       const char *optimizer, lng init_seed)
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

	/* 位置 0 是 void 返回位置；SQL 的三个输入对应位置 1、2、3。 */
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

	if (is_int_nil(batch_size) || batch_size <= 0)
		throw(SQL, "sql.cnn_train", SQLSTATE(22023) "batch_size must be a positive integer");

	if (sql_cnn_blank(optimizer) || (strcmp(optimizer, "sgd") != 0 && strcmp(optimizer, "adam") != 0))
		throw(SQL, "sql.cnn_train", SQLSTATE(22023) "optimizer must be 'sgd' or 'adam'");

	if (is_dbl_nil(learning_rate) || !isfinite(learning_rate) || learning_rate <= 0)
		throw(SQL, "sql.cnn_train", SQLSTATE(22023) "learning_rate must be finite and positive");

	/*
	 * 后续在这里接入：
	 * 加载模型 -> 执行数据查询 -> 构建 batch
	 * -> 前向 -> 损失 -> 反向 -> 更新参数。
	 *
	 * ****当前骨架不执行 data_query，也不修改模型！！
	 */
	throw(SQL, "sql.cnn_train", SQLSTATE(0A000) "CNN training is not implemented yet");
}
