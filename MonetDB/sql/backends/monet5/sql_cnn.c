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
