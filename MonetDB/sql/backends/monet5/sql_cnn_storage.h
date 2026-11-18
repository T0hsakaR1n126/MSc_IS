/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#ifndef SQL_CNN_STORAGE_H
#define SQL_CNN_STORAGE_H

#include "sql_monet_backend.h"
#include "sql_cnn_model.h"

/*
 * 当前只保存和读取初始化版本：Version=1、Status=initialized、Epoch=Step=0。
 * 四个描述字符串由调用者按约定生成；存储层保存原文，不解析 JSON。
 * 读取时使用同一结构描述预期配置，逐项检查数据库记录是否兼容。
 * 完整训练 checkpoint 的保存和恢复后续再接入。
 */
typedef struct {
	const char *model_id;
	int version;
	const char *architecture;
	const char *preprocess_spec;
	const char *class_labels;
	const char *training_spec;
	bool use_adam;
} CnnInitialModelInfo;

/*
 * cntxt 和 m 必须属于同一个 SQL 客户端上下文。
 * 普通 SGD 的 adam 必须为 NULL；Adam 的 adam 必须指向有效状态。
 * save 拒绝已有 ModelId；step 和 Adam 初始状态必须为零，seed 必须能
 * 用非负 SQL BIGINT 表示。参数以 float32 小端格式逐张量保存。
 *
 * 两个函数都沿用调用者事务，不自行 COMMIT 或 ROLLBACK。
 * 失败会返回 MonetDB 异常字符串，并将活动事务标为失败。
 * 成功仅表示当前事务内操作完成，持久化仍需外层提交。
 */
str cnn_model_save_initial(Client cntxt, mvc *m, const CnnInitialModelInfo *info, const CnnModel *model, const CnnAdamState *adam);

/* 全部读取和校验成功后才写入输出对象；失败时输出内容保持原样。 */
str cnn_model_load_initial(Client cntxt, mvc *m, const CnnInitialModelInfo *expected, CnnModel *model, CnnAdamState *adam);

#endif /* SQL_CNN_STORAGE_H */
