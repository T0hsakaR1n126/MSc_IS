/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#ifndef SQL_CNN_DATA_H
#define SQL_CNN_DATA_H

#include "sql_monet_backend.h"
#include "sql_cnn_batch.h"

/*
 * 从固定的 cnn.TrainingImage / cnn.ImageChannel 表读取指定图片并打包。
 * image_ids 的顺序就是输出顺序；重复 ID 允许，并保留对应的重复样本。
 * 任一图片缺失、格式不符或通道无效都会使整个加载失败，不静默丢样本。
 *
 * out 必须先初始化为空：CnnBatch batch = {0};
 * 成功后 batch 拥有独立数据，不再依赖查询结果；失败时 out 保持原样。
 * cntxt 与 m 必须属于同一个 SQL 上下文。函数检查原用户的 SELECT 权限，
 * 沿用调用者事务，不自行 COMMIT / ROLLBACK；失败会标记活动事务失败。
 *
 * 本函数不执行用户 data_query、不抽样、不洗牌，也不按 Split 过滤。
 * 调用者负责从相应数据集划分中选择 ID；本接口只负责有序加载。
 */
str cnn_batch_load_ids(Client cntxt, mvc *m, const int64_t *image_ids, size_t count, CnnBatch *out);

#endif /* SQL_CNN_DATA_H */
