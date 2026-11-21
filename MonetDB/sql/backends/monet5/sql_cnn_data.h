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

/*
 * 保存一张已经解码为 64×64 RGB uint8 三平面的图片。
 * image 只借用输入数据，调用结束后仍由调用者管理这些通道内存。
 * split 必须是 train / validation / test；本函数不自动划分数据集，
 * 不解码 JPEG/PNG、不 resize，也不根据图片内容执行去重。
 *
 * 原用户需有两张图片表的 SELECT 和 INSERT 权限，并具有 cnn schema
 * 的管理权限（当前 MonetDB 的序列取号要求；通常为管理员或 schema
 * 所有者）。重复 ImageId 返回 23505，不覆盖已有图片。
 * 三个通道和图片记录沿用同一外层事务；
 * 失败标记事务失败，成功仍需外层提交，不自行 COMMIT / ROLLBACK。
 * 通道引用来自持久化 BIGINT 序列；回滚可产生编号空洞，不影响引用。
 */
str cnn_image_store_rgb(Client cntxt, mvc *m, const CnnImageView *image, const char *split);

#endif /* SQL_CNN_DATA_H */
