/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#ifndef SQL_CNN_FORWARD_H
#define SQL_CNN_FORWARD_H

#include "gdk_cnn_forward.h"
#include "sql_cnn_batch.h"
#include "sql_cnn_model.h"

/*
 * 将已加载的固定模型和 batch 适配到 GDK 的完整前向接口。
 * workspace 必须先按 batch->count 初始化；本函数不分配内存。
 * 只读取 model 参数和 batch 输入，不修改它们或模型的 seed、step。
 * 不执行 SQL、不保存参数，也不计算损失；成功后读取 workspace->logits。
 * 失败时 ready=false，中间缓存不可使用，但可重新前向或销毁工作区。
 */
CnnOpsResult cnn_model_forward(const CnnModel *model, const CnnBatch *batch, CnnForwardWorkspace *workspace);

#endif /* SQL_CNN_FORWARD_H */
