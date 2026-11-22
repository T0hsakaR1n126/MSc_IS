/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#ifndef SQL_CNN_BACKWARD_H
#define SQL_CNN_BACKWARD_H

#include "gdk_cnn_backward.h"
#include "sql_cnn_batch.h"
#include "sql_cnn_model.h"

/*
 * 只适配 CE -> FC -> GAP 这段反向，输出分类层参数梯度和 d_pool2。
 * model、batch 标签顺序必须对应 forward 最近一次成功前向，期间不能
 * 更新模型参数或用其他 batch 覆盖前向缓存；本接口只能检查数量匹配。
 * backward 必须先按相同 batch size 初始化。
 * 不读取数据库、不修改模型或 seed/step，不更新权重，也不保存梯度。
 * 失败时 backward->ready=false；前向缓存仍由调用者持有。
 */
CnnOpsResult cnn_model_head_backward(const CnnModel *model, const CnnBatch *batch, const CnnForwardWorkspace *forward, CnnHeadBackwardWorkspace *backward);

#endif /* SQL_CNN_BACKWARD_H */
