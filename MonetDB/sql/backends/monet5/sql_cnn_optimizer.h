/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#ifndef SQL_CNN_OPTIMIZER_H
#define SQL_CNN_OPTIMIZER_H

#include "gdk_cnn_optimizer.h"
#include "gdk_cnn_backward.h"
#include "sql_cnn_model.h"

/*
 * 将六组参数梯度按已有模型布局打包，再调用 GDK 的普通 SGD。
 * forward、backward、backward->head 必须 ready=true，batch 正数且相同。
 * 调用者必须保证它们属于当前 model 的同一次前向/反向，期间未更新
 * 模型或覆盖缓存；数量和 ready 检查无法自动证明这种对应关系。
 * model、各工作区应为独立对象。由上层训练配置决定使用哪种优化器。
 *
 * 成功后内存参数更新，step 加一，并使三个 ready 均失效，不能重复
 * 消费同一梯度；seed、梯度数值、缓存指针和 Adam 状态不变。
 * 失败时参数、step、seed 和三个 ready 均不变，可修正学习率后重试。
 * step 不得超过持久化格式的非负 BIGINT 范围；不分配堆内存、不执行
 * SQL，也不保存训练后的模型。这一步仅更新内存模型，不是完整训练。
 */
CnnOpsResult cnn_model_sgd_step(CnnModel *model, CnnForwardWorkspace *forward, CnnBackwardWorkspace *backward, double learning_rate);

#endif /* SQL_CNN_OPTIMIZER_H */
