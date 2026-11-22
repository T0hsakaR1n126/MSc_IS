/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#ifndef GDK_CNN_OPTIMIZER_H
#define GDK_CNN_OPTIMIZER_H

#include "gdk_cnn.h"

/*
 * 普通 SGD：parameters[i] -= learning_rate * gradients[i]。
 * 无 momentum、weight decay，也不再次对已经平均的梯度除以 batch。
 * 参数、梯度均为连续 float32 数组，count 为元素数且必须正数、精确相等。
 * 学习率为正有限 double；候选新值以 double 计算后转回 float32。
 *
 * 先检查全部参数、梯度和候选值；只有全部有限且候选值位于 float32
 * 有限范围内时才进行第二遍写入。检查失败时参数数组保持原样。
 * 参数与梯度数组不得有任何重叠；调用期间由调用者独占参数并保持
 * 梯度不变。本函数不修改梯度、不分配内存，不管理模型 step 或数据库。
 */
gdk_export CnnOpsResult cnn_sgd_step(float *parameters, size_t parameter_count, const float *gradients, size_t gradient_count, double learning_rate);

/*
 * Adam 的参数、梯度、一阶矩 m 和二阶矩 v 都是连续 float32 数组。
 * 四个数组互不重叠，元素数相等且大于零。调用期间参数和状态由调用者
 * 独占，梯度保持不变。step 是即将执行的更新编号，第一次为 1。
 * beta1、beta2 取 [0,1)，学习率和 epsilon 为正有限值。
 *
 * 使用 double 计算矩、偏差修正及参数候选值，epsilon 加在平方根外。
 * v 必须非负。只有全部候选值都在 float32 有限范围内，才写入参数和矩。
 * 失败时三个数组保持原样。本函数不分配内存，直接使用已平均的梯度。
 */
gdk_export CnnOpsResult cnn_adam_step(float *parameters, size_t parameter_count, const float *gradients, size_t gradient_count, float *first_moment, size_t first_count, float *second_moment, size_t second_count, uint64_t step, double learning_rate, double beta1, double beta2, double epsilon);

#endif /* GDK_CNN_OPTIMIZER_H */
