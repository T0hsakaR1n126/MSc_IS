/*
 * SPDX-License-Identifier: MPL-2.0
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0.  If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * For copyright information, see the file debian/copyright.
 */

#ifndef SQL_CNN_TRAIN_H
#define SQL_CNN_TRAIN_H

#include "sql_cnn_storage.h"

#define CNN_TRAIN_DATA_QUERY \
	"SELECT ImageId FROM cnn.TrainingImage WHERE Split = 'train' ORDER BY ImageId"

str cnn_train_initialized_model(Client cntxt, mvc *m, const CnnInitialModelInfo *info,
	int epochs, size_t batch_size, double learning_rate);

#endif
