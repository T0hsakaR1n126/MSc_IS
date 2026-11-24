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
#include "sql_cnn_train.h"
#include "sql_cnn_data.h"
#include "sql_cnn_forward.h"
#include "sql_cnn_backward.h"
#include "sql_cnn_optimizer.h"
#include "sql_execute.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
	int64_t *ids;
	size_t count;
	uint64_t id_hash;
} CnnSplit;

/* The query is always a fixed literal. SQLstatementIntern runs as administrator,
 * so check the original caller's table rights before asking for any IDs. */
static str
cnn_load_split(Client cntxt, mvc *m, const char *query, CnnSplit *split)
{
	sql_schema *schema = mvc_bind_schema(m, "cnn");
	sql_table *table = schema ? mvc_bind_table(m, schema, "trainingimage") : NULL;
	if (table == NULL)
		throw(SQL, "cnn.train", SQLSTATE(42S02) "cnn.TrainingImage is missing");
	if (!table_privs(m, table, PRIV_SELECT))
		throw(SQL, "cnn.train", SQLSTATE(42501) "SELECT privilege required on cnn.TrainingImage");

	res_table *result = NULL;
	BAT *column = NULL;
	int64_t *ids = NULL;
	str msg = SQLstatementIntern(cntxt, query, "cnn.select_images", true, false, &result);
	if (msg != MAL_SUCCEED)
		goto cleanup;
	if (result == NULL || result->next != NULL || result->nr_cols != 1 ||
	    result->cols == NULL || (column = BATdescriptor(result->cols[0].b)) == NULL ||
	    column->ttype != TYPE_lng) {
		msg = createException(SQL, "cnn.train", SQLSTATE(22000) "Expected one BIGINT ImageId column");
		goto cleanup;
	}
	const size_t count = BATcount(column);
	if (count == 0) {
		msg = createException(SQL, "cnn.train", SQLSTATE(22023) "Train, validation and test splits must all contain images");
		goto cleanup;
	}
	if (count > SIZE_MAX / sizeof(*ids)) {
		msg = createException(SQL, "cnn.train", SQLSTATE(22003) "Too many CNN images");
		goto cleanup;
	}
	ids = GDKmalloc(count * sizeof(*ids));
	if (ids == NULL) {
		msg = createException(SQL, "cnn.train", SQLSTATE(HY013) MAL_MALLOC_FAIL);
		goto cleanup;
	}

	/* Record a reproducible fingerprint of the ordered ImageId list, not the
	 * contents of the image files. Encode each ID as eight little-endian bytes. */
	uint64_t id_hash = UINT64_C(14695981039346656037);
	BATiter it = bat_iterator(column);
	for (size_t i = 0; i < count; i++) {
		lng id = *(const lng *) BUNtloc(&it, i);
		if (id < 0 || (i > 0 && id <= ids[i - 1])) {
			msg = createException(SQL, "cnn.train", SQLSTATE(22000) "ImageIds must be sorted, unique and nonnegative");
			break;
		}
		ids[i] = id;
		uint64_t value = (uint64_t) id;
		for (int byte = 0; byte < 8; byte++, value >>= 8)
			id_hash = (id_hash ^ (value & UINT64_C(255))) * UINT64_C(1099511628211);
	}
	bat_iterator_end(&it);
	if (msg == MAL_SUCCEED) {
		*split = (CnnSplit) {ids, count, id_hash};
		ids = NULL;
	}
cleanup:
	GDKfree(ids);
	if (column != NULL)
		BBPunfix(column->batCacheid);
	res_tables_destroy(result);
	return msg;
}

static void
cnn_shuffle(int64_t *ids, size_t count, uint64_t *rng)
{
	for (size_t i = count; i > 1; i--) {
		uint64_t bound = i;
		uint64_t threshold = (UINT64_C(0) - bound) % bound;
		uint64_t draw;
		do {
			draw = cnn_splitmix64_next(rng);
		} while (draw < threshold);
		size_t j = (size_t) (draw % bound);
		int64_t tmp = ids[i - 1];
		ids[i - 1] = ids[j];
		ids[j] = tmp;
	}
}

static str
cnn_run_split(Client cntxt, mvc *m, const int64_t *ids, size_t count,
	size_t batch_size, CnnModel *model, CnnAdamState *adam, double learning_rate,
	bool training, CnnEpochMetric *metric)
{
	const lng started = GDKusec();
	CnnForwardWorkspace forward = {0};
	CnnBackwardWorkspace backward = {0};
	CnnBatch batch = {0};
	size_t tp = 0, tn = 0, fp = 0, fn = 0;
	double sum_loss = 0;
	str msg = MAL_SUCCEED;
	for (size_t offset = 0; offset < count;) {
		if (GDKexiting() || TIMEOUT_TEST(&cntxt->qryctx)) {
			msg = createException(SQL, "cnn.train", SQLSTATE(57014) "CNN training cancelled");
			goto cleanup;
		}
		const size_t remaining = count - offset;
		const size_t actual_batch = remaining < batch_size ? remaining : batch_size;
		if (forward.batch != actual_batch) {
			cnn_forward_workspace_destroy(&forward);
			cnn_backward_workspace_destroy(&backward);
			CnnOpsResult rc = cnn_forward_workspace_init(&forward, actual_batch);
			if (training && rc == CNN_OPS_OK)
				rc = cnn_backward_workspace_init(&backward, actual_batch);
			if (rc != CNN_OPS_OK) {
				msg = createException(SQL, "cnn.train", SQLSTATE(HY000) "Cannot allocate CNN workspace (status %d)", (int) rc);
				goto cleanup;
			}
		}
		msg = cnn_batch_load_ids(cntxt, m, ids + offset, actual_batch, &batch);
		if (msg != MAL_SUCCEED)
			goto cleanup;
		CnnOpsResult rc = cnn_model_forward(model, &batch, &forward);
		if (rc != CNN_OPS_OK) {
			msg = createException(SQL, "cnn.train", SQLSTATE(22000) "CNN forward failed (status %d)", (int) rc);
			goto cleanup;
		}
		for (size_t i = 0; i < actual_batch; i++) {
			bool predicted_dog = forward.logits[2 * i + 1] > forward.logits[2 * i];
			if (batch.labels[i] == 1) {
				if (predicted_dog)
					tp++;
				else
					fn++;
			} else if (predicted_dog) {
				fp++;
			} else {
				tn++;
			}
		}
		double loss = 0;
		if (training) {
			rc = cnn_model_backward(model, &batch, &forward, &backward);
			if (rc == CNN_OPS_OK) {
				loss = backward.head.mean_loss;
				rc = adam ? cnn_model_adam_step(model, adam, &forward, &backward, learning_rate)
					  : cnn_model_sgd_step(model, &forward, &backward, learning_rate);
			}
		} else {
			rc = cnn_softmax_cross_entropy(forward.logits, 2 * actual_batch, batch.labels,
				actual_batch, actual_batch, 2, NULL, 0, &loss);
		}
		if (rc != CNN_OPS_OK) {
			msg = createException(SQL, "cnn.train", SQLSTATE(22000) "CNN %s failed (status %d)",
				training ? "backward or update" : "evaluation", (int) rc);
			goto cleanup;
		}
		sum_loss += loss * (double) actual_batch;
		cnn_batch_destroy(&batch);
		offset += actual_batch;
	}
	metric->loss = sum_loss / (double) count;
	metric->accuracy = (double) (tp + tn) / (double) count;
	/* Macro F1. A class with no actual or predicted samples contributes zero. */
	const double dog_denominator = 2.0 * (double) tp + (double) fp + (double) fn;
	const double cat_denominator = 2.0 * (double) tn + (double) fp + (double) fn;
	metric->f1 = (dog_denominator ? (double) tp / dog_denominator : 0.0) +
		     (cat_denominator ? (double) tn / cat_denominator : 0.0);
	metric->runtime = (double) (GDKusec() - started) / 1000000.0;
cleanup:
	cnn_batch_destroy(&batch);
	cnn_backward_workspace_destroy(&backward);
	cnn_forward_workspace_destroy(&forward);
	return msg;
}

str
cnn_train_initialized_model(Client cntxt, mvc *m, const CnnInitialModelInfo *info,
	int epochs, size_t batch_size, double learning_rate)
{
	static const char *const queries[] = {
		CNN_TRAIN_DATA_QUERY ";",
		"SELECT ImageId FROM cnn.TrainingImage WHERE Split = 'validation' ORDER BY ImageId;",
		"SELECT ImageId FROM cnn.TrainingImage WHERE Split = 'test' ORDER BY ImageId;"
	};
	CnnSplit splits[3] = {0};
	CnnModel model;
	CnnAdamState adam_state;
	CnnAdamState *adam = info->use_adam ? &adam_state : NULL;
	int64_t *shuffled = NULL;
	str msg = cnn_model_check_trainable(cntxt, m, info);
	if (msg == MAL_SUCCEED)
		msg = cnn_model_load_initial(cntxt, m, info, &model, adam);
	for (size_t i = 0; i < 3 && msg == MAL_SUCCEED; i++)
		msg = cnn_load_split(cntxt, m, queries[i], &splits[i]);
	if (msg != MAL_SUCCEED)
		goto cleanup;

	const size_t steps_per_epoch = 1 + (splits[0].count - 1) / batch_size;
	if ((uint64_t) steps_per_epoch > (uint64_t) GDK_lng_max / (uint64_t) epochs) {
		msg = createException(SQL, "cnn.train", SQLSTATE(22003) "Requested number of updates exceeds BIGINT");
		goto cleanup;
	}
	shuffled = GDKmalloc(splits[0].count * sizeof(*shuffled));
	if (shuffled == NULL) {
		msg = createException(SQL, "cnn.train", SQLSTATE(HY013) MAL_MALLOC_FAIL);
		goto cleanup;
	}

	char rate[64];
	int n = snprintf(rate, sizeof(rate), "%.17g", learning_rate);
	if (n < 0 || (size_t) n >= sizeof(rate)) {
		msg = createException(SQL, "cnn.train", SQLSTATE(22000) "Cannot encode learning rate");
		goto cleanup;
	}
	char *decimal_comma = strchr(rate, ',');
	if (decimal_comma)
		*decimal_comma = '.';
	const char *optimizer = info->use_adam
		? "\"name\":\"adam\",\"beta1\":0.9,\"beta2\":0.999,\"epsilon\":1e-8"
		: "\"name\":\"sgd\",\"momentum\":0";
	char training_spec[1024];
	n = snprintf(training_spec, sizeof(training_spec),
		"{\"format_version\":1,\"stage\":\"training\",\"epochs\":%d,\"batch_size\":%zu,"
		"\"optimizer\":{%s,\"learning_rate\":%s},\"shuffle\":\"splitmix64-fisher-yates-v1\","
		"\"metric_f1\":\"macro\",\"dataset_ids\":{\"algorithm\":\"fnv1a64-le\","
		"\"train\":{\"count\":%zu,\"hash\":\"%016" PRIx64 "\"},"
		"\"validation\":{\"count\":%zu,\"hash\":\"%016" PRIx64 "\"},"
		"\"test\":{\"count\":%zu,\"hash\":\"%016" PRIx64 "\"}}}",
		epochs, batch_size, optimizer, rate, splits[0].count, splits[0].id_hash,
		splits[1].count, splits[1].id_hash, splits[2].count, splits[2].id_hash);
	if (n < 0 || (size_t) n >= sizeof(training_spec)) {
		msg = createException(SQL, "cnn.train", SQLSTATE(22000) "Cannot encode training specification");
		goto cleanup;
	}

	CnnInitialModelInfo snapshot = *info;
	snapshot.training_spec = training_spec;
	uint64_t rng = model.seed;
	for (int epoch = 1; epoch <= epochs; epoch++) {
		memcpy(shuffled, splits[0].ids, splits[0].count * sizeof(*shuffled));
		cnn_shuffle(shuffled, splits[0].count, &rng);
		CnnEpochMetric metrics[] = {{.split = "train"}, {.split = "validation"}, {.split = "test"}};
		msg = cnn_run_split(cntxt, m, shuffled, splits[0].count, batch_size, &model, adam, learning_rate, true, &metrics[0]);
		if (msg == MAL_SUCCEED)
			msg = cnn_run_split(cntxt, m, splits[1].ids, splits[1].count, batch_size, &model, NULL, 0, false, &metrics[1]);
		if (msg == MAL_SUCCEED && epoch == epochs)
			msg = cnn_run_split(cntxt, m, splits[2].ids, splits[2].count, batch_size, &model, NULL, 0, false, &metrics[2]);
		if (msg != MAL_SUCCEED)
			goto cleanup;
		snapshot.version = epoch + 1;
		msg = cnn_model_save_epoch(cntxt, m, &snapshot, epoch, &model, adam, metrics, epoch == epochs ? 3 : 2);
		if (msg != MAL_SUCCEED)
			goto cleanup;
	}
cleanup:
	GDKfree(shuffled);
	for (size_t i = 0; i < 3; i++)
		GDKfree(splits[i].ids);
	if (msg != MAL_SUCCEED && m->session->tr != NULL && m->session->tr->active && m->session->status == 0)
		m->session->status = -1;
	return msg;
}
