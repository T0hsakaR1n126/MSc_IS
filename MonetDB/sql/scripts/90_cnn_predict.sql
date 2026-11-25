-- SPDX-License-Identifier: MPL-2.0
--
-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0.  If a copy of the MPL was not distributed with this
-- file, You can obtain one at https://mozilla.org/MPL/2.0/.
--
-- For copyright information, see the file debian/copyright.

-- 使用最新已训练版本，对指定划分中按 ImageId 排序的图片执行前向计算。
CREATE FUNCTION sys.cnn_predict(
    model_id STRING,
    image_split STRING,
    max_images INTEGER
)
RETURNS TABLE (
    ModelVersion INTEGER,
    ImageId BIGINT,
    Label INTEGER,
    PredictedLabel INTEGER,
    CatProbability DOUBLE,
    DogProbability DOUBLE
)
EXTERNAL NAME sql.cnn_predict;
