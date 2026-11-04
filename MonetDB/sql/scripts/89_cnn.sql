-- SPDX-License-Identifier: MPL-2.0
--
-- This Source Code Form is subject to the terms of the Mozilla Public
-- License, v. 2.0.  If a copy of the MPL was not distributed with this
-- file, You can obtain one at https://mozilla.org/MPL/2.0/.
--
-- For copyright information, see the file debian/copyright.

-- 本脚本在新数据库首次初始化 SQL 时执行。已有数据库需要单独安装一次或迁移，不是可重复执行的幂等启动脚本。
-- 只创建空的存储表，不会初始化具体模型记录、参数张量或优化器状态张量。
CREATE SCHEMA cnn;

-- Version 标识一次保存的模型状态。SEED 是 SQL 保留词，因此该列使用带双引号的小写名称。
-- TrainingSpec 记录批次配置、优化器及其超参数、初始化与洗牌算法、
-- 数据集及划分标识，以及复现训练所需的执行设置。
CREATE TABLE cnn.Model (
    ModelId VARCHAR(128) NOT NULL,
    Version INTEGER NOT NULL CHECK (Version > 0),
    Architecture CLOB NOT NULL,
    Status VARCHAR(32) NOT NULL,
    InputShape VARCHAR(128) NOT NULL,
    InputLayout VARCHAR(32) NOT NULL,
    DType VARCHAR(32) NOT NULL,
    PreprocessSpec CLOB NOT NULL,
    ClassLabels CLOB NOT NULL,
    TrainingSpec CLOB NOT NULL,
    Epoch INTEGER NOT NULL DEFAULT 0 CHECK (Epoch >= 0),
    Step BIGINT NOT NULL DEFAULT 0 CHECK (Step >= 0),
    "seed" BIGINT NOT NULL,
    CONSTRAINT cnn_model_pk PRIMARY KEY (ModelId, Version)
);

-- 每行保存一个完整张量，由模型、版本、层和张量名称共同标识。
-- Shape、Layout 和 DType 描述如何解释 TensorValue 中的二进制数据。
CREATE TABLE cnn.ModelTensor (
    ModelId VARCHAR(128) NOT NULL,
    Version INTEGER NOT NULL,
    LayerId VARCHAR(64) NOT NULL,
    TensorName VARCHAR(64) NOT NULL,
    TensorRole VARCHAR(32) NOT NULL,
    Shape VARCHAR(128) NOT NULL,
    Layout VARCHAR(32) NOT NULL,
    DType VARCHAR(32) NOT NULL,
    TensorValue BLOB NOT NULL,
    CONSTRAINT cnn_model_tensor_pk
        PRIMARY KEY (ModelId, Version, LayerId, TensorName),
    CONSTRAINT cnn_model_tensor_model_fk
        FOREIGN KEY (ModelId, Version)
        REFERENCES cnn.Model (ModelId, Version)
);

-- 优化器状态通过模型、版本、层和张量名称组成的键关联到对应参数。
-- StateName 用于区分不同状态，例如 Adam 的一阶矩和二阶矩。
CREATE TABLE cnn.OptimizerTensor (
    ModelId VARCHAR(128) NOT NULL,
    Version INTEGER NOT NULL,
    LayerId VARCHAR(64) NOT NULL,
    TensorName VARCHAR(64) NOT NULL,
    StateName VARCHAR(32) NOT NULL,
    Shape VARCHAR(128) NOT NULL,
    DType VARCHAR(32) NOT NULL,
    TensorValue BLOB NOT NULL,
    CONSTRAINT cnn_optimizer_tensor_pk
        PRIMARY KEY (ModelId, Version, LayerId, TensorName, StateName),
    CONSTRAINT cnn_optimizer_tensor_parameter_fk
        FOREIGN KEY (ModelId, Version, LayerId, TensorName)
        REFERENCES cnn.ModelTensor (ModelId, Version, LayerId, TensorName)
);

-- 每个 epoch、每个数据集划分保存一条聚合指标记录。
-- Runtime 的单位为秒，Accuracy 和 F1 的取值范围为 [0, 1]。
CREATE TABLE cnn.TrainingMetric (
    ModelId VARCHAR(128) NOT NULL,
    Version INTEGER NOT NULL,
    Epoch INTEGER NOT NULL CHECK (Epoch >= 0),
    Split VARCHAR(32) NOT NULL,
    Loss DOUBLE NOT NULL CHECK (Loss >= 0),
    Accuracy DOUBLE NOT NULL CHECK (Accuracy BETWEEN 0 AND 1),
    F1 DOUBLE NOT NULL CHECK (F1 BETWEEN 0 AND 1),
    Runtime DOUBLE NOT NULL CHECK (Runtime >= 0),
    CONSTRAINT cnn_training_metric_pk
        PRIMARY KEY (ModelId, Version, Epoch, Split),
    CONSTRAINT cnn_training_metric_model_fk
        FOREIGN KEY (ModelId, Version)
        REFERENCES ml.Model (ModelId, Version)
);

-- 将 SQL 入口绑定到原生 MAL/C 实现。当前 C 入口仍为骨架，
-- 对合法训练请求返回 SQLSTATE 0A000，表示训练功能尚未实现。
CREATE PROCEDURE sys.cnn_train(
    model_id STRING,
    data_query STRING,
    epochs INTEGER,
    batch_size INTEGER,
    optimizer STRING,
    learning_rate DOUBLE
)
EXTERNAL NAME sql.cnn_train;
