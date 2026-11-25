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

-- 每行保存一个 64×64 通道的原始 uint8 字节，引用使用持久化主键。
-- 显式使用 BIGINT 序列；编号可以有空洞，不依赖内存地址或 BAT 标识。
CREATE SEQUENCE cnn.image_channel_ref_seq
    AS BIGINT START WITH 1;

CREATE TABLE cnn.ImageChannel (
    ChannelRef BIGINT NOT NULL
        DEFAULT NEXT VALUE FOR cnn.image_channel_ref_seq,
    ChannelValue BLOB NOT NULL,
    CONSTRAINT cnn_image_channel_pk PRIMARY KEY (ChannelRef),
    CONSTRAINT cnn_image_channel_length_ck
        CHECK (OCTET_LENGTH(ChannelValue) = 4096)
);

-- 当前原型固定使用 64×64 RGB 图片。三个引用分别指向 R、G、B 通道。
-- 图片导入需要保证通道语义正确，并在同一事务内保存通道和图片记录。
CREATE TABLE cnn.TrainingImage (
    ImageId BIGINT NOT NULL CHECK (ImageId >= 0),
    Split VARCHAR(16) NOT NULL
        CHECK (Split IN ('train', 'validation', 'test')),
    Label INTEGER NOT NULL CHECK (Label IN (0, 1)),
    Height INTEGER NOT NULL CHECK (Height = 64),
    Width INTEGER NOT NULL CHECK (Width = 64),
    Channels INTEGER NOT NULL CHECK (Channels = 3),
    RRef BIGINT NOT NULL,
    GRef BIGINT NOT NULL,
    BRef BIGINT NOT NULL,
    CONSTRAINT cnn_training_image_pk PRIMARY KEY (ImageId),
    CONSTRAINT cnn_training_image_r_fk
        FOREIGN KEY (RRef) REFERENCES cnn.ImageChannel (ChannelRef),
    CONSTRAINT cnn_training_image_g_fk
        FOREIGN KEY (GRef) REFERENCES cnn.ImageChannel (ChannelRef),
    CONSTRAINT cnn_training_image_b_fk
        FOREIGN KEY (BRef) REFERENCES cnn.ImageChannel (ChannelRef)
);

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

-- 当前 cnn_train 每个 epoch 保存一条 train 指标。Split 仍用于查询历史指标。
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
        REFERENCES cnn.Model (ModelId, Version)
);

-- 创建并保存固定 CNN 的初始化版本；不执行训练。
-- 同名模型已存在时返回错误，避免重新初始化覆盖已有参数。
-- init_seed 使用 BIGINT；SEED 是保留词，因此参数使用此名称。
CREATE PROCEDURE sys.cnn_create(
    model_id STRING,
    optimizer STRING,
    init_seed BIGINT
)
EXTERNAL NAME sql.cnn_create;

-- 从初始化模型开始训练。data_query 支持固定的训练集 ImageId 查询。
-- 每轮保存参数、优化器状态与训练指标。验证集和测试集由 sys.cnn_predict 读取。
CREATE PROCEDURE sys.cnn_train(
    model_id STRING,
    data_query STRING,
    epochs INTEGER,
    batch_size INTEGER,
    optimizer STRING,
    learning_rate DOUBLE
)
EXTERNAL NAME sql.cnn_train;
