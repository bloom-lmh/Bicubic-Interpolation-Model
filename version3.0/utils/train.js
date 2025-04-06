const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { promisify } = require("util");
const readFile = promisify(fs.readFile);
const readdir = promisify(fs.readdir);

// ===================== 配置参数 =====================
const METADATA_PATH = "./data/train/metadata.json";
const TRAIN_DIR = {
  X_IMAGE: "./data/train/X/",
  X_OFFSET: "./data/train/offset/",
  Y: "./data/train/Y/",
};
const MODEL_DIR = "file://./model/";
const EPOCHS = 100;
const LEARNING_RATE = 1e-4;

// ===================== 数据加载 =====================
// 统一数据加载接口
async function loadDynamicTensor(dirPath, dataType) {
  // 加载元数据
  const metadata = JSON.parse(await readFile(METADATA_PATH));
  const files = await readdir(dirPath);
  const tensorMap = new Map();

  for (const file of files) {
    if (!file.endsWith(".bin")) continue;

    const [id] = file.split(".");
    const buffer = await readFile(path.join(dirPath, file));
    const dataStartOffset = 12; // 头信息占12字节

    // 解析头信息
    const header = {
      height: buffer.readUInt32LE(0),
      width: buffer.readUInt32LE(4),
      channels: buffer.readUInt32LE(8),
    };

    // 关键修复：正确转换字节到 Float32Array
    const dataBuffer = buffer.buffer.slice(
      buffer.byteOffset + dataStartOffset,
      buffer.byteOffset +
        dataStartOffset +
        header.height * header.width * header.channels * 4
    );

    const float32Data = new Float32Array(dataBuffer);

    // 验证元素数量
    console.assert(
      float32Data.length === header.height * header.width * header.channels,
      `数据长度不匹配: 预期 ${
        header.height * header.width * header.channels
      }，实际 ${float32Data.length}`
    );

    tensorMap.set(
      id,
      tf.tensor3d(float32Data, [header.height, header.width, header.channels])
    );
  }

  return tensorMap;
}
// ===================== 模型定义 =====================

function buildDynamicModel() {
  const inputImage = tf.input({ shape: [null, null, 4] });
  const inputOffset = tf.input({ shape: [null, null, 2] });

  // ===================== 修复1：残差块尺寸对齐 =====================
  const x = tf.layers
    .conv2d({
      filters: 32,
      kernelSize: 3,
      padding: "same", // 保持尺寸不变
      activation: "relu",
    })
    .apply(inputImage);

  const res = tf.layers
    .conv2d({
      filters: 32,
      kernelSize: 3,
      padding: "same", // 必须与 x 的 padding 一致
    })
    .apply(x);

  const resOut = tf.layers.add().apply([x, res]);

  // ===================== 修复2：转置卷积尺寸计算 =====================
  const up = tf.layers
    .conv2dTranspose({
      filters: 16,
      kernelSize: 4,
      strides: 4,
      padding: "same", // 确保输出尺寸是输入的4倍
      kernelInitializer: "glorotUniform",
    })
    .apply(resOut);

  // ===================== 其他层保持不变 =====================
  const att = tf.layers
    .conv2d({
      filters: 1,
      kernelSize: 1,
      activation: "sigmoid",
      padding: "same",
    })
    .apply(up);

  const attended = tf.layers.multiply().apply([up, att]);

  // ===================== 修复3：偏移量投影层尺寸对齐 =====================
  const offsetProj = tf.layers
    .conv2d({
      filters: 16,
      kernelSize: 1,
      padding: "same", // 保持与 attended 相同的尺寸
    })
    .apply(inputOffset);

  const merged = tf.layers.concatenate().apply([attended, offsetProj]);

  // ===================== 输出层 =====================
  const output = tf.layers
    .conv2d({
      filters: 16,
      kernelSize: 3,
      padding: "same",
      activation: "tanh",
      kernelInitializer: "glorotUniform",
    })
    .apply(merged);

  return tf.model({ inputs: [inputImage, inputOffset], outputs: output });
}
// ===================== 训练流程 =====================
async function train() {
  // 统一加载接口
  const [imageTensors, offsetTensors, yTensors] = await Promise.all([
    loadDynamicTensor(TRAIN_DIR.X_IMAGE, "X"),
    loadDynamicTensor(TRAIN_DIR.X_OFFSET, "offset"),
    loadDynamicTensor(TRAIN_DIR.Y, "Y"),
  ]);

  // 验证样本一致性
  const ids = Array.from(imageTensors.keys());
  if (!ids.every((id) => offsetTensors.has(id) && yTensors.has(id))) {
    throw new Error("训练样本ID不匹配");
  }

  // 构建模型
  const model = buildDynamicModel();
  model.summary();

  // 编译模型时添加额外指标（示例添加MAE）
  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: "meanSquaredError",
    metrics: ["mae"], // 平均绝对误差
  });

  // 初始化训练记录器
  const metrics = {
    loss: [],
    mae: [],
    epochTime: [],
    batchTimes: [],
  };

  // 自定义训练循环
  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    const epochStart = Date.now();
    let totalLoss = 0;
    let totalMae = 0;

    console.log(`\n═════════ Epoch ${epoch + 1}/${EPOCHS} ═════════`);

    for (const [index, id] of ids.entries()) {
      const batchStart = Date.now();

      const inputs = [
        imageTensors.get(id).expandDims(0),
        offsetTensors.get(id).expandDims(0),
      ];
      const target = yTensors.get(id).expandDims(0);

      // 训练并获取指标
      const [loss, mae] = await model.trainOnBatch(inputs, target);

      // 记录指标
      totalLoss += loss;
      totalMae += mae;

      // 实时打印batch指标
      const batchTime = (Date.now() - batchStart) / 1000;
      metrics.batchTimes.push(batchTime);

      console.log(
        `Batch ${index + 1}/${ids.length} | ` +
          `Loss: ${loss.toFixed(8)} | ` +
          `MAE: ${mae.toFixed(8)} | ` +
          `Time: ${batchTime.toFixed(2)}s`
      );
    }

    // 计算epoch指标
    const avgLoss = totalLoss / ids.length;
    const avgMae = totalMae / ids.length;
    const epochTime = (Date.now() - epochStart) / 1000;

    // 记录epoch指标
    metrics.loss.push(avgLoss);
    metrics.mae.push(avgMae);
    metrics.epochTime.push(epochTime);

    // 打印epoch摘要
    console.log(
      `\n↳ Epoch Summary | ` +
        `Avg Loss: ${avgLoss.toFixed(4)} | ` +
        `Avg MAE: ${avgMae.toFixed(4)} | ` +
        `Time: ${epochTime.toFixed(2)}s\n`
    );
  }

  // 训练结束输出总结
  console.log("\n═════════ Training Summary ═════════");
  console.log(`Total Epochs: ${EPOCHS}`);
  console.log(`Final Loss: ${metrics.loss.slice(-1)[0].toFixed(4)}`);
  console.log(`Best Loss: ${Math.min(...metrics.loss).toFixed(4)}`);
  console.log(
    `Average Batch Time: ${(
      metrics.batchTimes.reduce((a, b) => a + b, 0) / metrics.batchTimes.length
    ).toFixed(2)}s`
  );

  await model.save(MODEL_DIR);
}

// ===================== 执行 =====================
train().catch(console.error);
