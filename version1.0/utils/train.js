const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { promisify } = require("util");
const readFile = promisify(fs.readFile);
const readdir = promisify(fs.readdir);

// ===================== 配置参数 =====================
const TRAIN_DIR = {
  X: "./data/train/X/",
  Y: "./data/train/Y/",
};
const MODEL_DIR = "file://C:/Users/13575/Desktop/wp-project/model/pretrained"; // 获取绝对路径
const INPUT_FEATURES = 32 * 32 * 4; // 修正为4通道
const OUTPUT_FEATURES = 16;
const BATCH_SIZE = 64;
const EPOCHS = 100;
const LEARNING_RATE = 1e-4;

// ===================== 数据加载 =====================

async function loadBinData(dirPath, features) {
  const files = (await readdir(dirPath)).filter((f) => f.endsWith(".bin"));
  const tensors = [];

  for (const file of files) {
    const filePath = path.join(dirPath, file);
    const buffer = await readFile(filePath);
    const float32Data = new Float32Array(buffer.buffer);

    // 验证数据完整性
    if (float32Data.length % features !== 0) {
      throw new Error(
        `文件 ${file} 数据量不兼容: 总浮点数 ${float32Data.length} 无法被特征数 ${features} 整除`
      );
    }

    const samples = float32Data.length / features;
    tensors.push(tf.tensor2d(float32Data, [samples, features]));
  }

  return tf.concat(tensors);
}

// ===================== 模型定义 =====================
function buildModel() {
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      units: 256,
      activation: "relu",
      inputShape: [INPUT_FEATURES],
    })
  );

  model.add(tf.layers.dense({ units: 128, activation: "relu" }));
  model.add(tf.layers.dense({ units: OUTPUT_FEATURES, activation: "linear" }));

  model.compile({
    optimizer: tf.train.adam(LEARNING_RATE),
    loss: "meanSquaredError",
  });

  return model;
}

// ===================== 训练流程 =====================

// 修改后的训练代码
async function train() {
  await tf.tidy(async () => {
    const [X, Y] = await Promise.all([
      loadBinData(TRAIN_DIR.X, INPUT_FEATURES),
      loadBinData(TRAIN_DIR.Y, OUTPUT_FEATURES),
    ]);

    const model = buildModel();
    model.summary();

    await model.fit(X, Y, {
      epochs: EPOCHS,
      batchSize: BATCH_SIZE,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(
            `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(
              4
            )}, val_loss=${logs.val_loss.toFixed(4)}`
          );
        },
      },
    });
    await model.save(MODEL_DIR);
    console.log("模型已保存至:", path.resolve(MODEL_DIR));
  });
}

// ===================== 执行 =====================
train().catch(console.error);
