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
const MODEL_DIR = "file://C:/Users/13575/Desktop/wp-project/model/pretrained";
const INPUT_FEATURES = 4 * 4 * 4 + 2;
const OUTPUT_FEATURES = 16;
const BATCH_SIZE = 512;
const EPOCHS = 50;
const LEARNING_RATE = 1e-5;

// ===================== 数据加载 =====================
async function loadBinData(dirPath, features) {
  try {
    const files = (await readdir(dirPath)).filter((f) => f.endsWith(".bin"));
    if (files.length === 0) throw new Error(`目录 ${dirPath} 中没有.bin文件`);

    const tensors = [];
    for (const file of files) {
      const filePath = path.join(dirPath, file);
      const buffer = await readFile(filePath);
      const float32Data = new Float32Array(buffer.buffer);

      if (float32Data.length % features !== 0) {
        console.error(`跳过损坏文件 ${file}`);
        continue;
      }

      const tensor = tf.tensor2d(float32Data, [
        float32Data.length / features,
        features,
      ]);
      tensors.push(tensor);
      console.log(`${file} 加载成功，形状: ${tensor.shape}`);
    }

    return tf.concat(tensors);
  } catch (err) {
    console.error("数据加载失败:", err);
    process.exit(1);
  }
}

// ===================== 模型定义 =====================

function buildModel() {
  const model = tf.sequential();
  // 输入层（减少单元数）
  model.add(
    tf.layers.dense({
      units: 64, // 原为128
      activation: "relu",
      kernelInitializer: "heNormal",
      inputShape: [INPUT_FEATURES],
      kernelConstraint: tf.constraints.maxNorm({ maxValue: 1.0 }), // 降低约束
    })
  );

  // 隐藏层（简化结构）
  model.add(
    tf.layers.dense({
      units: 32, // 原为64
      activation: "relu",
      kernelInitializer: "heNormal",
    })
  );
  // 输出
  model.add(
    tf.layers.dense({
      units: OUTPUT_FEATURES,
      activation: "linear",
      kernelInitializer: "zeros",
      useBias: false, // 禁用偏置项
    })
  );
  // 编译配置
  model.compile({
    optimizer: tf.train.sgd(0.001),
    loss: "meanSquaredError",
  });
  return model;
}
// ===================== 训练流程 =====================
async function train() {
  tf.ENV.set("WEBGL_FORCE_F16_TEXTURES", false);

  try {
    // 加载数据
    const [X_raw, Y] = await Promise.all([
      loadBinData(TRAIN_DIR.X, INPUT_FEATURES),
      loadBinData(TRAIN_DIR.Y, OUTPUT_FEATURES),
    ]);

    // 数据验证（关键步骤）
    console.log(
      "输入数据范围:",
      X_raw.min().dataSync(),
      "~",
      X_raw.max().dataSync()
    );
    if (X_raw.min().dataSync()[0] < 0 || X_raw.max().dataSync()[0] > 1) {
      throw new Error("输入数据应已归一化至[0,1]");
    }
    const X = X_raw.clone(); // 直接使用归一化数据

    console.log("\n输出数据验证:");
    console.log("Y最小值:", Y.min().dataSync());
    console.log("Y最大值:", Y.max().dataSync());
    console.log("Y样本和均值:", Y.sum(1).mean().dataSync());

    // 构建模型
    const model = buildModel();
    model.summary();

    // 早停机制
    let bestValLoss = Infinity;
    let wait = 0;
    const patience = 5;

    await model.fit(X, Y, {
      epochs: EPOCHS,
      batchSize: BATCH_SIZE,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          // 损失值检查
          if (isNaN(logs.loss) || isNaN(logs.val_loss)) {
            throw new Error(`第${epoch + 1}轮损失值为NaN`);
          }

          // 早停逻辑
          if (logs.val_loss < bestValLoss) {
            bestValLoss = logs.val_loss;
            wait = 0;
          } else {
            wait++;
            if (wait >= patience) {
              console.log(`早停触发于第 ${epoch + 1} 轮`);
              model.stopTraining = true;
            }
          }
          console.log(
            `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(
              6
            )}, val_loss=${logs.val_loss.toFixed(6)}`
          );
        },
      },
    });

    await model.save(MODEL_DIR);
    console.log("模型已保存至:", path.resolve(MODEL_DIR));
  } catch (err) {
    console.error("训练失败:", err);
    process.exit(1);
  }
}

// ===================== 执行 =====================
train();
