const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { promisify } = require("util");
const readFile = promisify(fs.readFile);
const PNG = require("pngjs").PNG; // 用于可视化

// ===================== 配置 =====================
const TEST_METADATA = "./data/test/metadata.json";
const TEST_DIR = {
  X_IMAGE: "./data/test/X/",
  X_OFFSET: "./data/test/offset/",
  Y_GT: "./data/test/Y/", // 可选的真实权重
};
const MODEL_PATH = "file://./model/1e-4/model.json";

// ===================== 改进后的数据加载函数 =====================
async function loadDynamicTensor(dirPath) {
  const files = await promisify(fs.readdir)(dirPath);
  const tensorMap = new Map();

  for (const file of files) {
    if (!file.endsWith(".bin")) continue;

    const filePath = path.join(dirPath, file);
    const { tensor, header } = await loadTensorWithHeader(filePath);

    // 验证数据完整性
    validateTensorData(file, tensor, header);

    tensorMap.set(path.parse(file).name, tensor);
  }
  return tensorMap;
}

async function loadTensorWithHeader(filePath) {
  const buffer = await fs.promises.readFile(filePath);

  // 读取头信息（前12字节）
  const header = {
    height: buffer.readUInt32LE(0),
    width: buffer.readUInt32LE(4),
    channels: buffer.readUInt32LE(8),
  };

  // 提取数据部分（从第12字节开始）
  const dataBuffer = buffer.slice(12);
  const float32Data = new Float32Array(
    dataBuffer.buffer,
    dataBuffer.byteOffset,
    dataBuffer.byteLength / 4
  );

  return {
    tensor: tf.tensor3d(float32Data, [
      header.height,
      header.width,
      header.channels,
    ]),
    header,
  };
}

function validateTensorData(filename, tensor, header) {
  // 验证数据尺寸
  const expectedSize = header.height * header.width * header.channels;
  const actualSize = tensor.size;

  if (actualSize !== expectedSize) {
    throw new Error(`数据尺寸验证失败：${filename}
      预期：${expectedSize} (${header.height}x${header.width}x${header.channels})
      实际：${actualSize}`);
  }

  // 添加其他验证规则（可选）
  if (header.height <= 0 || header.width <= 0 || header.channels <= 0) {
    throw new Error(`非法头信息：${filename} ${JSON.stringify(header)}`);
  }
}

// ===================== 测试函数 =====================
async function testModel() {
  try {
    const model = await tf.loadLayersModel(MODEL_PATH);
    console.log("✅ 模型加载成功");

    // 加载测试数据
    const [testImages, testOffsets] = await Promise.all([
      loadDynamicTensor(TEST_DIR.X_IMAGE),
      loadDynamicTensor(TEST_DIR.X_OFFSET),
    ]);

    const testId = Array.from(testImages.keys())[0];
    const X_image = testImages.get(testId).expandDims(0);
    const X_offset = testOffsets.get(testId).expandDims(0);

    // 执行预测
    const predY = model.predict([X_image, X_offset]);
    console.log("📊 预测形状:", predY.shape);

    // 分批验证权重和
    await validateSumWeights(predY);

    // 抽样可视化
    await visualizeWeights(predY.squeeze());
    // 预测后调用（示例查看中心点）
    const centerX = Math.floor(predY.shape[2] / 2);
    const centerY = Math.floor(predY.shape[1] / 2);
    await inspectPixelWeights(predY, centerX, centerY);

    // 也可以查看边缘点
    await inspectPixelWeights(predY, 0, 0);
    // 释放预测结果
    tf.dispose(predY);
  } catch (error) {
    console.error("‼️ 测试失败:", error);
  }
}
async function inspectPixelWeights(predY, x = 100, y = 100) {
  let weights = null;
  try {
    // 增强输入验证
    if (!predY || predY.isDisposed) {
      console.log("⚠️ 输入张量无效或已释放");
      return;
    }

    // 确保坐标在有效范围内
    const [_, H, W, C] = predY.shape || [];
    if (x < 0 || x >= (W || 0) || y < 0 || y >= (H || 0)) {
      console.log(
        `❌ 坐标超出范围 (最大坐标: X=${(W || 0) - 1}, Y=${(H || 0) - 1})`
      );
      return;
    }

    // 提取指定位置权重
    weights = predY.slice([0, y, x, 0], [1, 1, 1, 16]).squeeze();
    const weightValues = Array.from(weights.dataSync());

    // 格式化为科学计数法和原始值
    console.log(`\n🔍 像素 (X=${x}, Y=${y}) 的 16 个权重值：`);
    console.log("┌──────┬──────────────────────┬─────────────────┐");
    console.log("│ 通道 │ 原始值                │ 科学计数法      │");
    console.log("├──────┼──────────────────────┼─────────────────┤");
    weightValues.forEach((v, i) => {
      console.log(
        `│ ${i.toString().padStart(2)}  │ ${v.toFixed(10).padEnd(20)} │ ${v
          .toExponential(4)
          .padEnd(15)} │`
      );
    });
    console.log("└──────┴──────────────────────┴─────────────────┘");

    // 验证权重和
    const sum = weightValues.reduce((a, b) => a + b, 0);
    console.log(`\nΣ 权重和: ${sum.toFixed(8)} (理论应为 1.0)`);

    // 检查负值
    const negativeWeights = weightValues.filter((v) => v < 0);
    if (negativeWeights.length > 0) {
      console.log(`⚠️ 发现 ${negativeWeights.length} 个负权重值`);
    }
  } catch (error) {
    console.error("‼️ 权重检查失败:", error.message);
  } finally {
    // 安全清理内存
    if (weights && !weights.isDisposed) {
      tf.dispose(weights);
    }
  }
}

// ===================== 分批验证函数 =====================
async function validateSumWeights(predY) {
  const [H, W, C] = predY.shape.slice(1);

  let totalSum = 0;
  const batchSize = 128;

  for (let h = 0; h < H; h += batchSize) {
    const hEnd = Math.min(h + batchSize, H);
    for (let w = 0; w < W; w += batchSize) {
      const wEnd = Math.min(w + batchSize, W);
      const patch = predY.slice([0, h, w, 0], [1, hEnd - h, wEnd - w, C]);

      const sum = patch.sum(-1).dataSync();
      const patchAvg = sum.reduce((a, b) => a + b, 0) / sum.length;
      totalSum += patchAvg * (hEnd - h) * (wEnd - w);

      tf.dispose([patch, sum]);
      await tf.nextFrame();
    }
  }

  const globalAvg = totalSum / (H * W);
  console.log(`全局权重平均值: ${globalAvg.toFixed(6)}`);
}

// ===================== 可视化工具 =====================
async function visualizeWeights(weightsTensor) {
  // 抽样 100x100 区域
  const sample = weightsTensor.slice([0, 0, 0], [100, 100, 16]);
  const firstChannel = sample.slice([0, 0, 0], [100, 100, 1]);

  await tensorToPNG(firstChannel, "weight_sample.png");
  tf.dispose([sample, firstChannel]);
}

async function tensorToPNG(tensor, filename) {
  const data = await tensor.data();
  const [height, width] = tensor.shape;

  const png = new PNG({
    width,
    height,
    colorType: 2, // RGB
  });

  for (let i = 0; i < data.length; i++) {
    const val = Math.min(255, Math.max(0, data[i] * 255));
    png.data[i * 3] = val; // R
    png.data[i * 3 + 1] = val; // G
    png.data[i * 3 + 2] = val; // B
  }

  return new Promise((resolve) => {
    png.pack().pipe(fs.createWriteStream(filename)).on("finish", resolve);
  });
}

// 启动测试
testModel();
