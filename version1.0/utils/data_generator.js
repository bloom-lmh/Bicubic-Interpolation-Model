const tf = require("@tensorflow/tfjs"); // 使用纯 JavaScript 版本
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

// ===================== 配置参数 =====================
const SCALE_FACTOR = 4; // 超分放大倍数
const PATCH_SIZE = 4;
const OUTPUT_DIR = {
  TRAIN_X: "./data/train/X/",
  TRAIN_Y: "./data/train/Y/",
};
const RAW_HR_DIR = "./data/raw/DIV2K_train_HR/";

// ===================== 图像加载函数 =====================

async function loadImageTensor(imagePath) {
  const { data, info } = await sharp(imagePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  console.log(data, info);

  // 强制对齐到 SCALE_FACTOR * PATCH_SIZE 的整数倍
  const alignedWidth =
    Math.floor(info.width / (SCALE_FACTOR * PATCH_SIZE)) *
    (SCALE_FACTOR * PATCH_SIZE);
  const alignedHeight =
    Math.floor(info.height / (SCALE_FACTOR * PATCH_SIZE)) *
    (SCALE_FACTOR * PATCH_SIZE);

  return tf.tidy(() => {
    const tensor = tf
      .tensor3d(new Uint8Array(data), [info.height, info.width, info.channels])
      .slice([0, 0, 0], [alignedHeight, alignedWidth, info.channels])
      .div(255.0);

    return {
      tensor,
      metadata: { ...info, width: alignedWidth, height: alignedHeight },
    };
  });
}
// ===================== 下采样函数 =====================
async function downscaleImage(hrTensor, metadata, scale) {
  // 将 Tensor 转换为 Sharp 兼容的 Buffer
  const buffer = await tf.tidy(() => {
    return hrTensor.mul(255).toInt().dataSync();
  });
  // 缩小原图
  const targetWidth = Math.floor(metadata.width / scale);
  const targetHeight = Math.floor(metadata.height / scale);
  // 双三次下采样
  const downscaledBuffer = await sharp(Buffer.from(buffer), {
    raw: {
      width: metadata.width,
      height: metadata.height,
      channels: metadata.channels,
    },
  })
    .resize(targetWidth, targetHeight, { kernel: sharp.kernel.cubic })
    .raw()
    .toBuffer();
  // 低分辨率的图片tensor
  return tf.tidy(() => {
    return tf
      .tensor3d(new Uint8Array(downscaledBuffer), [
        targetHeight,
        targetWidth,
        metadata.channels,
      ])
      .div(255.0);
  });
}

// ===================== 核心逻辑（无需修改） =====================

function extractPatches(tensor, patchSize, stride, scaleFactor = 1) {
  return tf.tidy(() => {
    // 获取低分辨率tensor形状
    const [height, width, channels] = tensor.shape;

    const patches = [];
    const coordinates = []; // 存储坐标信息

    const numY = Math.floor((height - patchSize) / stride) + 1;
    const numX = Math.floor((width - patchSize) / stride) + 1;

    for (let y = 0; y < numY; y++) {
      const yStart = y * stride;
      for (let x = 0; x < numX; x++) {
        const xStart = x * stride;
        const patch = tensor.slice(
          [yStart, xStart, 0],
          [patchSize, patchSize, channels]
        );
        patches.push(patch.flatten());
        coordinates.push({
          x: xStart,
          y: yStart,
          scaledX: xStart * scaleFactor, // 对应的HR坐标
          scaledY: yStart * scaleFactor,
        });
      }
    }

    return {
      patches: tf.stack(patches),
      coordinates, // 返回坐标信息
    };
  });
}

function calculate4x4Weights(dx, dy) {
  const cubicKernel = (t, a = -0.5) => {
    t = Math.abs(t);
    if (t >= 2) return 0;
    if (t >= 1) return a * (t ​** 3 - 5 * t ​** 2 + 8 * t - 4);
    return (a + 2) * t ​** 3 - (a + 3) * t ​** 2 + 1;
  };

  // 计算x/y方向权重并截断负值
  const xWeights = Array.from({ length: 4 }, (_, i) => {
    return Math.max(0, cubicKernel(dx - (i - 1))); 
  });
  const yWeights = Array.from({ length: 4 }, (_, i) => {
    return Math.max(0, cubicKernel(dy - (i - 1)));
  });

  // 生成4x4权重矩阵（二次截断负值）
  let weights = [];
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      let weight = xWeights[x] * yWeights[y];
      weight = Math.max(0, weight);
      weight = Number(weight.toFixed(10)); // 截断到小数点后10位
      weights.push(weight);
    }
  }


  // 强制非零分母并归一化
  const sum = weights.reduce((a, b) => a + b, 0);
  const eps = 1e-6;
  return sum > eps ? weights.map(w => w / sum) : weights.fill(0);
}
function calculateBicubicWeightsForPatches(lrCoordinates, scale) {
  return tf.tidy(() => {
    const weights = [];
    for (const coord of lrCoordinates) {
      // 计算LR块中心在HR中的浮点偏移
      const centerX = coord.x + PATCH_SIZE / 2;
      const centerY = coord.y + PATCH_SIZE / 2;
      const lrX = (centerX + 0.5) / scale - 0.5;
      const lrY = (centerY + 0.5) / scale - 0.5;

      // 计算dx/dy并限制范围
      const dx = Math.max(0, Math.min(0.99999999, lrX - Math.floor(lrX)));
      const dy = Math.max(0, Math.min(0.99999999, lrY - Math.floor(lrY)));

      // 获取16个权重并归一化
      const weights4x4 = calculate4x4Weights(dx, dy);
      weights.push(weights4x4);
    }
    return tf.tensor2d(weights);
  });
}
// ===================== 数据保存 =====================
async function saveTensorAsBinary(tensor, filePath) {
  const data = await tensor.data();
 const processedData = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    processedData[i] = Math.max(0, data[i]);
  }

  const sum = processedData.reduce((a, b) => a + b, 0);
  if (sum === 0) {
    processedData.fill(0); // 安全处理全零情况
  } else {
    for (let i = 0; i < processedData.length; i++) {
      processedData[i] /= sum; // 重新归一化
    }
  }

  fs.writeFileSync(filePath, Buffer.from(processedData.buffer));
}

// ===================== 主流程 =====================
async function generateTrainingData() {
  // 创建输出目录
  [OUTPUT_DIR.TRAIN_X, OUTPUT_DIR.TRAIN_Y].forEach((dir) => {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  });
  const hrImages = fs
    .readdirSync(RAW_HR_DIR)
    .filter((f) => [".png", ".jpg"].includes(path.extname(f).toLowerCase()));

  for (const imgFile of hrImages) {
    try {
      const hrPath = path.join(RAW_HR_DIR, imgFile);
      console.log(`Processing: ${imgFile}`);

      // 加载并对齐图像
      const { tensor: hrTensor, metadata } = await loadImageTensor(hrPath);

      // 生成低分辨率图像（尺寸自动对齐）
      const lrTensor = await downscaleImage(hrTensor, metadata, SCALE_FACTOR);

      // ========== 关键修改：严格按比例提取块 ==========
      // 提取分块并获取坐标信息
      const lrResult = extractPatches(lrTensor, PATCH_SIZE, PATCH_SIZE);
      const hrResult = extractPatches(
        hrTensor,
        PATCH_SIZE * SCALE_FACTOR,
        PATCH_SIZE * SCALE_FACTOR,
        SCALE_FACTOR // 传入放大倍数
      );

      // 解构赋值必须同步完成
      const lrPatches = lrResult.patches;
      const hrPatches = hrResult.patches;
      // 添加调试日志

      // 强制验证块数量一致性
      if (lrPatches.shape[0] !== hrPatches.shape[0]) {
        throw new Error(
          `块数量不匹配: LR=${lrPatches.shape[0]} vs HR=${hrPatches.shape[0]}`
        );
      }
      const weights = calculateBicubicWeightsForPatches(
        lrResult.coordinates,
        SCALE_FACTOR
      );
      // 保存数据
      const baseName = path.basename(imgFile, path.extname(imgFile));
      console.log("lr形状:", lrResult.patches);
      console.log("权重形状：", weights);
      await saveTensorAsBinary(
        lrResult.patches,
        path.join(OUTPUT_DIR.TRAIN_X, `${baseName}.bin`)
      );

      await saveTensorAsBinary(
        weights,
        path.join(OUTPUT_DIR.TRAIN_Y, `${baseName}.bin`)
      );

      // 内存清理
      tf.dispose([hrTensor, lrTensor, lrPatches, hrPatches]);
    } catch (err) {
      console.error(`处理失败: ${imgFile}`, err);
    }
  }
}

// ===================== 执行入口 =====================
generateTrainingData()
  .then(() => console.log("数据生成完成!"))
  .catch((err) => console.error("致命错误:", err));
