const tf = require("@tensorflow/tfjs");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

// ===================== 配置参数 =====================
const SCALE_FACTOR = 4;
const PATCH_SIZE = 4;
const OUTPUT_DIR = {
  TRAIN_X: "./data/train/X/",
  TRAIN_Y: "./data/train/Y/",
};
const RAW_HR_DIR = "./data/raw/DIV2K_train_HR/";
const BATCH_SIZE = 10000; // 新增批次控制参数

// ===================== 图像加载函数 =====================
async function loadImageTensor(imagePath) {
  const { data, info } = await sharp(imagePath)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const minSize = SCALE_FACTOR * PATCH_SIZE;
  if (info.width < minSize || info.height < minSize) {
    throw new Error(`图像 ${imagePath} 尺寸过小，需至少 ${minSize}x${minSize}`);
  }

  const alignedWidth = Math.floor(info.width / minSize) * minSize;
  const alignedHeight = Math.floor(info.height / minSize) * minSize;

  return tf.tidy(() => {
    const tensor = tf
      .tensor3d(new Uint8Array(data), [info.height, info.width, 4])
      .slice([0, 0, 0], [alignedHeight, alignedWidth, 4])
      .div(255.0);
    return {
      tensor,
      metadata: { ...info, width: alignedWidth, height: alignedHeight },
    };
  });
}

// ===================== 下采样函数 =====================
async function downscaleImage(hrTensor, metadata, scale) {
  const buffer = await tf.tidy(() => hrTensor.mul(255).toInt().dataSync());

  const targetWidth = Math.floor(metadata.width / scale);
  const targetHeight = Math.floor(metadata.height / scale);

  const downscaledBuffer = await sharp(Buffer.from(buffer), {
    raw: { width: metadata.width, height: metadata.height, channels: 4 },
  })
    .resize(targetWidth, targetHeight, { kernel: sharp.kernel.cubic })
    .raw()
    .toBuffer();

  return tf.tidy(() => {
    const lrTensor = tf
      .tensor3d(new Uint8Array(downscaledBuffer), [
        targetHeight,
        targetWidth,
        4,
      ])
      .div(255.0);
    if (lrTensor.shape[0] < 4 || lrTensor.shape[1] < 4) {
      throw new Error("LR图像尺寸不足以提取4x4邻域");
    }
    return lrTensor;
  });
}

// ===================== 核心逻辑 =====================
function extractSinglePixelData(hrTensor, lrTensor, x, y, scale) {
  return tf.tidy(() => {
    const lrX = (x + 0.5) / scale - 0.5;
    const lrY = (y + 0.5) / scale - 0.5;

    const x0 = Math.floor(lrX) - 1;
    const y0 = Math.floor(lrY) - 1;

    const xStart = Math.max(x0, 0);
    const yStart = Math.max(y0, 0);
    const xEnd = Math.min(x0 + 4, lrTensor.shape[1]);
    const yEnd = Math.min(y0 + 4, lrTensor.shape[0]);

    const validPatch = lrTensor.slice(
      [yStart, xStart, 0],
      [yEnd - yStart, xEnd - xStart, 4]
    );

    const padTop = Math.max(0, -y0);
    const padBottom = Math.max(0, y0 + 4 - lrTensor.shape[0]);
    const padLeft = Math.max(0, -x0);
    const padRight = Math.max(0, x0 + 4 - lrTensor.shape[1]);

    const paddedPatch = tf.pad3d(validPatch, [
      [padTop, padBottom],
      [padLeft, padRight],
      [0, 0],
    ]);

    const dx = lrX - Math.floor(lrX);
    const dy = lrY - Math.floor(lrY);

    const weights = calculate4x4Weights(dx, dy);

    return {
      input: [...paddedPatch.flatten().dataSync(), dx, dy],
      target: weights,
    };
  });
}

function calculate4x4Weights(dx, dy) {
  const cubicKernel = (t) => {
    t = Math.abs(t);
    const a = -0.5;
    if (t >= 2.0) return 0.0;
    if (t >= 1.0) return Math.max(0, a * (t ** 3 - 5 * t ** 2 + 8 * t - 4));
    return Math.max(0, (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1);
  };

  const xWeights = Array.from({ length: 4 }, (_, i) =>
    cubicKernel(dx - (i - 1))
  );
  const yWeights = Array.from({ length: 4 }, (_, i) =>
    cubicKernel(dy - (i - 1))
  );

  let weights = [];
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      weights.push(xWeights[x] * yWeights[y]);
    }
  }

  const sum = weights.reduce((a, b) => a + b, 0);
  return sum > 1e-6 ? weights.map((w) => w / sum) : new Array(16).fill(0.0);
}

// ===================== 数据保存 =====================
async function saveBatch(bufferX, bufferY, xStream, yStream) {
  // 校验数据对齐
  const xSamples = Math.floor(bufferX.length / 66);
  const ySamples = Math.floor(bufferY.length / 16);
  const validSamples = Math.min(xSamples, ySamples);

  // 截取有效数据
  const validX = bufferX.slice(0, validSamples * 66);
  const validY = bufferY.slice(0, validSamples * 16);

  // 写入文件
  xStream.write(Buffer.from(new Float32Array(validX).buffer));
  yStream.write(Buffer.from(new Float32Array(validY).buffer));

  // 保留剩余数据
  return {
    remainX: bufferX.slice(validSamples * 66),
    remainY: bufferY.slice(validSamples * 16),
  };
}

// ===================== 主流程优化 =====================
async function generateTrainingData() {
  [OUTPUT_DIR.TRAIN_X, OUTPUT_DIR.TRAIN_Y].forEach((dir) => {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  });

  const hrImages = fs
    .readdirSync(RAW_HR_DIR)
    .filter((f) => [".png", ".jpg"].includes(path.extname(f).toLowerCase()));

  for (const imgFile of hrImages) {
    let xStream = null,
      yStream = null;
    try {
      const hrPath = path.join(RAW_HR_DIR, imgFile);
      console.log(`Processing: ${imgFile}`);

      // 加载图像
      const { tensor: hrTensor, metadata } = await loadImageTensor(hrPath);

      // 创建写入流
      const baseName = path.basename(imgFile, path.extname(imgFile));
      xStream = fs.createWriteStream(
        path.join(OUTPUT_DIR.TRAIN_X, `${baseName}.bin`)
      );
      yStream = fs.createWriteStream(
        path.join(OUTPUT_DIR.TRAIN_Y, `${baseName}.bin`)
      );

      // 生成LR图像
      const lrTensor = await downscaleImage(hrTensor, metadata, SCALE_FACTOR);

      // 初始化缓冲区
      let bufferX = [],
        bufferY = [];

      // 处理像素
      for (let y = 0; y < metadata.height; y++) {
        for (let x = 0; x < metadata.width; x++) {
          const { input, target } = extractSinglePixelData(
            hrTensor,
            lrTensor,
            x,
            y,
            SCALE_FACTOR
          );

          // 校验维度
          if (input.length !== 66 || target.length !== 16) {
            throw new Error(`数据维度错误 (x=${x}, y=${y})`);
          }

          bufferX.push(...input);
          bufferY.push(...target);

          // 批次处理
          if (
            bufferX.length >= BATCH_SIZE * 66 ||
            bufferY.length >= BATCH_SIZE * 16
          ) {
            const { remainX, remainY } = await saveBatch(
              bufferX,
              bufferY,
              xStream,
              yStream
            );
            bufferX = remainX;
            bufferY = remainY;
          }
        }
      }

      // 处理剩余数据
      if (bufferX.length > 0 || bufferY.length > 0) {
        await saveBatch(bufferX, bufferY, xStream, yStream);
      }
    } catch (err) {
      console.error(`处理失败: ${imgFile}`, err);
    } finally {
      // 安全关闭流
      if (xStream) xStream.end();
      if (yStream) yStream.end();
      tf.disposeVariables();
    }
  }
}

// ===================== 执行入口 =====================
generateTrainingData()
  .then(() => console.log("数据生成完成!"))
  .catch((err) => console.error("致命错误:", err));
