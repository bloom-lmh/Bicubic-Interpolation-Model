const tf = require("@tensorflow/tfjs-node");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

// ===================== 配置参数 =====================
const SCALE_FACTOR = 4;
const OUTPUT_DIR = {
  TRAIN_X: "./data/train/X/",
  TRAIN_OFFSET: "./data/train/offset/",
  TRAIN_Y: "./data/train/Y/",
  TRAIN_WEIGHT: "./data/train/weight/", // 新增权重目录
};
const RAW_HR_DIR = "./data/raw/DIV2K_train_HR/";
const METADATA_PATH = path.join(OUTPUT_DIR.TRAIN_X, "../metadata.json");

// ===================== 元数据工具 =====================
async function updateMetadata(sampleId, H_lr, W_lr, H_sr, W_sr) {
  let metadata = {};
  if (fs.existsSync(METADATA_PATH)) {
    metadata = JSON.parse(await fs.promises.readFile(METADATA_PATH, "utf8"));
  }

  metadata[sampleId] = {
    H_lr: Math.floor(H_lr),
    W_lr: Math.floor(W_lr),
    H_sr: Math.floor(H_sr),
    W_sr: Math.floor(W_sr),
    channels: {
      X: 4,
      offset: 2,
      Y: 16,
      weight: 16, // 新增权重通道
    },
  };

  const tempPath = METADATA_PATH + ".tmp";
  await fs.promises.writeFile(tempPath, JSON.stringify(metadata, null, 2));
  await fs.promises.rename(tempPath, METADATA_PATH);
}

// ===================== 图像处理器 =====================
class ImageProcessor {
  static async loadAndAlign(imagePath, alignFactor) {
    const { data, info } = await sharp(imagePath)
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const alignedWidth = Math.floor(info.width / alignFactor) * alignFactor;
    const alignedHeight = Math.floor(info.height / alignFactor) * alignFactor;

    return tf.tidy(() => ({
      tensor: tf
        .tensor3d(new Uint8Array(data), [info.height, info.width, 4])
        .slice([0, 0, 0], [alignedHeight, alignedWidth, 4])
        .div(255.0),
      metadata: { ...info, width: alignedWidth, height: alignedHeight },
    }));
  }

  static async downscale(hrTensor, metadata, scale) {
    const buffer = await tf.tidy(() => hrTensor.mul(255).toInt().dataSync());

    const lrBuffer = await sharp(Buffer.from(buffer), {
      raw: { width: metadata.width, height: metadata.height, channels: 4 },
    })
      .resize(
        Math.floor(metadata.width / scale),
        Math.floor(metadata.height / scale),
        {
          kernel: sharp.kernel.cubic,
        }
      )
      .raw()
      .toBuffer();

    return tf
      .tensor3d(new Uint8Array(lrBuffer), [
        Math.floor(metadata.height / scale),
        Math.floor(metadata.width / scale),
        4,
      ])
      .div(255.0);
  }

  // 新增：计算亮度图
  static calculateLuma(tensor) {
    return tf.tidy(() => {
      return tensor
        .slice([0, 0, 0], [-1, -1, 3]) // 去除alpha通道
        .mul(tf.tensor1d([0.2126, 0.7152, 0.0722])) // BT.709标准
        .sum(2); // 沿通道维度求和
    });
  }
}

// ===================== 高级数据生成器 =====================
class AdvancedDataGenerator {
  static generate(lrTensor, hrHeight, hrWidth, scale) {
    return tf.tidy(() => {
      // 1. 计算亮度图
      const luma = ImageProcessor.calculateLuma(lrTensor);
      const lumaData = luma.dataSync();
      const lrHeight = lrTensor.shape[0];
      const lrWidth = lrTensor.shape[1];

      // 2. 创建输出缓冲区
      const offsets = tf.buffer([hrHeight, hrWidth, 2]);
      const weights = tf.buffer([hrHeight, hrWidth, 16]);
      const adaptiveWeights = tf.buffer([hrHeight, hrWidth, 16]); // 新增自适应权重

      for (let y_hr = 0; y_hr < hrHeight; y_hr++) {
        for (let x_hr = 0; x_hr < hrWidth; x_hr++) {
          // 计算亚像素偏移
          const { dx, dy, x_lr, y_lr } = this.calculateSubpixelOffset(
            x_hr,
            y_hr,
            scale
          );
          offsets.set(dx, y_hr, x_hr, 0);
          offsets.set(dy, y_hr, x_hr, 1);

          // 计算基础双三次权重
          const baseWeights = this.calculateBicubicWeights(dx, dy);

          // 计算自适应权重
          const adaptiveFactors = this.calculateAdaptiveFactors(
            x_lr,
            y_lr,
            lumaData,
            lrWidth,
            lrHeight
          );

          // 合并权重
          const finalWeights = baseWeights.map(
            (w, i) => w * adaptiveFactors[i]
          );
          const sumWeights = finalWeights.reduce((a, b) => a + b, 0);

          // 归一化并存储
          const normalizedWeights =
            sumWeights > 0
              ? finalWeights.map((w) => w / sumWeights)
              : new Array(16).fill(0);

          for (let i = 0; i < 16; i++) {
            weights.set(baseWeights[i], y_hr, x_hr, i);
            adaptiveWeights.set(normalizedWeights[i], y_hr, x_hr, i);
          }
        }
      }

      return {
        offsets: offsets.toTensor(),
        weights: weights.toTensor(),
        adaptiveWeights: adaptiveWeights.toTensor(), // 新增自适应权重
      };
    });
  }

  static calculateSubpixelOffset(x_hr, y_hr, scale) {
    const x_lr = (x_hr + 0.5) / scale;
    const y_lr = (y_hr + 0.5) / scale;
    return {
      dx: x_lr - Math.floor(x_lr) - 0.5,
      dy: y_lr - Math.floor(y_lr) - 0.5,
      x_lr: Math.floor(x_lr),
      y_lr: Math.floor(y_lr),
    };
  }

  static calculateBicubicWeights(dx, dy, a = -0.5) {
    const cubic = (t) => {
      t = Math.abs(t);
      return t >= 2
        ? 0
        : t >= 1
        ? a * (t ** 3 - 5 * t ** 2 + 8 * t - 4)
        : (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1;
    };

    const grid = [];
    for (let j = -1; j <= 2; j++) {
      for (let i = -1; i <= 2; i++) {
        grid.push(cubic(i - dx) * cubic(j - dy));
      }
    }

    const sum = grid.reduce((a, b) => a + b, 0);
    return sum > 1e-6 ? grid.map((w) => w / sum) : new Array(16).fill(0);
  }

  // 新增：计算自适应因子
  static calculateAdaptiveFactors(x_lr, y_lr, lumaData, lrWidth, lrHeight) {
    const factors = new Array(16).fill(1);
    const centerIdx =
      Math.min(lrWidth - 1, Math.max(0, y_lr)) * lrWidth +
      Math.min(lrWidth - 1, Math.max(0, x_lr));
    const centerLuma = lumaData[centerIdx];

    // 分析局部对比度
    let minLuma = Infinity,
      maxLuma = -Infinity;
    for (let dy = -1; dy <= 2; dy++) {
      for (let dx = -1; dx <= 2; dx++) {
        const px = Math.min(lrWidth - 1, Math.max(0, x_lr + dx));
        const py = Math.min(lrHeight - 1, Math.max(0, y_lr + dy));
        const luma = lumaData[py * lrWidth + px];
        minLuma = Math.min(minLuma, luma);
        maxLuma = Math.max(maxLuma, luma);
      }
    }
    const contrast = maxLuma - minLuma;

    // 确定区域类型
    const isEdge = contrast > 0.3; // 高对比度视为边缘
    const isFlat = contrast < 0.1; // 低对比度视为平坦区域

    // 计算每个位置的调整因子
    for (let j = -1; j <= 2; j++) {
      for (let i = -1; i <= 2; i++) {
        const idx = (j + 1) * 4 + (i + 1);
        const px = Math.min(lrWidth - 1, Math.max(0, x_lr + i));
        const py = Math.min(lrHeight - 1, Math.max(0, y_lr + j));
        const currentLuma = lumaData[py * lrWidth + px];
        const lumaDiff = Math.abs(currentLuma - centerLuma);

        if (isEdge) {
          // 边缘区域：增强相似像素权重
          factors[idx] = 1.0 + 0.5 * (1 - lumaDiff / 0.3);
        } else if (isFlat) {
          // 平坦区域：抑制噪声
          factors[idx] = Math.max(0.7, 1 - lumaDiff / 0.2);
        } else {
          // 纹理区域：中等增强
          factors[idx] = 0.8 + 0.4 * Math.exp(-lumaDiff / 0.15);
        }
      }
    }

    return factors;
  }
}

// ===================== 数据存储 =====================
class DataSaver {
  static async saveWithHeader(tensor, filePath, expectedShape) {
    const tensorFloat = tensor.toFloat();
    const header = Buffer.alloc(12);
    header.writeUInt32LE(expectedShape[0], 0);
    header.writeUInt32LE(expectedShape[1], 4);
    header.writeUInt32LE(expectedShape[2], 8);

    const data = await tensorFloat.data();
    const dataBuffer = Buffer.from(data.buffer);

    await fs.promises.writeFile(filePath, Buffer.concat([header, dataBuffer]));
  }
}

// ===================== 主处理流程 =====================
async function processImages() {
  // 初始化目录
  Object.values(OUTPUT_DIR).forEach((dir) => {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  });

  const hrImages = fs
    .readdirSync(RAW_HR_DIR)
    .filter((f) => [".png", ".jpg"].includes(path.extname(f).toLowerCase()));

  for (const imgFile of hrImages) {
    try {
      console.log(`Processing: ${imgFile}`);
      const hrPath = path.join(RAW_HR_DIR, imgFile);
      const baseName = path.basename(imgFile, path.extname(imgFile));

      // 1. 加载和对齐图像
      const { tensor: hrTensor, metadata } = await ImageProcessor.loadAndAlign(
        hrPath,
        SCALE_FACTOR
      );

      // 2. 计算尺寸
      const H_sr = metadata.height;
      const W_sr = metadata.width;
      const H_lr = H_sr / SCALE_FACTOR;
      const W_lr = W_sr / SCALE_FACTOR;

      // 3. 更新元数据
      await updateMetadata(baseName, H_lr, W_lr, H_sr, W_sr);

      // 4. 生成数据
      const lrTensor = await ImageProcessor.downscale(
        hrTensor,
        metadata,
        SCALE_FACTOR
      );
      const { offsets, weights, adaptiveWeights } =
        AdvancedDataGenerator.generate(lrTensor, H_sr, W_sr, SCALE_FACTOR);

      // 5. 保存数据
      await DataSaver.saveWithHeader(
        lrTensor,
        path.join(OUTPUT_DIR.TRAIN_X, `${baseName}.bin`),
        [lrTensor.shape[0], lrTensor.shape[1], 4]
      );

      await DataSaver.saveWithHeader(
        offsets,
        path.join(OUTPUT_DIR.TRAIN_OFFSET, `${baseName}.bin`),
        [offsets.shape[0], offsets.shape[1], 2]
      );

      await DataSaver.saveWithHeader(
        weights,
        path.join(OUTPUT_DIR.TRAIN_Y, `${baseName}.bin`),
        [weights.shape[0], weights.shape[1], 16]
      );

      // 新增：保存自适应权重
      await DataSaver.saveWithHeader(
        adaptiveWeights,
        path.join(OUTPUT_DIR.TRAIN_WEIGHT, `${baseName}.bin`),
        [adaptiveWeights.shape[0], adaptiveWeights.shape[1], 16]
      );

      tf.dispose([hrTensor, lrTensor, offsets, weights, adaptiveWeights]);
    } catch (err) {
      console.error(`Error processing ${imgFile}:`, err.message);
    }
  }
}

// ===================== 执行入口 =====================
processImages()
  .then(() => console.log("Processing completed!"))
  .catch((err) => console.error("Fatal error:", err));
