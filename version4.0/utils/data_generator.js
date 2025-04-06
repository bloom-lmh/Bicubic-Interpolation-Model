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
};
const RAW_HR_DIR = "./data/raw/DIV2K_train_HR/";
const METADATA_PATH = path.join(OUTPUT_DIR.TRAIN_X, "../metadata.json"); // 元数据路径
// ===================== 新增元数据工具方法 =====================
async function updateMetadata(sampleId, H_lr, W_lr, H_sr, W_sr) {
  // 读取现有元数据或初始化空对象
  let metadata = {};
  if (fs.existsSync(METADATA_PATH)) {
    const content = await fs.promises.readFile(METADATA_PATH, "utf8");
    metadata = JSON.parse(content);
  }

  // 更新条目
  metadata[sampleId] = {
    H_lr: Math.floor(H_lr),
    W_lr: Math.floor(W_lr),
    H_sr: Math.floor(H_sr),
    W_sr: Math.floor(W_sr),
    channels: {
      X: 4,
      offset: 2,
      Y: 16, // 新增Y通道数记录
    },
  };

  // 原子写入：先写入临时文件再重命名
  const tempPath = METADATA_PATH + ".tmp";
  await fs.promises.writeFile(tempPath, JSON.stringify(metadata, null, 2));
  await fs.promises.rename(tempPath, METADATA_PATH);
}
// ===================== 图像处理工具 =====================
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
      raw: {
        width: metadata.width,
        height: metadata.height,
        channels: 4,
      },
    })
      .resize(
        Math.floor(metadata.width / scale),
        Math.floor(metadata.height / scale),
        { kernel: sharp.kernel.cubic }
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
}

// ===================== 数据生成器（方案二） =====================
class DataGenerator {
  static generate(lrTensor, hrHeight, hrWidth, scale) {
    return tf.tidy(() => {
      // 1. 生成 (dx, dy) 偏移量 [H_sr, W_sr, 2]
      const offsets = tf.buffer([hrHeight, hrWidth, 2]);

      // 2. 生成权重矩阵 [H_sr, W_sr, 16]
      const weights = tf.buffer([hrHeight, hrWidth, 16]);

      for (let y_hr = 0; y_hr < hrHeight; y_hr++) {
        for (let x_hr = 0; x_hr < hrWidth; x_hr++) {
          // 计算亚像素偏移 (dx, dy)
          const { dx, dy } = this.calculateSubpixelOffset(x_hr, y_hr, scale);
          offsets.set(dx, y_hr, x_hr, 0);
          offsets.set(dy, y_hr, x_hr, 1);

          // 计算双三次权重
          const weightValues = this.calculateBicubicWeights(dx, dy);
          for (let i = 0; i < 16; i++) {
            weights.set(weightValues[i], y_hr, x_hr, i);
          }
        }
      }
      // 2. 验证权重数值
      const weightSample = weights.values.slice(0, 16); // 第一个像素的权重
      const sumWeights = weightSample.reduce((a, b) => a + b, 0);

      return {
        offsets: offsets.toTensor(),
        weights: weights.toTensor(),
      };
    });
  }

  static calculateSubpixelOffset(x_hr, y_hr, scale) {
    /*  const x_lr = (x_hr + 0.5) / scale - 0.5;
    const y_lr = (y_hr + 0.5) / scale - 0.5;
    return {
      dx: x_lr - Math.floor(x_lr),
      dy: y_lr - Math.floor(y_lr),
    }; */
    // 正确计算方法：将HR坐标映射到LR网格中心
    const x_lr = (x_hr + 0.5) / scale;
    const y_lr = (y_hr + 0.5) / scale;

    return {
      dx: x_lr - Math.floor(x_lr) - 0.5, // 中心对齐
      dy: y_lr - Math.floor(y_lr) - 0.5,
    };
  }

  static calculateBicubicWeights(dx, dy, a = -0.5) {
    // 修正：使用标准双三次插值坐标偏移
    const cubic = (t) => {
      t = Math.abs(t);
      return t >= 2
        ? 0
        : t >= 1
        ? a * (t * t * t - 5 * t * t + 8 * t - 4)
        : (a + 2) * t * t * t - (a + 3) * t * t + 1;
    };

    // 关键修正：使用标准4x4网格坐标
    const grid = [
      [
        cubic(1 + dx) * cubic(1 + dy),
        cubic(dx) * cubic(1 + dy),
        cubic(1 - dx) * cubic(1 + dy),
        cubic(2 - dx) * cubic(1 + dy),
      ],
      [
        cubic(1 + dx) * cubic(dy),
        cubic(dx) * cubic(dy),
        cubic(1 - dx) * cubic(dy),
        cubic(2 - dx) * cubic(dy),
      ],
      [
        cubic(1 + dx) * cubic(1 - dy),
        cubic(dx) * cubic(1 - dy),
        cubic(1 - dx) * cubic(1 - dy),
        cubic(2 - dx) * cubic(1 - dy),
      ],
      [
        cubic(1 + dx) * cubic(2 - dy),
        cubic(dx) * cubic(2 - dy),
        cubic(1 - dx) * cubic(2 - dy),
        cubic(2 - dx) * cubic(2 - dy),
      ],
    ];

    // 展平为数组并归一化
    const weights = grid.flat(2);
    const sum = weights.reduce((a, b) => a + b, 0);
    return sum > 1e-6 ? weights.map((w) => w / sum) : new Array(16).fill(0);
  }
}

// ===================== 数据存储优化 =====================
class DataSaver {
  static async saveWithHeader(tensor, filePath, expectedShape) {
    // 确保 tensor 是 Float32 类型
    const tensorFloat = tensor.toFloat();
    const header = Buffer.alloc(12);
    header.writeUInt32LE(expectedShape[0], 0); // Height
    header.writeUInt32LE(expectedShape[1], 4); // Width
    header.writeUInt32LE(expectedShape[2], 8); // Channels

    // 直接获取 Float32Array 的底层 ArrayBuffer
    const data = await tensorFloat.data();
    const dataBuffer = Buffer.from(data.buffer);

    // 写入文件
    await fs.promises.writeFile(filePath, Buffer.concat([header, dataBuffer]));
  }
}

// ===================== 修改后的主处理流程 =====================
async function processImages() {
  // 初始化目录
  [OUTPUT_DIR.TRAIN_X, OUTPUT_DIR.TRAIN_OFFSET, OUTPUT_DIR.TRAIN_Y].forEach(
    (dir) => {
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    }
  );

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

      // 3. 立即更新元数据
      await updateMetadata(baseName, H_lr, W_lr, H_sr, W_sr);

      // 4. 生成并保存数据（保持原有逻辑）
      const lrTensor = await ImageProcessor.downscale(
        hrTensor,
        metadata,
        SCALE_FACTOR
      );
      const { offsets, weights } = DataGenerator.generate(
        lrTensor,
        H_sr,
        W_sr,
        SCALE_FACTOR
      );

      // 修改主处理流程中的保存代码
      await DataSaver.saveWithHeader(
        lrTensor,
        path.join(OUTPUT_DIR.TRAIN_X, `${baseName}.bin`),
        [lrTensor.shape[0], lrTensor.shape[1], 4] // [H_lr, W_lr, 4]
      );

      await DataSaver.saveWithHeader(
        offsets,
        path.join(OUTPUT_DIR.TRAIN_OFFSET, `${baseName}.bin`),
        [offsets.shape[0], offsets.shape[1], 2] // [H_sr, W_sr, 2]
      );

      await DataSaver.saveWithHeader(
        weights,
        path.join(OUTPUT_DIR.TRAIN_Y, `${baseName}.bin`),
        [weights.shape[0], weights.shape[1], 16] // 关键修改：16通道
      );

      tf.dispose([hrTensor, lrTensor, offsets, weights]);
    } catch (err) {
      console.error(`Error processing ${imgFile}:`, err.message);
    }
  }
}

// ===================== 执行入口 =====================
processImages()
  .then(() => console.log("Processing completed!"))
  .catch((err) => console.error("Fatal error:", err));
