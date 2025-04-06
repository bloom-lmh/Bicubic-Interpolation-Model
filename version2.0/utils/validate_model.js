const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { promisify } = require("util");
const readFile = promisify(fs.readFile);
const PNG = require("pngjs").PNG; // 需要安装 pngjs 包
const ssim = require("ssim.js");
// ===================== 配置 =====================
const TEST_DIR = {
  X: "./data/test/X/", // 测试集输入 (低分辨率图像块)
  Y: "./data/test/Y/", // 测试集标签 (双三次插值权重)
  HR: "./data/test/HR/", // 高分辨率原图 (用于质量评估)
};
const MODEL_DIR = "file://./model/pretrained/";
const SCALE_FACTOR = 4; // 超分比例
const BATCH_SIZE = 64; // 验证批量大小

// ===================== 工具函数 =====================
async function loadTestData(dirPath, features) {
  // 复用训练时的数据加载逻辑
  const files = (await fs.promises.readdir(dirPath)).filter((f) =>
    f.endsWith(".bin")
  );
  const tensors = [];

  for (const file of files.slice(0, 10)) {
    // 只加载前10个文件加速验证
    const buffer = await readFile(path.join(dirPath, file));
    const data = new Float32Array(buffer.buffer);
    const tensor = tf.tensor2d(data, [data.length / features, features]);
    tensors.push(tensor);
  }

  return tf.concat(tensors);
}
// 1. 下采样函数
/* function downsample(hrImage, scale) {
  const lrWidth = Math.floor(hrImage.width / scale);
  const lrHeight = Math.floor(hrImage.height / scale);
  const lrData = new Uint8Array(lrWidth * lrHeight * 3); // RGB格式

  for (let y = 0; y < lrHeight; y++) {
    for (let x = 0; x < lrWidth; x++) {
      const srcX = x * scale;
      const srcY = y * scale;
      const srcIdx = (srcY * hrImage.width + srcX) * 3;
      const dstIdx = (y * lrWidth + x) * 3;

      lrData[dstIdx] = hrImage.data[srcIdx]; // R
      lrData[dstIdx + 1] = hrImage.data[srcIdx + 1]; // G
      lrData[dstIdx + 2] = hrImage.data[srcIdx + 2]; // B
    }
  }

  return {
    width: lrWidth,
    height: lrHeight,
    data: lrData,
  };
} */
async function downsample(hrImage, scale) {
  const buffer = Buffer.from(hrImage.data);
  const { width, height } = hrImage;

  const lrBuffer = await sharp(buffer, {
    raw: { width, height, channels: 3 },
  })
    .resize(Math.floor(width / scale), Math.floor(height / scale), {
      kernel: "cubic",
      fastShrinkOnLoad: false, // 关闭快速缩小以保持质量
    })
    .toFormat("rgb")
    .toBuffer();

  return {
    width: Math.floor(width / scale),
    height: Math.floor(height / scale),
    data: new Uint8Array(lrBuffer),
  };
}
// 2. 保存图像函数
function saveImage(imageData, filePath) {
  const png = new PNG({
    width: imageData.width,
    height: imageData.height,
    colorType: 2, // RGB模式
  });
  png.data = Buffer.from(imageData.data);
  fs.writeFileSync(filePath, PNG.sync.write(png));
}
// 4. 提取图像块
function extract4x4Channel(lrImage, x, y, channel) {
  const patch = [];
  for (let dy = 0; dy < 4; dy++) {
    for (let dx = 0; dx < 4; dx++) {
      const idx = ((y + dy) * lrImage.width + (x + dx)) * 3;
      switch (channel) {
        case "R":
          patch.push(lrImage.data[idx] / 255);
          break;
        case "G":
          patch.push(lrImage.data[idx + 1] / 255);
          break;
        case "B":
          patch.push(lrImage.data[idx + 2] / 255);
          break;
        case "Y": // 亮度通道
          const r = lrImage.data[idx] / 255;
          const g = lrImage.data[idx + 1] / 255;
          const b = lrImage.data[idx + 2] / 255;
          patch.push(0.299 * r + 0.587 * g + 0.114 * b);
          break;
      }
    }
  }
  return patch;
}
function validateHRDimensions(hrImage, scale) {
  if (hrImage.width % scale !== 0 || hrImage.height % scale !== 0) {
    throw new Error(`HR图像尺寸必须为${scale}的整数倍`);
  }
}
function extractPatches(lrImage) {
  const patches = [];
  for (let y = 0; y < lrImage.height - 4; y++) {
    for (let x = 0; x < lrImage.width - 4; x++) {
      // 四通道特征: R(16) + G(16) + B(16) + Y(16) = 64
      const features = [
        ...extract4x4Channel(lrImage, x, y, "R"),
        ...extract4x4Channel(lrImage, x, y, "G"),
        ...extract4x4Channel(lrImage, x, y, "B"),
        ...extract4x4Channel(lrImage, x, y, "Y"),
      ];
      // 添加坐标偏移量
      const dx = (x % 4) / 4;
      const dy = (y % 4) / 4;
      patches.push([...features, dx, dy]); // 64 + 2 = 66
    }
  }
  return patches;
}

// 6. 重组超分图像
function assemblePatches(patchesTensor, targetWidth) {
  const patches = patchesTensor.arraySync();
  const channels = 3;
  const patchSize = 4;

  const totalPatches = patches.length;
  const rows = Math.ceil(Math.sqrt(totalPatches));
  const srImage = new Uint8Array(
    rows * patchSize * rows * patchSize * channels
  );

  let patchIdx = 0;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < rows; x++) {
      if (patchIdx >= totalPatches) break;

      const patch = patches[patchIdx];
      for (let dy = 0; dy < patchSize; dy++) {
        for (let dx = 0; dx < patchSize; dx++) {
          const srcIdx = dy * patchSize * channels + dx * channels;
          const dstY = y * patchSize + dy;
          const dstX = x * patchSize + dx;
          const dstIdx = (dstY * targetWidth + dstX) * channels;

          srImage[dstIdx] = Math.min(255, patch[srcIdx] * 255); // R
          srImage[dstIdx + 1] = Math.min(255, patch[srcIdx + 1] * 255); // G
          srImage[dstIdx + 2] = Math.min(255, patch[srcIdx + 2] * 255); // B
        }
      }
      patchIdx++;
    }
  }

  return {
    width: targetWidth,
    height: Math.ceil(patchIdx / rows) * patchSize,
    data: srImage,
  };
}
// 1. 加载PNG图像
function loadPNG(filePath) {
  const data = fs.readFileSync(filePath);
  return PNG.sync.read(data);
}
// ===================== 修改后的图像处理函数 =====================
async function applyModel(lrImage, model) {
  try {
    // 1. 提取图像块
    const patches = extractPatches(lrImage);

    // 2. 转换为Tensor (确保特征维度为66)
    const features = tf.tensor2d(
      patches.flat(),
      [patches.length, 66] // 显式指定维度
    );

    // 3. 预测权重 (保持为Tensor)
    const weights = model.predict(features);

    // 4. 应用插值 (直接传递Tensor)
    const srPatches = applyBicubic(patches, weights);

    // 5. 重组图像
    return assemblePatches(srPatches, lrImage.width * SCALE_FACTOR);
  } catch (err) {
    console.error("applyModel失败:", err);
    throw err;
  }
}
function calculatePSNR(hrImage, srImage) {
  // 1. 裁剪到相同尺寸
  const width = Math.min(hrImage.width, srImage.width);
  const height = Math.min(hrImage.height, srImage.height);
  const hrData = cropImage(hrImage, width, height);
  const srData = cropImage(srImage, width, height);

  // 2. 转换为 Tensor
  const hrTensor = tf.tensor3d(hrData, [height, width, 3]);
  const srTensor = tf.tensor3d(srData, [height, width, 3]);

  // 3. 计算 MSE
  const mse = tf.losses.meanSquaredError(hrTensor, srTensor).dataSync()[0];

  // 4. 计算 PSNR (假设图像为 8-bit)
  const maxPixelValue = 255;
  return 20 * Math.log10(maxPixelValue / Math.sqrt(mse));
}

function cropImage(image, targetWidth, targetHeight) {
  const channels = 3;
  const cropped = new Uint8Array(targetWidth * targetHeight * channels);

  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const srcIdx = (y * image.width + x) * channels;
      const dstIdx = (y * targetWidth + x) * channels;
      cropped[dstIdx] = image.data[srcIdx];
      cropped[dstIdx + 1] = image.data[srcIdx + 1];
      cropped[dstIdx + 2] = image.data[srcIdx + 2];
    }
  }
  return cropped;
}
// ===================== 工具函数 =====================
function rgbToYMatrix(imageData) {
  const yMatrix = [];
  const { data, width, height } = imageData;

  for (let y = 0; y < height; y++) {
    const row = [];
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 3; // 假设输入为 3 通道数据
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      // BT.601 转换并钳制到 [0, 255]
      const yValue = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      row.push(Math.min(255, Math.max(0, yValue))); // 确保值在 0-255
    }
    yMatrix.push(row);
  }
  return yMatrix;
}

function calculateSSIM(hrImage, srImage) {
  // 转换为 Y 通道矩阵
  const hrY = rgbToYMatrix(hrImage);
  const srY = rgbToYMatrix(srImage);

  // 计算最小公共尺寸（确保为整数）
  const minWidth = Math.min(hrImage.width, srImage.width);
  const minHeight = Math.min(hrImage.height, srImage.height);

  // 精确裁剪（避免浮点误差）
  const croppedHrY = hrY
    .slice(0, minHeight)
    .map((row) => row.slice(0, minWidth));
  const croppedSrY = srY
    .slice(0, minHeight)
    .map((row) => row.slice(0, minWidth));

  // 调试：输出裁剪后的尺寸
  console.log("HR Y通道尺寸:", croppedHrY.length, "x", croppedHrY[0].length);
  console.log("SR Y通道尺寸:", croppedSrY.length, "x", croppedSrY[0].length);
  console.assert(
    croppedHrY.length === croppedSrY.length,
    `行数不匹配: HR=${croppedHrY.length}, SR=${croppedSrY.length}`
  );
  console.assert(
    croppedHrY[0].length === croppedSrY[0].length,
    `列数不匹配: HR=${croppedHrY[0].length}, SR=${croppedSrY[0].length}`
  );
  // 调用 ssim.js
  return ssim.ssim(croppedHrY, croppedSrY).mssim;
}
// ===================== 验证流程 =====================
async function validateModel() {
  try {
    // 1. 加载测试数据
    const [X_test, Y_test] = await Promise.all([
      loadTestData(TEST_DIR.X, 66), // 输入特征: 4x4x4像素块 + 2个坐标偏移
      loadTestData(TEST_DIR.Y, 16), // 标签权重: 16个插值权重
    ]);

    // 2. 加载模型
    const model = await tf.loadLayersModel(`${MODEL_DIR}/model.json`);
    model.summary();

    // 3. 预测权重并计算指标
    const yPred = model.predict(X_test);
    const testLoss = tf.losses.meanSquaredError(Y_test, yPred).dataSync()[0];

    console.log(`
    ========== 验证报告 ==========
    测试集样本数: ${X_test.shape[0]}
    MSE损失: ${testLoss.toFixed(6)}
    权重范围: [${yPred.min().dataSync()[0].toFixed(4)}, ${yPred
      .max()
      .dataSync()[0]
      .toFixed(4)}]
    权重和均值: ${yPred.sum(1).mean().dataSync()[0].toFixed(4)} (理论应为1.0)
    `);

    // 4. 可视化权重对比 (随机选5个样本)
    const sampleIndices = tf.util
      .createShuffledIndices(X_test.shape[0])
      .slice(0, 5);
    sampleIndices.forEach((i) => {
      const trueWeights = Y_test.slice([i, 0], [1, 16]).arraySync()[0];
      const predWeights = yPred.slice([i, 0], [1, 16]).arraySync()[0];

      console.log(`
      样本 ${i} 权重对比:
      真实值: [${trueWeights.map((v) => v.toFixed(3)).join(", ")}]
      预测值: [${predWeights.map((v) => v.toFixed(3)).join(", ")}]
      `);
    });

    // 5. 超分辨率重建质量评估
    const hrImage = await loadPNG(TEST_DIR.HR + "0801.png"); // 加载一张HR原图
    validateHRDimensions(hrImage, SCALE_FACTOR); // SCALE_FACTOR=4
    const lrImage = downsample(hrImage, SCALE_FACTOR); // 生成LR图像

    const srImage = await applyModel(lrImage, model); // 模型超分重建

    // 计算PSNR
    const psnr = calculatePSNR(hrImage, srImage);
    console.log(`\n超分重建质量: PSNR = ${psnr.toFixed(2)} dB`);
    // 计算SSIM
    /*  const ssimValue = calculateSSIM(hrImage, srImage);
    console.log(`
      ========== 质量报告SSIM ==========
      SSIM: ${ssimValue.toFixed(4)}
      `); */
    console.log(srImage);

    // 6. 保存可视化结果
    saveImage(lrImage, "./rebuild/lr.png");
    saveImage(srImage, "./rebuild/sr.png");
    saveImage(hrImage, "./rebuild/hr.png");
  } catch (err) {
    console.error("验证失败:", err);
  }
}

// ===================== 修正后的插值函数 =====================
function applyBicubic(patches, weightsTensor) {
  return tf.tidy(() => {
    // 将Tensor转换为JavaScript数组
    const weights = weightsTensor.arraySync();
    const srPatches = [];

    weights.forEach((w, i) => {
      const patch = patches[i];
      const srBlock = new Float32Array(4 * 4 * 3);

      // 应用权重插值 (示例)
      for (let c = 0; c < 3; c++) {
        for (let dy = 0; dy < 4; dy++) {
          for (let dx = 0; dx < 4; dx++) {
            const idx = dy * 4 + dx;
            srBlock[dy * 4 * 3 + dx * 3 + c] = patch
              .slice(c * 16, (c + 1) * 16)
              .reduce((sum, val, i) => sum + val * w[i], 0);
          }
        }
      }
      srPatches.push(...srBlock);
    });

    return tf.tensor3d(srPatches, [patches.length, 4, 4 * 3]);
  });
}

// ===================== 执行 =====================
validateModel();
