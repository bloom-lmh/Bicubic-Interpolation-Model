const sharp = require("sharp");
const { createCanvas, loadImage } = require("canvas");
const fs = require("fs");
const HRID = "0829";
const LR_IMAGEPATH = `./cp_image/lr_images/${HRID}_downsample.png`;
const REBUILD_HR_IMAGEPATH = `./cp_image/rebuild_hr_images/${HRID}_rebuild_bicubic.png`;
/**
 * 双三次插值算法实现
 * @param {ImageData} input - 输入图像数据
 * @param {number} scale - 缩放倍数
 * @param {number} [a=-0.5] - 三次样条参数(默认Catmull-Rom样条)
 * @returns {ImageData} 超分后的图像数据
 */
function bicubicInterpolation(input, scale, a = -0.5) {
  const { width: w, height: h, data } = input;
  const nw = Math.round(w * scale);
  const nh = Math.round(h * scale);

  // 创建输出画布
  const canvas = createCanvas(nw, nh);
  const ctx = canvas.getContext("2d");
  const output = ctx.createImageData(nw, nh);

  // 三次样条权重函数
  const cubicWeight = (x) => {
    x = Math.abs(x);
    if (x <= 1) return (a + 2) * x ** 3 - (a + 3) * x ** 2 + 1;
    if (x <= 2) return a * x ** 3 - 5 * a * x ** 2 + 8 * a * x - 4 * a;
    return 0;
  };

  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      // 计算原始坐标
      const ox = x / scale;
      const oy = y / scale;

      // 确定周围16个像素
      const x0 = Math.floor(ox) - 1;
      const y0 = Math.floor(oy) - 1;

      let r = 0,
        g = 0,
        b = 0,
        a = 0;
      let weightSum = 0;

      // 遍历4x4邻域
      for (let m = 0; m < 4; m++) {
        for (let n = 0; n < 4; n++) {
          const px = Math.min(w - 1, Math.max(0, x0 + m));
          const py = Math.min(h - 1, Math.max(0, y0 + n));

          // 计算权重
          const wx = cubicWeight(ox - px);
          const wy = cubicWeight(oy - py);
          const weight = wx * wy;

          // 获取像素值
          const idx = (py * w + px) * 4;
          r += data[idx] * weight;
          g += data[idx + 1] * weight;
          b += data[idx + 2] * weight;
          a += data[idx + 3] * weight;

          weightSum += weight;
        }
      }

      // 归一化并写入输出
      const outIdx = (y * nw + x) * 4;
      output.data[outIdx] = Math.round(r / weightSum);
      output.data[outIdx + 1] = Math.round(g / weightSum);
      output.data[outIdx + 2] = Math.round(b / weightSum);
      output.data[outIdx + 3] = Math.round(a / weightSum);
    }
  }

  return output;
}

/**
 * 执行超分辨率处理
 * @param {string} inputPath - 输入图像路径
 * @param {string} outputPath - 输出图像路径
 * @param {number} scale - 缩放倍数
 * @param {number} [a=-0.5] - 三次样条参数
 */
async function superResolve(inputPath, outputPath, scale, a = -0.5) {
  try {
    // 使用sharp加载图像并获取原始数据
    const { data, info } = await sharp(inputPath)
      .raw()
      .ensureAlpha()
      .toBuffer({ resolveWithObject: true });

    // 转换为ImageData格式
    const inputImage = {
      width: info.width,
      height: info.height,
      data: new Uint8ClampedArray(data),
    };

    // 执行插值
    console.time("Bicubic Interpolation");
    const outputImage = bicubicInterpolation(inputImage, scale, a);
    console.timeEnd("Bicubic Interpolation");

    // 保存结果
    const canvas = createCanvas(outputImage.width, outputImage.height);
    const ctx = canvas.getContext("2d");
    ctx.putImageData(outputImage, 0, 0);

    await sharp(canvas.toBuffer()).png({ quality: 100 }).toFile(outputPath);

    console.log(
      `超分完成: ${info.width}x${info.height} → ${outputImage.width}x${outputImage.height}`
    );
  } catch (err) {
    console.error("处理失败:", err);
  }
}

// 使用示例
superResolve(
  LR_IMAGEPATH, // 低分辨率输入图像
  REBUILD_HR_IMAGEPATH, // 超分结果保存路径
  4, // 放大4倍
  -0.5 // Mitchell-Netravali参数
);
