const sharp = require("sharp");
const { createCanvas, loadImage } = require("canvas");
const fs = require("fs");
const pc = require("./compare_performance");
const { HRID } = require("./config");
const LR_IMAGEPATH = `./cp_image/lr_images/${HRID}_downsample.png`;
const REBUILD_HR_IMAGEPATH = `./cp_image/rebuild_hr_images/${HRID}/nearest.png`;

/**
 * 最近邻插值算法实现
 * @param {ImageData} input - 输入图像数据
 * @param {number} scale - 缩放倍数
 * @returns {ImageData} 超分后的图像数据
 */
function nearestNeighborInterpolation(input, scale) {
  const { width: w, height: h, data } = input;
  const nw = Math.round(w * scale);
  const nh = Math.round(h * scale);

  // 创建输出画布
  const canvas = createCanvas(nw, nh);
  const ctx = canvas.getContext("2d");
  const output = ctx.createImageData(nw, nh);

  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      // 计算原始坐标（直接取最近的像素）
      const ox = Math.round(x / scale);
      const oy = Math.round(y / scale);

      // 确保坐标在范围内
      const px = Math.min(w - 1, Math.max(0, ox));
      const py = Math.min(h - 1, Math.max(0, oy));

      // 获取最近像素的值
      const idx = (py * w + px) * 4;
      const outIdx = (y * nw + x) * 4;

      output.data[outIdx] = data[idx]; // R
      output.data[outIdx + 1] = data[idx + 1]; // G
      output.data[outIdx + 2] = data[idx + 2]; // B
      output.data[outIdx + 3] = data[idx + 3]; // A
    }
  }

  return output;
}

/**
 * 执行超分辨率处理（最近邻插值）
 * @param {string} inputPath - 输入图像路径
 * @param {string} outputPath - 输出图像路径
 * @param {number} scale - 缩放倍数
 */
async function superResolveNearest(inputPath, outputPath, scale) {
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
    console.time("Nearest Neighbor Interpolation");
    let outputImage;
    pc(() => (outputImage = nearestNeighborInterpolation(inputImage, scale)), {
      testItem: "nearest",
    });
    console.timeEnd("Nearest Neighbor Interpolation");

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

superResolveNearest(
  LR_IMAGEPATH, // 低分辨率输入图像
  REBUILD_HR_IMAGEPATH, // 超分结果保存路径
  4 // 放大4倍
);
// 使用示例
/* pc(
  () =>
    superResolveNearest(
      LR_IMAGEPATH, // 低分辨率输入图像
      REBUILD_HR_IMAGEPATH, // 超分结果保存路径
      4 // 放大4倍
    ),
  {
    testItem: "nearest",
  }
); */
