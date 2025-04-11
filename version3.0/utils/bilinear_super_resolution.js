const sharp = require("sharp");
const { createCanvas, loadImage } = require("canvas");
const fs = require("fs");
const pc = require("./compare_performance");
const { HRID } = require("./config");
const LR_IMAGEPATH = `./cp_image/lr_images/${HRID}_downsample.png`;
const REBUILD_HR_IMAGEPATH = `./cp_image/rebuild_hr_images/${HRID}/bilinear.png`;

function bilinearInterpolation(input, scale) {
  const { width: w, height: h, data } = input;
  const nw = Math.round(w * scale);
  const nh = Math.round(h * scale);

  // 创建输出画布
  const canvas = createCanvas(nw, nh);
  const ctx = canvas.getContext("2d");
  const output = ctx.createImageData(nw, nh);

  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      // 计算原始坐标（浮点数）
      const ox = x / scale;
      const oy = y / scale;

      // 确定最近的4个像素
      const x1 = Math.floor(ox);
      const y1 = Math.floor(oy);
      const x2 = Math.min(w - 1, x1 + 1);
      const y2 = Math.min(h - 1, y1 + 1);

      // 计算权重系数
      const dx = ox - x1;
      const dy = oy - y1;
      const w1 = (1 - dx) * (1 - dy);
      const w2 = dx * (1 - dy);
      const w3 = (1 - dx) * dy;
      const w4 = dx * dy;

      // 获取4个像素的RGBA值
      const idx1 = (y1 * w + x1) * 4;
      const idx2 = (y1 * w + x2) * 4;
      const idx3 = (y2 * w + x1) * 4;
      const idx4 = (y2 * w + x2) * 4;

      // 加权计算新像素值
      const outIdx = (y * nw + x) * 4;
      for (let i = 0; i < 4; i++) {
        output.data[outIdx + i] = Math.round(
          data[idx1 + i] * w1 +
            data[idx2 + i] * w2 +
            data[idx3 + i] * w3 +
            data[idx4 + i] * w4
        );
      }
    }
  }

  return output;
}

async function superResolveBilinear(inputPath, outputPath, scale) {
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
    console.time("Bilinear Interpolation");
    let outputImage;
    pc(() => (outputImage = bilinearInterpolation(inputImage, scale)), {
      testItem: "bilinear",
    });
    console.timeEnd("Bilinear Interpolation");

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
superResolveBilinear(
  LR_IMAGEPATH, // 低分辨率输入图像
  REBUILD_HR_IMAGEPATH, // 超分结果保存路径
  4 // 放大4倍
);
