const sharp = require("sharp");
const { createCanvas, loadImage } = require("canvas");
const fs = require("fs");
const pc = require("./compare_performance");
const HRID = "0829";
const LR_IMAGEPATH = `./cp_image/lr_images/${HRID}_downsample.png`;
const REBUILD_HR_IMAGEPATH = `./cp_image/rebuild_hr_images/${HRID}_rebuild_lanczos.png`;

function lanczosKernel(x, a = 3) {
  if (x === 0) return 1;
  if (Math.abs(x) > a) return 0;
  const px = Math.PI * x;
  return (a * Math.sin(px) * Math.sin(px / a)) / (px * px);
}

function lanczosInterpolation(input, scale, a = 3) {
  const { width: w, height: h, data } = input;
  const nw = Math.round(w * scale);
  const nh = Math.round(h * scale);

  // 创建输出画布
  const canvas = createCanvas(nw, nh);
  const ctx = canvas.getContext("2d");
  const output = ctx.createImageData(nw, nh);

  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      // 计算原始坐标
      const ox = x / scale;
      const oy = y / scale;

      // 确定采样窗口边界
      const xMin = Math.max(0, Math.floor(ox) - a + 1);
      const xMax = Math.min(w - 1, Math.floor(ox) + a);
      const yMin = Math.max(0, Math.floor(oy) - a + 1);
      const yMax = Math.min(h - 1, Math.floor(oy) + a);

      let r = 0,
        g = 0,
        b = 0,
        alpha = 0;
      let weightSum = 0;

      // 遍历采样窗口
      for (let sy = yMin; sy <= yMax; sy++) {
        for (let sx = xMin; sx <= xMax; sx++) {
          // 计算权重
          const dx = ox - sx;
          const dy = oy - sy;
          const wx = lanczosKernel(dx, a);
          const wy = lanczosKernel(dy, a);
          const weight = wx * wy;

          // 获取像素值
          const idx = (sy * w + sx) * 4;
          r += data[idx] * weight;
          g += data[idx + 1] * weight;
          b += data[idx + 2] * weight;
          alpha += data[idx + 3] * weight;
          weightSum += weight;
        }
      }

      // 归一化并写入输出
      const outIdx = (y * nw + x) * 4;
      output.data[outIdx] = Math.round(r / weightSum);
      output.data[outIdx + 1] = Math.round(g / weightSum);
      output.data[outIdx + 2] = Math.round(b / weightSum);
      output.data[outIdx + 3] = Math.round(alpha / weightSum);
    }
  }

  return output;
}

async function superResolveLanczos(inputPath, outputPath, scale, a = 3) {
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
    console.time("Lanczos Interpolation");
    const outputImage = lanczosInterpolation(inputImage, scale, a);
    console.timeEnd("Lanczos Interpolation");

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
pc(
  () =>
    superResolveLanczos(
      LR_IMAGEPATH, // 低分辨率输入图像
      REBUILD_HR_IMAGEPATH, // 超分结果保存路径
      4, // 放大4倍
      3 // Lanczos窗口大小
    ),
  {
    testItem: "lanczos",
  }
);
