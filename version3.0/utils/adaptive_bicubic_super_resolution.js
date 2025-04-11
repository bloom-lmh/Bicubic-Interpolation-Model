const sharp = require("sharp");
const { createCanvas, loadImage } = require("canvas");
const fs = require("fs");
const pc = require("./compare_performance");
const { HRID } = require("./config");
const MN = -0.5;
const LR_IMAGEPATH = `./cp_image/lr_images/${HRID}_downsample.png`;
const REBUILD_HR_IMAGEPATH = `./cp_image/rebuild_hr_images/${HRID}/adaptive_bicubic_${MN}.png`;

function ultimateBicubicInterpolation(input, scale, a = -0.5) {
  const { width: w, height: h, data } = input;
  const nw = Math.round(w * scale);
  const nh = Math.round(h * scale);

  // 创建输出画布
  const canvas = createCanvas(nw, nh);
  const ctx = canvas.getContext("2d");
  const output = ctx.createImageData(nw, nh);

  // 预计算亮度图
  const lumaData = new Float32Array(w * h);
  for (let i = 0; i < data.length; i += 4) {
    lumaData[i / 4] =
      data[i] * 0.2126 + data[i + 1] * 0.7152 + data[i + 2] * 0.0722;
  }

  // 三次样条权重函数（优化版）
  const cubicWeight = (() => {
    const cache = new Map();
    return (x) => {
      x = Math.abs(x).toFixed(2);
      if (!cache.has(x)) {
        const xVal = parseFloat(x);
        let res = 0;
        if (xVal <= 1) res = (a + 2) * xVal ** 3 - (a + 3) * xVal ** 2 + 1;
        else if (xVal <= 2)
          res = a * xVal ** 3 - 5 * a * xVal ** 2 + 8 * a * xVal - 4 * a;
        cache.set(x, res);
      }
      return cache.get(x);
    };
  })();

  // 局部对比度分析
  const analyzeLocalContrast = (x, y) => {
    const radius = 2;
    let sum = 0,
      sumSq = 0,
      count = 0;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const px = Math.min(w - 1, Math.max(0, x + dx));
        const py = Math.min(h - 1, Math.max(0, y + dy));
        const luma = lumaData[py * w + px];
        sum += luma;
        sumSq += luma * luma;
        count++;
      }
    }
    const variance = (sumSq - (sum * sum) / count) / count;
    return {
      contrast: Math.sqrt(variance),
      isFlat: variance < 10,
      isEdge: variance > 50,
    };
  };

  // 智能权重调整策略
  const getAdaptiveWeight = (baseWeight, centerIdx, currentIdx, regionType) => {
    const lumaDiff = Math.abs(
      lumaData[centerIdx / 4] - lumaData[currentIdx / 4]
    );

    // 边缘区域处理
    if (regionType.isEdge) {
      const edgePreserveFactor = 1.0 + 0.5 * Math.min(1, lumaDiff / 50);
      return baseWeight * edgePreserveFactor;
    }

    // 平坦区域处理
    if (regionType.isFlat) {
      const noiseSuppress = Math.max(0.5, 1 - lumaDiff / 30);
      return baseWeight * noiseSuppress;
    }

    // 普通纹理区域
    const similarity = Math.exp(-lumaDiff / 20);
    return baseWeight * (0.8 + 0.4 * similarity);
  };

  // 主处理循环
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) {
      const ox = x / scale;
      const oy = y / scale;
      const centerX = Math.min(w - 1, Math.max(0, Math.round(ox)));
      const centerY = Math.min(h - 1, Math.max(0, Math.round(oy)));
      const centerIdx = (centerY * w + centerX) * 4;
      const regionType = analyzeLocalContrast(centerX, centerY);

      let r = 0,
        g = 0,
        b = 0,
        a = 0;
      let weightSum = 0;
      const x0 = Math.floor(ox) - 1;
      const y0 = Math.floor(oy) - 1;

      // 4x4邻域处理
      for (let m = 0; m < 4; m++) {
        for (let n = 0; n < 4; n++) {
          const px = Math.min(w - 1, Math.max(0, x0 + m));
          const py = Math.min(h - 1, Math.max(0, y0 + n));
          const idx = (py * w + px) * 4;

          // 计算基础权重
          const wx = cubicWeight(ox - px);
          const wy = cubicWeight(oy - py);
          let weight = wx * wy;

          // 应用智能调整
          if (px !== centerX || py !== centerY) {
            weight = getAdaptiveWeight(weight, centerIdx, idx, regionType);
          }

          // 累加计算
          r += data[idx] * weight;
          g += data[idx + 1] * weight;
          b += data[idx + 2] * weight;
          a += data[idx + 3] * weight;
          weightSum += weight;
        }
      }

      // 输出结果
      const outIdx = (y * nw + x) * 4;
      output.data[outIdx] = Math.round(r / weightSum);
      output.data[outIdx + 1] = Math.round(g / weightSum);
      output.data[outIdx + 2] = Math.round(b / weightSum);
      output.data[outIdx + 3] = Math.round(a / weightSum);
    }
  }

  return output;
}

async function superResolveUltimate(inputPath, outputPath, scale, a = -0.5) {
  try {
    // 加载图像
    const { data, info } = await sharp(inputPath)
      .raw()
      .ensureAlpha()
      .toBuffer({ resolveWithObject: true });

    // 执行插值
    console.time("Ultimate Bicubic Interpolation");
    const outputImage = ultimateBicubicInterpolation(
      {
        width: info.width,
        height: info.height,
        data: new Uint8ClampedArray(data),
      },
      scale,
      a
    );
    console.timeEnd("Ultimate Bicubic Interpolation");

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
pc(() => superResolveUltimate(LR_IMAGEPATH, REBUILD_HR_IMAGEPATH, 4, MN), {
  testItem: "adaptive_bicubic",
});
