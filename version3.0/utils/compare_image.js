const sharp = require("sharp");
const SSIM = require("ssim.js");
const PSNR = require("psnr");
const fs = require("fs-extra");
const path = require("path");
const MODEL = "1e-3-30";
const HRID = "0829";
const MN = -1;
const HR_IMAGEPATH = `../cp_image/hr_images/${HRID}.png`;
const REBUILD_HR_IMAGEPATH_MODEL = `../cp_image/rebuild_hr_images/${HRID}_rebuild_${MODEL}.png`;
const REBUILD_HR_IMAGEPATH_BICUBIC = `../cp_image/rebuild_hr_images/${HRID}_rebuild_bicubic_${MN}.png`;
const REBUILD_HR_IMAGEPATH_NN = `../cp_image/rebuild_hr_images/${HRID}_rebuild_nearest.png`;
const REBUILD_HR_IMAGEPATH_BILINEAR = `../cp_image/rebuild_hr_images/${HRID}_rebuild_bilinear.png`;
const REBUILD_HR_IMAGEPATH_LANCZOS = `../cp_image/rebuild_hr_images/${HRID}_rebuild_lanczos.png`;
const REBUILD_HR_IMAGEPATH_ESPCN_THICK = `../cp_image/rebuild_hr_images/${HRID}_rebuild_espcn_thick.png`;
const REBUILD_HR_IMAGEPATH_ESPCN_MEDIUM = `../cp_image/rebuild_hr_images/${HRID}_rebuild_espcn_medium.png`;
const OR_DIFFOAPARH = `../cp_image/or_diff/`;
class ImageComparator {
  constructor() {
    this.metrics = {
      psnr: {
        name: "PSNR",
        value: null,
        unit: "dB",
        desc: "峰值信噪比 (越高越好)",
        format: (v) =>
          v === Infinity ? "∞" : typeof v === "number" ? v.toFixed(2) : "N/A",
      },
      ssim: {
        name: "SSIM",
        value: null,
        unit: "",
        desc: "结构相似性 (1为完美匹配)",
        format: (v) => (typeof v === "number" ? v.toFixed(4) : "N/A"),
      },
      mse: {
        name: "MSE",
        value: null,
        unit: "",
        desc: "均方误差 (越低越好)",
        format: (v) => (typeof v === "number" ? v.toFixed(2) : "N/A"),
      },
    };
    this.img1 = null;
    this.img2 = null;
  }

  async loadImages(imgPath1, imgPath2) {
    try {
      this.img1 = await sharp(imgPath1)
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
      this.img2 = await sharp(imgPath2)
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

      if (!this.img1 || !this.img2) {
        throw new Error("图像加载失败");
      }

      if (
        this.img1.info.width !== this.img2.info.width ||
        this.img1.info.height !== this.img2.info.height
      ) {
        throw new Error(
          `图像尺寸不匹配: ${imgPath1}(${this.img1.info.width}x${this.img1.info.height}) vs ${imgPath2}(${this.img2.info.width}x${this.img2.info.height})`
        );
      }
    } catch (error) {
      throw new Error(`图像加载错误: ${error.message}`);
    }
  }

  calculateMetrics() {
    if (!this.img1 || !this.img2) {
      throw new Error("请先加载图像");
    }

    const { width, height } = this.img1.info;
    const pixels = width * height;

    try {
      // 转换为灰度图像计算
      const img1Data = this.toGrayScale(
        this.img1.data,
        this.img1.info.channels
      );
      const img2Data = this.toGrayScale(
        this.img2.data,
        this.img2.info.channels
      );

      // 计算MSE
      let mse = 0;
      for (let i = 0; i < img1Data.length; i++) {
        mse += Math.pow(img1Data[i] - img2Data[i], 2);
      }
      mse /= pixels;
      this.metrics.mse.value = mse;

      // 计算PSNR
      const maxPixelValue = 255;
      if (mse === 0) {
        this.metrics.psnr.value = Infinity;
      } else {
        this.metrics.psnr.value =
          10 * Math.log10((maxPixelValue * maxPixelValue) / mse);
      }

      // 计算SSIM
      const ssimResult = SSIM.ssim(
        { data: img1Data, width, height },
        { data: img2Data, width, height },
        { windowSize: 11 }
      );

      this.metrics.ssim.value = ssimResult.mssim;
    } catch (error) {
      throw new Error(`指标计算错误: ${error.message}`);
    }
  }

  toGrayScale(data, channels) {
    const grayData = new Uint8Array(data.length / channels);
    for (let i = 0; i < grayData.length; i++) {
      const r = data[i * channels];
      const g = data[i * channels + 1];
      const b = data[i * channels + 2];
      grayData[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }
    return grayData;
  }

  async generateDiffImage(outputPath) {
    if (!this.img1 || !this.img2) {
      throw new Error("请先加载图像");
    }

    try {
      const { width, height } = this.img1.info;
      const diffBuffer = Buffer.alloc(this.img1.data.length);

      for (let i = 0; i < this.img1.data.length; i += 4) {
        const diff = Math.abs(this.img1.data[i] - this.img2.data[i]) / 255;
        diffBuffer[i] = 255; // R
        diffBuffer[i + 1] = Math.round(255 * (1 - diff)); // G
        diffBuffer[i + 2] = Math.round(255 * (1 - diff)); // B
        diffBuffer[i + 3] = 255; // Alpha
      }

      await sharp(diffBuffer, {
        raw: { width, height, channels: 4 },
      })
        .composite([
          { input: this.img1.data, raw: this.img1.info, blend: "over" },
        ])
        .png()
        .toFile(outputPath);
    } catch (error) {
      throw new Error(`差异图生成失败: ${error.message}`);
    }
  }

  printReport() {
    console.log("╔════════════════════════════════════╗");
    console.log("║       专业图像质量评估报告        ║");
    console.log("╠══════════════╦════════╦═══════════╣");
    console.log("║ 指标名称     ║ 值     ║ 说明      ║");
    console.log("╠══════════════╬════════╬═══════════╣");

    Object.values(this.metrics).forEach((metric) => {
      // 使用预定义格式化函数处理值
      const formattedValue = metric.format(metric.value);

      console.log(
        `║ ${metric.name.padEnd(12)} ║ ${formattedValue.padStart(
          6
        )} ${metric.unit.padEnd(2)} ║ ${metric.desc} ║`
      );
    });

    console.log("╚══════════════╩════════╩═══════════╝");
  }
}

async function compare(comparator, rebuildHRImagePath) {
  try {
    const comparator = new ImageComparator();

    // 1. 加载图像
    await comparator.loadImages(
      path.join(__dirname, HR_IMAGEPATH),
      path.join(__dirname, rebuildHRImagePath)
    );

    // 2. 计算质量指标
    comparator.calculateMetrics();
    // 3. 打印报告前添加调试信息
    console.log("调试信息 - 计算后的指标值:", {
      psnr: comparator.metrics.psnr.value,
      ssim: comparator.metrics.ssim.value,
      mse: comparator.metrics.mse.value,
    });

    // 4. 生成差异可视化图
    let diffPath;
    if (rebuildHRImagePath.includes("bicubic")) {
      diffPath = path.join(
        __dirname,
        OR_DIFFOAPARH,
        `diff_${HRID}_bicubic_${MN}.png`
      );
    } else {
      diffPath = path.join(
        __dirname,
        OR_DIFFOAPARH,
        `diff_${HRID}_${MODEL}.png`
      );
    }

    await comparator.generateDiffImage(diffPath);
    // 5. 打印专业报告
    comparator.printReport();

    console.log(`图像对比完成，差异图已保存为 diff_${HRID}_${MODEL}.png`);
  } catch (error) {
    console.error("❌ 错误详情:", {
      message: error.message,
      stack: error.stack,
    });
    process.exit(1);
  }
}
// 使用示例（带错误处理）
(async () => {
  const comparator = new ImageComparator();
  await compare(comparator, REBUILD_HR_IMAGEPATH_MODEL);
  await compare(comparator, REBUILD_HR_IMAGEPATH_BICUBIC);
  await compare(comparator, REBUILD_HR_IMAGEPATH_NN);
  await compare(comparator, REBUILD_HR_IMAGEPATH_BILINEAR);
  await compare(comparator, REBUILD_HR_IMAGEPATH_LANCZOS);
  await compare(comparator, REBUILD_HR_IMAGEPATH_ESPCN_THICK);
  await compare(comparator, REBUILD_HR_IMAGEPATH_ESPCN_MEDIUM);
})();
