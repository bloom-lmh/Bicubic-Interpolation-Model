const sharp = require("sharp");
const SSIM = require("ssim.js");
const PSNR = require("psnr");
const fs = require("fs-extra");
const path = require("path");
const { createObjectCsvWriter } = require("csv-writer");
const MODEL = "1e-3-30";
//const { HRID } = require("./config"); // 假设config.js导出的是数组如: module.exports = { HRID: ['001', '002', '003'] }
const HRID = [
  "0801",
  "0802",
  "0803",
  "0807",
  "0829",
  "0843",
  "0855",
  "0884",
  "0886",
];
const MN = -0.5;

// 结果存储对象
const results = {
  metrics: [],
  summary: {},
};

// CSV写入器配置
const csvWriter = createObjectCsvWriter({
  path: path.join(__dirname, "../cp_image/metrics_report.csv"),
  header: [
    { id: "imageId", title: "IMAGE_ID" },
    { id: "method", title: "METHOD" },
    { id: "psnr", title: "PSNR(dB)" },
    { id: "ssim", title: "SSIM" },
    { id: "mse", title: "MSE" },
  ],
});

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

  printReport(imageId, method) {
    console.log(`\n╔══════════════════════════════════════════════╗`);
    console.log(`║       专业图像质量评估报告 (${imageId}-${method})      ║`);
    console.log(`╠══════════════╦════════╦═══════════╦══════════╣`);
    console.log(`║ 指标名称     ║ 值     ║ 单位      ║ 说明     ║`);
    console.log(`╠══════════════╬════════╬═══════════╬══════════╣`);

    Object.values(this.metrics).forEach((metric) => {
      const formattedValue = metric.format(metric.value);
      console.log(
        `║ ${metric.name.padEnd(12)} ║ ${formattedValue.padStart(
          6
        )} ║ ${metric.unit.padEnd(9)} ║ ${metric.desc.padEnd(8)} ║`
      );
    });

    console.log(`╚══════════════╩════════╩═══════════╩══════════╝`);

    // 存储结果
    results.metrics.push({
      imageId,
      method,
      psnr: this.metrics.psnr.value,
      ssim: this.metrics.ssim.value,
      mse: this.metrics.mse.value,
    });
  }
}

async function compare(imageId, method) {
  try {
    const comparator = new ImageComparator();
    const hrImagePath = path.join(
      __dirname,
      `../cp_image/hr_images/${imageId}.png`
    );
    const rebuildHRImagePath = path.join(
      __dirname,
      `../cp_image/rebuild_hr_images/${imageId}/${method}.png`
    );
    const orDiffPath = path.join(
      __dirname,
      `../cp_image/or_diff/diff_${imageId}_${method}.png`
    );

    // 1. 加载图像
    await comparator.loadImages(hrImagePath, rebuildHRImagePath);

    // 2. 计算质量指标
    comparator.calculateMetrics();

    // 3. 生成差异可视化图
    await comparator.generateDiffImage(orDiffPath);

    // 4. 打印专业报告
    comparator.printReport(imageId, method);

    console.log(`图像对比完成，差异图已保存_${imageId}_${method}.png`);
  } catch (error) {
    console.error(`❌ ${imageId}-${method} 错误详情:`, {
      message: error.message,
      stack: error.stack,
    });
  }
}

// 计算平均值
function calculateAverages() {
  const methods = [...new Set(results.metrics.map((item) => item.method))];

  methods.forEach((currentMethod) => {
    // 重命名参数避免冲突
    const methodResults = results.metrics.filter(
      (item) => item.method === currentMethod
    );
    const count = methodResults.length;

    results.summary[currentMethod] = {
      avgPsnr:
        methodResults.reduce(
          (sum, item) => sum + (item.psnr === Infinity ? 100 : item.psnr),
          0
        ) / count,
      avgSsim: methodResults.reduce((sum, item) => sum + item.ssim, 0) / count,
      avgMse: methodResults.reduce((sum, item) => sum + item.mse, 0) / count,
    };
  });
}

// 输出CSV报告
async function exportToCSV() {
  try {
    // 写入详细数据
    await csvWriter.writeRecords(results.metrics);

    // 追加平均值数据
    const summaryWriter = createObjectCsvWriter({
      path: path.join(__dirname, "../cp_image/metrics_report.csv"),
      header: [
        { id: "imageId", title: "IMAGE_ID" },
        { id: "method", title: "METHOD" },
        { id: "psnr", title: "PSNR(dB)" },
        { id: "ssim", title: "SSIM" },
        { id: "mse", title: "MSE" },
      ],
      append: true,
    });

    const summaryRecords = Object.entries(results.summary).map(
      ([methodName, data]) => ({
        imageId: "AVERAGE",
        method: methodName, // 使用 methodName 替代 method
        psnr: data.avgPsnr.toFixed(2),
        ssim: data.avgSsim.toFixed(4),
        mse: data.avgMse.toFixed(2),
      })
    );

    await summaryWriter.writeRecords(summaryRecords);

    console.log("\n✅ 评估报告已成功导出到 metrics_report.csv");
  } catch (error) {
    console.error("CSV导出失败:", error);
  }
}
// 所有需要比较的方法
const COMPARE_METHODS = [
  MODEL,
  `espcn_medium`,
  `espcn_thick`,
  `lanczos`,
  `bicubic_${MN}`,
  `bilinear`,
  `nearest`,
  `adaptive_bicubic_${MN}`,
];

// 主执行函数
(async () => {
  try {
    // 确保HRID是数组
    const imageIds = Array.isArray(HRID) ? HRID : [HRID];

    // 并行处理所有图片和方法
    await Promise.all(
      imageIds.map((imageId) =>
        Promise.all(COMPARE_METHODS.map((method) => compare(imageId, method)))
      )
    );

    // 计算平均值
    calculateAverages();

    // 导出CSV
    await exportToCSV();

    // 打印汇总报告
    console.log("\n════════════════════ 汇总报告 ════════════════════");
    console.log("方法\t\t平均PSNR\t平均SSIM\t平均MSE");
    console.log("────────────────────────────────────────────────");
    Object.entries(results.summary).forEach(([method, data]) => {
      console.log(
        `${method.padEnd(16)}${data.avgPsnr
          .toFixed(2)
          .padStart(8)} dB\t${data.avgSsim
          .toFixed(4)
          .padStart(8)}\t${data.avgMse.toFixed(2).padStart(8)}`
      );
    });
  } catch (error) {
    console.error("主流程错误:", error);
  }
})();
