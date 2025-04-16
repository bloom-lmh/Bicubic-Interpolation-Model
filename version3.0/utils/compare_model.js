/* // bicubic_weight_comparison.js
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { promisify } = require("util");
const PNG = require("pngjs").PNG;

// ===================== 配置 =====================
const learn_rate = "adaptive-1e-3-30"; // 可配置为命令行参数
const OUTPUT_DIR = path.join("cp_model", learn_rate);

// 确保输出目录存在
if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

const CONFIG = {
  metadataPath: "./data/test/metadata.json",
  directories: {
    lrImages: "./data/test/X/",
    offsets: "./data/test/offset/",
    gtWeights: "./data/test/Y/",
  },
  modelPath: `file://./model/${learn_rate}/model.json`,
  output: {
    sampleWeights: path.join(OUTPUT_DIR, "weight_sample.png"),
    gtHistogram: path.join(OUTPUT_DIR, "gt_weights_hist.png"),
    predHistogram: path.join(OUTPUT_DIR, "pred_weights_hist.png"),
    comparisonTable: path.join(OUTPUT_DIR, "comparison.txt"),
  },
};

// ===================== 工具函数 =====================
const readDir = promisify(fs.readdir);
const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);

async function loadTensorWithHeader(filePath) {
  const buffer = await fs.promises.readFile(filePath);

  const header = {
    height: buffer.readUInt32LE(0),
    width: buffer.readUInt32LE(4),
    channels: buffer.readUInt32LE(8),
  };

  const dataBuffer = buffer.slice(12);
  const float32Data = new Float32Array(
    dataBuffer.buffer,
    dataBuffer.byteOffset,
    dataBuffer.byteLength / 4
  );

  return {
    tensor: tf.tensor3d(float32Data, [
      header.height,
      header.width,
      header.channels,
    ]),
    header,
  };
}

function validateTensor(tensor, header, filename) {
  const expectedSize = header.height * header.width * header.channels;
  if (tensor.size !== expectedSize) {
    throw new Error(
      `数据验证失败 ${filename}\n预期: ${expectedSize}\n实际: ${tensor.size}`
    );
  }
}

// ===================== 核心逻辑 =====================
class WeightComparator {
  constructor(config) {
    this.config = config;
    this.model = null;
    this.metadata = null;
  }

  async initialize() {
    this.metadata = JSON.parse(await readFile(this.config.metadataPath));
    this.model = await tf.loadLayersModel(this.config.modelPath);
    console.log("✅ 系统初始化完成");
  }

  async loadDataset() {
    const loadData = async (dir) => {
      const files = await readDir(dir);
      const tensorMap = new Map();

      for (const file of files.filter((f) => f.endsWith(".bin"))) {
        const filePath = path.join(dir, file);
        const { tensor, header } = await loadTensorWithHeader(filePath);
        validateTensor(tensor, header, file);
        tensorMap.set(path.parse(file).name, tensor);
      }
      return tensorMap;
    };

    const [images, offsets, weights] = await Promise.all([
      loadData(this.config.directories.lrImages),
      loadData(this.config.directories.offsets),
      loadData(this.config.directories.gtWeights),
    ]);

    return { images, offsets, weights };
  }

  async compareWeights(gtTensor, predTensor) {
    console.log("\n🔬 开始权重对比分析");

    // 全局误差分析
    const mse = tf.losses.meanSquaredError(gtTensor, predTensor);
    console.log(`📉 全局MSE: ${mse.dataSync()[0].toExponential(3)}`);
    tf.dispose(mse);

    // 通道级对比
    const channelErrors = [];
    for (let c = 0; c < 16; c++) {
      const gtChannel = gtTensor.slice([0, 0, c], [-1, -1, 1]);
      const predChannel = predTensor.slice([0, 0, c], [-1, -1, 1]);
      const channelMSE = tf.losses.meanSquaredError(gtChannel, predChannel);
      channelErrors.push(channelMSE.dataSync()[0]);
      tf.dispose([gtChannel, predChannel, channelMSE]);
    }
    console.log("📊 各通道MSE:");
    channelErrors.forEach((err, i) =>
      console.log(`通道 ${i.toString().padStart(2)}: ${err.toExponential(3)}`)
    );

    // 可视化对比
    await this.generateHistograms(gtTensor, predTensor);
    await this.samplePixelComparison(gtTensor, predTensor);
  }

  async generateHistograms(gtTensor, predTensor) {
    // 创建直方图目录
    const histDir = path.join(OUTPUT_DIR, "histograms");
    if (!fs.existsSync(histDir)) {
      fs.mkdirSync(histDir, { recursive: true });
    }

    await this.createHistogram(
      gtTensor,
      path.join(histDir, "gt_histogram.png")
    );
    await this.createHistogram(
      predTensor,
      path.join(histDir, "pred_histogram.png")
    );
  }

  async createHistogram(tensor, filename) {
    const data = await tensor.data();
    const bins = new Array(20).fill(0);

    data.forEach((v) => {
      const bin = Math.min(Math.floor(v * 20), 19);
      bins[bin]++;
    });

    const png = new PNG({ width: 800, height: 400, colorType: 2 });
    const maxCount = Math.max(...bins);

    bins.forEach((count, i) => {
      const barHeight = (count / maxCount) * 400;
      for (let y = 399; y >= 400 - barHeight; y--) {
        for (let x = i * 40; x < (i + 1) * 40; x++) {
          const idx = (y * 800 + x) * 3;
          png.data[idx] = 255; // 红色通道
          png.data[idx + 1] = 0; // 绿色通道
          png.data[idx + 2] = 0; // 蓝色通道
        }
      }
    });

    await new Promise((resolve) =>
      png.pack().pipe(fs.createWriteStream(filename)).on("finish", resolve)
    );
  }

  // 修改 samplePixelComparison 方法中的这部分
  async samplePixelComparison(gtTensor, predTensor) {
    const sampleCoord = {
      x: Math.floor(gtTensor.shape[1] / 2),
      y: Math.floor(gtTensor.shape[0] / 2),
    };

    // 修复数组访问方式
    const gtValues = gtTensor
      .slice([sampleCoord.y, sampleCoord.x, 0], [1, 1, 16])
      .arraySync()
      .flat(2);
    const predValues = predTensor
      .slice([sampleCoord.y, sampleCoord.x, 0], [1, 1, 16])
      .arraySync()
      .flat(2);

    console.log(`\n🔍 中心点 (${sampleCoord.x},${sampleCoord.y}) 权重对比:`);
    console.log("┌──────┬───────────────────┬───────────────────┬───────────┐");
    console.log("│ 通道 │ 真实权重          │ 预测权重          │ 差异(%)   │");
    console.log("├──────┼───────────────────┼───────────────────┼───────────┤");

    gtValues.forEach((gt, i) => {
      const pred = predValues[i];
      const diff = (((pred - gt) / (Math.abs(gt) + 1e-8)) * 100).toFixed(2);
      console.log(
        `│ ${i.toString().padStart(2)}  │ ${gt
          .toExponential(4)
          .padEnd(15)} │ ` +
          `${pred.toExponential(4).padEnd(15)} │ ${diff.padStart(7)}% │`
      );
    });
    console.log("└──────┴───────────────────┴───────────────────┴───────────┘");
    // 保存对比结果到文件
    const tableContent = [
      "┌──────┬───────────────────┬───────────────────┬───────────┐",
      "│ 通道 │ 真实权重          │ 预测权重          │ 差异(%)   │",
      "├──────┼───────────────────┼───────────────────┼───────────┤",
      ...gtValues.map((gt, i) => {
        const pred = predValues[i];
        const diff = (((pred - gt) / (Math.abs(gt) + 1e-8)) * 100).toFixed(2);
        return (
          `│ ${i.toString().padStart(2)}  │ ${gt
            .toExponential(4)
            .padEnd(15)} │ ` +
          `${pred.toExponential(4).padEnd(15)} │ ${diff.padStart(7)}% │`
        );
      }),
      "└──────┴───────────────────┴───────────────────┴───────────┘",
    ].join("\n");

    await writeFile(CONFIG.output.comparisonTable, tableContent);
    console.log(`📝 对比表格已保存至: ${CONFIG.output.comparisonTable}`);
  }
  async visualizeSampleWeights(tensor) {
    // 创建可视化目录
    const visDir = path.join(OUTPUT_DIR, "visualizations");
    if (!fs.existsSync(visDir)) {
      fs.mkdirSync(visDir, { recursive: true });
    }

    const outputPath = path.join(visDir, "weight_samples");
    if (!fs.existsSync(outputPath)) {
      fs.mkdirSync(outputPath);
    }

    // 保存多个通道的可视化结果
    for (let c = 0; c < 3; c++) {
      // 示例保存前3个通道
      const channel = tensor.slice([0, 0, c], [-1, -1, 1]);
      const png = await this.tensorToPNG(channel);
      await writeFile(path.join(outputPath, `channel_${c}.png`), png);
      tf.dispose(channel);
    }
    console.log(`🖼️ 多通道可视化已保存至: ${outputPath}`);
  }

  async tensorToPNG(tensor) {
    const data = await tensor.data();
    const [height, width] = tensor.shape;

    const png = new PNG({ width, height, colorType: 2 });
    data.forEach((v, i) => {
      const val = Math.min(255, Math.max(0, v * 255));
      png.data[i * 3] = val; // R
      png.data[i * 3 + 1] = val; // G
      png.data[i * 3 + 2] = val; // B
    });

    return PNG.sync.write(png);
  }
  async execute() {
    try {
      await this.initialize();
      const { images, offsets, weights } = await this.loadDataset();

      const [sampleId] = Array.from(images.keys());
      const inputTensor = images.get(sampleId).expandDims(0);
      const offsetTensor = offsets.get(sampleId).expandDims(0);
      const gtTensor = weights.get(sampleId);

      const predTensor = this.model
        .predict([inputTensor, offsetTensor])
        .squeeze();
      console.log("🧠 预测完成，开始分析...");

      await this.compareWeights(gtTensor, predTensor);
      await this.visualizeSampleWeights(predTensor);

      tf.dispose([inputTensor, offsetTensor, gtTensor, predTensor]);
    } catch (error) {
      console.error("‼️ 执行失败:", error.message);
    }
  }

  async visualizeSampleWeights(tensor) {
    const sample = tensor.slice([0, 0, 0], [100, 100, 16]);
    const firstChannel = sample.slice([0, 0, 0], [100, 100, 1]);

    const png = new PNG({ width: 100, height: 100, colorType: 2 });
    const data = await firstChannel.data();

    data.forEach((v, i) => {
      const val = Math.min(255, Math.max(0, v * 255));
      png.data[i * 3] = val; // R
      png.data[i * 3 + 1] = val; // G
      png.data[i * 3 + 2] = val; // B
    });

    await new Promise((resolve) =>
      png
        .pack()
        .pipe(fs.createWriteStream(this.config.output.sampleWeights))
        .on("finish", resolve)
    );
    console.log(`🖼️ 权重采样图已保存至: ${this.config.output.sampleWeights}`);
  }
}

// ===================== 执行入口 =====================
(async () => {
  try {
    // 二次确认输出目录
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
      console.log(`📂 已创建输出目录: ${OUTPUT_DIR}`);
    }

    const comparator = new WeightComparator(CONFIG);
    await comparator.execute();
    console.log("🏁 程序执行完成");

    // 打开结果目录（仅限macOS）
    if (process.platform === "darwin") {
      require("child_process").exec(`open ${path.resolve(OUTPUT_DIR)}`);
    }
  } catch (error) {
    console.error("‼️ 初始化失败:", error.message);
    process.exit(1);
  }
})();
 */
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { createCanvas } = require("@napi-rs/canvas"); // 更高效的内存管理
const { promisify } = require("util");
const { pipeline } = require("stream/promises");
const PNG = require("pngjs").PNG;

// ===================== 增强配置 =====================
const CONFIG = {
  learnRate: "adaptive-1e-3-30",
  metadataPath: "./data/test/metadata.json",
  directories: {
    lrImages: "./data/test/X/",
    offsets: "./data/test/offset/",
    gtWeights: "./data/test/Y/",
  },
  modelPath: "file://./model/adaptive-1e-3-30/model.json",
  output: {
    root: "./analysis_results",
    histograms: "histograms",
    tables: "tables",
    visualizations: "visuals",
  },
  performance: {
    chunkSize: 1e5, // 分块处理大小
    maxBins: 1000, // 最大分箱数
    canvasDPI: 300, // 输出分辨率
  },
};

// ===================== 工具函数 =====================
const readDir = promisify(fs.readdir);
const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);

async function initDirectories() {
  const paths = Object.values(CONFIG.output).map((p) =>
    path.join(CONFIG.output.root, p)
  );

  await Promise.all(
    paths.map(async (p) => {
      if (!fs.existsSync(p)) {
        await fs.promises.mkdir(p, { recursive: true });
      }
    })
  );
}

// ===================== 流式数据加载 =====================
class StreamLoader {
  static async *tensorIterator(tensor) {
    const size = tensor.size;
    for (let i = 0; i < size; i += CONFIG.performance.chunkSize) {
      const end = Math.min(i + CONFIG.performance.chunkSize, size);
      const slice = tensor.reshape([size]).slice([i], [end - i]);
      const data = await slice.data();
      yield data;
      tf.dispose(slice);
      await new Promise((resolve) => setImmediate(resolve));
    }
  }

  static async processTensor(tensor) {
    const stats = {
      min: Infinity,
      max: -Infinity,
      sum: 0,
      count: 0,
    };

    for await (const chunk of this.tensorIterator(tensor)) {
      const chunkStats = chunk.reduce(
        (acc, val) => ({
          min: Math.min(acc.min, val),
          max: Math.max(acc.max, val),
          sum: acc.sum + val,
          count: acc.count + 1,
        }),
        { ...stats }
      );

      Object.assign(stats, chunkStats);
    }

    return {
      ...stats,
      mean: stats.sum / stats.count,
      range: stats.max - stats.min,
    };
  }
}

// ===================== 高性能可视化 =====================
class HistogramGenerator {
  static async createComparison(gtTensor, predTensor, filename) {
    // 并行计算统计指标
    const [gtStats, predStats] = await Promise.all([
      StreamLoader.processTensor(gtTensor),
      StreamLoader.processTensor(predTensor),
    ]);

    // 动态分箱策略
    const binWidth = this.calculateBinWidth(gtStats, predStats);
    const binCount = Math.min(
      Math.ceil(
        (Math.max(gtStats.max, predStats.max) -
          Math.min(gtStats.min, predStats.min)) /
          binWidth
      ),
      CONFIG.performance.maxBins
    );

    // 创建Canvas
    const canvas = createCanvas(2400, 1200);
    const ctx = canvas.getContext("2d");
    this.setupCanvas(ctx, canvas);

    // 流式构建直方图
    await this.drawHistogram(ctx, gtTensor, gtStats, binWidth, binCount, "gt");
    await this.drawHistogram(
      ctx,
      predTensor,
      predStats,
      binWidth,
      binCount,
      "pred"
    );

    // 添加标注
    this.drawAnnotations(ctx, canvas, gtStats, predStats, binWidth);

    // 保存输出
    await this.saveOutput(canvas, filename);
  }

  static calculateBinWidth(gtStats, predStats) {
    const n = Math.sqrt(gtStats.count + predStats.count);
    const range = Math.max(gtStats.range, predStats.range);
    return (2 * (gtStats.mean + predStats.mean)) / Math.cbrt(n);
  }

  static setupCanvas(ctx, canvas) {
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = `bold ${12 * (canvas.width / 1200)}px Arial`;
    ctx.textBaseline = "top";
  }

  static async drawHistogram(ctx, tensor, stats, binWidth, binCount, type) {
    const bins = new Uint32Array(binCount);
    const minVal = Math.min(stats.min, stats.min);

    for await (const chunk of StreamLoader.tensorIterator(tensor)) {
      chunk.forEach((value) => {
        const bin = Math.min(
          Math.floor((value - minVal) / binWidth),
          binCount - 1
        );
        if (bin >= 0) bins[bin]++;
      });
    }

    const maxCount = Math.max(...bins);
    const color =
      type === "gt" ? "rgba(255,99,132,0.6)" : "rgba(54,162,235,0.6)";

    ctx.fillStyle = color;
    const barWidth = (ctx.canvas.width - 200) / binCount;

    bins.forEach((count, i) => {
      const height = (count / maxCount) * (ctx.canvas.height - 200);
      ctx.fillRect(
        100 + i * barWidth,
        ctx.canvas.height - 100 - height,
        barWidth * 0.8,
        height
      );
    });
  }

  static drawAnnotations(ctx, canvas, gtStats, predStats, binWidth) {
    ctx.fillStyle = "#333333";
    ctx.textAlign = "center";
    ctx.fillText("Weight Value Distribution", canvas.width / 2, 50);

    // X轴标注
    Array.from({ length: 5 }).forEach((_, i) => {
      const x = 100 + (i * (canvas.width - 200)) / 4;
      const value = (gtStats.min + i * binWidth * 5).toFixed(2);
      ctx.fillText(value, x, canvas.height - 70);
    });

    // 图例
    ctx.fillStyle = "#666666";
    ctx.fillRect(150, 100, 300, 120);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText(`Ground Truth (μ=${gtStats.mean.toFixed(4)})`, 300, 120);
    ctx.fillText(`Predicted (μ=${predStats.mean.toFixed(4)})`, 300, 150);
  }

  static async saveOutput(canvas, filename) {
    const buffer = canvas.toBuffer("image/png", {
      compressionLevel: 3,
      filters: canvas.PNG_FILTER_NONE,
    });

    await writeFile(filename, buffer);
    console.log(`📈 直方图已保存: ${filename}`);
  }
}

// ===================== 分析引擎 =====================
class AnalysisEngine {
  constructor() {
    this.model = null;
  }

  async initialize() {
    await initDirectories();
    this.model = await tf.loadLayersModel(CONFIG.modelPath);
    console.log("✅ 系统初始化完成 | 模型加载成功");
  }

  async execute() {
    try {
      const [images, offsets, weights] = await Promise.all([
        this.loadDataset("lrImages"),
        this.loadDataset("offsets"),
        this.loadDataset("gtWeights"),
      ]);

      const sampleId = Array.from(images.keys())[0];
      console.log(`🔍 分析样本: ${sampleId}`);

      const { gtWeights, predWeights } = await this.processSample(
        images.get(sampleId),
        offsets.get(sampleId),
        weights.get(sampleId)
      );

      await this.generateReports(gtWeights, predWeights);
      await this.cleanup([gtWeights, predWeights]);
    } catch (error) {
      console.error("‼️ 分析流程异常:", error);
      process.exit(1);
    }
  }

  async loadDataset(type) {
    const dir = CONFIG.directories[type];
    const files = (await readDir(dir)).filter((f) => f.endsWith(".bin"));

    const dataset = new Map();
    await Promise.all(
      files.map(async (file) => {
        const tensor = await this.loadTensor(path.join(dir, file));
        dataset.set(path.parse(file).name, tensor);
      })
    );

    return dataset;
  }

  async loadTensor(filePath) {
    const buffer = await readFile(filePath);
    const header = {
      height: buffer.readUInt32LE(0),
      width: buffer.readUInt32LE(4),
      channels: buffer.readUInt32LE(8),
    };

    const data = new Float32Array(
      buffer.buffer,
      buffer.byteOffset + 12,
      (buffer.length - 12) / 4
    );

    return tf.tensor3d(data, [header.height, header.width, header.channels]);
  }

  async processSample(image, offset, gt) {
    const pred = this.model
      .predict([image.expandDims(0), offset.expandDims(0)])
      .squeeze();

    return {
      gtWeights: gt,
      predWeights: pred,
    };
  }

  async generateReports(gt, pred) {
    const histPath = path.join(
      CONFIG.output.root,
      CONFIG.output.histograms,
      "weight_comparison.png"
    );

    await HistogramGenerator.createComparison(gt, pred, histPath);
    console.log("📊 可视化报告生成完成");
  }

  async cleanup(tensors) {
    await Promise.all(tensors.map((t) => tf.dispose(t) || true));
    console.log("🧹 内存清理完成");
  }
}

// ===================== 执行入口 =====================
(async () => {
  try {
    const engine = new AnalysisEngine();
    await engine.initialize();
    await engine.execute();
    console.log("🏁 分析流程成功结束");
  } catch (error) {
    console.error("‼️ 系统级错误:", error);
    process.exit(2);
  }
})();
