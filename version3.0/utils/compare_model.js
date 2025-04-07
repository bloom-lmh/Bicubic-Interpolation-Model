// bicubic_weight_comparison.js
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { promisify } = require("util");
const PNG = require("pngjs").PNG;

// ===================== 配置 =====================
const learn_rate = "1e-4-20"; // 可配置为命令行参数
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
