const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const { promisify } = require("util");
const readFile = promisify(fs.readFile);
const readdir = promisify(fs.readdir);

// ===================== 配置参数 =====================
const TRAIN_DIR = {
  X: "./data/train/X/",
  Y: "./data/train/Y/",
};
const INPUT_FEATURES = 4 * 4 * 4 + 2;
const OUTPUT_FEATURES = 16;
const BATCH_SIZE = 5000; // 根据内存调整批次大小

// ===================== 流式验证工具 =====================
async function* batchIterator(filePath, features) {
  const buffer = await readFile(filePath);
  const float32Data = new Float32Array(buffer.buffer);
  const totalSamples = float32Data.length / features;

  for (let i = 0; i < totalSamples; i += BATCH_SIZE) {
    const batchStart = i * features;
    const batchEnd = Math.min((i + BATCH_SIZE) * features, float32Data.length);
    yield float32Data.slice(batchStart, batchEnd);
  }
}

async function validateFile(filePath, features, checks) {
  let fileStats = {
    min: Infinity,
    max: -Infinity,
    sum: 0,
    count: 0,
    nanCount: 0,
    infCount: 0,
    sumErrors: [],
  };

  // 分块处理数据
  for await (const batchData of batchIterator(filePath, features)) {
    const tensor = tf.tidy(() => {
      return tf.tensor2d(batchData, [batchData.length / features, features]);
    });

    // 执行所有检查
    const batchStats = await checks.run(tensor);

    // 合并统计结果
    fileStats.min = Math.min(fileStats.min, batchStats.min);
    fileStats.max = Math.max(fileStats.max, batchStats.max);
    fileStats.sum += batchStats.sum;
    fileStats.count += batchStats.count;
    fileStats.nanCount += batchStats.nanCount;
    fileStats.infCount += batchStats.infCount;
    fileStats.sumErrors.push(...batchStats.sumErrors);

    tf.dispose(tensor);
  }

  return fileStats;
}

// ===================== 验证逻辑配置 =====================
const validationConfig = {
  X: {
    features: INPUT_FEATURES,
    range: [-5, 5],
    run: async (tensor) => {
      const data = tensor.dataSync();
      return {
        min: tensor.min().dataSync()[0],
        max: tensor.max().dataSync()[0],
        sum: tensor.sum().dataSync()[0],
        count: tensor.size,
        nanCount: data.filter((v) => isNaN(v)).length,
        infCount: data.filter((v) => !isFinite(v)).length,
        sumErrors: [], // X不检查权重和
      };
    },
  },
  Y: {
    features: OUTPUT_FEATURES,
    range: [-0.5, 1.5],
    run: async (tensor) => {
      const data = tensor.dataSync();
      const sums = tensor.sum(1).dataSync();

      return {
        min: tensor.min().dataSync()[0],
        max: tensor.max().dataSync()[0],
        sum: tensor.sum().dataSync()[0],
        count: tensor.shape[0],
        nanCount: data.filter((v) => isNaN(v)).length,
        infCount: data.filter((v) => !isFinite(v)).length,
        sumErrors: sums.filter((s) => Math.abs(s - 1) > 1e-5),
      };
    },
  },
};

// ===================== 主验证流程 =====================
async function main() {
  try {
    console.log("开始流式数据验证...");

    // 1. 文件基础检查
    console.log("\n[阶段1] 文件完整性检查");
    await checkFiles(TRAIN_DIR.X, INPUT_FEATURES);
    await checkFiles(TRAIN_DIR.Y, OUTPUT_FEATURES);

    // 2. 详细数据验证
    console.log("\n[阶段2] 深度数据验证");
    const globalStats = {
      X: { min: Infinity, max: -Infinity, sum: 0, count: 0, nan: 0, inf: 0 },
      Y: {
        min: Infinity,
        max: -Infinity,
        sum: 0,
        count: 0,
        nan: 0,
        inf: 0,
        sumErrors: [],
      },
    };

    for (const [dataType, dirPath] of Object.entries(TRAIN_DIR)) {
      console.log(`\n=== 验证 ${dataType} 数据 ===`);
      const files = (await readdir(dirPath)).filter((f) => f.endsWith(".bin"));
      const config = validationConfig[dataType];

      for (const file of files) {
        const filePath = path.join(dirPath, file);
        console.log(`处理文件: ${file}`);

        const stats = await validateFile(filePath, config.features, config);

        // 合并全局统计
        globalStats[dataType].min = Math.min(
          globalStats[dataType].min,
          stats.min
        );
        globalStats[dataType].max = Math.max(
          globalStats[dataType].max,
          stats.max
        );
        globalStats[dataType].sum += stats.sum;
        globalStats[dataType].count += stats.count;
        globalStats[dataType].nan += stats.nanCount;
        globalStats[dataType].inf += stats.infCount;
        if (dataType === "Y") globalStats.Y.sumErrors.push(...stats.sumErrors);
      }
    }

    // 3. 最终验证
    console.log("\n[阶段3] 全局验证");
    validateGlobalStats(globalStats);

    console.log("\n所有检查通过 ✅");
  } catch (err) {
    console.error("\n‼️ 验证失败:", err.message);
    process.exit(1);
  }
}

// ===================== 辅助函数 =====================
async function checkFiles(dirPath, features) {
  console.log(`检查目录: ${dirPath}`);
  const files = await readdir(dirPath);
  let total = 0;

  for (const file of files.filter((f) => f.endsWith(".bin"))) {
    const filePath = path.join(dirPath, file);
    const stats = await fs.promises.stat(filePath);
    if (stats.size === 0) throw new Error(`空文件: ${file}`);
    total++;
  }
  console.log(`✓ 共发现 ${total} 个合法文件`);
}

function validateGlobalStats(stats) {
  // X数据验证
  console.log("\n[输入特征X]");
  console.log(
    `数值范围: [${stats.X.min.toFixed(4)}, ${stats.X.max.toFixed(4)}]`
  );
  console.log(`样本数量: ${stats.X.count}`);
  console.log(`NaN/Inf数量: ${stats.X.nan}/${stats.X.inf}`);
  if (stats.X.min < -5 || stats.X.max > 5) {
    throw new Error("X数据范围异常");
  }

  // Y数据验证
  console.log("\n[输出权重Y]");
  console.log(
    `数值范围: [${stats.Y.min.toFixed(4)}, ${stats.Y.max.toFixed(4)}]`
  );
  console.log(`样本数量: ${stats.Y.count}`);
  console.log(`NaN/Inf数量: ${stats.Y.nan}/${stats.Y.inf}`);
  console.log(`权重和异常样本: ${stats.Y.sumErrors.length}`);
  if (stats.Y.min < -0.5 || stats.Y.max > 1.5) {
    throw new Error("Y数据范围异常");
  }
  if (stats.Y.sumErrors.length > 0) {
    throw new Error(`发现 ${stats.Y.sumErrors.length} 个权重和异常样本`);
  }
}

// ===================== 执行 =====================
main();
