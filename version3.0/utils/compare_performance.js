const fs = require("fs");
const path = require("path");
const { performance } = require("perf_hooks");

// 配置输出目录
const REPORT_DIR = "../cp_performance";

/**
 * 简化版性能测试工具
 * @param {Function} func - 待测试函数（支持异步）
 * @param {Object} options - 配置项
 * @param {string} options.testName - 测试名称
 * @param {number} [options.runs=5] - 测试次数
 */
async function simplePerformanceTest(func, options) {
  const { testItem, testName = "performance_test", runs = 2 } = options;
  const results = [];
  let outputDir = path.join(__dirname, REPORT_DIR, testItem);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  // CSV表头
  const headers =
    "Run,Timestamp,Execution Time (ms),CPU Usage (ms),Memory Usage (MB)";
  results.push(headers);

  for (let i = 0; i < runs; i++) {
    // 1. 记录初始状态
    const memBefore = process.memoryUsage();
    const cpuBefore = process.cpuUsage();
    const startTime = performance.now();

    // 2. 执行函数
    await func();

    // 3. 计算指标
    const memAfter = process.memoryUsage();
    const cpuAfter = process.cpuUsage();
    const endTime = performance.now();

    // 4. 收集结果
    const row = [
      i + 1,
      new Date().toISOString(),
      (endTime - startTime).toFixed(2),
      ((cpuAfter.user - cpuBefore.user) / 1000).toFixed(2),
      ((memAfter.heapUsed - memBefore.heapUsed) / 1024 / 1024).toFixed(2),
    ].join(",");

    results.push(row);
  }

  // 5. 写入CSV文件
  const filename = `${testName}.csv`;
  const filepath = path.join(outputDir, filename);
  fs.writeFileSync(filepath, results.join("\n"));

  console.log(`✅ 测试完成！结果已保存到: ${filepath}`);
  console.log("\n测试结果概览:");
  console.log(results.join("\n"));
}
module.exports = simplePerformanceTest;
