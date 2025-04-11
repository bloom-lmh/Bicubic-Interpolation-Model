const fs = require("fs");
const path = require("path");
const { performance } = require("perf_hooks");

async function accuratePerformanceTest(func, options) {
  const { testItem, runs = 2 } = options;
  const results = [
    "Run,Timestamp,Execution Time (ms),CPU Time (ms),Memory (MB)",
  ];

  // 预热运行（避免冷启动误差）
  for (let i = 0; i < 2; i++) await func();

  for (let i = 0; i < runs; i++) {
    // 1. 重置内存基准
    global.gc?.(); // 显式触发GC（需--expose-gc）
    const memBefore = process.memoryUsage.rss(); // 改用RSS内存
    const cpuBefore = process.cpuUsage();

    // 2. 执行计时
    const start = performance.now();
    await func();
    const execTime = performance.now() - start;

    // 3. 计算指标
    const memAfter = process.memoryUsage.rss();
    const cpuDiff = process.cpuUsage(cpuBefore);
    const cpuTime = (cpuDiff.user + cpuDiff.system) / 1000; // 修正CPU时间计算

    // 4. 记录结果
    results.push(
      [
        i + 1,
        new Date().toISOString(),
        execTime.toFixed(2),
        cpuTime.toFixed(2),
        (memAfter / 1024 / 1024).toFixed(2),
      ].join(",")
    );
  }

  // 保存结果（同原代码）
  const outputDir = path.join(__dirname, "../cp_performance", testItem);
  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(
    path.join(outputDir, `${testItem}_performance.csv`),
    results.join("\n")
  );
}
module.exports = accuratePerformanceTest;
