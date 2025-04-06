const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs-node");

async function loadTensorWithHeader(filePath) {
  const buffer = await fs.promises.readFile(filePath);
  const header = {
    height: buffer.readUInt32LE(0),
    width: buffer.readUInt32LE(4),
    channels: buffer.readUInt32LE(8),
  };

  // 从第12字节开始，直接解析为 Float32Array
  const dataStart = 12;
  const float32Buffer = buffer.slice(dataStart);
  const data = new Float32Array(
    float32Buffer.buffer,
    float32Buffer.byteOffset,
    float32Buffer.byteLength / 4
  );

  return {
    tensor: tf.tensor3d(data, [header.height, header.width, header.channels]),
    header,
  };
}

async function validateSample(sampleId) {
  const metadata = JSON.parse(
    await fs.promises.readFile("./data/train/metadata.json")
  )[sampleId];

  // 加载所有数据
  const [X, offset, Y] = await Promise.all([
    loadTensorWithHeader(`./data/train/X/${sampleId}.bin`),
    loadTensorWithHeader(`./data/train/offset/${sampleId}.bin`),
    loadTensorWithHeader(`./data/train/Y/${sampleId}.bin`),
  ]);

  // 1. 验证形状匹配
  console.log("\n=== 形状验证 ===");
  console.assert(
    X.header.height === metadata.H_lr &&
      X.header.width === metadata.W_lr &&
      X.header.channels === 4,
    "X 形状不匹配"
  );
  console.assert(
    offset.header.height === metadata.H_sr &&
      offset.header.width === metadata.W_sr &&
      offset.header.channels === 2,
    "Offset 形状不匹配"
  );
  console.assert(
    Y.header.height === metadata.H_sr &&
      Y.header.width === metadata.W_sr &&
      Y.header.channels === 16,
    "Y 形状不匹配"
  );

  console.log("\n=== 数值范围验证 ===");
  const offsetData = await offset.tensor.data();
  const Ydata = await Y.tensor.data();

  // 1. 安全计算范围
  function safeMinMax(arr, start, end) {
    let min = Infinity;
    let max = -Infinity;
    for (let i = start; i < end; i++) {
      const val = arr[i];
      if (val < min) min = val;
      if (val > max) max = val;
    }
    return { min, max };
  }

  // 2. 采样部分数据
  const sampleSize = 1000;
  const dxRange = safeMinMax(offsetData, 0, sampleSize);
  const dyStart = Math.floor(offsetData.length / 2);
  const dyRange = safeMinMax(offsetData, dyStart, dyStart + sampleSize);

  console.log(
    `Offset范围: dx [${dxRange.min.toFixed(3)}, ${dxRange.max.toFixed(
      3
    )}], dy [${dyRange.min.toFixed(3)}, ${dyRange.max.toFixed(3)}]`
  );

  // 3. 验证权重范围
  const weightSample = Array.from(Ydata.subarray(0, 1000));
  const weightRange = safeMinMax(weightSample, 0, weightSample.length);
  console.log(
    `Y权重范围: [${weightRange.min.toFixed(3)}, ${weightRange.max.toFixed(3)}]`
  );
  console.assert(
    weightRange.min >= -0.75 && weightRange.max <= 2.0,
    "Y权重超出双三次合理范围"
  );

  // 3. 随机采样验证权重
  console.log("\n=== 随机采样验证 ===");
  const randomPositions = Array.from({ length: 5 }, () => ({
    y: Math.floor(Math.random() * metadata.H_sr),
    x: Math.floor(Math.random() * metadata.W_sr),
  }));

  for (const { y, x } of randomPositions) {
    const offsetVal = offset.tensor.slice([y, x, 0], [1, 1, 2]).dataSync();
    const weights = Y.tensor.slice([y, x, 0], [1, 1, 16]).dataSync();
    const sum = weights.reduce((a, b) => a + b, 0);

    console.log(
      `位置 (${y}, ${x}): dx=${offsetVal[0].toFixed(
        3
      )}, dy=${offsetVal[1].toFixed(3)}, 权重和=${sum.toFixed(5)}`
    );
    console.assert(
      Math.abs(sum - 1) < 0.01,
      `权重和不接近1，误差: ${Math.abs(sum - 1)}`
    );
  }

  tf.dispose([X.tensor, offset.tensor, Y.tensor]);
  console.log("\n验证通过！");
}

// 执行验证（示例验证0001.bin）
validateSample("0001").catch(console.error);
