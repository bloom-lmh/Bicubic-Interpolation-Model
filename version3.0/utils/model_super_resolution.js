const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const PNG = require("pngjs").PNG;
const sharp = require("sharp");
const pc = require("./compare_performance");
const MODEL = "1e-3-30";
const HRID = "0829";
const LR_IMAGEPATH = `./cp_image/lr_images/${HRID}_downsample.png`;
const HR_IMAGEPATH = `./cp_image/hr_images/${HRID}.png`;
const REBUILD_HR_IMAGEPATH = `./cp_image/rebuild_hr_images/${HRID}_rebuild_${MODEL}.png`;
const SCALE_FACTOR = 4;
// 降采样函数（使用Lanczos滤波）
async function downsampleImage(hrPath, scaleFactor = 4) {
  const hrImage = await sharp(hrPath);
  const metadata = await hrImage.metadata();

  const lrWidth = Math.floor(metadata.width / scaleFactor);
  const lrHeight = Math.floor(metadata.height / scaleFactor);

  return hrImage
    .resize(lrWidth, lrHeight, {
      kernel: sharp.kernel.lanczos3, // 使用Lanczos3降采样
    })
    .toBuffer();
}
function generateOffset(H_lr, W_lr, H_sr, W_sr) {
  const scale_h = H_sr / H_lr;
  const scale_w = W_sr / W_lr;

  // 生成SR像素的网格坐标
  const x_sr = tf.range(0, W_sr);
  const y_sr = tf.range(0, H_sr);
  const [x_grid, y_grid] = tf.meshgrid(x_sr, y_sr);

  // 转换为LR坐标系中的位置
  const x_lr = x_grid.div(scale_w);
  const y_lr = y_grid.div(scale_h);

  // 计算dx和dy（相对于LR像素中心的偏移）
  const x_center = tf.floor(x_lr).add(0.5);
  const dx = x_lr.sub(x_center);

  const y_center = tf.floor(y_lr).add(0.5);
  const dy = y_lr.sub(y_center);

  // 合并为X_offset [H_sr, W_sr, 2]
  return tf.stack([dx, dy], 2);
}
async function predictWeights(model, X_image, X_offset) {
  // 预处理X_image（归一化等）
  const inputTensor = X_image.toFloat().div(255.0); // 假设图像为0-255

  // 扩展维度为批量形式 [1, H, W, C]
  const batchedX = inputTensor.expandDims(0);
  const batchedOffset = X_offset.expandDims(0);

  // 预测权重
  const pred = model.predict([batchedX, batchedOffset]);
  return pred.squeeze(); // 移除批量维度 [H_sr, W_sr, 16]
}
function applyWeights(X_image, weights, H_sr, W_sr) {
  const [H_lr, W_lr, C] = X_image.shape;
  const scale = H_sr / H_lr;

  // 1. 获取每个SR像素对应的16个LR邻居的坐标
  const x_sr = tf.range(0, W_sr);
  const y_sr = tf.range(0, H_sr);
  const [x_grid, y_grid] = tf.meshgrid(x_sr, y_sr);

  const x_lr = x_grid.div(scale);
  const y_lr = y_grid.div(scale);

  // 基准坐标是左上角的邻居 (i-1, j-1)
  const x_base = tf.floor(x_lr).sub(1);
  const y_base = tf.floor(y_lr).sub(1);

  // 2. 生成16个邻居的坐标偏移 (0-3 in each direction)
  const offsets = [];
  for (let dy = 0; dy < 4; dy++) {
    for (let dx = 0; dx < 4; dx++) {
      offsets.push([dy, dx]);
    }
  }

  // 3. 为每个SR像素收集16个邻居的像素值
  const neighborValues = [];
  for (const [dy, dx] of offsets) {
    // 计算邻居坐标并转换为整数
    const x_neighbor = x_base.add(dx).toInt();
    const y_neighbor = y_base.add(dy).toInt();

    // 处理边界条件：使用镜像填充
    const x_clipped = tf.clipByValue(x_neighbor, 0, W_lr - 1);
    const y_clipped = tf.clipByValue(y_neighbor, 0, H_lr - 1);

    // 创建索引张量 [H_sr, W_sr, 2] 并确保是int32
    const indices = tf.stack([y_clipped, x_clipped], -1).toInt();

    // 收集邻居像素值 [H_sr, W_sr, C]
    const neighbor = tf.gatherND(X_image, indices);
    neighborValues.push(neighbor);
  }

  // 4. 应用权重进行加权求和
  const weightSlices = tf.split(weights, 16, -1);

  let srImage = tf.zeros([H_sr, W_sr, C]);
  for (let i = 0; i < 16; i++) {
    const weighted = neighborValues[i].mul(weightSlices[i]);
    srImage = srImage.add(weighted);
  }

  // 5. 确保像素值在合理范围内
  srImage = tf.clipByValue(srImage, 0, 255).round().cast("int32");

  return srImage;
}
async function superResolve(modelPath, lrImage, H_lr, W_lr, H_sr, W_sr) {
  const model = await tf.loadLayersModel(
    `file://./model/${modelPath}/model.json`
  );
  const X_offset = generateOffset(H_lr, W_lr, H_sr, W_sr);
  const weights = await predictWeights(model, lrImage, X_offset);

  const srImage = applyWeights(lrImage, weights, H_sr, W_sr);

  // 清理中间张量
  tf.dispose([X_offset, weights]);

  return srImage;
}
function validateOffset(X_offset) {
  // 检查形状 [H_sr, W_sr, 2]
  console.log("X_offset形状:", X_offset.shape);

  // 抽取部分样本查看 dx 和 dy
  const sampleOffset = X_offset.slice([0, 0], [5, 5]).dataSync();
  console.log("部分 X_offset:", Array.from(sampleOffset));

  // 检查数值范围
  const dx = X_offset.gather(2, 0).squeeze(); // 提取所有 dx
  const dy = X_offset.gather(2, 1).squeeze(); // 提取所有 dy

  const dxStats = {
    min: dx.min().dataSync()[0],
    max: dx.max().dataSync()[0],
    mean: dx.mean().dataSync()[0],
  };
  const dyStats = {
    min: dy.min().dataSync()[0],
    max: dy.max().dataSync()[0],
    mean: dy.mean().dataSync()[0],
  };
  console.log("dx范围:", dxStats);
  console.log("dy范围:", dyStats);
}
function validateWeights(weights) {
  // 检查形状 [H_sr, W_sr, 16]
  console.log("weights形状:", weights.shape);
  // 抽取部分样本权重
  const sampleWeights = weights.slice([0, 0], [5, 5]).dataSync();
  console.log("部分 weights 值:", Array.from(sampleWeights));

  // 检查权重是否归一化（每个 SR 像素的 16 权重之和应为 1）
  const weightsSum = weights.reshape([-1, 16]).sum(1);
  const sumStats = {
    min: weightsSum.min().dataSync()[0],
    max: weightsSum.max().dataSync()[0],
    mean: weightsSum.mean().dataSync()[0],
  };
  console.log("权重和统计:", sumStats);
}
async function loadPNG(path) {
  return new Promise((resolve, reject) => {
    fs.createReadStream(path)
      .pipe(new PNG())
      .on("parsed", function () {
        // 将像素数据转换为 Tensor [H, W, C]
        const data = new Uint8Array(this.data);
        const tensor = tf.tensor3d(data, [this.height, this.width, 4], "int32");
        resolve(tensor);
      })
      .on("error", (err) => reject(err));
  });
}

function getImageMetadata(lrImage, scale = 4) {
  const [H_lr, W_lr] = lrImage.shape;
  return [H_lr, W_lr, H_lr * scale, W_lr * scale];
}
// 使用示例
async function main() {
  // 1. 降采样高清图得到低分辨率图
  const lrBuffer = await downsampleImage(HR_IMAGEPATH, SCALE_FACTOR);
  fs.writeFileSync(LR_IMAGEPATH, lrBuffer); // 保存降采样后的图

  const lrImage = await loadPNG(LR_IMAGEPATH);
  const [H_lr, W_lr, H_sr, W_sr] = getImageMetadata(lrImage);

  const srImage = await superResolve(MODEL, lrImage, H_lr, W_lr, H_sr, W_sr);

  // 保存结果
  const buffer = await tf.node.encodePng(srImage);
  fs.writeFileSync(REBUILD_HR_IMAGEPATH, buffer);

  // 清理内存
  tf.dispose([lrImage, srImage]);
}
pc(() => main(), {
  testItem: `model_${MODEL}`,
});
