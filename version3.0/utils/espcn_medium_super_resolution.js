const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const pc = require("./compare_performance");
const Upscaler = require("upscaler/node"); // this is important!
const x4 = require("@upscalerjs/esrgan-medium/4x");
// 配置路径
const HRID = "0829";
const LR_IMAGEPATH = `./cp_image/lr_images/${HRID}_downsample.png`;
const REBUILD_HR_IMAGEPATH = `./cp_image/rebuild_hr_images/${HRID}_rebuild_espcn_medium.png`;

async function main() {
  const upscaler = new Upscaler({
    model: x4,
  });
  const image = tf.node.decodeImage(fs.readFileSync(LR_IMAGEPATH), 3);
  const tensor = await upscaler.upscale(image);
  const upscaledTensor = await tf.node.encodePng(tensor);
  fs.writeFileSync(REBUILD_HR_IMAGEPATH, upscaledTensor);

  // dispose the tensors!
  image.dispose();
  tensor.dispose();
}

// 执行
// 使用示例
pc(() => main(), {
  testItem: "espcn_medium",
});
