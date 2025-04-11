const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const pc = require("./compare_performance");
const Upscaler = require("upscaler/node"); // this is important!
const x4 = require("@upscalerjs/esrgan-medium/4x");
const { HRID } = require("./config");
// 配置路径
/* const HRID = "0829"; */
const LR_IMAGEPATH = `./cp_image/lr_images/${HRID}_downsample.png`;
const REBUILD_HR_IMAGEPATH = `./cp_image/rebuild_hr_images/${HRID}/espcn_medium.png`;

async function main() {
  const upscaler = new Upscaler({
    model: x4,
  });
  const image = tf.node.decodeImage(fs.readFileSync(LR_IMAGEPATH), 3);
  let tensor;
  await pc(async () => (tensor = await upscaler.upscale(image)), {
    testItem: "espcn_medium",
  });
  const upscaledTensor = await tf.node.encodePng(tensor);
  fs.writeFileSync(REBUILD_HR_IMAGEPATH, upscaledTensor);

  // dispose the tensors!
  image.dispose();
  tensor.dispose();
}
main();
// 执行
// 使用示例
/*  */
