const tf = require("@tensorflow/tfjs-node");
const InputError = require("../exceptions/InputError");

async function predictClassification(model, image) {
  try {
    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat();

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;

    const label = confidenceScore > 0.5 ? "Cancer" : "Non-cancer";

    let suggestion;

    if (label === "Cancer") {
      suggestion = "Kamu kena kanker hahaha";
    }

    if (label === "Non-cancer") {
      suggestion = "tumben sehat, ga kena kanker";
    }

    return { label, suggestion };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan dalam melakukan prediksi`);
  }
}

module.exports = predictClassification;
