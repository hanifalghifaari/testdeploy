// pages/api/predict.js
import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";

export default async function handler(req, res) {
  if (req.method === "POST") {
    try {
      const { inputData } = req.body;

      // Load model dari file
      const modelPath = path.join(process.cwd(), "model", "Densenet_model.h5");
      const model = await tf.loadLayersModel(`file://${modelPath}`);

      // Lakukan preprocessing data input jika diperlukan
      // const processedData = preprocess(inputData);

      // Buat prediksi
      const prediction = model.predict(tf.tensor([inputData]));
      const result = prediction.arraySync();

      res.status(200).json({ result });
    } catch (error) {
      console.error("Error in prediction:", error);
      res.status(500).json({ error: "Prediction failed" });
    }
  } else {
    res.status(405).json({ error: "Method not allowed" });
  }
}
