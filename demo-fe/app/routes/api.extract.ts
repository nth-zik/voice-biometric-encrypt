import type { ActionFunctionArgs } from "@remix-run/node";
import { json } from "@remix-run/node";
import { resolve, dirname } from "path";
import fs from "fs";
import { promisify } from "util";
import ffmpeg from "fluent-ffmpeg";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
import { Readable } from "stream";

export async function action({ request }: ActionFunctionArgs) {
  const formData = await request.formData();
  const audioFile = formData.get("file");

  if (!audioFile) {
    return json({ error: "No file provided" }, { status: 400 });
  }

  // Chuyển file thành một Readable stream
  const buffer = await audioFile.arrayBuffer();
  const readableStream = Readable.from(Buffer.from(buffer));

  const tempFilePath = resolve(__dirname, "../../temp.webm");
  const writeStream = fs.createWriteStream(tempFilePath, {
    encoding: "binary",
  });

  readableStream.pipe(writeStream);

  await new Promise((resolve) => writeStream.on("finish", resolve));

  // Đường dẫn cho file đích (file WAV)
  const outputFilePath = resolve(__dirname, "./../output.wav");

  // Chuyển đổi từ webm sang wav
  await new Promise((resolve, reject) => {
    ffmpeg(tempFilePath)
      .toFormat("wav")
      .audioFrequency(16000)
      .on("end", resolve)
      .on("error", reject)
      .save(outputFilePath);
  });

  // Đọc file WAV và gửi về (hoặc lưu trữ)
  formData.delete("file");
  const wavBuffer = await promisify(fs.readFile)(outputFilePath);
  const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });

  formData.append("file", wavBlob);

  await promisify(fs.unlink)(tempFilePath);
  await promisify(fs.unlink)(outputFilePath);

  try {
    // Gửi file tới API Flask
    const response = await fetch(
      `${process.env.BACKEND_ENDPOINT}/api/extract`,
      {
        method: "POST",
        body: formData,
      }
    );

    if (!response.ok) {
      throw new Error(`Failed to fetch: ${response.statusText}`);
    }

    const data = await response.json();
    return json(data);
  } catch (error) {
    console.error("Error sending file to Flask API:", error);
    return json({ error: `${error}` }, { status: 500 });
  }
}
