import { useEffect, useRef, useState } from "react";
import { useFetcher } from "@remix-run/react";
import WaveSurfer from "wavesurfer.js";
import RecordPlugin from "wavesurfer.js/dist/plugins/record.esm.js";
import {
  Card,
  CardBody,
  CardHeader,
  Button,
  Typography,
  Select,
  Option,
  IconButton,
} from "@material-tailwind/react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCheck, faX } from "@fortawesome/free-solid-svg-icons";

export default function AudioRecorder() {
  const [wavesurfer, setWaveSurfer] = useState(null);
  const [record, setRecord] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const mode = useRef("");
  const [isPaused, setIsPaused] = useState(false);
  const [scrollingWaveform, setScrollingWaveform] = useState(true);
  const [progressTime, setProgressTime] = useState("00:00");
  const [audioBlob, setAudioBlob] = useState(null);
  const [verificationBlob, setVerificationBlob] = useState(null);
  const [hex, setHex] = useState(null);
  const [verifyHex, setVerifyHex] = useState(null);
  const saveHexFetcher = useFetcher();
  const verifyFetcher = useFetcher();
  const micRef = useRef(null);
  const recordingsRef = useRef(null);
  const verifyContainerRef = useRef(null);
  const pauseButtonRef = useRef(null);
  const micList = useRef([]);
  const micSelectRef = useRef(null);
  const audioFileInputRef = useRef(null);
  const verificationFileInputRef = useRef(null);
  const [micSelect, setMicSelect] = useState(null);

  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedHex = localStorage.getItem("savedHex");
      setHex(savedHex);
    }
  }, []);
  useEffect(() => {
    if (saveHexFetcher.state === "idle" && saveHexFetcher.data) {
      const extractedHex = saveHexFetcher.data.hex_embedding; // Giả sử API trả về { hex: "your_hex_value" }
      setHex(extractedHex);
      localStorage.setItem("hex", extractedHex);
    }
  }, [saveHexFetcher.state, saveHexFetcher.data]);
  useEffect(() => {
    if (verifyFetcher.state === "idle" && verifyFetcher.data) {
      const extractedHex = verifyFetcher.data.new_hex; // Giả sử API trả về { hex: "your_hex_value" }
      setVerifyHex(extractedHex);
    }
  }, [verifyFetcher.state, verifyFetcher.data]);

  useEffect(() => {
    initializeWaveSurfer();
    return () => {
      if (wavesurfer) wavesurfer.destroy();
    };
  }, [scrollingWaveform]);

  const initializeWaveSurfer = () => {
    if (wavesurfer) {
      wavesurfer.destroy();
    }
    const ws = WaveSurfer.create({
      container: micRef.current,
      waveColor: "rgb(200, 0, 200)",
      progressColor: "rgb(100, 0, 100)",
    });

    const recordPlugin = ws.registerPlugin(
      RecordPlugin.create({
        scrollingWaveform,
        renderRecordedAudio: false,
      })
    );

    recordPlugin.on("record-end", handleRecordEnd);
    recordPlugin.on("record-progress", handleRecordProgress);
    setWaveSurfer(ws);
    setRecord(recordPlugin);
  };

  const handleRecordEnd = (blob) => {
    let container = recordingsRef.current;
    // Check if we are in verification mode or main record mode
    if (mode.current === "login") {
      setVerificationBlob(blob); // Save the blob for verification if in verification mode
      container = verifyContainerRef.current;
    } else if (mode.current === "register") {
      setAudioBlob(blob); // Save the blob for audio if in main record mode
    }
    const recordedUrl = URL.createObjectURL(blob);

    const recordedWaveSurfer = WaveSurfer.create({
      container,
      waveColor: "rgb(200, 100, 0)",
      progressColor: "rgb(100, 50, 0)",
      url: recordedUrl,
    });

    const playButton = document.createElement("button");
    playButton.textContent = "Play";
    playButton.className = "px-4 py-2 bg-blue-500 text-white rounded mr-2";
    playButton.onclick = () => recordedWaveSurfer.playPause();
    recordedWaveSurfer.on("pause", () => (playButton.textContent = "Play"));
    recordedWaveSurfer.on("play", () => (playButton.textContent = "Pause"));
    container.appendChild(playButton);

    const downloadLink = document.createElement("a");
    downloadLink.href = recordedUrl;
    downloadLink.download = `recording.${
      blob.type.split(";")[0].split("/")[1] || "webm"
    }`;
    downloadLink.textContent = "Tải file voice";
    container.appendChild(downloadLink);
  };

  const handleRecordProgress = (time) => {
    const formattedTime = [
      Math.floor((time % 3600000) / 60000),
      Math.floor((time % 60000) / 1000),
    ]
      .map((v) => (v < 10 ? "0" + v : v))
      .join(":");
    setProgressTime(formattedTime);
  };

  const togglePause = () => {
    if (record.isPaused()) {
      record.resumeRecording();
      setIsPaused(false);
    } else {
      record.pauseRecording();
      setIsPaused(true);
    }
  };

  const toggleRecording = () => {
    if (record.isRecording() || record.isPaused()) {
      record.stopRecording();
      setIsRecording(false);
      setIsPaused(false);
    } else {
      const deviceId = micSelect;
      setIsRecording(true);
      record.startRecording({ deviceId });
    }
  };

  const registerVoice = () => {
    console.log("click");
    mode.current = "register";
    toggleRecording();
  };
  const loginVoice = () => {
    mode.current = "login";
    toggleRecording();
  };

  const handleSaveHex = async () => {
    if (audioBlob) {
      const formData = new FormData();
      formData.append("file", audioBlob);

      const data = await saveHexFetcher.submit(formData, {
        action: "/api/extract",
        method: "post",
        encType: "multipart/form-data",
      });
      console.log(data);
    }
  };

  const handleVerifyVoice = () => {
    console.log(verificationBlob);
    console.log(hex);
    if (verificationBlob && hex) {
      const formData = new FormData();
      formData.append("file", verificationBlob);
      formData.append("reference_hex", hex);

      verifyFetcher.submit(formData, {
        action: "/api/verify",
        method: "post",
        encType: "multipart/form-data",
      });
    }
  };

  useEffect(() => {
    RecordPlugin.getAvailableAudioDevices().then((devices) => {
      const selectElement = micSelectRef.current;
      devices.forEach((device) => {
        // const option = document.createElement("option");
        // option.value = device.deviceId;
        // option.text = device.label || device.deviceId;
        micList.current.push(device);
      });
    });
  }, []);

  const handleAudioUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      const container = recordingsRef.current;
      const uploadedWaveSurfer = WaveSurfer.create({
        container,
        waveColor: "rgb(0, 200, 0)",
        progressColor: "rgb(0, 100, 0)",
        url: url,
      });
      uploadedWaveSurfer.on("ready", () => {
        setAudioBlob(file); // Lưu tệp âm thanh đã tải lên vào audioBlob
      });
      const playButton = document.createElement("button");
      playButton.textContent = "Play";
      playButton.className = "px-4 py-2 bg-blue-500 text-white rounded mr-2";
      playButton.onclick = () => uploadedWaveSurfer.playPause();
      uploadedWaveSurfer.on("pause", () => (playButton.textContent = "Play"));
      uploadedWaveSurfer.on("play", () => (playButton.textContent = "Pause"));
      container.appendChild(playButton);
    }
  };

  const handleVerificationUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      const container = verifyContainerRef.current;
      const uploadedWaveSurfer = WaveSurfer.create({
        container,
        waveColor: "rgb(0, 100, 200)",
        progressColor: "rgb(0, 50, 100)",
        url: url,
      });
      uploadedWaveSurfer.on("ready", () => {
        setVerificationBlob(file); // Lưu tệp âm thanh đã tải lên vào verificationBlob
      });
      const playButton = document.createElement("button");
      playButton.textContent = "Play";
      playButton.className = "px-4 py-2 bg-blue-500 text-white rounded mr-2";
      playButton.onclick = () => uploadedWaveSurfer.playPause();
      uploadedWaveSurfer.on("pause", () => (playButton.textContent = "Play"));
      uploadedWaveSurfer.on("play", () => (playButton.textContent = "Pause"));
      container.appendChild(playButton);
    }
  };

  return (
    <div className="pt-4">
      {/* <h1 className="text-2xl mb-2 text-center" style={{ marginTop: 0 }}>
        Voice Hashing and Verification
      </h1> */}

      <Card className="mt-8">
        <CardHeader color="green" className="px-2">
          <Typography variant="h3">Ghi âm giọng nói để xác thực</Typography>
        </CardHeader>
        <CardBody>
          {/* Phần ghi âm cho giọng nói cần xác thực */}
          <div className="flex gap-3">
            <Button
              variant="gradient"
              className="px-4 py-2 bg-blue-500 text-white rounded mr-2"
              onClick={registerVoice}
            >
              {isRecording ? "Dừng ghi âm" : "Bắt đầu ghi âm"}
            </Button>
            <Button
              variant="gradient"
              className="px-4 py-2 bg-blue-500 text-white rounded"
              ref={pauseButtonRef}
              onClick={togglePause}
              style={{ display: isRecording ? "inline" : "none" }}
            >
              {isPaused ? "Tiếp tục" : "Dừng"}
            </Button>

            {micList.current.length > 1 && (
              <Select
                label="Chọn mic"
                value={micSelect}
                onChange={(val) => setMicSelect(val)}
              >
                {micList.current?.map((device) => (
                  <Option key={device.deviceId} value={device.deviceId}>
                    {device.label || device.deviceId}
                  </Option>
                ))}
              </Select>
            )}

            <label style={{ display: "inline-block" }}>
              <input
                type="checkbox"
                checked={scrollingWaveform}
                onChange={(e) => setScrollingWaveform(e.target.checked)}
              />{" "}
              Cuộn waveform
            </label>
            <input
              type="file"
              accept="audio/*"
              ref={audioFileInputRef}
              onChange={handleAudioUpload}
            />
          </div>

          <p>{progressTime}</p>
          <div
            ref={micRef}
            style={{
              border: "1px solid #ddd",
              borderRadius: "4px",
              marginTop: "1rem",
            }}
          ></div>

          <div ref={recordingsRef} style={{ margin: "1rem 0" }}></div>
          <Button
            variant="gradient"
            loading={
              saveHexFetcher.state == "submitting" ||
              saveHexFetcher.state == "loading"
            }
            onClick={handleSaveHex}
            // disabled={!audioBlob}
            className="px-4 py-2 bg-green-500 text-white rounded disabled:opacity-50"
          >
            Đăng ký hex
          </Button>
        </CardBody>
      </Card>
      <Card className="mt-8">
        <CardHeader color="green" className="px-2">
          <Typography variant="h3">
            Tải lên mẫu giọng nói để xác thực
          </Typography>
        </CardHeader>
        <CardBody className="flex flex-col gap-3">
          {/* Phần xác thực giọng nói */}
          <div>
            <h2>Xác thực giọng nói</h2>
          </div>
          <div ref={verifyContainerRef} style={{ margin: "1rem 0" }}></div>
          <div className="flex gap-3">
            <Button
              variant="gradient"
              className="px-4 py-2 bg-blue-500 text-white rounded"
              onClick={loginVoice}
            >
              {isRecording
                ? "Dừng ghi âm giọng nói để xác thực"
                : "Ghi âm giọng nói để xác thực"}
            </Button>
            <Button
              variant="gradient"
              className="px-4 py-2 bg-blue-500 text-white rounded"
              ref={pauseButtonRef}
              onClick={togglePause}
              style={{ display: isRecording ? "inline" : "none" }}
            >
              {isPaused ? "Resume" : "Pause"}
            </Button>
          </div>
          <input
            type="file"
            accept="audio/*"
            ref={verificationFileInputRef}
            onChange={handleVerificationUpload}
          />
          <div>
            <Button
              loading={
                verifyFetcher.state == "submitting" ||
                verifyFetcher.state == "loading"
              }
              onClick={handleVerifyVoice}
              // disabled={!verificationBlob || !hex}
              className="px-4 py-2 bg-red-500 text-white rounded disabled:opacity-50"
            >
              Đăng nhập
            </Button>
          </div>

          {verifyFetcher.data?.verificationResult && (
            <p>
              {verifyFetcher.data.verificationResult === "match"
                ? "Voice matches!"
                : "Voice does not match."}
            </p>
          )}
        </CardBody>
      </Card>
      <Card className="mt-8">
        <CardHeader color="green" className="px-2">
          <Typography variant="h3">Mã hex</Typography>
        </CardHeader>
        <CardBody>
          {verifyFetcher.data ? (
            <div className="flex flex-row justify-center align-center items-center">
              <FontAwesomeIcon
                size="4x"
                color={verifyFetcher.data?.verified ? "green" : "red"}
                icon={verifyFetcher.data?.verified ? faCheck : faX}
              />
              <Typography variant="h3">
                {verifyFetcher.data?.verified
                  ? "Giọng nói hợp lệ"
                  : "Không hợp lệ"}
              </Typography>
            </div>
          ) : (
            <></>
          )}
          <Typography variant="h4" className="mb-2">
            Điểm Jaccard
          </Typography>
          <Typography variant="h5" className="text-balance break-all">
            {verifyFetcher.data?.similarity || ""}
          </Typography>

          <Typography variant="h4" className="mb-2">
            Hex đã đăng ký
          </Typography>
          <div className="text-balance break-all overflow-scroll max-h-[200px]">
            {hex || ""}
          </div>
          <Typography variant="h4" className="mb-2">
            Hex đăng nhập
          </Typography>
          <div className="text-balance break-all overflow-scroll max-h-[200px]">
            {verifyHex || ""}
          </div>
        </CardBody>
      </Card>
    </div>
  );
}
