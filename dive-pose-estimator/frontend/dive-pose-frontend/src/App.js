import React, { useState } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import 'bootstrap/dist/css/bootstrap.min.css';
import {
  Chart as ChartJS,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
  Legend
);

export default function DivePoseUI() {
  const [file, setFile] = useState(null);
  const [rotate, setRotate] = useState(true);
  const [stageDetection, setStageDetection] = useState(false);
  const [startTime, setStartTime] = useState(0);
  const [endTime, setEndTime] = useState(0);
  const [boardHeight, setBoardHeight] = useState(5);
  const [diverHeight, setDiverHeight] = useState(1.5);

  const [processing, setProcessing] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  const handleUpload = async () => {
    if (!file) return;
    setProcessing(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("rotate", rotate);
    formData.append("stage_detection", stageDetection);
    formData.append("start_time", startTime);
    formData.append("end_time", endTime);
    formData.append("board_height", boardHeight);
    formData.append("diver_height", diverHeight);

    try {
      const res = await axios.post("http://localhost:8000/upload/", formData);
      const { video_url, metrics_url } = res.data;

      setTimeout(async () => {
        const videoRes = await axios.get(`http://localhost:8000${video_url}`, {
          responseType: "blob",
        });
        const metricsRes = await axios.get(`http://localhost:8000${metrics_url}`);

        setVideoUrl(URL.createObjectURL(videoRes.data));
        setMetrics(JSON.parse(metricsRes.data));
        setProcessing(false);
      }, 5000);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        "Upload failed. Please try again."
      );
      setProcessing(false);
    }
  };

  const renderLineChart = (label, data, color) => (
    <div className="bg-white p-4 rounded shadow mb-4 card mb-4 p-3 shadow-sm">
      <h3 className="font-medium text-lg mb-2 card-title h5 mb-3">{label}</h3>
      <Line
        data={{
          labels: data.map((_, i) => i),
          datasets: [{
            label: label,
            data,
            borderColor: color,
            backgroundColor: color,
            fill: false
          }]
        }}
      />
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-100 p-8 bg-light py-5 min-vh-100">
      <div className="max-w-4xl mx-auto bg-white p-6 rounded shadow container bg-white p-4 rounded-3 shadow-sm">
        <h1 className="text-2xl font-bold mb-6 text-center display-6 mb-4 text-center">Dive Pose Estimation</h1>
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4 alert alert-danger">
            {error}
          </div>
        )}
        <div className="flex flex-col items-center mb-6 d-flex flex-column align-items-center mb-4">
          <label className="mb-2 w-full form-label w-100">
            <span className="block mb-1">Video File</span>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setFile(e.target.files[0])}
              className="mb-4 form-control mb-3"
            />
          </label>
          <label className="mb-2 w-full flex items-center form-check w-100 mb-2">
            <input
              type="checkbox"
              checked={rotate}
              onChange={(e) => setRotate(e.target.checked)}
              className="mr-2 form-check-input me-2"
            />
            Rotate Video
          </label>
          <label className="mb-2 w-full flex items-center form-check w-100 mb-2">
            <input
              type="checkbox"
              checked={stageDetection}
              onChange={(e) => setStageDetection(e.target.checked)}
              className="mr-2 form-check-input me-2"
            />
            Stage Detection
          </label>
          <label className="mb-2 w-full form-label w-100">
            <span className="block mb-1">Dive Start Time (s)</span>
            <input
              type="number"
              step="any"
              placeholder="Start Time (s)"
              value={startTime}
              onChange={(e) => setStartTime(e.target.value)}
              className="form-control mb-2"
            />
          </label>
          <label className="mb-2 w-full form-label w-100">
            <span className="block mb-1">Dive End Time (s)</span>
            <input
              type="number"
              step="any"
              placeholder="End Time (s)"
              value={endTime}
              onChange={(e) => setEndTime(e.target.value)}
              className="form-control mb-2"
            />
          </label>
          <label className="mb-2 w-full form-label w-100">
            <span className="block mb-1">Board Height (m)</span>
            <input
              type="number"
              step="any"
              placeholder="Board Height (m)"
              value={boardHeight}
              onChange={(e) => setBoardHeight(e.target.value)}
              className="form-control mb-2"
            />
          </label>
          <label className="mb-4 w-full form-label w-100">
            <span className="block mb-1">Diver Height (m)</span>
            <input
              type="number"
              step="any"
              placeholder="Diver Height (m)"
              value={diverHeight}
              onChange={(e) => setDiverHeight(e.target.value)}
              className="form-control mb-3"
            />
          </label>
          <button
            onClick={handleUpload}
            disabled={processing || !file}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded disabled:opacity-50 btn btn-primary px-4 py-2"
          >
            {processing ? "Processing..." : "Upload & Analyze"}
          </button>
        </div>

        {videoUrl && (
          <div className="mb-6 mb-4">
            <h2 className="text-xl font-semibold mb-2 h5 mb-3">Pose Estimation Output</h2>
            <video controls src={videoUrl} className="w-full rounded-lg shadow-md w-100 rounded-3 shadow-sm" />
          </div>
        )}

        {metrics && (
          <div className="mt-8 mt-5">
            <h2 className="text-xl font-semibold mb-4 h5 mb-3">Performance Metrics</h2>
            {renderLineChart("Diver Height (m)", metrics.filtered_metrics.diver_heights, '#3b82f6')}
            {renderLineChart("Total Rotation Over Time (°)", metrics.filtered_metrics.total_rotation_over_time, '#f97316')}
            {renderLineChart("Rotation Rate (°/s)", metrics.filtered_metrics.rotation_rate, '#10b981')}
          </div>
        )}
      </div>
    </div>
  );
}
