import React, { useState } from "react";
import axios from "axios";
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
import { Line } from "react-chartjs-2";
// import "chart.js/auto";

export default function App() {
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [videoUrl, setVideoUrl] = useState(null);
  const [metrics, setMetrics] = useState(null);

  const handleUpload = async () => {
    if (!file) return;
    setProcessing(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://localhost:8000/upload/", formData);
      const { video_url, metrics_url } = res.data;

      // Wait a few seconds to simulate processing
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
      console.error("Upload failed:", err);
      setProcessing(false);
    }
  };

  return (
    <div className="p-6 max-w-xl mx-auto">
      <h1 className="text-xl font-bold mb-4">Dive Pose Estimation</h1>

      <input
        type="file"
        accept="video/*"
        onChange={(e) => setFile(e.target.files[0])}
        className="mb-4"
      />
      <button
        onClick={handleUpload}
        disabled={processing || !file}
        className="bg-blue-500 text-white px-4 py-2 rounded mb-6"
      >
        {processing ? "Processing..." : "Upload & Analyze"}
      </button>

      {videoUrl && (
        <div className="mb-6">
          <h2 className="font-semibold">Pose Estimation Video</h2>
          <video controls src={videoUrl} className="w-full mt-2" />
        </div>
      )}

      {metrics && (
        <>
          {/* Existing joint_angles chart */}
          <h2 className="font-semibold mb-2">Joint Angles</h2>
          <Line
            data={{
              labels: metrics.joint_angles.Torso.map((_, i) => i),
              datasets: Object.entries(metrics.joint_angles).map(([joint, values], idx) => ({
                label: joint,
                data: values,
                borderColor: ['#3b82f6', '#f97316', '#10b981', '#e11d48'][idx % 4],
                fill: false,
              }))
            }}
          />

          {/* New filtered_metrics charts */}
          <h2 className="font-semibold mt-8 mb-2">Diver Heights</h2>
          <Line
            data={{
              labels: metrics.filtered_metrics.diver_heights.map((_, i) => i),
              datasets: [{
                label: 'Diver Heights (m)',
                data: metrics.filtered_metrics.diver_heights,
                borderColor: '#3b82f6',
                fill: false,
              }]
            }}
            options={{ responsive: true }}
          />

          <h2 className="font-semibold mt-8 mb-2">Total Rotation Over Time</h2>
          <Line
            data={{
              labels: metrics.filtered_metrics.total_rotation_over_time.map((_, i) => i),
              datasets: [{
                label: 'Total Rotation (°)',
                data: metrics.filtered_metrics.total_rotation_over_time,
                borderColor: '#f97316',
                fill: false,
              }]
            }}
            options={{ responsive: true }}
          />

          <h2 className="font-semibold mt-8 mb-2">Rotation Rate</h2>
          <Line
            data={{
              labels: metrics.filtered_metrics.rotation_rate.map((_, i) => i),
              datasets: [{
                label: 'Rotation Rate (°/s)',
                data: metrics.filtered_metrics.rotation_rate,
                borderColor: '#10b981',
                fill: false,
              }]
            }}
            options={{ responsive: true }}
          />
        </>
      )}

    </div>
  );
}
