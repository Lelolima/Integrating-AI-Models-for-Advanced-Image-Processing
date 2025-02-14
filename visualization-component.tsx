import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

const TrainingVisualization = () => {
  // Sample training history data
  const history = Array.from({length: 10}, (_, i) => ({
    epoch: i + 1,
    trainLoss: [0.8, 0.6, 0.5, 0.4, 0.35, 0.32, 0.3, 0.28, 0.27, 0.26][i],
    valLoss: [0.85, 0.7, 0.6, 0.55, 0.52, 0.5, 0.48, 0.47, 0.46, 0.45][i],
    trainAcc: [0.7, 0.8, 0.85, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.94][i],
    valAcc: [0.65, 0.75, 0.8, 0.82, 0.84, 0.85, 0.86, 0.86, 0.87, 0.87][i]
  }));

  return (
    <div className="w-full max-w-6xl mx-auto p-4">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-4">Training Metrics</h2>
        <div className="grid grid-cols-1 gap-6">
          {/* Loss Plot */}
          <div className="border rounded-lg p-4">
            <h3 className="text-xl font-semibold mb-4">Loss Over Time</h3>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={history}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="epoch" 
                    label={{ 
                      value: 'Epoch', 
                      position: 'insideBottom', 
                      offset: -5 
                    }}
                  />
                  <YAxis
                    label={{
                      value: 'Loss',
                      angle: -90,
                      position: 'insideLeft'
                    }}
                  />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="trainLoss"
                    name="Training Loss"
                    stroke="#1f77b4"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="valLoss"
                    name="Validation Loss"
                    stroke="#ff7f0e"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          
          {/* Accuracy Plot */}
          <div className="border rounded-lg p-4">
            <h3 className="text-xl font-semibold mb-4">Accuracy Over Time</h3>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={history}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="epoch"
                    label={{ 
                      value: 'Epoch', 
                      position: 'insideBottom', 
                      offset: -5 
                    }}
                  />
                  <YAxis
                    label={{
                      value: 'Accuracy',
                      angle: -90,
                      position: 'insideLeft'
                    }}
                  />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="trainAcc"
                    name="Training Accuracy"
                    stroke="#2ca02c"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="valAcc"
                    name="Validation Accuracy"
                    stroke="#d62728"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingVisualization;