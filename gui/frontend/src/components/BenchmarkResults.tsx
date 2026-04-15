import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, TrendingUp, AlertCircle, CheckCircle } from 'lucide-react';

interface BenchmarkData {
  generated: string;
  n_pools: number;
  chance_level: number;
  run_A: {
    label: string;
    accuracy: number;
    delta_vs_chance: number;
  };
  run_B: {
    label: string;
    accuracy: number;
    delta_vs_chance: number;
    delta_vs_A: number;
  };
  run_C: {
    label: string;
    accuracy: number;
    delta_vs_chance: number;
    delta_vs_A: number;
  };
}

export const BenchmarkResults: React.FC = () => {
  const [benchmarkData, setBenchmarkData] = useState<BenchmarkData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchBenchmarkData();
  }, []);

  const fetchBenchmarkData = async () => {
    try {
      const response = await fetch('/api/benchmark/results');
      if (!response.ok) throw new Error('Failed to fetch benchmark data');
      const data = await response.json();
      setBenchmarkData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error || !benchmarkData) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center">
          <AlertCircle className="h-5 w-5 text-red-600 mr-2" />
          <span className="text-red-800">{error || 'No benchmark data available'}</span>
        </div>
      </div>
    );
  }

  const chartData = [
    {
      name: 'Chance',
      accuracy: benchmarkData.chance_level * 100,
      fill: '#9CA3AF',
    },
    {
      name: 'Run A',
      accuracy: benchmarkData.run_A.accuracy * 100,
      fill: '#60A5FA',
      delta: `+${(benchmarkData.run_A.delta_vs_chance * 100).toFixed(1)}pp`,
    },
    {
      name: 'Run B',
      accuracy: benchmarkData.run_B.accuracy * 100,
      fill: '#34D399',
      delta: `+${(benchmarkData.run_B.delta_vs_A * 100).toFixed(1)}pp vs A`,
    },
    {
      name: 'Run C',
      accuracy: benchmarkData.run_C.accuracy * 100,
      fill: '#F59E0B',
      delta: `+${(benchmarkData.run_C.delta_vs_A * 100).toFixed(1)}pp vs A`,
    },
  ];

  const getConfoundAssessment = () => {
    const deltaCA = benchmarkData.run_C.accuracy - benchmarkData.run_A.accuracy;
    const deltaBC = benchmarkData.run_B.accuracy - benchmarkData.run_C.accuracy;

    if (deltaCA < 0.05) {
      return {
        level: 'low',
        icon: CheckCircle,
        color: 'text-green-600',
        bgColor: 'bg-green-50',
        borderColor: 'border-green-200',
        message: 'Low confound risk. Attribution head performs meaningful classification.',
      };
    } else if (deltaCA >= 0.05 && deltaBC > 0.10) {
      return {
        level: 'moderate',
        icon: Activity,
        color: 'text-yellow-600',
        bgColor: 'bg-yellow-50',
        borderColor: 'border-yellow-200',
        message: 'Moderate restructuring. Both conditioning and head contribute.',
      };
    } else {
      return {
        level: 'high',
        icon: AlertCircle,
        color: 'text-orange-600',
        bgColor: 'bg-orange-50',
        borderColor: 'border-orange-200',
        message: 'Further analysis needed. Consider counterfactual codeword tests.',
      };
    }
  };

  const assessment = getConfoundAssessment();
  const AssessmentIcon = assessment.icon;

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-4">CARA Attribution Probe Results</h2>
        <p className="text-gray-600 mb-6">
          Generated: {new Date(benchmarkData.generated).toLocaleString()}
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">Run A: Base Model</h3>
            <p className="text-3xl font-bold text-blue-600">
              {(benchmarkData.run_A.accuracy * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-blue-700 mt-1">
              +{(benchmarkData.run_A.delta_vs_chance * 100).toFixed(1)}pp vs chance
            </p>
          </div>

          <div className="bg-green-50 rounded-lg p-4">
            <h3 className="font-semibold text-green-900 mb-2">Run B: CARA Head</h3>
            <p className="text-3xl font-bold text-green-600">
              {(benchmarkData.run_B.accuracy * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-green-700 mt-1">
              +{(benchmarkData.run_B.delta_vs_A * 100).toFixed(1)}pp vs base
            </p>
          </div>

          <div className="bg-yellow-50 rounded-lg p-4">
            <h3 className="font-semibold text-yellow-900 mb-2">Run C: CARA Linear</h3>
            <p className="text-3xl font-bold text-yellow-600">
              {(benchmarkData.run_C.accuracy * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-yellow-700 mt-1">
              +{(benchmarkData.run_C.delta_vs_A * 100).toFixed(1)}pp vs base
            </p>
          </div>
        </div>

        <div className="h-64 mb-6">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Bar dataKey="accuracy" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className={`${assessment.bgColor} ${assessment.borderColor} border rounded-lg p-4`}>
          <div className="flex items-start">
            <AssessmentIcon className={`h-5 w-5 ${assessment.color} mr-2 mt-0.5`} />
            <div>
              <h4 className="font-semibold text-gray-900 mb-1">
                Control-Token Confound Assessment
              </h4>
              <p className="text-gray-700">{assessment.message}</p>
            </div>
          </div>
        </div>

        <div className="mt-6 space-y-2">
          <h4 className="font-semibold text-gray-900">Key Insights</h4>
          <ul className="space-y-1 text-sm text-gray-600">
            <li className="flex items-center">
              <TrendingUp className="h-4 w-4 mr-2 text-gray-400" />
              CARA adds {(benchmarkData.run_B.delta_vs_A * 100).toFixed(1)} percentage points 
              beyond base model implicit signal
            </li>
            <li className="flex items-center">
              <Activity className="h-4 w-4 mr-2 text-gray-400" />
              Fine-tuning {benchmarkData.run_C.delta_vs_A > 0.05 ? 'significantly' : 'minimally'} 
              restructured DiT representations
            </li>
            <li className="flex items-center">
              <CheckCircle className="h-4 w-4 mr-2 text-gray-400" />
              Attribution head contributes {((benchmarkData.run_B.accuracy - benchmarkData.run_C.accuracy) * 100).toFixed(1)}pp 
              additional accuracy
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
