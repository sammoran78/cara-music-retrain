import React, { useState } from 'react';
import { DatasetDownloadProgress } from '../components/DatasetDownloadProgress';
import { BenchmarkResults } from '../components/BenchmarkResults';

const tabs = [
  { id: 'dataset-download', label: 'Dataset Download' },
  { id: 'benchmark-results', label: 'Benchmark Results' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('dataset-download');

  return (
    <main style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px', fontFamily: 'Arial, sans-serif' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ marginBottom: '8px' }}>CARA Attribution PoC</h1>
        <p style={{ color: '#555', margin: 0 }}>
          Monitor subset dataset downloads and inspect benchmark outputs from the web GUI.
        </p>
      </div>

      <div style={{ display: 'flex', gap: '12px', marginBottom: '24px', borderBottom: '1px solid #ddd', paddingBottom: '12px' }}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              border: '1px solid #ccc',
              borderRadius: '8px',
              padding: '10px 14px',
              background: activeTab === tab.id ? '#2563eb' : '#fff',
              color: activeTab === tab.id ? '#fff' : '#111',
              cursor: 'pointer',
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'dataset-download' ? <DatasetDownloadProgress /> : <BenchmarkResults />}
    </main>
  );
}
