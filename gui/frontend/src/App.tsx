import React, { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { DatasetPage } from './pages/DatasetPage';
import { PoolCreatorPage } from './pages/PoolCreatorPage';
import { PoolViewerPage } from './pages/PoolViewerPage';
import { FinetuneDiffusionPage } from './pages/FinetuneDiffusionPage';
import { FinetuneContextDiffusionPage } from './pages/FinetuneContextDiffusionPage';
import { FinetuneAutoregressivePage } from './pages/FinetuneAutoregressivePage';
import { FinetuneHybridPage } from './pages/FinetuneHybridPage';
import { AzureRunsPage } from './pages/AzureRunsPage';
import { TestPrepPage } from './pages/TestPrepPage';
import { TestingPage } from './pages/TestingPage';
import { BenchmarksPage } from './pages/BenchmarksPage';
import { DocumentationPage } from './pages/DocumentationPage';
import { NAV_ITEMS, type ViewId } from './nav';

export const App: React.FC = () => {
  const [view, setView] = useState<ViewId>('dataset');
  const [navOpen, setNavOpen] = useState<boolean>(false);

  const activeMeta = NAV_ITEMS.find((n) => n.id === view);

  const renderView = (currentView: ViewId) => {
    switch (currentView) {
      case 'dataset':
        return <DatasetPage />;
      case 'pool-creator':
        return <PoolCreatorPage />;
      case 'pool-viewer':
        return <PoolViewerPage />;
      case 'finetune-diffusion':
        return <FinetuneDiffusionPage />;
      case 'finetune-context-diffusion':
        return <FinetuneContextDiffusionPage onNavigate={setView} />;
      case 'finetune-autoregressive':
        return <FinetuneAutoregressivePage />;
      case 'finetune-hybrid':
        return <FinetuneHybridPage />;
      case 'azure-runs':
        return <AzureRunsPage />;
      case 'test-prep':
        return <TestPrepPage />;
      case 'testing':
        return <TestingPage />;
      case 'benchmarks':
        return <BenchmarksPage />;
      case 'documentation':
        return <DocumentationPage />;
      default:
        return <DatasetPage />;
    }
  };

  return (
    <div className="app-layout">
      <Sidebar active={view} onSelect={setView} open={navOpen} onOpenChange={setNavOpen} />

      <main className="app-main">
        <div className="app-main-inner">
          <div className="app-topbar">
            <div className="app-topbar-title">
              <span className="dim mono" style={{ fontSize: 12, letterSpacing: '0.18em' }}>
                {activeMeta?.groupLabel.toUpperCase()}
              </span>
              <span className="topbar-sep">/</span>
              <span>{activeMeta?.label}</span>
            </div>
          </div>

          {renderView(view)}

          <footer
            style={{
              fontSize: 12,
              color: 'var(--text-mute)',
              textAlign: 'center',
              paddingTop: 12,
            }}
          >
            Backend: <span className="mono">/api/data/*</span> + <span className="mono">/api/azureml/*</span> · Proxy via Vite dev server on{' '}
            <span className="mono">127.0.0.1:5173</span>
          </footer>
        </div>
      </main>
    </div>
  );
};
