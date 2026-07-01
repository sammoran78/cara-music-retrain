import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Cloud,
  ExternalLink,
  FileJson,
  FlaskConical,
  Loader2,
  Play,
  RefreshCw,
  ShieldCheck,
  XCircle,
} from 'lucide-react';
import { PageHeader } from './PageHeader';

interface AzureJob {
  name?: string | null;
  display_name?: string | null;
  status?: string | null;
  compute?: string | null;
  environment?: string | null;
  created_at?: string | null;
  start_time?: string | null;
  end_time?: string | null;
  studio_url?: string | null;
  test_id?: string;
  test_name?: string;
}

interface TestDefinition {
  id: string;
  name: string;
  label: string;
  description: string;
  job_file: string;
  environment_file?: string;
  compute: string;
  environment: string;
  input_path?: string | null;
  gpu: boolean;
  prerequisites: string[];
  warnings: string[];
  latest_job?: AzureJob | null;
}

interface AuditEvent {
  timestamp_utc?: string;
  action?: string;
  test_id?: string;
  test_name?: string;
  job_name?: string;
  compute?: string;
  environment?: string;
  input_path?: string | null;
  error?: string;
}

interface TestPrepState {
  definitions: TestDefinition[];
  history: AzureJob[];
  audit: AuditEvent[];
  gpu_warning: string;
  submissions?: Record<string, {
    running: boolean;
    status: string;
    started_at?: string | null;
    finished_at?: string | null;
    job_name?: string | null;
    error?: string | null;
  }>;
  run_all: {
    running: boolean;
    current_test_id?: string | null;
    last_message?: string | null;
    last_error?: string | null;
    submitted_jobs?: string[];
  };
}

interface JobReport {
  job_name: string;
  report: Record<string, unknown>;
  metadata: Record<string, unknown>;
  output_artifact_location?: string | null;
  report_path?: string | null;
}

interface LiveMonitor {
  job_name: string;
  observed_at: string;
  last_azure_heartbeat?: string | null;
  azure_cache_refreshed_at?: string | null;
  azure_cache_age_seconds?: number | null;
  elapsed_seconds?: number | null;
  status: string;
  compute?: string | null;
  environment?: string | null;
  studio_url?: string | null;
  logs_downloadable: boolean;
  heartbeat_fresh: boolean;
  heartbeat_error?: string | null;
  message: string;
  timeline: Array<{
    observed_at: string;
    status?: string | null;
    compute?: string | null;
    environment?: string | null;
  }>;
}

type TabId = '01' | '02' | '03' | '04' | 'history' | 'audit';

const fetchJson = async <T,>(url: string, init?: RequestInit): Promise<T> => {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new Error('Dashboard backend is unavailable. Start or restart ./start-cara-console.sh, then refresh this page.');
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body?.detail;
    throw new Error(typeof detail === 'string' ? detail : detail?.message || `${url} HTTP ${response.status}`);
  }
  return body as T;
};

const formatTimestamp = (value?: string | null): string => {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

const phaseStatus = (job?: AzureJob | null): 'PASS' | 'FAIL' | 'RUNNING' | 'NOT RUN' => {
  const value = String(job?.status || '').toLowerCase();
  if (['completed', 'succeeded', 'finished'].includes(value)) return 'PASS';
  if (['failed', 'canceled', 'cancelled'].includes(value)) return 'FAIL';
  if (value) return 'RUNNING';
  return 'NOT RUN';
};

const statusClass = (status: ReturnType<typeof phaseStatus>): string => {
  if (status === 'PASS') return 'status-done';
  if (status === 'FAIL') return 'status-error';
  if (status === 'RUNNING') return 'status-running';
  return 'status-queued';
};

const definitionStatus = (definition: TestDefinition, state?: TestPrepState | null): ReturnType<typeof phaseStatus> =>
  state?.submissions?.[definition.id]?.running ? 'RUNNING' : phaseStatus(definition.latest_job);

const isActiveCloudState = (status?: string | null): boolean => {
  const value = String(status || '').toLowerCase();
  return Boolean(value) && !['completed', 'succeeded', 'finished', 'failed', 'canceled', 'cancelled'].includes(value);
};

const compactResource = (value?: string | null): string => {
  if (!value) return '—';
  const parts = value.split('/').filter(Boolean);
  return parts[parts.length - 1] || value;
};

const formatDuration = (seconds?: number | null): string => {
  if (seconds == null) return '—';
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes}m ${remainder}s`;
};

export const TestPrepPage: React.FC = () => {
  const [state, setState] = useState<TestPrepState | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>('01');
  const [report, setReport] = useState<JobReport | null>(null);
  const [liveMonitor, setLiveMonitor] = useState<LiveMonitor | null>(null);
  const [monitorLoading, setMonitorLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [action, setAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setState(await fetchJson<TestPrepState>('/api/azureml/test-prep'));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh Azure ML test-prep state');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = window.setInterval(refresh, 15000);
    return () => window.clearInterval(interval);
  }, [refresh]);

  const activeDefinition = useMemo(
    () => state?.definitions.find((definition) => definition.id === activeTab) || null,
    [activeTab, state?.definitions],
  );

  const loadMonitor = useCallback(async (jobName?: string | null, silent = false) => {
    if (!jobName) return;
    try {
      if (!silent) setMonitorLoading(true);
      setLiveMonitor(await fetchJson<LiveMonitor>(`/api/azureml/test-prep/jobs/${encodeURIComponent(jobName)}/monitor`));
    } catch (err) {
      if (!silent) setError(err instanceof Error ? err.message : 'Failed to load Azure ML live state');
    } finally {
      if (!silent) setMonitorLoading(false);
    }
  }, []);

  useEffect(() => {
    const jobName = activeDefinition?.latest_job?.name;
    if (!jobName) {
      setLiveMonitor(null);
      return;
    }
    setLiveMonitor(null);
    loadMonitor(jobName);
    const interval = window.setInterval(() => loadMonitor(jobName, true), 10000);
    return () => window.clearInterval(interval);
  }, [activeDefinition?.latest_job?.name, loadMonitor]);

  const loadReport = useCallback(async (jobName?: string | null) => {
    if (!jobName) return;
    try {
      setAction(`report:${jobName}`);
      setReport(await fetchJson<JobReport>(`/api/azureml/test-prep/jobs/${encodeURIComponent(jobName)}/report`));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load Azure ML test report');
    } finally {
      setAction(null);
    }
  }, []);

  const runPhase = useCallback(async (definition: TestDefinition) => {
    const prerequisiteWarning = definition.warnings.join('\n');
    let allowOverride = false;
    if (prerequisiteWarning) {
      allowOverride = window.confirm(`${prerequisiteWarning}\n\nRun this phase anyway using the advanced prerequisite override?`);
      if (!allowOverride) return;
    }
    let confirmGpu = false;
    if (definition.gpu) {
      confirmGpu = window.confirm(state?.gpu_warning || 'This will start GPU compute on gpu-smoke-h100. Confirm before running.');
      if (!confirmGpu) return;
    }
    try {
      setAction(`run:${definition.id}`);
      await fetchJson(`/api/azureml/test-prep/${definition.id}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm_gpu: confirmGpu, allow_prerequisite_override: allowOverride }),
      });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to submit phase ${definition.id}`);
    } finally {
      setAction(null);
    }
  }, [refresh, state?.gpu_warning]);

  const registerEnvironment = useCallback(async (definition: TestDefinition) => {
    if (!window.confirm(`Register ${definition.environment} in Azure ML from ${definition.environment_file}?`)) return;
    try {
      setAction(`environment:${definition.id}`);
      await fetchJson(`/api/azureml/test-prep/${definition.id}/register-environment`, { method: 'POST' });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to register environment for phase ${definition.id}`);
    } finally {
      setAction(null);
    }
  }, [refresh]);

  const cancelPhase = useCallback(async (definition: TestDefinition) => {
    const job = definition.latest_job;
    const jobName = job?.name;
    if (!job || !jobName || !isActiveCloudState(job.status)) return;
    if (!window.confirm(`Cancel Azure ML job ${jobName} for phase ${definition.id}?\n\nThis stops the active cloud run. Any completed earlier evidence remains preserved.`)) return;
    try {
      setAction(`cancel:${jobName}`);
      await fetchJson(`/api/azureml/test-prep/${definition.id}/jobs/${encodeURIComponent(jobName)}/cancel`, { method: 'POST' });
      await refresh();
      await loadMonitor(jobName, true);
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to cancel phase ${definition.id}`);
    } finally {
      setAction(null);
    }
  }, [loadMonitor, refresh]);

  const runAll = useCallback(async () => {
    if (!window.confirm(state?.gpu_warning || 'Run All will start GPU compute on gpu-smoke-h100. Confirm before running.')) return;
    try {
      setAction('run-all');
      await fetchJson('/api/azureml/test-prep/run-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirm_gpu: true }),
      });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start the sequential test-prep run');
    } finally {
      setAction(null);
    }
  }, [refresh, state?.gpu_warning]);

  return (
    <div style={{ display: 'grid', gap: 20 }}>
      <PageHeader
        kicker="Operations · Azure Machine Learning"
        title={<>Test Prep / <em>Environment Validation</em></>}
        description={
          <>
            Validate one shared private audio dataset across CPU access, H100 CUDA, AudioCraft, and Stable Audio Tools before any fine-tuning begins.
            Every cloud phase is versioned and written into the CARA research audit trail.
          </>
        }
        actions={
          <div className="row" style={{ gap: 10 }}>
            <button className="btn btn-ghost" onClick={refresh} type="button" disabled={loading}>
              <RefreshCw size={16} className={loading ? 'spin' : ''} /> Refresh
            </button>
            <button className="btn" onClick={runAll} type="button" disabled={Boolean(action) || Boolean(state?.run_all.running)}>
              {action === 'run-all' || state?.run_all.running ? <Loader2 size={15} className="spin" /> : <Play size={15} />} Run All Test-Prep Phases
            </button>
          </div>
        }
      />

      {error ? <div className="error-banner">{error}</div> : null}
      <div className="test-prep-warning">
        <AlertTriangle size={17} />
        <div>
          <strong>GPU cost gate:</strong> {state?.gpu_warning || 'GPU phases require confirmation before submission.'}
          <div className="dim">Test prep never submits training jobs, online endpoints, Marketplace resources, or gpu-fulltraining-h100 work.</div>
        </div>
      </div>

      {state?.run_all.running || state?.run_all.last_error ? (
        <section className="card">
          <div className="card-header">
            <div className="card-title">Sequential Run All</div>
            <div className="card-meta">{state.run_all.running ? `phase ${state.run_all.current_test_id || 'starting'}` : 'stopped'}</div>
          </div>
          <div className={state.run_all.last_error ? 'error-banner' : 'dim'}>
            {state.run_all.last_error || state.run_all.last_message}
          </div>
        </section>
      ) : null}

      <section className="test-prep-tabs" aria-label="Test prep sections">
        {state?.definitions.map((definition) => {
          const value = definitionStatus(definition, state);
          return (
            <button className={`test-prep-tab${activeTab === definition.id ? ' is-active' : ''}`} key={definition.id} type="button" onClick={() => setActiveTab(definition.id as TabId)}>
              <span className="mono">{definition.id}</span>
              <span>{definition.label}</span>
              <span className={`status-pill ${statusClass(value)}`}>{value}</span>
            </button>
          );
        })}
        <button className={`test-prep-tab${activeTab === 'history' ? ' is-active' : ''}`} type="button" onClick={() => setActiveTab('history')}>
          <span><FlaskConical size={14} /></span><span>Test History</span>
        </button>
        <button className={`test-prep-tab${activeTab === 'audit' ? ' is-active' : ''}`} type="button" onClick={() => setActiveTab('audit')}>
          <span><ShieldCheck size={14} /></span><span>Research Audit Trail</span>
        </button>
      </section>

      {activeDefinition ? (
        <section className="split-2 test-prep-grid">
          <div className="card">
            <div className="card-header">
              <div>
                <div className="card-title">{activeDefinition.id} · {activeDefinition.label}</div>
                <div className="card-meta mono">{activeDefinition.name}</div>
              </div>
              <span className={`status-pill ${statusClass(definitionStatus(activeDefinition, state))}`}>{definitionStatus(activeDefinition, state)}</span>
            </div>
            <p className="dim">{activeDefinition.description}</p>
            {activeDefinition.warnings.length ? (
              <div className="test-prep-warning">
                <AlertTriangle size={16} />
                <div>{activeDefinition.warnings.join(' ')}</div>
              </div>
            ) : null}
            {state?.submissions?.[activeDefinition.id]?.running ? (
              <div className="dim" style={{ marginBottom: 12 }}>
                Azure ML is accepting the phase submission in the background.
              </div>
            ) : null}
            {state?.submissions?.[activeDefinition.id]?.error ? (
              <div className="error-banner">{state.submissions[activeDefinition.id].error}</div>
            ) : null}
            <div className="mini-list">
              <div><span className="k">Compute</span><span className="v mono">{activeDefinition.compute}</span></div>
              <div><span className="k">Environment</span><span className="v mono">{activeDefinition.environment}</span></div>
              <div><span className="k">Input datastore path</span><span className="v mono">{activeDefinition.input_path || 'not required'}</span></div>
              <div><span className="k">Job definition</span><span className="v mono">{activeDefinition.job_file}</span></div>
            </div>
            <div className="row" style={{ gap: 10, marginTop: 16, flexWrap: 'wrap', alignItems: 'center' }}>
              <button
                className="btn"
                type="button"
                disabled={Boolean(action) || Boolean(state?.submissions?.[activeDefinition.id]?.running) || isActiveCloudState(activeDefinition.latest_job?.status)}
                onClick={() => runPhase(activeDefinition)}
              >
                {action === `run:${activeDefinition.id}` || state?.submissions?.[activeDefinition.id]?.running || isActiveCloudState(activeDefinition.latest_job?.status)
                  ? <Loader2 size={15} className="spin" />
                  : <Play size={15} />}
                {state?.submissions?.[activeDefinition.id]?.running
                  ? 'Submitting to Azure ML...'
                  : isActiveCloudState(activeDefinition.latest_job?.status)
                    ? `Cloud state: ${activeDefinition.latest_job?.status}`
                    : `Run ${activeDefinition.id} ${activeDefinition.label} Test`}
              </button>
              {activeDefinition.latest_job && isActiveCloudState(activeDefinition.latest_job.status) ? (
                <button
                  className="btn btn-ghost"
                  type="button"
                  disabled={Boolean(action)}
                  onClick={() => cancelPhase(activeDefinition)}
                  style={{ padding: 0, border: 0, background: 'transparent', minHeight: 0 }}
                >
                  {action === `cancel:${activeDefinition.latest_job.name}` ? <Loader2 size={14} className="spin" /> : null}
                  Cancel run
                </button>
              ) : null}
              {activeDefinition.environment_file ? (
                <button className="btn btn-ghost" type="button" disabled={Boolean(action)} onClick={() => registerEnvironment(activeDefinition)}>
                  {action === `environment:${activeDefinition.id}` ? <Loader2 size={15} className="spin" /> : <Cloud size={15} />}
                  Register Environment
                </button>
              ) : null}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div className="card-title">Latest Run</div>
              <div className="card-meta">{formatTimestamp(activeDefinition.latest_job?.created_at)}</div>
            </div>
            {activeDefinition.latest_job ? (
              <>
                <div className="mini-list">
                  <div><span className="k">Azure ML job ID</span><span className="v mono">{activeDefinition.latest_job.name}</span></div>
                  <div><span className="k">Cloud state</span><span className="v">{activeDefinition.latest_job.status}</span></div>
                  <div><span className="k">Compute target</span><span className="v mono">{compactResource(activeDefinition.latest_job.compute)}</span></div>
                  <div><span className="k">Environment</span><span className="v mono">{compactResource(activeDefinition.latest_job.environment)}</span></div>
                </div>
                <div className="row" style={{ gap: 10, marginTop: 16, flexWrap: 'wrap' }}>
                  {activeDefinition.latest_job.studio_url ? (
                    <a className="btn btn-ghost" href={activeDefinition.latest_job.studio_url} target="_blank" rel="noreferrer">
                      <ExternalLink size={15} /> Open in Studio
                    </a>
                  ) : null}
                  <button className="btn btn-ghost" type="button" disabled={Boolean(action)} onClick={() => loadReport(activeDefinition.latest_job?.name)}>
                    {action === `report:${activeDefinition.latest_job.name}` ? <Loader2 size={15} className="spin" /> : <FileJson size={15} />} View Report
                  </button>
                  <button className="btn btn-ghost" type="button" disabled={monitorLoading} onClick={() => loadMonitor(activeDefinition.latest_job?.name)}>
                    {monitorLoading ? <Loader2 size={15} className="spin" /> : <RefreshCw size={15} />} Refresh Live State
                  </button>
                </div>
              </>
            ) : <div className="pool-empty-state">This phase has not been submitted yet.</div>}
          </div>
        </section>
      ) : null}

      {liveMonitor ? (
        <section className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Live Azure State Monitor</div>
              <div className="card-meta mono">{liveMonitor.job_name}</div>
            </div>
            <span className={`status-pill ${statusClass(phaseStatus({ status: liveMonitor.status }))}`}>{liveMonitor.status}</span>
          </div>
          <div className="mini-list">
            <div><span className="k">Elapsed since submission</span><span className="v mono">{formatDuration(liveMonitor.elapsed_seconds)}</span></div>
            <div><span className="k">Dashboard response</span><span className="v mono">{formatTimestamp(liveMonitor.observed_at)}</span></div>
            <div><span className="k">Last recorded heartbeat</span><span className="v mono">{formatTimestamp(liveMonitor.last_azure_heartbeat)}</span></div>
            <div><span className="k">Azure cache refreshed</span><span className="v mono">{formatTimestamp(liveMonitor.azure_cache_refreshed_at)}</span></div>
            <div><span className="k">Cache freshness</span><span className="v">{liveMonitor.heartbeat_fresh ? 'fresh Azure state' : `cached Azure state · ${formatDuration(liveMonitor.azure_cache_age_seconds)} old`}</span></div>
            <div><span className="k">Environment</span><span className="v mono">{liveMonitor.environment || '—'}</span></div>
            <div><span className="k">Downloadable logs</span><span className="v">{liveMonitor.logs_downloadable ? 'available' : 'not exposed by Azure yet'}</span></div>
          </div>
          <p className="dim">{liveMonitor.message}</p>
          {liveMonitor.heartbeat_error ? <div className="dim mono">{liveMonitor.heartbeat_error}</div> : null}
          {liveMonitor.studio_url ? (
            <a className="btn btn-ghost" href={liveMonitor.studio_url} target="_blank" rel="noreferrer">
              <ExternalLink size={15} /> Open Live Azure Studio Logs
            </a>
          ) : null}
          <div className="log-stream" style={{ marginTop: 16 }}>
            {liveMonitor.timeline.map((event, index) => (
              <div className="log-line" key={`${event.observed_at}:${index}`}>
                {event.observed_at} · Azure heartbeat · {event.status || 'unknown'} · {event.environment || '—'}
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {activeTab === 'history' ? (
        <section className="card">
          <div className="card-header"><div className="card-title">Test History</div><div className="card-meta">{state?.history.length || 0} cloud jobs</div></div>
          <div className="table-scroll">
            <div className="run-table test-prep-history-table">
              <div className="run-row run-head"><span>Phase</span><span>Status</span><span>Job</span><span>Compute</span><span>Created</span><span>Report</span></div>
              {state?.history.map((job) => (
                <div className="run-row" key={job.name}>
                  <span className="mono">{job.test_id} · {job.test_name}</span>
                  <span className={`status-pill ${statusClass(phaseStatus(job))}`}>{phaseStatus(job)}</span>
                  <span className="mono">{job.name}</span>
                  <span className="mono">{compactResource(job.compute)}</span>
                  <span>{formatTimestamp(job.created_at)}</span>
                  <button className="btn btn-ghost" type="button" disabled={Boolean(action)} onClick={() => loadReport(job.name)}>View</button>
                </div>
              ))}
              {!state?.history.length ? <div className="pool-empty-state">No CARA test-prep jobs have been submitted.</div> : null}
            </div>
          </div>
        </section>
      ) : null}

      {activeTab === 'audit' ? (
        <section className="card">
          <div className="card-header"><div className="card-title">Research Audit Trail</div><div className="card-meta">{state?.audit.length || 0} local events</div></div>
          <div className="log-stream">
            {state?.audit.length ? state.audit.map((event, index) => (
              <div className="log-line" key={`${event.timestamp_utc}:${index}`}>
                {event.timestamp_utc} · {event.action} · {event.test_id || '—'} · {event.job_name || event.environment || event.error || '—'}
              </div>
            )) : <div className="log-line">No dashboard-triggered Azure ML audit events yet.</div>}
          </div>
        </section>
      ) : null}

      {report ? (
        <section className="split-2">
          <div className="card">
            <div className="card-header">
              <div className="card-title">Report JSON</div>
              <div className="card-meta mono">{report.job_name}</div>
            </div>
            <pre className="test-prep-json">{JSON.stringify(report.report, null, 2)}</pre>
          </div>
          <div className="card">
            <div className="card-header">
              <div className="card-title">Run Metadata</div>
              <div className="card-meta mono">{report.report_path || 'report.json'}</div>
            </div>
            <div className="mini-list">
              <div><span className="k">Output artifact location</span><span className="v mono">{report.output_artifact_location || 'Azure ML managed output'}</span></div>
            </div>
            <pre className="test-prep-json">{JSON.stringify(report.metadata, null, 2)}</pre>
          </div>
        </section>
      ) : null}

      <section className="card">
        <div className="card-header"><div className="card-title">Controlled Research Input</div><div className="card-meta">shared source dataset</div></div>
        <div className="mini-list">
          <div><span className="k">Datastore</span><span className="v mono">ds_cara_raw_audio</span></div>
          <div><span className="k">Mounted root</span><span className="v mono">azureml://datastores/ds_cara_raw_audio/paths/test-audio/</span></div>
          <div><span className="k">Audio folder</span><span className="v mono">data/freesound/</span></div>
          <div><span className="k">Manifest</span><span className="v mono">data/freesound_meta/test-manifest/tracks.csv</span></div>
        </div>
        <div className="dim" style={{ marginTop: 12 }}>
          MusicGen and Stable Audio use the same source audio and manifest. Model-specific adapter scripts begin only after phases 01–04 pass.
        </div>
      </section>
    </div>
  );
};
