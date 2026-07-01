import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  Boxes,
  Cloud,
  Cpu,
  ExternalLink,
  Loader2,
  RefreshCw,
  Search,
  Square,
  Terminal,
} from 'lucide-react';
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { PageHeader } from './PageHeader';

interface AzureStatus {
  configured: boolean;
  sdk_installed: boolean;
  python_executable?: string;
  packages?: Record<string, boolean>;
  missing_packages?: string[];
  connected: boolean;
  missing_settings: string[];
  error?: string;
  settings: {
    workspace_name: string;
    resource_group: string;
    datastore_name: string;
    raw_audio_path: string;
  };
  workspace?: {
    name?: string;
    location?: string;
    description?: string;
  };
}

interface AzureCompute {
  name?: string | null;
  type?: string | null;
  size?: string | null;
  location?: string | null;
  provisioning_state?: string | null;
  min_instances?: number | null;
  max_instances?: number | null;
}

interface AzureEnvironment {
  name?: string | null;
  version?: string | null;
  description?: string | null;
  image?: string | null;
}

interface AzureJob {
  name?: string | null;
  display_name?: string | null;
  description?: string | null;
  status?: string | null;
  experiment_name?: string | null;
  compute?: string | null;
  environment?: string | null;
  created_at?: string | null;
  start_time?: string | null;
  end_time?: string | null;
  studio_url?: string | null;
  tags: Record<string, string>;
  properties: Record<string, string>;
}

interface MetricPoint {
  key: string;
  value: number;
  step: number;
  timestamp: number;
}

interface AzureMetrics {
  run_id: string;
  latest: Record<string, number>;
  histories: Record<string, MetricPoint[]>;
  params: Record<string, string>;
  tags: Record<string, string>;
}

interface AzureLogs {
  job_name: string;
  files: Array<{ path: string; content: string; truncated: boolean }>;
}

interface AzureJobProgress {
  job_name: string;
  checked_at: string;
  status?: string | null;
  method: string;
  label: string;
  completed?: number | null;
  total?: number | null;
  unit?: string | null;
  percent?: number | null;
  elapsed_seconds?: number | null;
  estimated_remaining_seconds?: number | null;
  latest_observed_at?: string | null;
  note?: string | null;
  error?: string | null;
}

interface AzureJobProgressResponse {
  checked_at: string;
  progress: Record<string, AzureJobProgress>;
}

const fetchJson = async <T,>(url: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(url, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body?.detail || `${url} HTTP ${response.status}`);
  return body as T;
};

const statusClass = (status?: string | null): string => {
  const normalized = String(status || '').toLowerCase();
  if (['completed', 'succeeded', 'finished'].includes(normalized)) return 'status-done';
  if (['failed', 'canceled', 'cancelled'].includes(normalized)) return 'status-error';
  if (['running', 'starting', 'preparing', 'provisioning'].includes(normalized)) return 'status-running';
  return 'status-queued';
};

const isActiveStatus = (status?: string | null): boolean =>
  ['running', 'starting', 'preparing', 'provisioning', 'queued', 'notstarted'].includes(
    String(status || '').toLowerCase(),
  );

const formatTimestamp = (value?: string | null): string => {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
};

const parseTimestampMs = (value?: string | null): number | null => {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed;
};

const formatDuration = (seconds?: number | null): string => {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '—';
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
};

const elapsedSecondsForJob = (job?: AzureJob | null): number | null => {
  if (!job) return null;
  const start = parseTimestampMs(job.start_time) ?? parseTimestampMs(job.created_at);
  if (!start) return null;
  const end = parseTimestampMs(job.end_time) ?? Date.now();
  return Math.max(0, (end - start) / 1000);
};

const numericParam = (metrics: AzureMetrics | null, keys: string[]): number | null => {
  if (!metrics) return null;
  for (const key of keys) {
    const value = metrics.params?.[key] ?? metrics.tags?.[key];
    if (value !== undefined) {
      const parsed = Number(value);
      if (Number.isFinite(parsed) && parsed > 0) return parsed;
    }
  }
  return null;
};

const compactResource = (value?: string | null): string => {
  if (!value) return '—';
  const parts = value.split('/').filter(Boolean);
  return parts[parts.length - 1] || value;
};

const formatProgressPercent = (progress?: AzureJobProgress | null): string => (
  progress?.percent !== null && progress?.percent !== undefined ? `${progress.percent.toFixed(1)}%` : '—'
);

const progressCountLabel = (progress?: AzureJobProgress | null): string => {
  if (!progress || progress.completed === null || progress.completed === undefined || progress.total === null || progress.total === undefined) return '—';
  return `${Math.round(progress.completed).toLocaleString()} / ${Math.round(progress.total).toLocaleString()} ${progress.unit || ''}`.trim();
};

export const AzureRunsPage: React.FC = () => {
  const [status, setStatus] = useState<AzureStatus | null>(null);
  const [computes, setComputes] = useState<AzureCompute[]>([]);
  const [environments, setEnvironments] = useState<AzureEnvironment[]>([]);
  const [jobs, setJobs] = useState<AzureJob[]>([]);
  const [selectedName, setSelectedName] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<AzureJob | null>(null);
  const [metrics, setMetrics] = useState<AzureMetrics | null>(null);
  const [logs, setLogs] = useState<AzureLogs | null>(null);
  const [progressByJob, setProgressByJob] = useState<Record<string, AzureJobProgress>>({});
  const [progressCheckedAt, setProgressCheckedAt] = useState<string | null>(null);
  const [metricName, setMetricName] = useState('');
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [loading, setLoading] = useState(true);
  const [logsLoading, setLogsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const nextStatus = await fetchJson<AzureStatus>('/api/azureml/status');
      setStatus(nextStatus);
      if (!nextStatus.connected) {
        setComputes([]);
        setEnvironments([]);
        setJobs([]);
        setError(nextStatus.error || null);
        return;
      }
      const [nextComputes, nextEnvironments, nextJobs] = await Promise.all([
        fetchJson<AzureCompute[]>('/api/azureml/computes'),
        fetchJson<AzureEnvironment[]>('/api/azureml/environments?limit=200'),
        fetchJson<AzureJob[]>('/api/azureml/jobs?limit=200'),
      ]);
      setComputes(nextComputes);
      setEnvironments(nextEnvironments);
      setJobs(nextJobs);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh Azure ML workspace');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = window.setInterval(refresh, 15000);
    return () => window.clearInterval(interval);
  }, [refresh]);

  const refreshProgress = useCallback(async () => {
    if (!status?.connected) return;
    try {
      const payload = await fetchJson<AzureJobProgressResponse>('/api/azureml/job-progress?limit=200&active_only=true');
      setProgressByJob((current) => ({ ...current, ...payload.progress }));
      setProgressCheckedAt(payload.checked_at);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh Azure ML progress');
    }
  }, [status?.connected]);

  useEffect(() => {
    refreshProgress();
    const interval = window.setInterval(refreshProgress, 60000);
    return () => window.clearInterval(interval);
  }, [refreshProgress]);

  useEffect(() => {
    if (!selectedName || !status?.connected) {
      setSelectedJob(null);
      setMetrics(null);
      return;
    }
    let cancelled = false;
    Promise.all([
      fetchJson<AzureJob>(`/api/azureml/jobs/${encodeURIComponent(selectedName)}`),
      fetchJson<AzureMetrics>(`/api/azureml/jobs/${encodeURIComponent(selectedName)}/metrics`).catch(() => null),
      fetchJson<AzureJobProgress>(`/api/azureml/job-progress/${encodeURIComponent(selectedName)}`).catch(() => null),
    ])
      .then(([job, nextMetrics, nextProgress]) => {
        if (cancelled) return;
        setSelectedJob(job);
        setMetrics(nextMetrics);
        if (nextProgress) {
          setProgressByJob((current) => ({ ...current, [selectedName]: nextProgress }));
          setProgressCheckedAt(nextProgress.checked_at);
        }
        const keys = Object.keys(nextMetrics?.histories || {});
        setMetricName((current) => (current && keys.includes(current) ? current : keys[0] || ''));
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load Azure ML job');
      });
    return () => {
      cancelled = true;
    };
  }, [selectedName, status?.connected]);

  const visibleJobs = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return jobs.filter((job) => {
      const matchesStatus = statusFilter === 'all' || String(job.status || '').toLowerCase() === statusFilter;
      if (!matchesStatus) return false;
      if (!needle) return true;
      return [job.name, job.display_name, job.experiment_name, job.compute, job.environment]
        .some((value) => String(value || '').toLowerCase().includes(needle));
    });
  }, [jobs, search, statusFilter]);

  const metricKeys = Object.keys(metrics?.histories || {});
  const chartData = (metrics?.histories?.[metricName] || []).map((point) => ({
    ...point,
    label: point.step || point.timestamp,
  }));
  const elapsedSeconds = elapsedSecondsForJob(selectedJob);
  const maxObservedStep = Math.max(
    0,
    ...Object.values(metrics?.histories || {}).flat().map((point) => Number(point.step) || 0),
  );
  const maxConfiguredSteps = numericParam(metrics, ['max_steps', 'max_train_steps', 'trainer.max_steps', 'steps']);
  const estimatedRemainingSeconds = (
    elapsedSeconds !== null
    && maxConfiguredSteps !== null
    && maxObservedStep > 0
    && maxObservedStep < maxConfiguredSteps
  )
    ? elapsedSeconds * ((maxConfiguredSteps - maxObservedStep) / maxObservedStep)
    : null;
  const activeRuns = jobs.filter((job) => isActiveStatus(job.status)).length;
  const selectedProgress = selectedName ? progressByJob[selectedName] : null;

  const loadLogs = useCallback(async () => {
    if (!selectedName) return;
    try {
      setLogsLoading(true);
      setLogs(await fetchJson<AzureLogs>(`/api/azureml/jobs/${encodeURIComponent(selectedName)}/logs`));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download Azure ML logs');
    } finally {
      setLogsLoading(false);
    }
  }, [selectedName]);

  const cancelJob = useCallback(async () => {
    if (!selectedName || !window.confirm(`Hard stop Azure ML job ${selectedName}? This sends a cloud cancellation request and may interrupt checkpoint writing.`)) return;
    try {
      await fetchJson(`/api/azureml/jobs/${encodeURIComponent(selectedName)}/cancel`, { method: 'POST' });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to cancel Azure ML job');
    }
  }, [refresh, selectedName]);

  return (
    <div style={{ display: 'grid', gap: 20 }}>
      <PageHeader
        kicker="Operations · Azure Machine Learning"
        title={
          <>
            Monitor cloud <em>training runs</em>
          </>
        }
        description={
          <>
            Read workspace jobs, compute clusters, registered environments, MLflow metrics, and downloadable logs from Azure ML.
            Closing this dashboard does not stop cloud jobs; use the hard stop action on a selected active job when cancellation is intentional.
          </>
        }
        actions={
          <button className="btn btn-ghost" onClick={refresh} type="button" disabled={loading}>
            <RefreshCw size={16} className={loading ? 'spin' : ''} /> Refresh
          </button>
        }
      />

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="pool-summary-grid" aria-label="Azure ML overview">
        <div className={`pool-metric-card tone-${status?.connected ? 'good' : 'warn'}`}>
          <div className="pool-metric-top"><Cloud size={16} /> Workspace</div>
          <div className="pool-metric-value" style={{ fontSize: 22 }}>{status?.workspace?.name || status?.settings.workspace_name || 'Setup needed'}</div>
          <div className="pool-metric-meta">{status?.connected ? status?.workspace?.location || 'connected' : 'not connected'}</div>
        </div>
        <div className="pool-metric-card">
          <div className="pool-metric-top"><Activity size={16} /> Active jobs</div>
          <div className="pool-metric-value">{activeRuns.toLocaleString()}</div>
          <div className="pool-metric-meta">{jobs.length.toLocaleString()} recent workspace jobs</div>
        </div>
        <div className="pool-metric-card">
          <div className="pool-metric-top"><Cpu size={16} /> Compute clusters</div>
          <div className="pool-metric-value">{computes.length.toLocaleString()}</div>
          <div className="pool-metric-meta">workspace compute targets</div>
        </div>
        <div className="pool-metric-card">
          <div className="pool-metric-top"><Boxes size={16} /> Environments</div>
          <div className="pool-metric-value">{environments.length.toLocaleString()}</div>
          <div className="pool-metric-meta">registered versions loaded</div>
        </div>
      </section>

      {!status?.connected ? (
        <section className="card">
          <div className="card-header">
            <div className="card-title">Azure ML Connection Setup</div>
            <div className="card-meta">{loading ? 'checking local setup' : 'action required'}</div>
          </div>
          <div className="azure-setup-grid">
            <div className="check-list">
              <div className={`check-item${status?.sdk_installed ? ' is-checked' : ''}`}>
                <span className="check-icon">{status?.sdk_installed ? '✓' : '1'}</span>
                <span>Install Azure ML SDK v2 and MLflow packages</span>
                <span className="mono dim">{status?.sdk_installed ? 'ready' : status?.missing_packages?.join(', ') || 'missing'}</span>
              </div>
              <div className={`check-item${status?.configured ? ' is-checked' : ''}`}>
                <span className="check-icon">{status?.configured ? '✓' : '2'}</span>
                <span>Set workspace identifiers in <span className="mono">.env</span></span>
                <span className="mono dim">{status?.configured ? 'ready' : status?.missing_settings.join(', ') || 'missing'}</span>
              </div>
              <div className={`check-item${status?.connected ? ' is-checked' : ''}`}>
                <span className="check-icon">{status?.connected ? '✓' : '3'}</span>
                <span>Authenticate Azure CLI and refresh this page</span>
                <span className="mono dim">{status?.connected ? 'connected' : 'waiting'}</span>
              </div>
            </div>
            <div className="log-stream" style={{ height: 'auto', minHeight: 160 }}>
              <div className="log-line">az login --scope https://management.core.windows.net//.default</div>
              <div className="log-line">az extension add --name ml</div>
              <div className="log-line">python3 -m venv .venv-dashboard</div>
              <div className="log-line">.venv-dashboard/bin/python -m pip install -r gui/backend/requirements.txt</div>
              <div className="log-line">[ -e .env ] || cp .env.example .env</div>
              <div className="log-line"># Preserve an existing .env. Add only missing AZUREML_* values.</div>
              <div className="log-line"># Backend Python: {status?.python_executable || 'restart dashboard after install'}</div>
            </div>
          </div>
        </section>
      ) : (
        <>
          <section className="card">
            <div className="card-header">
              <div className="card-title">Workspace Jobs</div>
              <div className="card-meta">
                {visibleJobs.length.toLocaleString()} visible · jobs every 15s · progress every 60s{progressCheckedAt ? ` · ${new Date(progressCheckedAt).toLocaleTimeString()}` : ''}
              </div>
            </div>
            <div className="azure-toolbar">
              <label className="search-field" style={{ marginBottom: 0 }}>
                <Search size={16} />
                <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search jobs, experiments, compute, environments" />
              </label>
              <select value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
                <option value="all">All statuses</option>
                <option value="running">Running</option>
                <option value="queued">Queued</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
                <option value="canceled">Canceled</option>
              </select>
            </div>
            <div className="table-scroll azure-jobs-scroll">
              <div className="run-table azure-jobs-table">
                <div className="run-row run-head">
                  <span>Job</span><span>Status</span><span>Progress</span><span>Experiment</span><span>Compute</span><span>Environment</span><span>Created</span>
                </div>
                {visibleJobs.map((job) => {
                  const jobProgress = job.name ? progressByJob[job.name] : null;
                  const percent = Math.min(100, Math.max(0, jobProgress?.percent ?? 0));
                  return (
                    <button
                      className={`run-row azure-job-row${selectedName === job.name ? ' is-selected' : ''}`}
                      key={job.name || job.display_name}
                      type="button"
                      onClick={() => {
                        setSelectedName(job.name || null);
                        setLogs(null);
                      }}
                    >
                      <span className="mono">{job.display_name || job.name || '—'}</span>
                      <span className={`status-pill ${statusClass(job.status)}`}>{job.status || 'unknown'}</span>
                      <span className="azure-progress-cell">
                        <span className="mono">{formatProgressPercent(jobProgress)}</span>
                        <span className="bar azure-progress-bar" aria-label={`Progress ${formatProgressPercent(jobProgress)}`}>
                          <span className="bar-fill" style={{ width: `${percent}%` }} />
                        </span>
                      </span>
                      <span>{job.experiment_name || '—'}</span>
                      <span className="mono">{compactResource(job.compute)}</span>
                      <span className="mono">{compactResource(job.environment)}</span>
                      <span className="dim">{formatTimestamp(job.created_at)}</span>
                    </button>
                  );
                })}
                {!visibleJobs.length ? <div className="pool-empty-state">No Azure ML jobs match this view.</div> : null}
              </div>
            </div>
          </section>

          {selectedJob ? (
            <section className="split-2 azure-detail-grid">
              <div className="card">
                <div className="card-header">
                  <div>
                    <div className="card-title">{selectedJob.display_name || selectedJob.name}</div>
                    <div className="card-meta mono">{selectedJob.name}</div>
                  </div>
                  <span className={`status-pill ${statusClass(selectedJob.status)}`}>{selectedJob.status || 'unknown'}</span>
                </div>
                <div className="mini-list">
                  <div><span className="k">Experiment</span><span className="v">{selectedJob.experiment_name || '—'}</span></div>
                  <div><span className="k">Compute</span><span className="v mono">{compactResource(selectedJob.compute)}</span></div>
                  <div><span className="k">Environment</span><span className="v mono">{compactResource(selectedJob.environment)}</span></div>
                  <div><span className="k">Started</span><span className="v">{formatTimestamp(selectedJob.start_time)}</span></div>
                  <div><span className="k">Finished</span><span className="v">{formatTimestamp(selectedJob.end_time)}</span></div>
                  <div><span className="k">Elapsed</span><span className="v">{formatDuration(elapsedSeconds)}</span></div>
                  <div><span className="k">ETA</span><span className="v">{formatDuration(selectedProgress?.estimated_remaining_seconds ?? estimatedRemainingSeconds)}</span></div>
                  <div><span className="k">Step</span><span className="v">{maxObservedStep || '—'}{maxConfiguredSteps ? ` / ${maxConfiguredSteps}` : ''}</span></div>
                  <div><span className="k">Progress</span><span className="v">{formatProgressPercent(selectedProgress)} · {progressCountLabel(selectedProgress)}</span></div>
                </div>
                {selectedProgress ? (
                  <div className="azure-progress-detail">
                    <div className="row" style={{ justifyContent: 'space-between' }}>
                      <span className="dim">{selectedProgress.label}</span>
                      <span className="mono">{formatProgressPercent(selectedProgress)}</span>
                    </div>
                    <div className="bar">
                      <div className="bar-fill" style={{ width: `${Math.min(100, Math.max(0, selectedProgress.percent ?? 0))}%` }} />
                    </div>
                    <div className="mini-list" style={{ marginTop: 10 }}>
                      <div><span className="k">Observed</span><span className="v mono">{progressCountLabel(selectedProgress)}</span></div>
                      <div><span className="k">Last update</span><span className="v mono">{formatTimestamp(selectedProgress.latest_observed_at)}</span></div>
                      <div><span className="k">Checked</span><span className="v mono">{formatTimestamp(selectedProgress.checked_at)}</span></div>
                    </div>
                    {selectedProgress.error ? <div className="error-banner" style={{ marginTop: 10 }}>{selectedProgress.error}</div> : null}
                    <div className="callout" style={{ marginTop: 10 }}>{selectedProgress.note}</div>
                  </div>
                ) : null}
                <div className="row" style={{ gap: 10, marginTop: 16, flexWrap: 'wrap' }}>
                  {selectedJob.studio_url ? (
                    <a className="btn btn-ghost" href={selectedJob.studio_url} target="_blank" rel="noreferrer">
                      <ExternalLink size={15} /> Open in Studio
                    </a>
                  ) : null}
                  <button className="btn btn-ghost" type="button" onClick={loadLogs} disabled={logsLoading}>
                    {logsLoading ? <Loader2 size={15} className="spin" /> : <Terminal size={15} />} Download logs
                  </button>
                  {isActiveStatus(selectedJob.status) ? (
                    <button className="btn" type="button" onClick={cancelJob}>
                      <Square size={14} /> Hard stop job
                    </button>
                  ) : null}
                </div>
              </div>

              <div className="card">
                <div className="card-header">
                  <div className="card-title">MLflow Metrics</div>
                  <div className="card-meta">{metricKeys.length ? `${metricKeys.length} tracked metrics` : 'no MLflow metrics logged yet'}</div>
                </div>
                {metricKeys.length ? (
                  <>
                    <select value={metricName} onChange={(event) => setMetricName(event.target.value)}>
                      {metricKeys.map((key) => <option key={key}>{key}</option>)}
                    </select>
                    <div className="chart-wrap" style={{ height: 190, marginTop: 12 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData}>
                          <defs>
                            <linearGradient id="azureMetricGradient" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="0%" stopColor="#7c5cff" stopOpacity={0.7} />
                              <stop offset="100%" stopColor="#18d2ff" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <XAxis dataKey="step" tick={{ fontSize: 11 }} />
                          <YAxis tick={{ fontSize: 11 }} />
                          <Tooltip />
                          <Area type="monotone" dataKey="value" stroke="#7c5cff" fill="url(#azureMetricGradient)" isAnimationActive={false} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </>
                ) : <div className="pool-empty-state">The selected job has no MLflow metric history available.</div>}
              </div>
            </section>
          ) : null}

          {logs ? (
            <section className="card">
              <div className="card-header">
                <div className="card-title">Downloaded Logs</div>
                <div className="card-meta">{logs.files.length} text files</div>
              </div>
              <div className="log-stream">
                {logs.files.length ? logs.files.map((file) => (
                  <div key={file.path}>
                    <div className="log-line" style={{ color: '#fcd34d' }}>--- {file.path}{file.truncated ? ' (truncated)' : ''} ---</div>
                    <div className="log-line">{file.content}</div>
                  </div>
                )) : <div className="log-line">No downloadable text logs were returned.</div>}
              </div>
            </section>
          ) : null}

          <section className="split-2 azure-inventory-grid">
            <div className="card">
              <div className="card-header"><div className="card-title">Compute Inventory</div><div className="card-meta">{computes.length} targets</div></div>
              <div className="mini-list">
                {computes.map((compute) => (
                  <div key={compute.name}>
                    <span className="v mono">{compute.name}</span>
                    <span className="k">{compute.size || compute.type || '—'} · {compute.provisioning_state || 'unknown'} · {compute.min_instances ?? '—'}-{compute.max_instances ?? '—'} nodes</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="card">
              <div className="card-header"><div className="card-title">Environment Inventory</div><div className="card-meta">{environments.length} versions</div></div>
              <div className="mini-list">
                {environments.map((environment) => (
                  <div key={`${environment.name}:${environment.version}`}>
                    <span className="v mono">{environment.name}:{environment.version || '—'}</span>
                    <span className="k">{environment.description || environment.image || 'registered Azure ML environment'}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
};
