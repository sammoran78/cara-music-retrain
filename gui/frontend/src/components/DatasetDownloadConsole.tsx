import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Activity,
  CheckCircle2,
  CloudDownload,
  Database,
  FileAudio,
  FileCog,
  Gauge,
  Loader2,
  Pause,
  Play,
  TimerReset,
  Zap,
} from 'lucide-react';

interface DownloadProgressData {
  status: string;
  subset_mode: string;
  subset_role?: string | null;
  manifest_path: string;
  progress_path: string;
  output_dir: string;
  meta_dir: string;
  unavailable_log: string;
  total_requested: number;
  processed_count: number;
  remaining_count: number;
  percent_complete: number;
  downloaded_count: number;
  metadata_only_count: number;
  unavailable_count: number;
  audio_files_on_disk: number;
  metadata_files_on_disk: number;
  last_updated: number | null;
  last_batch_started_at?: number | null;
  last_batch_at?: number | null;
  last_error?: string | null;
  cached_metadata_hits: number;
  bulk_metadata_calls: number;
  single_metadata_fallback_calls: number;
  downloads_skipped_existing: number;
  skipped_non_confirmed: number;
  api_requests_used_today: number;
  job?: DownloadJobState;
  progress?: {
    active_batch?: ActiveBatch | null;
    activity_log?: ActivityLogEntry[];
  };
}

interface DownloadJobState {
  running?: boolean;
  requested_stop?: boolean;
  started_at?: number | null;
  finished_at?: number | null;
  completed_batches?: number;
  target_batches?: number;
  target_items?: number;
  last_message?: string | null;
}

interface ActivityLogEntry {
  ts?: number;
  level?: string;
  phase?: string;
  sound_id?: number;
  message?: string;
}

interface ActiveBatch {
  phase?: string;
  started_at?: number;
  updated_at?: number;
  requested?: number;
  total_in_batch?: number;
  completed_in_batch?: number;
  total_audio_downloads?: number;
  completed_audio_downloads?: number;
  active_downloads?: number;
  downloaded_bytes?: number;
  expected_bytes?: number;
  current_id?: number;
  active_ids?: number[];
  message?: string;
}

interface BatchSummary {
  subset_mode?: string;
  subset_role?: string;
  requested?: number;
  downloaded?: number;
  metadata_only?: number;
  unavailable?: number;
  cached_metadata_hits?: number;
  bulk_metadata_calls?: number;
  single_metadata_fallback_calls?: number;
  downloads_skipped_existing?: number;
  api_requests_used_today?: number;
  [key: string]: number | string | undefined;
}

interface NextResponse {
  status: 'batch_complete' | string;
  done: boolean;
  summary: BatchSummary;
  busy?: undefined;
}

interface BusyResponse {
  busy: true;
  status?: undefined;
  done?: undefined;
  summary?: undefined;
}

interface HistoryPoint {
  t: number;
  processed: number;
  pct: number;
}

const PROGRESS_POLL_MS = 2500;
const MAX_HISTORY_POINTS = 80;
const DEFAULT_BATCH_COUNT = 4;
const DEFAULT_TARGET_ITEMS = 2000;
const BUSY_RETRY_MS = 2000;
const NEXT_REQUEST_TIMEOUT_MS = 5 * 60 * 1000;

const numberFormat = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) return '–';
  return value.toLocaleString();
};

const relativeTime = (epochSeconds: number | null | undefined): string => {
  if (!epochSeconds) return '—';
  const diff = Date.now() / 1000 - epochSeconds;
  if (diff < 60) return `${Math.max(0, Math.floor(diff))}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return new Date(epochSeconds * 1000).toLocaleString();
};

const formatBytes = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) return '–';
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 * 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`;
};

const elapsedTime = (startSeconds: number | null | undefined): string => {
  if (!startSeconds) return '—';
  const elapsed = Math.max(0, Date.now() / 1000 - startSeconds);
  if (elapsed < 60) return `${Math.floor(elapsed)}s`;
  if (elapsed < 3600) return `${Math.floor(elapsed / 60)}m ${Math.floor(elapsed % 60)}s`;
  return `${Math.floor(elapsed / 3600)}h ${Math.floor((elapsed % 3600) / 60)}m`;
};

export const DatasetDownloadConsole: React.FC = () => {
  const [data, setData] = useState<DownloadProgressData | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [lastSummary, setLastSummary] = useState<BatchSummary | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  // controls
  const [subsetMode, setSubsetMode] = useState<string>('subset_role');
  const [subsetRole, setSubsetRole] = useState<string>('music_train_candidate');
  const [batchCount, setBatchCount] = useState<string>(String(DEFAULT_BATCH_COUNT));
  const [targetItems, setTargetItems] = useState<string>(String(DEFAULT_TARGET_ITEMS));
  const [skipAudio, setSkipAudio] = useState<boolean>(false);

  const [running, setRunning] = useState<boolean>(false);
  const [retryBusy, setRetryBusy] = useState<boolean>(false);

  const pollTimerRef = useRef<number | null>(null);
  const subsetModeRef = useRef<string>(subsetMode);
  const subsetRoleRef = useRef<string>(subsetRole);
  subsetModeRef.current = subsetMode;
  subsetRoleRef.current = subsetRole;

  const progressUrl = useMemo(() => {
    const params = new URLSearchParams();
    params.set('subset_mode', subsetMode);
    if (subsetMode === 'subset_role' && subsetRole.trim()) {
      params.set('subset_role', subsetRole.trim());
    }
    return `/api/data/download-progress?${params.toString()}`;
  }, [subsetMode, subsetRole]);

  const fetchProgress = useCallback(async () => {
    try {
      const res = await fetch(progressUrl);
      if (!res.ok) throw new Error(`progress HTTP ${res.status}`);
      const json: DownloadProgressData = await res.json();
      setData(json);
      setError(null);
      setHistory((prev) => {
        const next: HistoryPoint = {
          t: Date.now(),
          processed: json.processed_count,
          pct: json.percent_complete,
        };
        const last = prev[prev.length - 1];
        if (last && last.processed === next.processed && last.pct === next.pct) {
          return prev;
        }
        return [...prev, next].slice(-MAX_HISTORY_POINTS);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'unknown error');
    } finally {
      setLoading(false);
    }
  }, [progressUrl]);

  // poll progress continuously so stats are always current
  useEffect(() => {
    fetchProgress();
    if (pollTimerRef.current !== null) window.clearInterval(pollTimerRef.current);
    pollTimerRef.current = window.setInterval(fetchProgress, PROGRESS_POLL_MS);
    return () => {
      if (pollTimerRef.current !== null) window.clearInterval(pollTimerRef.current);
    };
  }, [fetchProgress]);

  useEffect(() => {
    const backendRunning = Boolean(data?.job?.running);
    setRunning(backendRunning);
    if (!backendRunning && data?.job?.finished_at && data?.job?.last_message) {
      setActionMessage(data.job.last_message);
    }
  }, [data?.job?.finished_at, data?.job?.last_message, data?.job?.running]);

  const parsedBatchCount = useMemo(
    () => Math.max(1, Math.min(25, Number(batchCount) || DEFAULT_BATCH_COUNT)),
    [batchCount],
  );

  const parsedTargetItems = useMemo(
    () => Math.max(1, Math.min(250000, Number(targetItems) || DEFAULT_TARGET_ITEMS)),
    [targetItems],
  );

  const computedRequestLoops = useMemo(
    () => Math.max(1, Math.ceil(parsedTargetItems / parsedBatchCount)),
    [parsedBatchCount, parsedTargetItems],
  );

  const effectiveRequestedItems = useMemo(
    () => parsedBatchCount * computedRequestLoops,
    [computedRequestLoops, parsedBatchCount],
  );

  const startJob = useCallback(async () => {
    const body = {
      subset_mode: subsetModeRef.current,
      subset_role: subsetModeRef.current === 'subset_role' ? subsetRoleRef.current.trim() : null,
      skip_audio: skipAudio,
      count: parsedBatchCount,
      target_items: parsedTargetItems,
    };
    const res = await fetch('/api/data/download/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json?.detail || `HTTP ${res.status}`);
    return json;
  }, [parsedBatchCount, parsedTargetItems, skipAudio]);

  const handleStart = useCallback(async () => {
    if (running) return;
    try {
      setActionMessage('Starting backend download job…');
      setRunning(true);
      await startJob();
      setActionMessage(null);
      fetchProgress();
    } catch (err) {
      setRunning(false);
      setActionMessage(err instanceof Error ? err.message : 'Failed to start download job');
    }
  }, [fetchProgress, running, startJob]);

  const handlePause = useCallback(async () => {
    try {
      const res = await fetch('/api/data/download/stop', { method: 'POST' });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.detail || `HTTP ${res.status}`);
      setActionMessage('Stop requested. The current backend batch will finish first.');
      fetchProgress();
    } catch (err) {
      setActionMessage(err instanceof Error ? err.message : 'Failed to stop download job');
    }
  }, [fetchProgress]);

  const handleRetryTemporaryUnavailable = useCallback(async () => {
    try {
      setRetryBusy(true);
      const res = await fetch('/api/data/download/retry-unavailable', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'temporary' }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.detail || `HTTP ${res.status}`);
      setActionMessage(
        json.removed_count > 0
          ? `Moved ${numberFormat(json.removed_count)} temporary unavailable IDs back to retryable.`
          : 'No temporary unavailable IDs were found to retry.',
      );
      fetchProgress();
    } catch (err) {
      setActionMessage(err instanceof Error ? err.message : 'Failed to retry unavailable IDs');
    } finally {
      setRetryBusy(false);
    }
  }, [fetchProgress]);

  const connection = useMemo(() => {
    if (error) return { dot: 'bad' as const, label: 'Offline' };
    if (loading) return { dot: 'warn' as const, label: 'Connecting' };
    return { dot: 'good' as const, label: 'Live' };
  }, [error, loading]);

  useEffect(() => {
    const dot = document.getElementById('connection-dot');
    const text = document.getElementById('connection-text');
    if (dot) {
      dot.classList.remove('warn', 'bad');
      if (connection.dot !== 'good') dot.classList.add(connection.dot);
    }
    if (text) text.textContent = connection.label;
  }, [connection]);

  const percent = data?.percent_complete ?? 0;
  const activeBatch = data?.progress?.active_batch ?? null;
  const batchTotal = activeBatch?.total_in_batch ?? activeBatch?.requested ?? 0;
  const batchCompleted = activeBatch?.completed_in_batch ?? 0;
  const batchPercent = batchTotal > 0 ? Math.min(100, (batchCompleted / batchTotal) * 100) : 0;
  const downloadedBytes = activeBatch?.downloaded_bytes ?? 0;
  const expectedBytes = activeBatch?.expected_bytes ?? 0;
  const bytePercent = expectedBytes > 0 ? Math.min(100, (downloadedBytes / expectedBytes) * 100) : 0;
  const activityLog = (data?.progress?.activity_log ?? []).slice(-30).reverse();
  const inFlightLabel = useMemo(() => {
    if (activeBatch) {
      const phase = activeBatch.phase || 'working';
      const updated = relativeTime(activeBatch.updated_at ?? null);
      if (activeBatch.message) return `backend: ${activeBatch.message} · updated ${updated}`;
      return `backend: ${phase} · updated ${updated}`;
    }
    if (running || data?.job?.last_message) {
      const completed = data?.job?.completed_batches ?? 0;
      const target = data?.job?.target_batches ?? computedRequestLoops;
      const message = running ? data?.job?.last_message || 'backend job running' : data?.job?.last_message || 'backend job idle';
      return `${message} · ${numberFormat(completed)} / ${numberFormat(target)} batches`;
    }
    return 'idle';
  }, [activeBatch, computedRequestLoops, data?.job, running]);

  const subsetLabel = useMemo(() => {
    if (subsetMode === 'confirmed_only') return 'Prefilter-confirmed rows';
    if (subsetMode === 'all_freesound') return 'All Freesound rows';
    if (subsetMode === 'subset_role') return subsetRole.trim() ? `Subset role · ${subsetRole.trim()}` : 'Subset role';
    return subsetMode;
  }, [subsetMode, subsetRole]);

  const statusChip = useMemo(() => {
    if (!data) return 'Idle';
    if (running) return 'Downloading';
    if (data.percent_complete >= 100 && data.total_requested > 0) return 'Complete';
    if (data.processed_count > 0) return 'Paused';
    return 'Ready';
  }, [data, running]);

  return (
    <div style={{ display: 'grid', gap: 20 }}>
      {/* HERO */}
      <section className="card" aria-label="Download overview">
        <div className="hero">
          <div className="hero-left">
            <div className="hero-kicker">Freesound · Music Attribution Pool</div>
            <h1 className="hero-title">
              A curated <em>25,000-item</em> music subset, downloaded on your terms.
            </h1>
            <p className="hero-sub">
              Each item is pulled by an explicit request from this page. Close the tab and the download halts.
              Return and it resumes exactly where disk state says you left off. The expansion beyond the original
              20,000 favors Acoustic/Folk, Jazz/Blues, World/Traditional, and Hip-Hop/Beats.
            </p>
            <div className="hero-stats">
              <span><strong>{numberFormat(data?.total_requested)}</strong> in subset</span>
              <span><strong>{numberFormat(data?.processed_count)}</strong> processed</span>
              <span><strong>{numberFormat(data?.remaining_count)}</strong> remaining</span>
              <span className="pill"><Activity size={13} /> {subsetLabel}</span>
            </div>
          </div>
          <div className="hero-right">
            <div
              className="ring"
              style={{ ['--progress' as any]: Math.max(0, Math.min(100, percent)) } as React.CSSProperties}
              role="img"
              aria-label={`Progress ${percent.toFixed(2)} percent`}
            >
              <div className="ring-inner">
                <div className="ring-pct">
                  {percent.toFixed(percent >= 10 ? 1 : 2)}
                  <span className="sym">%</span>
                </div>
                <div className="ring-label">{statusChip}</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* KPI */}
      <section className="kpi-grid" aria-label="Key metrics">
        <KPI
          label="Downloaded audio"
          value={data?.downloaded_count}
          icon={<FileAudio size={16} />}
          trend={`on disk · ${numberFormat(data?.audio_files_on_disk)}`}
        />
        <KPI
          label="Metadata only"
          value={data?.metadata_only_count}
          icon={<FileCog size={16} />}
          trend={`files · ${numberFormat(data?.metadata_files_on_disk)}`}
        />
        <KPI
          label="Unavailable"
          value={data?.unavailable_count}
          icon={<TimerReset size={16} />}
          trend="410 / 404 / throttled"
        />
        <KPI
          label="API used today"
          value={data?.api_requests_used_today}
          icon={<Zap size={16} />}
          trend="Freesound quota"
        />
        <KPI
          label="Cache hits"
          value={data?.cached_metadata_hits}
          icon={<Database size={16} />}
          trend={`bulk calls · ${numberFormat(data?.bulk_metadata_calls)}`}
        />
      </section>

      {/* CONTROLS + CHART */}
      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Launch Control</div>
            <div className="card-meta">
              session · <span className={running ? 'v-good' : 'dim'}>{running ? 'running' : 'idle'}</span>
            </div>
          </div>

          <div className="controls">
            <div className="control-row">
              <div className="field">
                <label>Subset source</label>
                <select
                  value={subsetMode}
                  onChange={(e) => setSubsetMode(e.target.value)}
                  disabled={running}
                >
                  <option value="subset_role">Official music subset</option>
                  <option value="confirmed_only">Prefilter-confirmed rows</option>
                  <option value="all_freesound">All Freesound rows</option>
                </select>
              </div>
              {subsetMode === 'subset_role' && (
                <div className="field">
                  <label>Manifest subset role</label>
                  <input
                    type="text"
                    value={subsetRole}
                    onChange={(e) => setSubsetRole(e.target.value)}
                    disabled={running}
                  />
                </div>
              )}
              <div className="field">
                <label>Items per request</label>
                <input
                  type="number"
                  min={1}
                  max={25}
                  value={batchCount}
                  onChange={(e) => setBatchCount(e.target.value)}
                  disabled={running}
                />
              </div>
              <div className="field">
                <label>Target items</label>
                <input
                  type="number"
                  min={1}
                  max={250000}
                  value={targetItems}
                  onChange={(e) => setTargetItems(e.target.value)}
                  disabled={running}
                />
              </div>
            </div>

            <div className="row" style={{ justifyContent: 'space-between' }}>
              <label className="toggle">
                <input
                  type="checkbox"
                  checked={skipAudio}
                  onChange={(e) => setSkipAudio(e.target.checked)}
                  disabled={running}
                />
                <span>Metadata only · skip audio</span>
              </label>

              {running ? (
                <button className="btn" onClick={handlePause}>
                  <Pause size={16} /> Pause
                </button>
              ) : (
                <button className="btn" onClick={handleStart}>
                  <Play size={16} /> Start / Resume Download
                </button>
              )}
            </div>

            <div className="row" style={{ justifyContent: 'flex-end' }}>
              <button
                className="btn"
                onClick={handleRetryTemporaryUnavailable}
                disabled={running || retryBusy || !data?.unavailable_count}
                title="Moves rate-limit, timeout, and network unavailable IDs back into the retry pool"
              >
                <TimerReset size={16} /> {retryBusy ? 'Preparing retry…' : 'Retry temporary unavailable'}
              </button>
            </div>

            <div className="bar" aria-hidden>
              <div className="bar-fill" style={{ width: `${Math.max(0, Math.min(100, percent))}%` }} />
            </div>

            <div className="kv">
              <div className="k">Active subset</div>
              <div className="v">{subsetLabel}</div>
            </div>
            <div className="kv">
              <div className="k">Planned run</div>
              <div className="v">
                target {numberFormat(parsedTargetItems)} items → {numberFormat(computedRequestLoops)} requests ×{' '}
                {numberFormat(parsedBatchCount)} items = up to{' '}
                {numberFormat(effectiveRequestedItems)} items
              </div>
            </div>
            <div className="kv">
              <div className="k">Last batch</div>
              <div className="v">{relativeTime(data?.last_batch_at ?? null)}</div>
            </div>
            <div className="kv">
              <div className="k">Batch started</div>
              <div className="v">{relativeTime(data?.last_batch_started_at ?? null)}</div>
            </div>
            <div className="kv">
              <div className="k">Current activity</div>
              <div className="v">{inFlightLabel}</div>
            </div>
            {lastSummary ? (
              <div className="kv">
                <div className="k">Last summary</div>
                <div className="v">
                  req {String(lastSummary.requested ?? 0)} · dl {String(lastSummary.downloaded ?? 0)} · meta{' '}
                  {String(lastSummary.metadata_only ?? 0)} · n/a {String(lastSummary.unavailable ?? 0)}
                </div>
              </div>
            ) : null}

            {actionMessage ? <div className="msg">{actionMessage}</div> : null}
            {data?.last_error ? (
              <div className="msg msg-err" style={{ whiteSpace: 'pre-wrap', maxHeight: 140, overflow: 'auto' }}>
                {data.last_error}
              </div>
            ) : null}
            {error ? <div className="msg msg-err">{error}</div> : null}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Live Throughput</div>
            <div className="card-meta">
              polling · {PROGRESS_POLL_MS / 1000}s
            </div>
          </div>
          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={history} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="pctGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#7c5cff" stopOpacity={0.7} />
                    <stop offset="100%" stopColor="#18d2ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="t" hide />
                <YAxis hide domain={[0, 100]} />
                <Tooltip
                  contentStyle={{
                    background: '#14171d',
                    border: '1px solid #242932',
                    borderRadius: 8,
                    fontSize: 12,
                    color: '#f3efe7',
                  }}
                  labelFormatter={(t: any) => new Date(Number(t)).toLocaleTimeString()}
                  formatter={(value: any, name: any) =>
                    name === 'pct'
                      ? [`${Number(value).toFixed(2)}%`, 'percent']
                      : [numberFormat(Number(value)), 'processed']
                  }
                />
                <Area
                  type="monotone"
                  dataKey="pct"
                  stroke="#7c5cff"
                  strokeWidth={2}
                  fill="url(#pctGrad)"
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="row" style={{ gap: 18, marginTop: 8 }}>
            <Stat icon={<Gauge size={14} />} label="remaining" value={numberFormat(data?.remaining_count)} />
            <Stat icon={<CloudDownload size={14} />} label="last update" value={relativeTime(data?.last_updated ?? null)} />
            <Stat icon={<CheckCircle2 size={14} />} label="complete" value={`${percent.toFixed(2)}%`} />
            {running ? (
              <Stat icon={<Loader2 size={14} className="spin" />} label="session" value="active" />
            ) : data?.job?.finished_at ? (
              <Stat icon={<CheckCircle2 size={14} />} label="session" value="finished" />
            ) : null}
          </div>
          {activeBatch ? (
            <div style={{ marginTop: 16 }}>
              <div className="kv">
                <div className="k">Active phase</div>
                <div className="v">{activeBatch.phase || 'working'} · {activeBatch.message || 'processing'}</div>
              </div>
              <div className="kv">
                <div className="k">Batch progress</div>
                <div className="v">
                  {numberFormat(batchCompleted)} / {numberFormat(batchTotal)} items · {batchPercent.toFixed(1)}%
                </div>
              </div>
              <div className="kv">
                <div className="k">Download bytes</div>
                <div className="v">
                  {formatBytes(downloadedBytes)} / {expectedBytes > 0 ? formatBytes(expectedBytes) : 'unknown'} ·{' '}
                  {expectedBytes > 0 ? `${bytePercent.toFixed(1)}%` : 'streaming'}
                </div>
              </div>
              <div className="kv">
                <div className="k">Active downloads</div>
                <div className="v">
                  {numberFormat(activeBatch.active_downloads ?? activeBatch.active_ids?.length ?? 0)}
                  {activeBatch.active_ids?.length ? ` · ${activeBatch.active_ids.slice(0, 6).join(', ')}` : ''}
                </div>
              </div>
              <div className="kv">
                <div className="k">Batch elapsed</div>
                <div className="v">
                  {elapsedTime(activeBatch.started_at)} · updated {relativeTime(activeBatch.updated_at ?? null)}
                </div>
              </div>
            </div>
          ) : null}
          <div style={{ marginTop: 16 }}>
            <div className="card-header" style={{ padding: 0, marginBottom: 8 }}>
              <div className="card-title">Live Activity Log</div>
              <div className="card-meta">latest {activityLog.length}</div>
            </div>
            <div
              className="paths"
              style={{
                maxHeight: 260,
                overflow: 'auto',
                border: '1px solid rgba(255,255,255,0.08)',
                borderRadius: 12,
                padding: 12,
              }}
            >
              {activityLog.length ? activityLog.map((entry, index) => (
                <div key={`${entry.ts ?? 0}-${index}`}>
                  <span className="dim">{relativeTime(entry.ts ?? null)}</span>
                  {entry.level && entry.level !== 'info' ? ` · ${entry.level}` : ''}
                  {entry.phase ? ` · ${entry.phase}` : ''}
                  {entry.sound_id ? ` · ${entry.sound_id}` : ''}
                  {' · '}
                  {entry.message || 'working'}
                </div>
              )) : (
                <div className="dim">No backend activity entries yet.</div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* EFFICIENCY + PATHS */}
      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Efficiency Counters</div>
            <div className="card-meta">last batch</div>
          </div>
          <div className="kv"><div className="k">Cached metadata hits</div><div className="v">{numberFormat(data?.cached_metadata_hits)}</div></div>
          <div className="kv"><div className="k">Bulk metadata calls</div><div className="v">{numberFormat(data?.bulk_metadata_calls)}</div></div>
          <div className="kv"><div className="k">Single fallback calls</div><div className="v">{numberFormat(data?.single_metadata_fallback_calls)}</div></div>
          <div className="kv"><div className="k">Existing audio skipped</div><div className="v">{numberFormat(data?.downloads_skipped_existing)}</div></div>
          <div className="kv"><div className="k">Skipped non-confirmed</div><div className="v">{numberFormat(data?.skipped_non_confirmed)}</div></div>
          <div className="kv"><div className="k">API used today</div><div className="v">{numberFormat(data?.api_requests_used_today)}</div></div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Tracked Paths</div>
            <div className="card-meta">relative to repo root</div>
          </div>
          <div className="paths">
            <div><span className="dim">manifest</span> · {data?.manifest_path}</div>
            <div><span className="dim">progress</span> · {data?.progress_path}</div>
            <div><span className="dim">audio</span> · {data?.output_dir}</div>
            <div><span className="dim">metadata</span> · {data?.meta_dir}</div>
            <div><span className="dim">unavailable</span> · {data?.unavailable_log}</div>
          </div>
        </div>
      </section>
    </div>
  );
};

const KPI: React.FC<{
  label: string;
  value: number | null | undefined;
  trend?: string;
  icon?: React.ReactNode;
}> = ({ label, value, trend, icon }) => (
  <div className="kpi">
    <div className="row" style={{ justifyContent: 'space-between' }}>
      <span className="kpi-label">{label}</span>
      <span className="dim" style={{ display: 'inline-flex' }}>{icon}</span>
    </div>
    <div className="kpi-value">{numberFormat(value)}</div>
    {trend ? <div className="kpi-trend">{trend}</div> : null}
  </div>
);

const Stat: React.FC<{ label: string; value: string; icon?: React.ReactNode }> = ({ label, value, icon }) => (
  <div className="row" style={{ gap: 6, fontSize: 12, color: 'var(--text-dim)' }}>
    <span style={{ display: 'inline-flex' }}>{icon}</span>
    <span>{label}</span>
    <span className="mono" style={{ color: 'var(--text)' }}>{value}</span>
  </div>
);

if (typeof document !== 'undefined' && !document.getElementById('cara-inline-anim')) {
  const style = document.createElement('style');
  style.id = 'cara-inline-anim';
  style.textContent = `.spin { animation: spin 1s linear infinite; } @keyframes spin { to { transform: rotate(360deg); } }`;
  document.head.appendChild(style);
}
