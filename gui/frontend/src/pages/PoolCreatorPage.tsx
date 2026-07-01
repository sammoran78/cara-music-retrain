import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Files,
  FolderPlus,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  ShieldAlert,
  Workflow,
} from 'lucide-react';
import { PageHeader } from './PageHeader';

interface PoolAllocationSummary {
  engine_version?: string;
  allocation_engine_version?: string;
  counts: Record<string, number>;
  pool_count: number;
  asset_count: number;
  candidate_asset_count?: number;
  assignment_count: number;
  duplicate_count: number;
  review_count: number;
  completed_download_ids?: number;
  downloaded_audio_files_on_disk?: number;
  manifest_requires_reconcile?: boolean;
  manifest_paths: {
    source_manifest_path: string;
    cara_pool_manifest_path: string;
    cara_pool_manifest_csv_path: string;
  };
  latest_run?: {
    run_id?: string;
    finished_at?: string | null;
    processed_assets?: number;
  } | null;
  rules: {
    max_pool_duration_seconds: number;
    max_artist_duration_seconds: number;
    repair_threshold: number;
    min_pool_code_edit_distance: number;
    target_pool_count?: number;
    artist_exception_min_duration_seconds?: number;
  };
}

interface PoolAllocationJobState {
  running?: boolean;
  requested_stop?: boolean;
  started_at?: number | null;
  finished_at?: number | null;
  processed_assets?: number;
  total_assets?: number;
  percent_complete?: number;
  current_phase?: string | null;
  current_asset?: string | null;
  current_asset_title?: string | null;
  current_pool_id?: string | null;
  counts?: Record<string, number>;
  last_message?: string | null;
  last_error?: string | null;
  latest_run_id?: string | null;
  options?: {
    subset_role?: string | null;
    only_downloaded?: boolean;
    limit?: number | null;
    allow_relaxed_metadata?: boolean;
    start_fresh?: boolean;
  } | null;
}

interface RunStatusResponse {
  job: PoolAllocationJobState;
  progress: PoolAllocationProgress;
  summary: PoolAllocationSummary;
}

interface ProgressLogEntry {
  ts?: string;
  phase?: string;
  level?: string;
  asset_id?: string | null;
  source_key?: string | null;
  pool_id?: string | null;
  message?: string;
}

interface PoolAllocationProgress {
  status: string;
  run_id?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  updated_at?: string | null;
  current_phase?: string | null;
  current_asset?: string | null;
  current_asset_title?: string | null;
  current_pool_id?: string | null;
  processed_assets: number;
  total_assets: number;
  percent_complete: number;
  counts: Record<string, number>;
  options?: {
    subset_role?: string | null;
    only_downloaded?: boolean;
    limit?: number | null;
    allow_relaxed_metadata?: boolean;
    start_fresh?: boolean;
  } | null;
  activity_log: ProgressLogEntry[];
}

interface PoolRow {
  pool_id: string;
  licence_class: string;
  territory: string;
  record_label?: string | null;
  rights_holder_group?: string | null;
  primary_genre: string;
  pool_family?: string | null;
  pool_type?: string | null;
  spillover_index?: number | null;
  included_primary_genres?: string[];
  asset_count: number;
  current_duration_seconds: number;
  remaining_capacity_seconds: number;
  top_artist_share: number;
  style_summary?: string | null;
  updated_at?: string | null;
}

interface AssignmentRow {
  assignment_id: string;
  asset_id: string;
  pool_id?: string | null;
  assignment_status: string;
  reason_codes: string[];
  review_required: boolean;
  assigned_at?: string | null;
  pool_was_created: boolean;
}

interface ReviewRow {
  assignment_id: string;
  asset_id: string;
  pool_id?: string | null;
  assignment_status: string;
  reason_codes: string[];
  assigned_at?: string | null;
}

const POLL_MS = 3000;

const numberFormat = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) return '–';
  return value.toLocaleString();
};

const percentFormat = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) return '–';
  return `${(value * 100).toFixed(1)}%`;
};

const formatDuration = (seconds: number | null | undefined): string => {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '–';
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return `${hours}h ${minutes}m`;
};

const formatTimestamp = (value: string | null | undefined): string => {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString();
};

const statusClass = (status: string): string => {
  if (status === 'assigned' || status === 'new_pool_created') return 'status-done';
  if (status === 'duplicate_found') return 'status-queued';
  if (status === 'review_required' || status === 'rejected') return 'status-error';
  return 'status-running';
};

const MetricCard: React.FC<{
  label: string;
  value: number | string;
  icon: React.ReactNode;
  tone?: 'good' | 'warn' | 'bad' | 'neutral';
  meta?: string;
}> = ({ label, value, icon, tone = 'neutral', meta }) => (
  <div className={`pool-metric-card tone-${tone}`}>
    <div className="pool-metric-top">
      <span className="pool-metric-icon">{icon}</span>
      <span className="pool-metric-label">{label}</span>
    </div>
    <div className="pool-metric-value">{value}</div>
    {meta ? <div className="pool-metric-meta">{meta}</div> : null}
  </div>
);

const EmptyState: React.FC<{ message: string }> = ({ message }) => (
  <div className="pool-empty-state">{message}</div>
);

export const PoolCreatorPage: React.FC = () => {
  const [summary, setSummary] = useState<PoolAllocationSummary | null>(null);
  const [job, setJob] = useState<PoolAllocationJobState | null>(null);
  const [progress, setProgress] = useState<PoolAllocationProgress | null>(null);
  const [pools, setPools] = useState<PoolRow[]>([]);
  const [assignments, setAssignments] = useState<AssignmentRow[]>([]);
  const [reviewQueue, setReviewQueue] = useState<ReviewRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [subsetRole, setSubsetRole] = useState<string>('music_train_candidate');
  const [onlyDownloaded, setOnlyDownloaded] = useState<boolean>(true);
  const [allowRelaxedMetadata, setAllowRelaxedMetadata] = useState<boolean>(true);
  const [engineVersion, setEngineVersion] = useState<'v1' | 'v2'>('v2');
  const [limit, setLimit] = useState<string>('');
  const pollRef = useRef<number | null>(null);

  const fetchAll = useCallback(async () => {
    try {
      const engineParam = `engine=${encodeURIComponent(engineVersion)}`;
      const [statusRes, poolsRes, assignmentsRes, reviewRes] = await Promise.all([
        fetch(`/api/data/pool-allocation/run-status?${engineParam}`),
        fetch(`/api/data/pool-allocation/pools?${engineParam}`),
        fetch(`/api/data/pool-allocation/assignments?limit=200&${engineParam}`),
        fetch(`/api/data/pool-allocation/review-queue?limit=200&${engineParam}`),
      ]);
      if (!statusRes.ok) throw new Error(`run-status HTTP ${statusRes.status}`);
      if (!poolsRes.ok) throw new Error(`pools HTTP ${poolsRes.status}`);
      if (!assignmentsRes.ok) throw new Error(`assignments HTTP ${assignmentsRes.status}`);
      if (!reviewRes.ok) throw new Error(`review-queue HTTP ${reviewRes.status}`);

      const statusJson: RunStatusResponse = await statusRes.json();
      const poolsJson: PoolRow[] = await poolsRes.json();
      const assignmentsJson: AssignmentRow[] = await assignmentsRes.json();
      const reviewJson: ReviewRow[] = await reviewRes.json();

      setSummary(statusJson.summary);
      setJob(statusJson.job);
      setProgress(statusJson.progress);
      setPools(poolsJson);
      setAssignments(assignmentsJson);
      setReviewQueue(reviewJson);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load pool allocator data');
    } finally {
      setLoading(false);
    }
  }, [engineVersion]);

  useEffect(() => {
    fetchAll();
    if (pollRef.current !== null) window.clearInterval(pollRef.current);
    pollRef.current = window.setInterval(fetchAll, POLL_MS);
    return () => {
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
    };
  }, [fetchAll]);

  const handleRun = useCallback(async (startFresh = false) => {
    try {
      setActionMessage(startFresh ? 'Starting fresh pool allocation run…' : 'Starting pool allocation run…');
      const body = {
        engine_version: engineVersion,
        subset_role: subsetRole.trim() || null,
        only_downloaded: onlyDownloaded,
        allow_relaxed_metadata: allowRelaxedMetadata,
        limit: limit.trim() ? Number(limit) : null,
        start_fresh: engineVersion === 'v2' ? true : startFresh,
      };
      const res = await fetch('/api/data/pool-allocation/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.detail || `HTTP ${res.status}`);
      setActionMessage(startFresh ? 'Fresh pool allocation run started.' : 'Pool allocation run started.');
      fetchAll();
    } catch (err) {
      setActionMessage(err instanceof Error ? err.message : 'Failed to start pool allocation run');
    }
  }, [allowRelaxedMetadata, engineVersion, fetchAll, limit, onlyDownloaded, subsetRole]);

  const handlePause = useCallback(async () => {
    try {
      const res = await fetch('/api/data/pool-allocation/stop', { method: 'POST' });
      const json = await res.json();
      if (!res.ok) throw new Error(json?.detail || `HTTP ${res.status}`);
      setActionMessage('Pause requested. The allocator will stop after the current asset.');
      fetchAll();
    } catch (err) {
      setActionMessage(err instanceof Error ? err.message : 'Failed to pause allocator');
    }
  }, [fetchAll]);

  const counts = summary?.counts ?? {};
  const isRunning = Boolean(job?.running);
  const candidateAssetCount = summary?.candidate_asset_count ?? summary?.asset_count ?? 0;
  const registryAssetCount = summary?.asset_count ?? 0;
  const pendingCandidateCount = Math.max(candidateAssetCount - registryAssetCount, 0);
  const totalOutcomes = useMemo(
    () =>
      ['assigned', 'new_pool_created', 'duplicate_found', 'review_required', 'rejected', 'unresolved'].reduce(
        (sum, key) => sum + (counts[key] ?? 0),
        0,
      ),
    [counts],
  );

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

  const canResume = !isRunning && progress?.status === 'paused';
  const statusLabel = isRunning
    ? 'Running'
    : canResume
      ? 'Paused run'
      : pendingCandidateCount > 0
        ? 'Needs resume'
        : summary?.latest_run
          ? 'Ready'
          : 'Not started';
  const rules = summary?.rules;
  const progressPercent = Math.max(0, Math.min(100, progress?.percent_complete ?? job?.percent_complete ?? 0));
  const activityLog = (progress?.activity_log ?? []).slice(-40).reverse();

  return (
    <div style={{ display: 'grid', gap: 20 }}>
      <PageHeader
        kicker="Data · Pool Creator"
        title={
          <>
            Build auditable <em>CaRA source pools</em> before training
          </>
        }
        description={
          <>
            Allocate licensed source audio into persistent non-semantic pool IDs with duplicate
            protection, capacity limits, artist concentration caps, and metadata-derived style
            compatibility.
          </>
        }
        actions={
          <span className={`status-pill ${isRunning ? 'status-running' : summary?.latest_run ? 'status-done' : 'status-queued'}`}>
            {isRunning ? <Loader2 size={12} className="spin" /> : <Workflow size={12} />}
            {statusLabel}
          </span>
        }
      />

      <section className="card">
        <div className="pool-creator-hero">
          <div className="pool-creator-copy">
            <div className="hero-kicker">Registered source pool allocation</div>
            <h2 className="page-title" style={{ margin: 0 }}>
              Licence first, rights compatible, <em>capacity bounded</em>.
            </h2>
            <p className="page-sub">
              {engineVersion === 'v2'
                ? 'v2 plans broad, duration-balanced pool families first, then spills into new registered pools when a 4-hour pool fills.'
                : 'v1 runs against the current manifest and writes its outputs to a dedicated `registry/pool_allocator/` registry without disturbing the legacy pool taxonomy.'}
            </p>
            <div className="row">
              <span className="pill">
                <Files size={13} /> {numberFormat(candidateAssetCount)} candidate assets
              </span>
              <span className="pill">
                <FolderPlus size={13} /> {numberFormat(summary?.pool_count)} pools
              </span>
              <span className="pill">
                <ShieldAlert size={13} /> {numberFormat(summary?.review_count)} in review
              </span>
            </div>
            {summary?.manifest_requires_reconcile ? (
              <div className="pool-warning-text">
                Source manifest is smaller than local downloaded audio. Starting a run will reconcile it against disk before allocation continues.
              </div>
            ) : null}
            {!summary?.manifest_requires_reconcile && pendingCandidateCount > 0 ? (
              <div className="pool-warning-text">
                {numberFormat(pendingCandidateCount)} candidate assets are still outside the current allocator registry. Run the allocator again to continue from the saved allocations.
              </div>
            ) : null}
          </div>

          <div className="pool-creator-side">
            <div className="kv">
              <span className="k">Latest run</span>
              <span className="v mono">{summary?.latest_run?.run_id || '—'}</span>
            </div>
            <div className="kv">
              <span className="k">Finished</span>
              <span className="v">{formatTimestamp(summary?.latest_run?.finished_at ?? null)}</span>
            </div>
            <div className="kv">
              <span className="k">Processed assets</span>
              <span className="v mono">{numberFormat(summary?.latest_run?.processed_assets)}</span>
            </div>
            <div className="kv">
              <span className="k">Assignment records</span>
              <span className="v mono">{numberFormat(summary?.assignment_count)}</span>
            </div>
          </div>
        </div>
      </section>

      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Run Controls</div>
            <div className="card-meta">
              {isRunning ? 'job running' : 'manifest-backed execution'}
            </div>
          </div>

          <div className="controls">
            <div className="control-row">
              <div className="field">
                <label>Allocator engine</label>
                <select
                  value={engineVersion}
                  onChange={(event) => setEngineVersion(event.target.value as 'v1' | 'v2')}
                  disabled={isRunning}
                >
                  <option value="v2">v2 · broad planned pools</option>
                  <option value="v1">v1 · asset reactive</option>
                </select>
              </div>
              <div className="field">
                <label>Subset role</label>
                <input
                  type="text"
                  value={subsetRole}
                  onChange={(e) => setSubsetRole(e.target.value)}
                  disabled={isRunning}
                  placeholder="music_train_candidate"
                />
              </div>
              <div className="field">
                <label>Optional limit</label>
                <input
                  type="number"
                  min={1}
                  value={limit}
                  onChange={(e) => setLimit(e.target.value)}
                  disabled={isRunning}
                  placeholder="process all"
                />
              </div>
            </div>

            <label className="toggle">
              <input
                type="checkbox"
                checked={onlyDownloaded}
                onChange={(e) => setOnlyDownloaded(e.target.checked)}
                disabled={isRunning}
              />
              Only include manifest rows whose `download_status` is `downloaded`
            </label>

            <label className="toggle">
              <input
                type="checkbox"
                checked={allowRelaxedMetadata}
                onChange={(e) => setAllowRelaxedMetadata(e.target.checked)}
                disabled={isRunning}
              />
              Relax missing rights metadata gates and allocate using licence, territory, genre, style, and caps when label/rightsholder fields are absent
            </label>

            <div className="row" style={{ justifyContent: 'space-between' }}>
              <div className="dim mono">
                {job?.last_message || 'No active job'}
              </div>
              <div className="row">
                {isRunning ? (
                  <button className="btn btn-ghost" type="button" onClick={handlePause}>
                    <Pause size={16} />
                    Pause
                  </button>
                ) : (
                  <>
                    {canResume ? (
                      <button className="btn btn-ghost" type="button" onClick={() => handleRun(false)}>
                        <Play size={16} />
                        Resume paused run
                      </button>
                    ) : null}
                    <button className="btn" type="button" onClick={() => handleRun(canResume)} disabled={isRunning}>
                      {isRunning ? <Loader2 size={16} className="spin" /> : canResume ? <RotateCcw size={16} /> : <Play size={16} />}
                      {canResume ? 'Start fresh run' : 'Run allocator'}
                    </button>
                  </>
                )}
              </div>
            </div>

            {canResume ? (
              <div className="msg">
                Paused at {numberFormat(progress?.processed_assets)} / {numberFormat(progress?.total_assets)} assets.
                `Resume paused run` continues from the saved checkpoint. `Start fresh run` clears the saved allocator state and rebuilds pools from scratch.
              </div>
            ) : null}

            {actionMessage ? <div className="msg">{actionMessage}</div> : null}
            {job?.last_error ? <div className="msg msg-err">{job.last_error}</div> : null}
            {error ? <div className="msg msg-err">{error}</div> : null}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Rule Snapshot</div>
            <div className="card-meta">public principles only</div>
          </div>
          <div className="kv">
            <span className="k">Engine</span>
            <span className="v mono">{summary?.allocation_engine_version || engineVersion}</span>
          </div>
          <div className="kv">
            <span className="k">Pool capacity</span>
            <span className="v mono">{formatDuration(rules?.max_pool_duration_seconds)}</span>
          </div>
          <div className="kv">
            <span className="k">Artist cap</span>
            <span className="v mono">{formatDuration(rules?.max_artist_duration_seconds)}</span>
          </div>
          <div className="kv">
            <span className="k">Repair threshold</span>
            <span className="v mono">{numberFormat(rules?.repair_threshold)}</span>
          </div>
          <div className="kv">
            <span className="k">Pool code distance</span>
            <span className="v mono">{numberFormat(rules?.min_pool_code_edit_distance)}</span>
          </div>
          {engineVersion === 'v2' ? (
            <>
              <div className="kv">
                <span className="k">Target pools</span>
                <span className="v mono">{numberFormat(rules?.target_pool_count)}</span>
              </div>
              <div className="kv">
                <span className="k">Artist exception floor</span>
                <span className="v mono">{formatDuration(rules?.artist_exception_min_duration_seconds)}</span>
              </div>
            </>
          ) : null}
          <div className="kv">
            <span className="k">Relaxed mode</span>
            <span className="v mono">{job?.options?.allow_relaxed_metadata ? 'enabled' : 'disabled'}</span>
          </div>
          <ul className="bullet-list" style={{ marginTop: 12 }}>
            <li>{engineVersion === 'v2' ? 'v2 groups by licence, territory, rights compatibility, and broad pool family before duration packing.' : 'Hard filters run in order: licence, territory, rights compatibility, capacity, artist cap, genre.'}</li>
            <li>Exact identifier or fingerprint duplicates are blocked from creating new allocations.</li>
            <li>{engineVersion === 'v2' ? 'General pools keep the artist cap; registered artist-concentrated pools are explicit exceptions.' : 'Borderline or incomplete metadata routes to read-only review in v1.'}</li>
          </ul>
        </div>
      </section>

      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Run Progress</div>
            <div className="card-meta">
              {numberFormat(progress?.processed_assets)} / {numberFormat(progress?.total_assets)} assets
            </div>
          </div>
          <div className="controls">
            <div className="bar" aria-label="Pool allocation progress">
              <div className="bar-fill" style={{ width: `${progressPercent}%` }} />
            </div>
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <span className="mono">{progressPercent.toFixed(1)}%</span>
              <span className="dim mono">{progress?.status || 'idle'}</span>
            </div>
            <div className="kv">
              <span className="k">Current phase</span>
              <span className="v mono">{progress?.current_phase || '—'}</span>
            </div>
            <div className="kv">
              <span className="k">Current asset</span>
              <span className="v mono">{progress?.current_asset || '—'}</span>
            </div>
            <div className="kv">
              <span className="k">Current title</span>
              <span className="v">{progress?.current_asset_title || '—'}</span>
            </div>
            <div className="kv">
              <span className="k">Current pool</span>
              <span className="v mono">{progress?.current_pool_id || '—'}</span>
            </div>
            <div className="kv">
              <span className="k">Started</span>
              <span className="v">{formatTimestamp(progress?.started_at)}</span>
            </div>
            <div className="kv">
              <span className="k">Updated</span>
              <span className="v">{formatTimestamp(progress?.updated_at)}</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Activity Log</div>
            <div className="card-meta">{numberFormat(activityLog.length)} recent events</div>
          </div>
          <div className="log-stream">
            {activityLog.length === 0 ? (
              <div className="log-line dim">No allocator events yet.</div>
            ) : (
              activityLog.map((entry, index) => (
                <div className="log-line" key={`${entry.ts || 'log'}-${index}`}>
                  [{formatTimestamp(entry.ts ?? null)}] {entry.phase || 'job'} · {entry.message || '—'}
                </div>
              ))
            )}
          </div>
        </div>
      </section>

      <section className="pool-summary-grid">
        <MetricCard
          label="Assigned"
          value={numberFormat(counts.assigned)}
          icon={<CheckCircle2 size={16} />}
          tone="good"
          meta={`${numberFormat(totalOutcomes)} total outcomes`}
        />
        <MetricCard
          label="New pools"
          value={numberFormat(counts.new_pool_created)}
          icon={<FolderPlus size={16} />}
          tone="neutral"
          meta={`${numberFormat(summary?.pool_count)} total pools`}
        />
        <MetricCard
          label="Duplicates"
          value={numberFormat(counts.duplicate_found)}
          icon={<Files size={16} />}
          tone="warn"
          meta={`${numberFormat(summary?.duplicate_count)} registry matches`}
        />
        <MetricCard
          label="Review required"
          value={numberFormat(counts.review_required)}
          icon={<AlertTriangle size={16} />}
          tone="bad"
          meta={`${numberFormat(summary?.review_count)} queued`}
        />
        <MetricCard
          label="Rejected"
          value={numberFormat(counts.rejected)}
          icon={<ShieldAlert size={16} />}
          tone="bad"
        />
        <MetricCard
          label="Unresolved"
          value={numberFormat(counts.unresolved)}
          icon={<Workflow size={16} />}
          tone="neutral"
        />
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">Pool Registry</div>
          <div className="card-meta">{numberFormat(pools.length)} visible pools</div>
        </div>
        <div className="paths" style={{ marginBottom: 14 }}>
          <div>Source manifest: <span className="mono">{summary?.manifest_paths?.source_manifest_path || '—'}</span></div>
          <div>CaRA pool manifest: <span className="mono">{summary?.manifest_paths?.cara_pool_manifest_path || '—'}</span></div>
          <div>CaRA pool CSV: <span className="mono">{summary?.manifest_paths?.cara_pool_manifest_csv_path || '—'}</span></div>
        </div>
        {pools.length === 0 ? (
          <EmptyState message="No allocator pools registered yet. Run the allocator to create or reuse pools." />
        ) : (
          <div className="table-scroll">
            <div className="run-table pool-table">
              <div className="run-row run-head" style={{ gridTemplateColumns: '1.8fr 1fr 0.8fr 1.1fr 0.9fr 0.8fr 0.9fr 0.9fr 0.8fr' }}>
                <span>Pool</span>
                <span>Licence</span>
                <span>Territory</span>
                <span>Rights</span>
                <span>Genre</span>
                <span>Type</span>
                <span>Assets</span>
                <span>Capacity</span>
                <span>Top artist</span>
              </div>
              {pools.map((pool) => (
                <div
                  className="run-row"
                  key={pool.pool_id}
                  style={{ gridTemplateColumns: '1.8fr 1fr 0.8fr 1.1fr 0.9fr 0.8fr 0.9fr 0.9fr 0.8fr' }}
                >
                  <span>
                    <div className="mono">{pool.pool_id}</div>
                    <div className="dim" style={{ fontSize: 12 }}>{pool.style_summary || 'No style summary'}</div>
                  </span>
                  <span className="mono">{pool.licence_class || '—'}</span>
                  <span className="mono">{pool.territory || '—'}</span>
                  <span>{pool.record_label || pool.rights_holder_group || '—'}</span>
                  <span>{pool.pool_family || pool.primary_genre || '—'}</span>
                  <span className="mono">{pool.pool_type || 'general'}</span>
                  <span className="mono">{numberFormat(pool.asset_count)}</span>
                  <span className="mono">
                    {formatDuration(pool.current_duration_seconds)} / {formatDuration(pool.current_duration_seconds + pool.remaining_capacity_seconds)}
                  </span>
                  <span className="mono">{percentFormat(pool.top_artist_share)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Recent Assignments</div>
            <div className="card-meta">{numberFormat(assignments.length)} rows</div>
          </div>
          {assignments.length === 0 ? (
            <EmptyState message="No assignment records yet." />
          ) : (
            <div className="table-scroll">
              <div className="run-table">
                <div className="run-row run-head" style={{ gridTemplateColumns: '1fr 1.4fr 0.9fr 1.4fr 0.8fr' }}>
                  <span>Asset</span>
                  <span>Pool</span>
                  <span>Status</span>
                  <span>Reason codes</span>
                  <span>Review</span>
                </div>
                {assignments.map((assignment) => (
                  <div
                    className="run-row"
                    key={`${assignment.assignment_id}-${assignment.asset_id}-${assignment.assigned_at || 'na'}`}
                    style={{ gridTemplateColumns: '1fr 1.4fr 0.9fr 1.4fr 0.8fr' }}
                  >
                    <span className="mono">{assignment.asset_id}</span>
                    <span className="mono">{assignment.pool_id || '—'}</span>
                    <span>
                      <span className={`status-pill ${statusClass(assignment.assignment_status)}`}>
                        {assignment.assignment_status}
                      </span>
                    </span>
                    <span className="mono code-list">{assignment.reason_codes.join(', ') || '—'}</span>
                    <span className={assignment.review_required ? 'v-bad mono' : 'dim mono'}>
                      {assignment.review_required ? 'yes' : 'no'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Review Queue</div>
            <div className="card-meta">{numberFormat(reviewQueue.length)} queued</div>
          </div>
          {reviewQueue.length === 0 ? (
            <EmptyState message="No review items. Incomplete rights metadata or fuzzy duplicate cases will appear here." />
          ) : (
            <div className="table-scroll">
              <div className="run-table">
                <div className="run-row run-head" style={{ gridTemplateColumns: '1fr 1.1fr 1.5fr' }}>
                  <span>Asset</span>
                  <span>Pool</span>
                  <span>Reasons</span>
                </div>
                {reviewQueue.map((row) => (
                  <div
                    className="run-row"
                    key={`${row.assignment_id}-${row.asset_id}-${row.assigned_at || 'na'}`}
                    style={{ gridTemplateColumns: '1fr 1.1fr 1.5fr' }}
                  >
                    <span className="mono">{row.asset_id}</span>
                    <span className="mono">{row.pool_id || '—'}</span>
                    <span className="mono code-list">{row.reason_codes.join(', ') || '—'}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
};
