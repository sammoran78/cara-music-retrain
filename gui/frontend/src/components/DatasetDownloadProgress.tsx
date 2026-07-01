import React, { useEffect, useMemo, useState } from 'react';

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
  cached_metadata_hits: number;
  bulk_metadata_calls: number;
  single_metadata_fallback_calls: number;
  downloads_skipped_existing: number;
  skipped_non_confirmed: number;
  api_requests_used_today: number;
}

interface HistoryPoint {
  recordedAt: string;
  processedCount: number;
  percentComplete: number;
}

interface DownloadJobState {
  status: string;
  started_at: number | null;
  finished_at: number | null;
  limit: number | null;
  skip_audio: boolean;
  subset_mode?: string;
  subset_role?: string | null;
  last_summary: Record<string, number> | null;
  error: string | null;
  thread_alive?: boolean;
}

const POLL_MS = 5000;
const MAX_HISTORY_POINTS = 24;

export const DatasetDownloadProgress: React.FC = () => {
  const [data, setData] = useState<DownloadProgressData | null>(null);
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<DownloadJobState | null>(null);
  const [limitInput, setLimitInput] = useState<string>('');
  const [skipAudio, setSkipAudio] = useState<boolean>(false);
  const [subsetMode, setSubsetMode] = useState<string>('subset_role');
  const [subsetRole, setSubsetRole] = useState<string>('music_train_candidate');
  const [actionBusy, setActionBusy] = useState<boolean>(false);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  const subsetModeLabel = useMemo(() => {
    const activeMode = job?.subset_mode || data?.subset_mode || subsetMode;
    const activeRole = job?.subset_role || data?.subset_role || subsetRole;
    if (activeMode === 'confirmed_only') {
      return 'Prefilter-confirmed rows only';
    }
    if (activeMode === 'subset_role') {
      return activeRole ? `Manifest subset role: ${activeRole}` : 'Manifest subset role';
    }
    if (activeMode === 'all_freesound') {
      return 'All Freesound rows';
    }
    return activeMode;
  }, [data?.subset_mode, data?.subset_role, job?.subset_mode, job?.subset_role, subsetMode, subsetRole]);

  const fetchJob = async () => {
    try {
      const response = await fetch('/api/data/download/job');
      if (!response.ok) {
        return;
      }
      const nextJob: DownloadJobState = await response.json();
      setJob(nextJob);
    } catch (err) {
      // best-effort; surface through progress error path only
    }
  };

  const handleStart = async () => {
    setActionBusy(true);
    setActionMessage(null);
    try {
      const parsedLimit = limitInput.trim() === '' ? null : Number(limitInput);
      if (parsedLimit !== null && (!Number.isFinite(parsedLimit) || parsedLimit <= 0)) {
        throw new Error('Limit must be a positive integer or blank');
      }
      const response = await fetch('/api/data/download/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          limit: parsedLimit,
          skip_audio: skipAudio,
          subset_mode: subsetMode,
          subset_role: subsetMode === 'subset_role' ? subsetRole.trim() : null,
        }),
      });
      const body = await response.json();
      if (!response.ok) {
        throw new Error(body?.detail || 'Failed to start download');
      }
      setActionMessage(
        body.status === 'already_running'
          ? 'Download already running'
          : 'Download started'
      );
      if (body.job) {
        setJob(body.job);
      }
    } catch (err) {
      setActionMessage(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setActionBusy(false);
    }
  };

  useEffect(() => {
    let mounted = true;

    const fetchProgress = async () => {
      try {
        const response = await fetch('/api/data/download-progress');
        if (!response.ok) {
          throw new Error('Failed to fetch dataset download progress');
        }
        const nextData: DownloadProgressData = await response.json();
        if (!mounted) {
          return;
        }
        setData(nextData);
        setError(null);
        setHistory((current: HistoryPoint[]) => {
          const nextPoint: HistoryPoint = {
            recordedAt: new Date().toISOString(),
            processedCount: nextData.processed_count,
            percentComplete: nextData.percent_complete,
          };
          const lastPoint = current[current.length - 1];
          if (
            lastPoint &&
            lastPoint.processedCount === nextPoint.processedCount &&
            lastPoint.percentComplete === nextPoint.percentComplete
          ) {
            return current;
          }
          return [...current, nextPoint].slice(-MAX_HISTORY_POINTS);
        });
      } catch (err) {
        if (!mounted) {
          return;
        }
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    fetchProgress();
    fetchJob();
    const intervalId = window.setInterval(() => {
      fetchProgress();
      fetchJob();
    }, POLL_MS);
    return () => {
      mounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  const statusLabel = useMemo(() => {
    if (!data) {
      return 'Unknown';
    }
    if (data.status === 'not_started') {
      return 'Not started';
    }
    if (data.percent_complete >= 100) {
      return 'Complete';
    }
    return 'In progress';
  }, [data]);

  if (loading) {
    return <div>Loading dataset download progress…</div>;
  }

  if (error) {
    return <div>Error loading dataset download progress: {error}</div>;
  }

  if (!data) {
    return <div>No dataset download progress available.</div>;
  }

  return (
    <div style={{ display: 'grid', gap: '16px' }}>
      <div>
        <h2>Dataset Download Progress</h2>
        <p>Tracks the Freesound subset download progress file and current files on disk.</p>
        <p style={{ margin: '6px 0 0', color: '#555' }}>
          Subset mode: <strong>{subsetModeLabel}</strong>
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '12px' }}>
        <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '12px' }}>
          <div>Status</div>
          <strong>{statusLabel}</strong>
        </div>
        <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '12px' }}>
          <div>Processed</div>
          <strong>{data.processed_count} / {data.total_requested}</strong>
        </div>
        <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '12px' }}>
          <div>Downloaded Audio</div>
          <strong>{data.downloaded_count}</strong>
        </div>
        <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '12px' }}>
          <div>Metadata Only</div>
          <strong>{data.metadata_only_count}</strong>
        </div>
        <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '12px' }}>
          <div>Unavailable</div>
          <strong>{data.unavailable_count}</strong>
        </div>
      </div>

      <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '12px' }}>
        <strong>Download Controls</strong>
        <p style={{ margin: '6px 0 12px', color: '#555' }}>
          Reuses the existing <code>data_pipeline/02_freesound_downloader.py</code> script.
          Progress JSON + manifest are updated as items complete, so re-running resumes automatically.
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center' }}>
          <label style={{ display: 'flex', flexDirection: 'column', fontSize: '14px' }}>
            <span>Subset source</span>
            <select
              value={subsetMode}
              onChange={(event: any) => setSubsetMode(event.target.value)}
              disabled={actionBusy || Boolean(job?.thread_alive)}
              style={{ padding: '6px', borderRadius: '6px', border: '1px solid #ccc', minWidth: '220px' }}
            >
              <option value="subset_role">Official music subset</option>
              <option value="confirmed_only">Prefilter-confirmed rows</option>
              <option value="all_freesound">All Freesound rows</option>
            </select>
          </label>
          {subsetMode === 'subset_role' ? (
            <label style={{ display: 'flex', flexDirection: 'column', fontSize: '14px' }}>
              <span>Manifest subset role</span>
              <input
                type="text"
                value={subsetRole}
                onChange={(event: any) => setSubsetRole(event.target.value)}
                disabled={actionBusy || Boolean(job?.thread_alive)}
                style={{ padding: '6px', borderRadius: '6px', border: '1px solid #ccc', width: '220px' }}
              />
            </label>
          ) : null}
          <label style={{ display: 'flex', flexDirection: 'column', fontSize: '14px' }}>
            <span>Batch limit (blank = all remaining)</span>
            <input
              type="number"
              min={1}
              value={limitInput}
              onChange={(event: any) => setLimitInput(event.target.value)}
              style={{ padding: '6px', borderRadius: '6px', border: '1px solid #ccc', width: '160px' }}
            />
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '14px' }}>
            <input
              type="checkbox"
              checked={skipAudio}
              onChange={(event: any) => setSkipAudio(Boolean(event.target.checked))}
            />
            <span>Metadata only (skip audio files)</span>
          </label>
          <button
            type="button"
            onClick={handleStart}
            disabled={actionBusy || Boolean(job?.thread_alive)}
            style={{
              padding: '10px 14px',
              borderRadius: '8px',
              border: '1px solid #1d4ed8',
              background: job?.thread_alive ? '#93c5fd' : '#2563eb',
              color: '#fff',
              cursor: job?.thread_alive || actionBusy ? 'not-allowed' : 'pointer',
            }}
          >
            {job?.thread_alive ? 'Download running…' : 'Start / Continue Download'}
          </button>
        </div>
        <div style={{ marginTop: '12px', display: 'grid', gap: '6px', fontSize: '14px' }}>
          <div>Job status: <strong>{job?.status ?? 'idle'}</strong></div>
          <div>Active subset: <strong>{subsetModeLabel}</strong></div>
          {job?.started_at ? (
            <div>Started: {new Date(job.started_at * 1000).toLocaleString()}</div>
          ) : null}
          {job?.finished_at ? (
            <div>Finished: {new Date(job.finished_at * 1000).toLocaleString()}</div>
          ) : null}
          {job?.last_summary ? (
            <div>
              Last batch summary: requested {job.last_summary.requested ?? 0}, downloaded {job.last_summary.downloaded ?? 0},
              metadata only {job.last_summary.metadata_only ?? 0}, unavailable {job.last_summary.unavailable ?? 0}
            </div>
          ) : null}
          {job?.error ? (
            <div style={{ color: '#b91c1c', whiteSpace: 'pre-wrap' }}>Job error: {job.error}</div>
          ) : null}
          {actionMessage ? (
            <div style={{ color: '#1d4ed8' }}>{actionMessage}</div>
          ) : null}
        </div>
      </div>

      <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '12px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
          <strong>Completion</strong>
          <span>{data.percent_complete.toFixed(2)}%</span>
        </div>
        <div style={{ width: '100%', height: '16px', background: '#eee', borderRadius: '999px', overflow: 'hidden' }}>
          <div
            style={{
              width: `${Math.max(0, Math.min(100, data.percent_complete))}%`,
              height: '100%',
              background: '#2563eb',
            }}
          />
        </div>
        <div style={{ marginTop: '8px' }}>Remaining items: {data.remaining_count}</div>
      </div>

      <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '12px' }}>
        <strong>Observed Progress Over Time</strong>
        {history.length === 0 ? (
          <div style={{ marginTop: '8px' }}>No history samples yet.</div>
        ) : (
          <div style={{ marginTop: '8px', display: 'grid', gap: '6px' }}>
            {history.map((point: HistoryPoint) => (
              <div key={point.recordedAt} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '14px' }}>
                <span>{new Date(point.recordedAt).toLocaleTimeString()}</span>
                <span>{point.processedCount} processed</span>
                <span>{point.percentComplete.toFixed(2)}%</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '12px' }}>
        <strong>Files on Disk</strong>
        <div style={{ marginTop: '8px', display: 'grid', gap: '6px' }}>
          <div>Audio files: {data.audio_files_on_disk}</div>
          <div>Metadata files: {data.metadata_files_on_disk}</div>
          <div>Last updated: {data.last_updated ? new Date(data.last_updated * 1000).toLocaleString() : 'Unknown'}</div>
        </div>
      </div>

      <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '12px' }}>
        <strong>Efficiency</strong>
        <div style={{ marginTop: '8px', display: 'grid', gap: '6px' }}>
          <div>Cached metadata hits: {data.cached_metadata_hits}</div>
          <div>Bulk metadata calls: {data.bulk_metadata_calls}</div>
          <div>Single metadata fallback calls: {data.single_metadata_fallback_calls}</div>
          <div>Existing audio skips: {data.downloads_skipped_existing}</div>
          <div>Skipped non-confirmed rows: {data.skipped_non_confirmed}</div>
          <div>API requests used today: {data.api_requests_used_today}</div>
        </div>
      </div>

      <div style={{ border: '1px solid #ddd', borderRadius: '8px', padding: '12px' }}>
        <strong>Tracked Paths</strong>
        <div style={{ marginTop: '8px', display: 'grid', gap: '6px', fontFamily: 'monospace', fontSize: '12px' }}>
          <div>Manifest: {data.manifest_path}</div>
          <div>Progress JSON: {data.progress_path}</div>
          <div>Audio output: {data.output_dir}</div>
          <div>Metadata output: {data.meta_dir}</div>
          <div>Unavailable log: {data.unavailable_log}</div>
        </div>
      </div>
    </div>
  );
};
