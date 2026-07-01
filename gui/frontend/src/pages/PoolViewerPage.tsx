import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AudioLines,
  ChevronRight,
  FileAudio,
  FolderOpen,
  Info,
  Loader2,
  Search,
  X,
} from 'lucide-react';
import { PageHeader } from './PageHeader';

interface PoolFolder {
  pool_id: string;
  pool_family?: string | null;
  licence_class?: string | null;
  territory?: string | null;
  primary_genres: string[];
  asset_count: number;
  duration_seconds: number;
  audio_available_count: number;
}

interface PoolAsset {
  asset_id: string;
  pool_id: string;
  source?: string | null;
  source_id?: string | number | null;
  title?: string | null;
  artist?: string | null;
  licence_class?: string | null;
  primary_genre?: string | null;
  secondary_genre?: string | null;
  pool_family?: string | null;
  duration_seconds: number;
  style_tags?: string[] | string | null;
  audio_available: boolean;
  audio_url?: string | null;
}

interface PoolAssetDetail extends PoolAsset {
  metadata: Record<string, unknown>;
}

type SortDirection = 'asc' | 'desc';

interface SortState<T extends string> {
  key: T;
  direction: SortDirection;
}

type PoolSortKey = 'pool_id' | 'pool_family' | 'licence_class' | 'asset_count' | 'duration_seconds' | 'audio_available_count';
type AssetSortKey = 'title' | 'artist' | 'primary_genre' | 'duration_seconds' | 'source_id' | 'audio_available';

const ENGINE = 'v2';

const numberFormat = (value: number | null | undefined): string => {
  if (value === null || value === undefined || Number.isNaN(value)) return '–';
  return value.toLocaleString();
};

const formatDuration = (seconds: number | null | undefined): string => {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '–';
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
};

const normalizeCell = (value: unknown): string | number | boolean => {
  if (typeof value === 'number' || typeof value === 'boolean') return value;
  if (Array.isArray(value)) return value.join(', ').toLowerCase();
  return String(value ?? '').toLowerCase();
};

const compareValues = (left: unknown, right: unknown, direction: SortDirection): number => {
  const a = normalizeCell(left);
  const b = normalizeCell(right);
  let result = 0;
  if (typeof a === 'number' && typeof b === 'number') {
    result = a - b;
  } else if (typeof a === 'boolean' && typeof b === 'boolean') {
    result = Number(a) - Number(b);
  } else {
    result = String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: 'base' });
  }
  return direction === 'asc' ? result : -result;
};

const sortIndicator = <T extends string>(sort: SortState<T>, key: T): string => {
  if (sort.key !== key) return '';
  return sort.direction === 'asc' ? ' ↑' : ' ↓';
};

const toggleSort = <T extends string>(sort: SortState<T>, key: T): SortState<T> => ({
  key,
  direction: sort.key === key && sort.direction === 'asc' ? 'desc' : 'asc',
});

const metadataValue = (value: unknown): string => {
  if (value === null || value === undefined || value === '') return '—';
  if (Array.isArray(value) || typeof value === 'object') return JSON.stringify(value, null, 2);
  return String(value);
};

const styleTagText = (value: PoolAsset['style_tags']): string => {
  if (!value) return '—';
  if (Array.isArray(value)) return value.join(', ') || '—';
  return String(value);
};

const WaveformCanvas: React.FC<{ audioUrl?: string | null }> = ({ audioUrl }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [status, setStatus] = useState<string>('No audio loaded');

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audioUrl) {
      setStatus(audioUrl ? 'Preparing waveform…' : 'No audio available');
      return;
    }

    const context = canvas.getContext('2d');
    if (!context) return;

    let cancelled = false;
    const drawPlaceholder = (label: string) => {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = '#11151c';
      context.fillRect(0, 0, canvas.width, canvas.height);
      context.fillStyle = '#7f8795';
      context.font = '13px ui-monospace, monospace';
      context.fillText(label, 18, canvas.height / 2);
    };

    drawPlaceholder('Loading waveform…');
    setStatus('Loading waveform…');

    const audioContext = new AudioContext();
    fetch(audioUrl)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.arrayBuffer();
      })
      .then((buffer) => audioContext.decodeAudioData(buffer))
      .then((audioBuffer) => {
        if (cancelled) return;
        const channel = audioBuffer.getChannelData(0);
        const width = canvas.width;
        const height = canvas.height;
        const step = Math.max(1, Math.floor(channel.length / width));
        const amp = height / 2;
        context.clearRect(0, 0, width, height);
        const gradient = context.createLinearGradient(0, 0, width, 0);
        gradient.addColorStop(0, '#7c5cff');
        gradient.addColorStop(1, '#18d2ff');
        context.fillStyle = '#10141b';
        context.fillRect(0, 0, width, height);
        context.strokeStyle = gradient;
        context.lineWidth = 1;
        context.beginPath();
        for (let x = 0; x < width; x += 1) {
          let min = 1;
          let max = -1;
          const offset = x * step;
          for (let i = 0; i < step && offset + i < channel.length; i += 1) {
            const sample = channel[offset + i];
            if (sample < min) min = sample;
            if (sample > max) max = sample;
          }
          context.moveTo(x, (1 + min) * amp);
          context.lineTo(x, (1 + max) * amp);
        }
        context.stroke();
        setStatus(`Waveform · ${formatDuration(audioBuffer.duration)}`);
      })
      .catch((error) => {
        if (!cancelled) {
          drawPlaceholder('Waveform unavailable for this file');
          setStatus(error instanceof Error ? `Waveform unavailable: ${error.message}` : 'Waveform unavailable');
        }
      })
      .finally(() => {
        audioContext.close().catch(() => undefined);
      });

    return () => {
      cancelled = true;
      audioContext.close().catch(() => undefined);
    };
  }, [audioUrl]);

  return (
    <div className="waveform-wrap">
      <canvas ref={canvasRef} width={920} height={132} aria-label="Audio waveform" />
      <div className="dim mono" style={{ fontSize: 12 }}>{status}</div>
    </div>
  );
};

export const PoolViewerPage: React.FC = () => {
  const [pools, setPools] = useState<PoolFolder[]>([]);
  const [assets, setAssets] = useState<PoolAsset[]>([]);
  const [selectedPool, setSelectedPool] = useState<PoolFolder | null>(null);
  const [selectedAsset, setSelectedAsset] = useState<PoolAssetDetail | null>(null);
  const [loadingPools, setLoadingPools] = useState(true);
  const [loadingAssets, setLoadingAssets] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [poolSearch, setPoolSearch] = useState('');
  const [fileSearch, setFileSearch] = useState('');
  const [poolSort, setPoolSort] = useState<SortState<PoolSortKey>>({ key: 'pool_id', direction: 'asc' });
  const [assetSort, setAssetSort] = useState<SortState<AssetSortKey>>({ key: 'title', direction: 'asc' });

  const fetchPools = useCallback(async () => {
    try {
      setLoadingPools(true);
      const response = await fetch(`/api/data/pool-viewer/pools?engine=${ENGINE}`);
      if (!response.ok) throw new Error(`pools HTTP ${response.status}`);
      const json: PoolFolder[] = await response.json();
      setPools(json);
      setSelectedPool((current) => current ?? json[0] ?? null);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load pool viewer data');
    } finally {
      setLoadingPools(false);
    }
  }, []);

  const fetchAssets = useCallback(async (pool: PoolFolder | null) => {
    if (!pool) {
      setAssets([]);
      return;
    }
    try {
      setLoadingAssets(true);
      const response = await fetch(`/api/data/pool-viewer/pools/${encodeURIComponent(pool.pool_id)}/assets?engine=${ENGINE}`);
      if (!response.ok) throw new Error(`pool assets HTTP ${response.status}`);
      const json: PoolAsset[] = await response.json();
      setAssets(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load pool contents');
    } finally {
      setLoadingAssets(false);
    }
  }, []);

  useEffect(() => {
    fetchPools();
  }, [fetchPools]);

  useEffect(() => {
    fetchAssets(selectedPool);
  }, [fetchAssets, selectedPool]);

  const openAsset = useCallback(async (asset: PoolAsset) => {
    try {
      setLoadingDetail(true);
      const response = await fetch(`/api/data/pool-viewer/assets/${encodeURIComponent(asset.asset_id)}?engine=${ENGINE}`);
      if (!response.ok) throw new Error(`asset HTTP ${response.status}`);
      const json: PoolAssetDetail = await response.json();
      setSelectedAsset(json);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to open file metadata');
    } finally {
      setLoadingDetail(false);
    }
  }, []);

  const filteredPools = useMemo(() => {
    const needle = poolSearch.trim().toLowerCase();
    return pools
      .filter((pool) => {
        if (!needle) return true;
        return [
          pool.pool_id,
          pool.pool_family,
          pool.licence_class,
          pool.territory,
          pool.primary_genres.join(' '),
        ].some((value) => String(value ?? '').toLowerCase().includes(needle));
      })
      .sort((a, b) => compareValues(a[poolSort.key], b[poolSort.key], poolSort.direction));
  }, [poolSearch, poolSort, pools]);

  const filteredAssets = useMemo(() => {
    const needle = fileSearch.trim().toLowerCase();
    return assets
      .filter((asset) => {
        if (!needle) return true;
        return [
          asset.asset_id,
          asset.title,
          asset.artist,
          asset.primary_genre,
          asset.secondary_genre,
          asset.source_id,
          styleTagText(asset.style_tags),
        ].some((value) => String(value ?? '').toLowerCase().includes(needle));
      })
      .sort((a, b) => compareValues(a[assetSort.key], b[assetSort.key], assetSort.direction));
  }, [assetSort, assets, fileSearch]);

  const totalDuration = pools.reduce((sum, pool) => sum + (pool.duration_seconds || 0), 0);
  const totalAssets = pools.reduce((sum, pool) => sum + (pool.asset_count || 0), 0);

  return (
    <div style={{ display: 'grid', gap: 20 }}>
      <PageHeader
        kicker="Data · Pool Viewer"
        title={
          <>
            Browse the <em>CaRA pool manifest</em> like folders
          </>
        }
        description={
          <>
            Inspect v2 Pool Creator outputs by pool, open allocated files, review metadata, and play
            source audio directly from the local Freesound cache.
          </>
        }
        actions={
          <span className="status-pill status-done">
            <FolderOpen size={12} />
            {numberFormat(pools.length)} pools
          </span>
        }
      />

      <section className="pool-summary-grid">
        <div className="pool-metric-card">
          <div className="pool-metric-top"><FolderOpen size={16} /> <span className="pool-metric-label">Pool folders</span></div>
          <div className="pool-metric-value">{numberFormat(pools.length)}</div>
          <div className="pool-metric-meta">from cara_pool_manifest_v2</div>
        </div>
        <div className="pool-metric-card">
          <div className="pool-metric-top"><FileAudio size={16} /> <span className="pool-metric-label">Allocated files</span></div>
          <div className="pool-metric-value">{numberFormat(totalAssets)}</div>
          <div className="pool-metric-meta">training manifest rows</div>
        </div>
        <div className="pool-metric-card">
          <div className="pool-metric-top"><AudioLines size={16} /> <span className="pool-metric-label">Total duration</span></div>
          <div className="pool-metric-value">{formatDuration(totalDuration)}</div>
          <div className="pool-metric-meta">pooled audio time</div>
        </div>
      </section>

      {error ? <div className="msg msg-err">{error}</div> : null}

      <section className="pool-viewer-layout">
        <div className="card pool-browser-panel">
          <div className="card-header">
            <div className="card-title">Pool Folders</div>
            <div className="card-meta">{numberFormat(filteredPools.length)} visible</div>
          </div>
          <label className="search-field">
            <Search size={14} />
            <input
              value={poolSearch}
              onChange={(event) => setPoolSearch(event.target.value)}
              placeholder="Search pools, families, licences…"
            />
          </label>
          {loadingPools ? (
            <div className="pool-empty-state"><Loader2 size={16} className="spin" /> Loading pools…</div>
          ) : (
            <div className="table-scroll pool-viewer-scroll">
              <div className="run-table pool-viewer-table">
                <button className="run-row run-head sortable-row" style={{ gridTemplateColumns: '1.7fr 1fr 0.7fr 0.9fr 0.8fr' }} type="button">
                  <span onClick={() => setPoolSort((sort) => toggleSort(sort, 'pool_id'))}>Pool{sortIndicator(poolSort, 'pool_id')}</span>
                  <span onClick={() => setPoolSort((sort) => toggleSort(sort, 'pool_family'))}>Family{sortIndicator(poolSort, 'pool_family')}</span>
                  <span onClick={() => setPoolSort((sort) => toggleSort(sort, 'licence_class'))}>Licence{sortIndicator(poolSort, 'licence_class')}</span>
                  <span onClick={() => setPoolSort((sort) => toggleSort(sort, 'asset_count'))}>Files{sortIndicator(poolSort, 'asset_count')}</span>
                  <span onClick={() => setPoolSort((sort) => toggleSort(sort, 'duration_seconds'))}>Time{sortIndicator(poolSort, 'duration_seconds')}</span>
                </button>
                {filteredPools.map((pool) => (
                  <button
                    className={`run-row pool-folder-row${selectedPool?.pool_id === pool.pool_id ? ' is-selected' : ''}`}
                    key={pool.pool_id}
                    onClick={() => setSelectedPool(pool)}
                    style={{ gridTemplateColumns: '1.7fr 1fr 0.7fr 0.9fr 0.8fr' }}
                    type="button"
                  >
                    <span>
                      <div className="mono"><ChevronRight size={12} /> {pool.pool_id}</div>
                      <div className="dim">{pool.primary_genres.slice(0, 3).join(', ') || 'No genres'}</div>
                    </span>
                    <span>{pool.pool_family || '—'}</span>
                    <span className="mono">{pool.licence_class || '—'}</span>
                    <span className="mono">{numberFormat(pool.asset_count)}</span>
                    <span className="mono">{formatDuration(pool.duration_seconds)}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="card pool-browser-panel">
          <div className="card-header">
            <div>
              <div className="card-title">Pool Contents</div>
              <div className="card-meta mono">{selectedPool?.pool_id || 'Select a pool'}</div>
            </div>
            <div className="card-meta">{numberFormat(filteredAssets.length)} files</div>
          </div>
          <label className="search-field">
            <Search size={14} />
            <input
              value={fileSearch}
              onChange={(event) => setFileSearch(event.target.value)}
              placeholder="Search titles, artists, genres, source IDs…"
            />
          </label>
          {loadingAssets ? (
            <div className="pool-empty-state"><Loader2 size={16} className="spin" /> Loading files…</div>
          ) : !selectedPool ? (
            <div className="pool-empty-state">Select a pool folder to view allocated files.</div>
          ) : filteredAssets.length === 0 ? (
            <div className="pool-empty-state">No files matched the current filter.</div>
          ) : (
            <div className="table-scroll pool-viewer-scroll">
              <div className="run-table pool-viewer-table">
                <button className="run-row run-head sortable-row" style={{ gridTemplateColumns: '1.6fr 1fr 0.8fr 0.8fr 0.8fr 0.6fr' }} type="button">
                  <span onClick={() => setAssetSort((sort) => toggleSort(sort, 'title'))}>File{sortIndicator(assetSort, 'title')}</span>
                  <span onClick={() => setAssetSort((sort) => toggleSort(sort, 'artist'))}>Artist{sortIndicator(assetSort, 'artist')}</span>
                  <span onClick={() => setAssetSort((sort) => toggleSort(sort, 'primary_genre'))}>Genre{sortIndicator(assetSort, 'primary_genre')}</span>
                  <span onClick={() => setAssetSort((sort) => toggleSort(sort, 'duration_seconds'))}>Time{sortIndicator(assetSort, 'duration_seconds')}</span>
                  <span onClick={() => setAssetSort((sort) => toggleSort(sort, 'source_id'))}>Source{sortIndicator(assetSort, 'source_id')}</span>
                  <span onClick={() => setAssetSort((sort) => toggleSort(sort, 'audio_available'))}>Audio{sortIndicator(assetSort, 'audio_available')}</span>
                </button>
                {filteredAssets.map((asset) => (
                  <button
                    className="run-row pool-file-row"
                    key={asset.asset_id}
                    onClick={() => openAsset(asset)}
                    style={{ gridTemplateColumns: '1.6fr 1fr 0.8fr 0.8fr 0.8fr 0.6fr' }}
                    type="button"
                  >
                    <span>
                      <div><FileAudio size={13} /> {asset.title || asset.asset_id}</div>
                      <div className="dim mono">{asset.asset_id}</div>
                    </span>
                    <span>{asset.artist || '—'}</span>
                    <span>{asset.primary_genre || '—'}</span>
                    <span className="mono">{formatDuration(asset.duration_seconds)}</span>
                    <span className="mono">{asset.source}:{asset.source_id}</span>
                    <span className={asset.audio_available ? 'v-good mono' : 'v-bad mono'}>
                      {asset.audio_available ? 'yes' : 'no'}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      {loadingDetail ? (
        <div className="modal-backdrop">
          <div className="metadata-modal">
            <Loader2 className="spin" size={18} /> Loading file metadata…
          </div>
        </div>
      ) : null}

      {selectedAsset ? (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Pool file metadata">
          <div className="metadata-modal">
            <div className="metadata-modal-header">
              <div>
                <div className="hero-kicker">Allocated source file</div>
                <h2>{selectedAsset.title || selectedAsset.asset_id}</h2>
                <div className="dim mono">{selectedAsset.pool_id}</div>
              </div>
              <button className="icon-button" type="button" onClick={() => setSelectedAsset(null)} aria-label="Close metadata modal">
                <X size={18} />
              </button>
            </div>

            <div className="audio-panel">
              {selectedAsset.audio_available && selectedAsset.audio_url ? (
                <>
                  <audio controls src={selectedAsset.audio_url} preload="metadata" style={{ width: '100%' }} />
                  <WaveformCanvas audioUrl={selectedAsset.audio_url} />
                </>
              ) : (
                <div className="pool-empty-state">No local audio file was found for this manifest row.</div>
              )}
            </div>

            <div className="metadata-quick-grid">
              <div className="kv"><span className="k">Asset</span><span className="v mono">{selectedAsset.asset_id}</span></div>
              <div className="kv"><span className="k">Source</span><span className="v mono">{selectedAsset.source}:{selectedAsset.source_id}</span></div>
              <div className="kv"><span className="k">Artist</span><span className="v">{selectedAsset.artist || '—'}</span></div>
              <div className="kv"><span className="k">Duration</span><span className="v mono">{formatDuration(selectedAsset.duration_seconds)}</span></div>
              <div className="kv"><span className="k">Genre</span><span className="v">{selectedAsset.primary_genre || '—'}</span></div>
              <div className="kv"><span className="k">Tags</span><span className="v">{styleTagText(selectedAsset.style_tags)}</span></div>
            </div>

            <div className="metadata-section-title">
              <Info size={14} /> Full manifest metadata
            </div>
            <div className="metadata-grid">
              {Object.entries(selectedAsset.metadata)
                .sort(([left], [right]) => left.localeCompare(right))
                .map(([key, value]) => (
                  <React.Fragment key={key}>
                    <div className="metadata-key mono">{key}</div>
                    <pre className="metadata-value">{metadataValue(value)}</pre>
                  </React.Fragment>
                ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
