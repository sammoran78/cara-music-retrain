import React, { useEffect, useMemo, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { PageHeader } from './PageHeader';

interface BenchRow {
  id: string;
  metric: string;
  description: string;
  higher_is_better: boolean;
  base_external_probe?: number | null;
  diffusion_native?: number | null;
  diffusion_external_probe?: number | null;
  ar_native?: number | null;
  hybrid_native?: number | null;
  base_external_probe_status?: string | null;
  diffusion_native_status?: string | null;
  diffusion_external_probe_status?: string | null;
  ar_native_status?: string | null;
  hybrid_native_status?: string | null;
  status: string;
}

interface BenchmarksPayload {
  format: string;
  benchmark_spec?: BenchmarkSpec;
  model_lanes?: ModelLane[];
  metric_rows?: MetricRow[];
  comparison_cards?: ComparisonCard[];
  repairability_matrix?: RepairabilityMatrix | null;
  repair_method_matrix?: RepairMethodMatrix | null;
  prediction_examples?: Record<string, PredictionExample[]>;
  openai_summary?: OpenAISummaryPayload;
  rows: BenchRow[];
  latest_results: {
    metrics_available: boolean;
    audio_available?: boolean;
    audio_complete?: boolean;
    audio_progress_percent?: number | null;
    metric_stage?: string;
    native_attribution_status?: string;
    external_probe_status?: string;
    latest_metrics?: {
      heldout_training_evaluation?: Record<string, HeldoutTrainingEvaluation>;
      lanes?: Record<string, LaneMetrics>;
    };
  };
  benchmark_prompt_set?: {
    locked: boolean;
    prompt_manifest_uri?: string | null;
    suite_ids?: string[];
    reason?: string;
  };
  latest_generated_audio_result?: EvaluationJobState | null;
  latest_generated_audio_smoke_result?: EvaluationJobState | null;
  latest_generated_audio_full_result?: EvaluationJobState | null;
  active_generated_audio_job?: EvaluationJobState | null;
  latest_attribution_scoring_result?: EvaluationJobState | null;
  active_attribution_scoring_job?: EvaluationJobState | null;
}

interface RepairabilityCell {
  status?: string | null;
  count?: number | null;
  rate?: number | null;
  labelled_count?: number | null;
}

interface RepairabilityRow {
  tier: string;
  label: string;
  [laneId: string]: string | RepairabilityCell;
}

interface RepairabilityMatrix {
  format: string;
  lanes: string[];
  rows: RepairabilityRow[];
}

interface RepairMethodMatrix {
  format: string;
  lanes: string[];
  rows: RepairMethodRow[];
}

interface RepairMethodRow {
  method: string;
  label: string;
  [laneId: string]: string | RepairabilityCell;
}

interface LaneMetrics {
  status?: string | null;
  count?: number | null;
  labelled_count?: number | null;
  tier_counts?: Record<string, number>;
  resolution_tier_counts?: Record<string, number>;
  repair_method_counts?: Record<string, number>;
  repairability?: {
    tier_counts?: Record<string, number>;
    tier_rates?: Record<string, number>;
    correct_tier_counts?: Record<string, number>;
    correct_tier_rates?: Record<string, number>;
  };
}

interface TopKCandidate {
  rank?: number | null;
  cara_pool_id?: string | null;
  cara_pool_family?: string | null;
  confidence?: number | null;
}

interface PredictionExample {
  model_id?: string | null;
  suite_id?: string | null;
  prompt_id?: string | null;
  audio_path?: string | null;
  tier?: string | null;
  expected_pool_id?: string | null;
  predicted_pool_id?: string | null;
  resolved_pool_id?: string | null;
  expected_family?: string | null;
  predicted_family?: string | null;
  resolved_family?: string | null;
  confidence?: number | null;
  registry_valid?: boolean | null;
  repair_method?: string | null;
  repair_distance?: number | null;
  exact?: boolean | null;
  repairable?: boolean | null;
  family_match?: boolean | null;
  top_k?: TopKCandidate[];
  prediction_status?: string | null;
  prediction_source?: string | null;
  prediction_error?: string | null;
  feature_alignment?: {
    requested_batch_size?: number | null;
    observed_feature_shapes?: number[][];
    usable_feature_count?: number | null;
    adjustments?: Array<{
      tap_index?: number | null;
      mode?: string | null;
      branches?: number | null;
      shape?: number[];
    }>;
    policy?: string | null;
  } | null;
}

interface HeldoutTrainingEvaluation {
  status?: string;
  reason?: string;
  report_path?: string;
  global_step?: number | string | null;
  heldout_evaluation?: Record<string, {
    samples?: number | null;
    batches?: number | null;
    loss?: number | null;
    metrics?: Record<string, number | null>;
  }>;
}

interface BenchmarkSpec {
  prompt_set_version: string;
  claim_language?: string;
  pool_count?: number | null;
  family_count?: number | null;
}

interface ModelLane {
  model_id: string;
  label: string;
  family: string;
  architecture: string;
  variant: string;
  status: string;
  baseline_role?: string;
}

interface MetricRow {
  model_id: string;
  variant: string;
  evidence_lane: string;
  suite_id: string;
  condition: string;
  metric_id: string;
  metric_label: string;
  value?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  status: string;
  higher_is_better: boolean;
}

interface ComparisonCard {
  id: string;
  title: string;
  metric_id: string;
  condition: string;
  candidate_value?: number | null;
  baseline_value?: number | null;
  delta?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  status: string;
}

interface OpenAISummaryPayload {
  configured: boolean;
  api_key_configured?: boolean;
  sdk_available?: boolean;
  configuration_message?: string;
  latest_path?: string;
  log_path?: string;
  latest?: {
    status?: string;
    created_at?: string;
    model?: string;
    summary?: string;
  } | null;
}

interface EvaluationJobState {
  job_name?: string | null;
  created_at?: string | null;
  status?: string | null;
  message?: string;
  output_path?: string | null;
  scope?: string | null;
  source_audio_job_name?: string | null;
  metrics_uri?: string | null;
  model_ids?: string[];
  suite_ids?: string[];
  planned_generations?: number | null;
  attribution_status?: string | null;
  audio_artifacts?: {
    available: boolean;
    wav_count?: number;
    generation_manifest_uri?: string | null;
  };
}

const fetchJson = async <T,>(url: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(url, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail || body));
  return body as T;
};

const BENCHMARKS_SESSION_CACHE_KEY = 'cara.benchmarks.payload.v1';

interface BenchmarksSessionCache {
  cachedAt: string;
  payload: BenchmarksPayload;
}

const readBenchmarksSessionCache = (): BenchmarksSessionCache | null => {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.sessionStorage.getItem(BENCHMARKS_SESSION_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<BenchmarksSessionCache>;
    if (!parsed || typeof parsed.cachedAt !== 'string' || !parsed.payload) return null;
    return parsed as BenchmarksSessionCache;
  } catch {
    return null;
  }
};

const writeBenchmarksSessionCache = (payload: BenchmarksPayload, cachedAt: string) => {
  if (typeof window === 'undefined') return;
  try {
    window.sessionStorage.setItem(BENCHMARKS_SESSION_CACHE_KEY, JSON.stringify({ cachedAt, payload }));
  } catch {
    // Session cache is a convenience only; benchmark state still comes from the backend.
  }
};

const fmt = (value?: number | null) => {
  if (value === null || value === undefined) return 'pending';
  if (Math.abs(value) <= 1) return `${(value * 100).toFixed(1)}%`;
  return value.toFixed(3);
};

const fmtCell = (value?: number | null, status?: string | null) => {
  if (value !== null && value !== undefined) return fmt(value);
  if (!status) return 'pending';
  if (status === 'not_applicable') return 'N/A';
  if (status === 'missing_predictions') return 'missing predictions';
  if (status === 'scored') return 'pending value';
  return status.replace(/_/g, ' ');
};

const fmtDelta = (value?: number | null) => {
  if (value === null || value === undefined) return 'pending';
  const sign = value > 0 ? '+' : '';
  return `${sign}${fmt(value)}`;
};

const fmtCountRate = (cell?: RepairabilityCell | string | null) => {
  if (!cell || typeof cell === 'string') return 'pending';
  if (cell.status === 'no_labelled_rows' || cell.labelled_count === 0) return 'no labelled rows';
  if (cell.status === 'not_applicable') return 'N/A';
  if (cell.status === 'missing_predictions') return 'missing predictions';
  if (cell.rate !== null && cell.rate !== undefined) {
    const count = cell.count !== null && cell.count !== undefined ? `${cell.count}` : '0';
    const denom = cell.labelled_count !== null && cell.labelled_count !== undefined ? `/${cell.labelled_count}` : '';
    return `${fmt(cell.rate)} · ${count}${denom}`;
  }
  return cell.status?.replace(/_/g, ' ') ?? 'pending';
};

const fmtBool = (value?: boolean | null) => (value === true ? 'yes' : value === false ? 'no' : 'pending');

const fmtShort = (value?: string | number | null) => {
  if (value === null || value === undefined || value === '') return 'none';
  return String(value);
};

const fmtMetric = (value?: number | null) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'pending';
  if (Math.abs(value) <= 1) return `${(value * 100).toFixed(1)}%`;
  return Number(value).toFixed(4);
};

const fmtFeatureAlignment = (alignment?: PredictionExample['feature_alignment']) => {
  if (!alignment) return null;
  const usable = alignment.usable_feature_count ?? 0;
  const modes = Array.from(new Set((alignment.adjustments ?? []).map((item) => item.mode).filter(Boolean)));
  const shapes = (alignment.observed_feature_shapes ?? [])
    .slice(0, 3)
    .map((shape) => `[${shape.join('x')}]`)
    .join(', ');
  return `${usable} tap(s) · ${modes.length ? modes.join(', ') : 'no adjustments'}${shapes ? ` · shapes ${shapes}` : ''}`;
};

const repairTierLabels: Record<string, string> = {
  exact_pool: 'Exact registry match',
  repairable_pool: 'Repaired to registry pool',
  family_or_genre: 'Family / genre only',
  unattributable: 'Unresolved',
};

const repairTierIds = ['exact_pool', 'repairable_pool', 'family_or_genre', 'unattributable'];

const mechanicalTierLabel = (tier?: string | null) => {
  const key = String(tier ?? '');
  return repairTierLabels[key] ?? (key.replace(/_/g, ' ') || 'unknown');
};

const correctnessTierForExample = (example: PredictionExample) => {
  if (example.exact) return 'correct exact pool';
  if (example.repairable) return 'correct repaired pool';
  if (example.family_match) return 'correct family fallback';
  if (example.prediction_status === 'exception') return 'extractor exception';
  return 'not correct for expected label';
};

const laneDisplay = (laneId: string) => {
  const labels: Record<string, string> = {
    base_external_probe: 'Base probe',
    diffusion_native: 'Diffusion native',
    diffusion_external_probe: 'Diffusion probe',
    context_diffusion_native: 'Context Diffusion native',
    context_diffusion_external_probe: 'Context Diffusion probe',
    base_musicgen_external_probe: 'MusicGen base probe',
    musicgen_native: 'MusicGen native',
    musicgen_external_probe: 'MusicGen probe',
  };
  return labels[laneId] ?? laneId.replace(/_/g, ' ');
};

const statusMeaning = [
  ['pending', 'No scored artifact has provided this metric yet.'],
  ['missing predictions', 'Audio exists, but that lane did not write native/probe CARA predictions.'],
  ['extractor failed', 'The scorer attempted native/probe attribution but failed internally; this is not a zero-percent model result.'],
  ['scored with extractor errors', 'Some predictions were scored and some failed; read the exception rate and examples before making a headline claim.'],
  ['pending value', 'The lane was scored, but this specific metric was not emitted by the current scorer.'],
  ['N/A', 'The lane is not applicable, usually because the base model has no native CARA output channel.'],
  ['correct repair', 'In the strict repairability ladder, this means repaired to the expected held-out pool, not merely any registry pool.'],
  ['mechanical repair', 'In the mechanical matrix, this means the raw output resolved to some registry pool before expected-label correctness is checked.'],
];

const statusClass = (status?: string | null) => {
  const normalized = (status ?? '').toLowerCase();
  if (normalized.includes('blocked') || normalized.includes('missing') || normalized.includes('failed') || normalized.includes('error')) return 'status-error';
  if (normalized.includes('ready') || normalized.includes('complete') || normalized.includes('scored') || normalized.includes('available')) {
    return 'status-done';
  }
  if (normalized.includes('pending') || normalized.includes('not run')) return 'status-queued';
  return 'status-running';
};

const listText = (values?: Array<string | number> | string | number | null) => {
  if (Array.isArray(values)) return values.length ? values.join(', ') : 'none';
  if (values === null || values === undefined || values === '') return 'none';
  return String(values);
};

const gateLabel = (job?: EvaluationJobState | null) => {
  const normalized = String(job?.status ?? '').toLowerCase();
  if (!job) return 'Pending';
  if (normalized === 'completed' || normalized === 'succeeded') return 'Passed';
  if (normalized === 'recorded') return 'Recorded';
  return job.status ?? 'Recorded';
};

const isSameOrAfter = (candidate?: EvaluationJobState | null, anchor?: EvaluationJobState | null) => {
  if (!candidate) return false;
  if (!anchor) return true;
  const candidateTime = Date.parse(String(candidate.created_at ?? ''));
  const anchorTime = Date.parse(String(anchor.created_at ?? ''));
  if (Number.isNaN(candidateTime) || Number.isNaN(anchorTime)) return true;
  return candidateTime >= anchorTime;
};

const laneGroups = [
  {
    id: 'diffusion',
    title: 'Diffusion CARA',
    subtitle: 'Stable Audio',
    modelIds: ['diffusion_cara_strong_full_modest_arch', 'stable_audio_cara_strong', 'stable_audio_cara_strong_full'],
    variants: ['cara_strong'],
    family: 'diffusion',
  },
  {
    id: 'context-diffusion',
    title: 'Context Diffusion',
    subtitle: 'Stable Audio + context',
    modelIds: ['context_diffusion_cara_strong_full'],
    variants: ['cara_strong_context_conditioned'],
    family: 'context_diffusion',
  },
  {
    id: 'musicgen',
    title: 'MusicGen CARA',
    subtitle: 'autoregressive',
    modelIds: ['musicgen_cara_strong_full', 'musicgen_cara_strong'],
    variants: ['cara_strong'],
    family: 'autoregressive',
  },
  {
    id: 'retrieval',
    title: 'Retrieval',
    subtitle: 'post-hoc floor',
    modelIds: ['retrieval_baseline'],
    variants: ['retrieval_baseline'],
  },
];

const headlineMetricIds = [
  { id: 'exact_pool_top1', label: 'Pool top-1', direction: 'higher' },
  { id: 'exact_pool_top3', label: 'Pool top-3', direction: 'higher' },
  { id: 'balanced_accuracy', label: 'Balanced acc.', direction: 'higher' },
  { id: 'macro_f1', label: 'Macro-F1', direction: 'higher' },
  { id: 'family_accuracy', label: 'Family acc.', direction: 'higher' },
  { id: 'ece', label: 'ECE', direction: 'lower' },
  { id: 'registry_valid_rate', label: 'Registry valid', direction: 'higher' },
];

const scoredMetricRank = (row: MetricRow, groupId: string) => {
  let rank = 0;
  if (row.value !== null && row.value !== undefined) rank += 100;
  if (row.status === 'scored') rank += 40;
  if (row.condition === 'tag_withheld') rank += 20;
  if (row.condition === 'mixed') rank += 10;
  if ((groupId === 'diffusion' || groupId === 'context-diffusion' || groupId === 'musicgen') && row.evidence_lane === 'native') rank += 30;
  if ((groupId === 'baseline' || groupId === 'retrieval') && row.evidence_lane === 'external_probe') rank += 15;
  if (row.status === 'missing_predictions') rank -= 20;
  return rank;
};

export const BenchmarksPage: React.FC = () => {
  const [initialBenchmarksCache] = useState<BenchmarksSessionCache | null>(() => readBenchmarksSessionCache());
  const [payload, setPayload] = useState<BenchmarksPayload | null>(() => initialBenchmarksCache?.payload ?? null);
  const [cacheLoadedAt, setCacheLoadedAt] = useState<string | null>(() => initialBenchmarksCache?.cachedAt ?? null);
  const [loading, setLoading] = useState<boolean>(false);
  const [summaryLoading, setSummaryLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  const loadBenchmarks = async () => {
    setLoading(true);
    setError(null);
    try {
      const nextPayload = await fetchJson<BenchmarksPayload>('/api/evaluation/benchmarks');
      const cachedAt = new Date().toISOString();
      setPayload(nextPayload);
      setCacheLoadedAt(cachedAt);
      writeBenchmarksSessionCache(nextPayload, cachedAt);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  const refreshSummary = async () => {
    setSummaryLoading(true);
    setSummaryError(null);
    try {
      const summary = await fetchJson<OpenAISummaryPayload['latest']>('/api/evaluation/benchmark-summary', {
        method: 'POST',
      });
      const nextPayload = payload ? {
        ...payload,
        openai_summary: {
          ...(payload.openai_summary ?? { configured: true }),
          configured: true,
          latest: summary,
        },
      } : null;
      setPayload((current) => current ? {
        ...current,
        openai_summary: {
          ...(current.openai_summary ?? { configured: true }),
          configured: true,
          latest: summary,
        },
      } : current);
      if (nextPayload) {
        const cachedAt = new Date().toISOString();
        setCacheLoadedAt(cachedAt);
        writeBenchmarksSessionCache(nextPayload, cachedAt);
      }
    } catch (err) {
      setSummaryError(err instanceof Error ? err.message : String(err));
    } finally {
      setSummaryLoading(false);
    }
  };

  useEffect(() => {
    if (!initialBenchmarksCache?.payload) void loadBenchmarks();
  }, []);

  const lanesByGroup = useMemo(() => {
    const lanes = payload?.model_lanes ?? [];
    return Object.fromEntries(
      laneGroups.map((group) => {
        const exact = lanes.find((lane) => group.modelIds.includes(lane.model_id));
        const byVariant = lanes.find(
          (lane) => group.variants.includes(lane.variant) && (!('family' in group) || lane.family === group.family),
        );
        return [group.id, exact ?? byVariant ?? null];
      }),
    ) as Record<string, ModelLane | null>;
  }, [payload?.model_lanes]);

  const metricFor = (group: (typeof laneGroups)[number], metricId: string) => {
    const lane = lanesByGroup[group.id];
    const rows = (payload?.metric_rows ?? []).filter((row) => {
      if (row.metric_id !== metricId) return false;
      if (lane) return row.model_id === lane.model_id;
      return group.modelIds.includes(row.model_id) || group.variants.includes(row.variant);
    });
    return (
      rows
        .slice()
        .sort((left, right) => scoredMetricRank(right, group.id) - scoredMetricRank(left, group.id))[0] ??
      null
    );
  };

  const readyLaneCount = (payload?.model_lanes ?? []).filter((lane) => statusClass(lane.status) === 'status-done').length;
  const blockedLaneCount = (payload?.model_lanes ?? []).filter((lane) => statusClass(lane.status) === 'status-error').length;
  const latestAudio = payload?.latest_generated_audio_result;
  const latestSmoke = payload?.latest_generated_audio_smoke_result ?? (latestAudio?.scope === 'smoke' ? latestAudio : null);
  const latestFull = payload?.latest_generated_audio_full_result ?? (latestAudio?.scope === 'full' ? latestAudio : null);
  const currentFull = isSameOrAfter(latestFull, latestSmoke) ? latestFull : null;
  const fullAudioComplete = Boolean(payload?.latest_results.audio_complete);
  const fullAudioProgress = payload?.latest_results.audio_progress_percent;
  const latestScore = currentFull?.job_name && payload?.latest_attribution_scoring_result?.source_audio_job_name
    ? payload.latest_attribution_scoring_result.source_audio_job_name === currentFull.job_name
      ? payload.latest_attribution_scoring_result
      : null
    : currentFull
      ? payload?.latest_attribution_scoring_result
      : null;
  const resultsStatus = payload?.latest_results.metrics_available
    ? payload.latest_results.native_attribution_status === 'missing_predictions' ||
      payload.latest_results.external_probe_status === 'missing_predictions'
      ? 'Scored: missing predictions'
      : 'Benchmark metrics ready'
    : currentFull && !fullAudioComplete
      ? `Full audio running${typeof fullAudioProgress === 'number' ? ` · ${fullAudioProgress.toFixed(1)}%` : ''}`
    : currentFull
      ? 'Full audio ready'
      : latestSmoke
        ? 'Full pending'
      : 'Waiting for benchmark run';
  const nextAction = payload?.active_generated_audio_job
    ? 'Generation running'
    : payload?.active_attribution_scoring_job
      ? 'Scoring running'
    : payload?.latest_results.metrics_available
      ? 'Review results'
        : !latestSmoke
          ? 'Run smoke test'
          : !currentFull
            ? 'Run full benchmark'
            : !fullAudioComplete
              ? 'Wait for full audio'
            : latestScore
              ? 'Review scoring'
	      : 'Run attribution scoring';
  const repairabilityMatrix = payload?.repairability_matrix;
  const repairabilityLanes = (repairabilityMatrix?.lanes ?? []).filter((laneId) =>
    ['diffusion_native', 'context_diffusion_native', 'musicgen_native', 'retrieval_native'].includes(laneId),
  );
  const repairMethodMatrix = payload?.repair_method_matrix;
  const repairMethodLanes = (repairMethodMatrix?.lanes ?? []).filter((laneId) =>
    ['diffusion_native', 'context_diffusion_native', 'musicgen_native', 'retrieval_native'].includes(laneId),
  );
  const laneMetrics = payload?.latest_results.latest_metrics?.lanes ?? {};
  const mechanicalLanes = ['diffusion_native', 'context_diffusion_native', 'musicgen_native', 'retrieval_native'].filter(
    (laneId) => laneMetrics[laneId],
  );
  const mechanicalRows = repairTierIds.map((tierId) => {
    const row: RepairabilityRow = { tier: tierId, label: repairTierLabels[tierId] ?? tierId };
    mechanicalLanes.forEach((laneId) => {
      const metrics = laneMetrics[laneId];
      const counts =
        metrics.resolution_tier_counts ??
        metrics.repairability?.tier_counts ??
        metrics.tier_counts ??
        {};
      const total =
        metrics.count ??
        Object.values(counts).reduce((sum, value) => sum + Number(value ?? 0), 0);
      const count = Number(counts[tierId] ?? 0);
      row[laneId] = {
        status: metrics.status ?? (total ? 'scored' : 'pending'),
        count,
        labelled_count: total,
        rate: total ? count / total : null,
      };
    });
    return row;
  });
  const predictionExamples = Object.entries(payload?.prediction_examples ?? {})
    .filter(([, examples]) => examples.length)
    .flatMap(([laneId, examples]) => examples.map((example) => ({ laneId, ...example })))
    .slice(0, 8);
  const heldoutTrainingEvaluation = payload?.latest_results.latest_metrics?.heldout_training_evaluation ?? {};
  const heldoutRows = [
    {
      id: 'diffusion_cara_strong_full_modest_arch',
      label: 'Diffusion CARA',
      evidence: heldoutTrainingEvaluation.diffusion_cara_strong_full_modest_arch,
    },
    {
      id: 'context_diffusion_cara_strong_full',
      label: 'Context Diffusion',
      evidence: heldoutTrainingEvaluation.context_diffusion_cara_strong_full,
    },
  ].filter((row) => row.evidence);
  const summaryPayload = payload?.openai_summary;
  const latestSummary = summaryPayload?.latest;

  return (
    <div style={{ display: 'grid', gap: 20 }}>
      <PageHeader
        kicker="Evaluation · Benchmarks"
        title={
          <>
            CARA-Strong <em>benchmark matrix</em>
          </>
        }
        description={
          <>
            One locked prompt set, one scoring contract, and side-by-side numbers for Diffusion CARA,
            Context Diffusion, MusicGen CARA, and retrieval controls.
          </>
        }
        actions={
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            <span className="dim mono" style={{ fontSize: 12 }}>
              {cacheLoadedAt ? `cached ${new Date(cacheLoadedAt).toLocaleString()}` : 'not fetched this session'}
            </span>
            <button className="btn btn-ghost" onClick={loadBenchmarks} disabled={loading} type="button">
              <RefreshCw size={16} /> {loading ? 'Refreshing' : 'Refresh Benchmarks'}
            </button>
          </div>
        }
      />

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="pool-summary-grid" aria-label="Benchmark headline status">
        <div className={`pool-metric-card tone-${payload?.benchmark_prompt_set?.locked ? 'good' : 'warn'}`}>
          <div className="pool-metric-top">Prompt set</div>
          <div className="pool-metric-value">{payload?.benchmark_spec?.prompt_set_version ?? 'v2'}</div>
          <div className="pool-metric-meta">{payload?.benchmark_prompt_set?.locked ? 'locked manifest' : 'lock before scoring'}</div>
        </div>
        <div className={`pool-metric-card tone-${latestSmoke ? 'good' : 'warn'}`}>
          <div className="pool-metric-top">Smoke gate</div>
          <div className="pool-metric-value" style={{ fontSize: 24 }}>{latestSmoke ? gateLabel(latestSmoke) : 'Pending'}</div>
          <div className="pool-metric-meta">{latestSmoke?.job_name ?? 'run smoke before full'}</div>
        </div>
        <div className={`pool-metric-card tone-${currentFull ? 'good' : latestSmoke ? 'warn' : 'bad'}`}>
          <div className="pool-metric-top">Full audio</div>
          <div className="pool-metric-value" style={{ fontSize: 22 }}>{resultsStatus}</div>
          <div className="pool-metric-meta">{currentFull?.job_name ?? (latestSmoke ? 'launch full locked set next' : 'blocked by smoke')}</div>
        </div>
        <div className="pool-metric-card">
          <div className="pool-metric-top">Next action</div>
          <div className="pool-metric-value" style={{ fontSize: 22 }}>{nextAction}</div>
          <div className="pool-metric-meta">{readyLaneCount}/{payload?.model_lanes?.length ?? 0} lanes ready · {blockedLaneCount} blocked</div>
        </div>
      </section>

      {payload?.active_generated_audio_job || payload?.active_attribution_scoring_job ? (
        <section className="card">
          <div className="card-header">
            <div className="card-title">Active benchmark job</div>
            <div className="card-meta">
              {payload.active_generated_audio_job?.status ?? payload.active_attribution_scoring_job?.status ?? 'running'}
            </div>
          </div>
          <div className="metric-list">
            <div>
              <span>Job</span>
              <strong className="mono">{payload.active_generated_audio_job?.job_name ?? payload.active_attribution_scoring_job?.job_name ?? 'unknown'}</strong>
            </div>
            <div>
              <span>Stage</span>
              <strong>{payload.active_generated_audio_job ? 'Generated audio' : 'Attribution scoring'}</strong>
            </div>
            <div>
              <span>Output</span>
              <strong className="mono">{payload.active_generated_audio_job?.output_path ?? payload.active_attribution_scoring_job?.output_path ?? 'pending'}</strong>
            </div>
          </div>
        </section>
      ) : null}

      <section className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Headline comparison matrix</div>
            <div className="dim" style={{ marginTop: 4, fontSize: 13 }}>
              Primary read: tag-withheld pool attribution. Pending cells mean the model is registered but not yet scored for that metric.
            </div>
          </div>
          <div className="card-meta">{payload?.latest_results.metric_stage ?? payload?.format ?? 'loading'}</div>
        </div>
        <div className="table-scroll">
          <div className="run-table" style={{ minWidth: 980 }}>
            <div className="run-row run-head" style={{ gridTemplateColumns: '1.15fr repeat(4, minmax(135px, 1fr)) 0.65fr' }}>
              <span>Metric</span>
              {laneGroups.map((group) => (
                <span key={group.id}>
                  {group.title}
                  <span className="dim" style={{ display: 'block', fontSize: 10, letterSpacing: 0 }}>{group.subtitle}</span>
                </span>
              ))}
              <span>Goal</span>
            </div>
            {headlineMetricIds.map((metric) => (
              <div className="run-row" key={metric.id} style={{ gridTemplateColumns: '1.15fr repeat(4, minmax(135px, 1fr)) 0.65fr' }}>
                <span>{metric.label}</span>
                {laneGroups.map((group) => {
                  const row = metricFor(group, metric.id);
                  return (
                    <span key={`${group.id}-${metric.id}`}>
                      <span className="mono">{fmtCell(row?.value, row?.status)}</span>
                      {row?.ci_low !== null && row?.ci_low !== undefined && row?.ci_high !== null && row?.ci_high !== undefined ? (
                        <span className="dim mono" style={{ display: 'block', fontSize: 11 }}>
                          {fmt(row.ci_low)} to {fmt(row.ci_high)}
                        </span>
                      ) : null}
                    </span>
                  );
                })}
                <span className="mono dim">{metric.direction}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div>
            <div className="card-title">OpenAI TLDR analysis</div>
            <div className="dim" style={{ marginTop: 4, fontSize: 13 }}>
              Cached abstract generated from the current benchmark matrices, repairability ladder, repair-method audit, and prediction examples.
            </div>
          </div>
          <div className="card-meta">{latestSummary?.created_at ? `cached ${latestSummary.created_at}` : summaryPayload?.configured ? 'no cache yet' : 'OpenAI not configured'}</div>
        </div>
        <div className="metric-list">
          <div>
            <span>Summary</span>
            <strong style={{ whiteSpace: 'pre-wrap' }}>
              {latestSummary?.summary ?? (summaryPayload?.configured
                ? 'No cached TLDR yet. Refresh to generate one with the OpenAI API credentials from .env.'
                : summaryPayload?.configuration_message ?? 'OpenAI TLDR generation is not configured.')}
            </strong>
          </div>
          <div>
            <span>Artifact</span>
            <strong className="mono">{summaryPayload?.latest_path ?? 'evaluation/generated/benchmark_tldr_latest.json'}</strong>
          </div>
          {summaryError ? (
            <div>
              <span>Error</span>
              <strong className="status-error">{summaryError}</strong>
            </div>
          ) : null}
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 14 }}>
          <button
            className="btn btn-ghost"
            onClick={refreshSummary}
            disabled={summaryLoading || !summaryPayload?.configured}
            type="button"
            title={summaryPayload?.configured ? 'Refresh cached OpenAI benchmark TLDR' : summaryPayload?.configuration_message ?? 'Configure OpenAI TLDR generation first'}
          >
            <RefreshCw size={16} /> {summaryLoading ? 'Refreshing TLDR' : 'Refresh TLDR'}
          </button>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Matrix status legend</div>
            <div className="dim" style={{ marginTop: 4, fontSize: 13 }}>
              These labels distinguish absent metrics from scored failures, so an empty cell is not mistaken for evidence.
            </div>
          </div>
          <div className="card-meta">read before headline claims</div>
        </div>
        <div className="metric-list">
          {statusMeaning.map(([label, meaning]) => (
            <div key={label}>
              <span className="mono">{label}</span>
              <strong>{meaning}</strong>
            </div>
          ))}
        </div>
      </section>

      {heldoutRows.length ? (
        <section className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Held-out prepared-audio evidence</div>
              <div className="dim" style={{ marginTop: 4, fontSize: 13 }}>
                Trainer validation/test metrics from prepared chunks with known CARA labels. This is separate from generated-audio repairability.
              </div>
            </div>
            <div className="card-meta">training artifact</div>
          </div>
          <div className="table-scroll">
            <div className="run-table" style={{ minWidth: 880 }}>
              <div className="run-row run-head" style={{ gridTemplateColumns: '1fr 0.8fr repeat(4, minmax(115px, 1fr))' }}>
                <span>Model</span>
                <span>Status</span>
                <span>Validation pool</span>
                <span>Validation family</span>
                <span>Test pool</span>
                <span>Test family</span>
              </div>
              {heldoutRows.map((row) => {
                const validation = row.evidence?.heldout_evaluation?.validation;
                const test = row.evidence?.heldout_evaluation?.test;
                return (
                  <div className="run-row" key={row.id} style={{ gridTemplateColumns: '1fr 0.8fr repeat(4, minmax(115px, 1fr))' }}>
                    <span>
                      {row.label}
                      <span className="dim mono" style={{ display: 'block', fontSize: 11 }}>
                        step {fmtShort(row.evidence?.global_step)}
                      </span>
                    </span>
                    <span className={`mono ${statusClass(row.evidence?.status)}`}>{row.evidence?.status ?? 'missing'}</span>
                    <span className="mono">{fmtMetric(validation?.metrics?.['cara/pool_top1'])}</span>
                    <span className="mono">{fmtMetric(validation?.metrics?.['cara/family_top1'])}</span>
                    <span className="mono">{fmtMetric(test?.metrics?.['cara/pool_top1'])}</span>
                    <span className="mono">{fmtMetric(test?.metrics?.['cara/family_top1'])}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </section>
      ) : null}

      <section className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Repairability ladder matrix</div>
            <div className="dim" style={{ marginTop: 4, fontSize: 13 }}>
              Strict accuracy ladder: exact, repairable, and family rows count only correct recovery against the expected held-out label. Unattributable includes wrong, missing, or unrepairable predictions.
            </div>
          </div>
          <div className="card-meta">native and probe lanes</div>
        </div>
        <div className="table-scroll">
          <div className="run-table" style={{ minWidth: 900 }}>
            <div className="run-row run-head" style={{ gridTemplateColumns: `1.1fr repeat(${Math.max(1, repairabilityLanes.length)}, minmax(145px, 1fr))` }}>
              <span>Tier</span>
              {repairabilityLanes.length ? repairabilityLanes.map((laneId) => <span key={laneId}>{laneDisplay(laneId)}</span>) : <span>Scoring pending</span>}
            </div>
            {(repairabilityMatrix?.rows ?? []).map((row) => (
              <div className="run-row" key={String(row.tier)} style={{ gridTemplateColumns: `1.1fr repeat(${Math.max(1, repairabilityLanes.length)}, minmax(145px, 1fr))` }}>
                <span>{String(row.label)}</span>
                {repairabilityLanes.length ? repairabilityLanes.map((laneId) => (
                  <span key={`${row.tier}-${laneId}`} className="mono">{fmtCountRate(row[laneId] as RepairabilityCell)}</span>
                )) : <span className="mono">pending</span>}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Mechanical resolution matrix</div>
            <div className="dim" style={{ marginTop: 4, fontSize: 13 }}>
              Diagnostic ladder: shows whether emitted CARA strings resolve to any registry pool before checking whether that pool is the expected held-out target.
            </div>
          </div>
          <div className="card-meta">candidate influence signal</div>
        </div>
        <div className="table-scroll">
          <div className="run-table" style={{ minWidth: 900 }}>
            <div className="run-row run-head" style={{ gridTemplateColumns: `1.1fr repeat(${Math.max(1, mechanicalLanes.length)}, minmax(145px, 1fr))` }}>
              <span>Resolution</span>
              {mechanicalLanes.length ? mechanicalLanes.map((laneId) => <span key={laneId}>{laneDisplay(laneId)}</span>) : <span>Scoring pending</span>}
            </div>
            {mechanicalRows.map((row) => (
              <div className="run-row" key={String(row.tier)} style={{ gridTemplateColumns: `1.1fr repeat(${Math.max(1, mechanicalLanes.length)}, minmax(145px, 1fr))` }}>
                <span>{String(row.label)}</span>
                {mechanicalLanes.length ? mechanicalLanes.map((laneId) => (
                  <span key={`${row.tier}-${laneId}`} className="mono">{fmtCountRate(row[laneId] as RepairabilityCell)}</span>
                )) : <span className="mono">pending</span>}
              </div>
            ))}
          </div>
        </div>
        <div className="info-banner" style={{ marginTop: 14 }}>
          Mechanical resolution is diagnostic, not accuracy. For Diffusion and Context Diffusion, the native head is a closed-set classifier over registry pools, so exact registry output is expected whenever the extractor succeeds. For MusicGen, this table is stricter evidence that the emitted suffix string decoded or repaired into the registry. The strict ladder above is the target-correct benchmark.
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Repair method matrix</div>
            <div className="dim" style={{ marginTop: 4, fontSize: 13 }}>
              Counts how predictions were resolved mechanically. Checksum repair fixes same-payload codewords with a bad check digit; unique payload edit-distance repair is capped at distance 2; family fallback is capped at distance 4.
            </div>
          </div>
          <div className="card-meta">resolution audit</div>
        </div>
        <div className="table-scroll">
          <div className="run-table" style={{ minWidth: 900 }}>
            <div className="run-row run-head" style={{ gridTemplateColumns: `1.1fr repeat(${Math.max(1, repairMethodLanes.length)}, minmax(145px, 1fr))` }}>
              <span>Method</span>
              {repairMethodLanes.length ? repairMethodLanes.map((laneId) => <span key={laneId}>{laneDisplay(laneId)}</span>) : <span>Scoring pending</span>}
            </div>
            {(repairMethodMatrix?.rows ?? []).map((row) => (
              <div className="run-row" key={String(row.method)} style={{ gridTemplateColumns: `1.1fr repeat(${Math.max(1, repairMethodLanes.length)}, minmax(145px, 1fr))` }}>
                <span>{String(row.label)}</span>
                {repairMethodLanes.length ? repairMethodLanes.map((laneId) => (
                  <span key={`${row.method}-${laneId}`} className="mono">{fmtCountRate(row[laneId] as RepairabilityCell)}</span>
                )) : <span className="mono">pending</span>}
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Prediction examples</div>
            <div className="dim" style={{ marginTop: 4, fontSize: 13 }}>
              These are native/probe outputs written by the scorer, not copied from expected labels. Top candidates appear only for scorers that emit top-k pool probabilities.
            </div>
          </div>
          <div className="card-meta">{predictionExamples.length ? `${predictionExamples.length} shown` : 'waiting for predictions'}</div>
        </div>
        {predictionExamples.length ? (
          <div className="metric-list">
            {predictionExamples.map((example, index) => (
              <div key={`${example.laneId}-${example.prompt_id}-${index}`}>
                <span>
                  {laneDisplay(example.laneId)} · {correctnessTierForExample(example)}
                  <span className="dim" style={{ display: 'block', fontSize: 11 }}>
                    mechanical {mechanicalTierLabel(example.tier)}
                  </span>
                </span>
                <strong>
                  <span className="mono">predicted {fmtShort(example.predicted_pool_id)}</span>
                  <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                    resolved {fmtShort(example.resolved_pool_id)} · expected {fmtShort(example.expected_pool_id)} · confidence {fmt(example.confidence)}
                  </span>
                  <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                    exact {fmtBool(example.exact)} · repair-correct {fmtBool(example.repairable)} · family-correct {fmtBool(example.family_match)} · registry-valid {fmtBool(example.registry_valid)}
                  </span>
                  <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                    method {example.repair_method ?? 'none'} · distance {example.repair_distance ?? 'n/a'} · family {example.resolved_family ?? example.predicted_family ?? 'none'}
                  </span>
                  {example.prediction_error ? (
                    <span className="status-error" style={{ display: 'block', fontSize: 12 }}>
                      error {example.prediction_error}
                    </span>
                  ) : null}
                  {fmtFeatureAlignment(example.feature_alignment) ? (
                    <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                      extractor alignment {fmtFeatureAlignment(example.feature_alignment)}
                    </span>
                  ) : null}
                  <span className="dim mono" style={{ display: 'block', fontSize: 11 }}>
                    {example.prompt_id ?? 'prompt'} · {example.audio_path ?? 'audio path pending'}
                  </span>
                  <span className="dim mono" style={{ display: 'block', fontSize: 11 }}>
                    top pools: {(example.top_k ?? []).slice(0, 3).length
                      ? (example.top_k ?? [])
                        .slice(0, 3)
                        .map((candidate) => `${candidate.rank ?? '?'}:${candidate.cara_pool_id ?? 'unknown'} ${fmt(candidate.confidence)}`)
                        .join(' · ')
                      : 'not emitted by this scorer'}
                  </span>
                </strong>
              </div>
            ))}
          </div>
        ) : (
          <div className="info-banner">
            No prediction examples are available yet. Run attribution scoring after the generated-audio benchmark; the scorer should write native_predictions.jsonl and scored_generation_manifest.jsonl.
          </div>
        )}
      </section>

      {(payload?.comparison_cards ?? []).length ? (
        <section className="pool-summary-grid" aria-label="Benchmark pairwise outcomes">
          {(payload?.comparison_cards ?? []).slice(0, 4).map((card) => (
            <div className="pool-metric-card" key={card.id}>
              <div className="pool-metric-top">{card.title}</div>
              <div className="pool-metric-value" style={{ fontSize: 28 }}>{fmtDelta(card.delta)}</div>
              <div className="pool-metric-meta">
                {card.metric_id.replace(/_/g, ' ')} · {card.condition.replace(/_/g, ' ')} · {card.status.replace(/_/g, ' ')}
              </div>
            </div>
          ))}
        </section>
      ) : null}

      <section className="card">
        <div className="card-header">
          <div className="card-title">Model readiness</div>
          <div className="card-meta">clear blockers only</div>
        </div>
        <div className="table-scroll">
          <div className="run-table" style={{ minWidth: 840 }}>
            <div className="run-row run-head" style={{ gridTemplateColumns: '1.25fr 0.9fr 0.9fr 1.05fr' }}>
              <span>Lane</span>
              <span>Role</span>
              <span>Architecture</span>
              <span>Outcome</span>
            </div>
            {laneGroups.map((group) => {
              const lane = lanesByGroup[group.id];
              return (
                <div className="run-row" key={group.id} style={{ gridTemplateColumns: '1.25fr 0.9fr 0.9fr 1.05fr' }}>
                  <span>{lane?.label ?? group.title}</span>
                  <span>{group.subtitle}</span>
                  <span>{lane?.architecture?.replace(/_/g, ' ') ?? 'pending registration'}</span>
                  <span>
                    <span className={`status-pill ${statusClass(lane?.status)}`}>{lane?.status ?? 'Blocked: missing registry lane'}</span>
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">Evidence contract</div>
          <div className="card-meta">plain-language claim</div>
        </div>
        <div className="metric-list">
          <div>
            <span>Claim</span>
            <strong>{payload?.benchmark_spec?.claim_language ?? 'Recoverable, confidence-scored pool-level attribution; not individual-track causality.'}</strong>
          </div>
          <div>
            <span>Prompt manifest</span>
            <strong>{payload?.benchmark_prompt_set?.locked ? 'Locked and reused across all lanes' : 'Must be locked before final comparisons'}</strong>
          </div>
          <div>
            <span>Latest smoke</span>
            <strong>{latestSmoke?.job_name ? `${gateLabel(latestSmoke)} · ${latestSmoke.job_name}` : 'not run yet'}</strong>
          </div>
          <div>
            <span>Current full audio</span>
            <strong>{currentFull?.job_name ? `${gateLabel(currentFull)} · ${currentFull.job_name}` : latestSmoke ? 'ready to launch after latest smoke' : 'not run yet'}</strong>
          </div>
          <div>
            <span>Last scoring run</span>
            <strong>{latestScore?.job_name ? `${latestScore.job_name} · ${latestScore.status ?? 'unknown'}` : 'not run yet'}</strong>
          </div>
        </div>
      </section>
    </div>
  );
};
