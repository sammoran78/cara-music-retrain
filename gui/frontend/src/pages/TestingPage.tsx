import React, { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, CheckCircle2, ListChecks, LockKeyhole, Play, RefreshCw, ShieldCheck } from 'lucide-react';
import { PageHeader } from './PageHeader';

interface SuiteOption {
  id: string;
  label: string;
  description: string;
  prompt_count: number;
  evidence_type: string;
}

interface RepairTier {
  id: string;
  label: string;
  description: string;
  counts_as_pool_success: boolean;
}

interface BenchmarkPromptSet {
  format?: string;
  locked: boolean;
  prompt_manifest_uri?: string | null;
  source_job_name?: string | null;
  can_lock?: boolean;
  reason?: string;
}

interface BenchmarkSpec {
  prompt_set_version: string;
  registry_hash?: string | null;
  manifest_hash?: string | null;
  claim_language?: string;
}

interface ModelLane {
  model_id: string;
  label: string;
  family: string;
  architecture: string;
  variant: string;
  generation_adapter?: string | null;
  native_prediction_adapter?: string | null;
  status: string;
  artifact_checks?: Array<{ status: string; path?: string | null; required?: boolean }>;
}

interface EvaluationJobState {
  job_name?: string | null;
  created_at?: string | null;
  action?: string | null;
  status?: string | null;
  active?: boolean;
  stage_label?: string;
  message?: string;
  studio_url?: string | null;
  output_path?: string | null;
  scope?: string | null;
  max_prompts?: number | null;
  metrics_uri?: string | null;
  source_audio_job_name?: string | null;
  generated_audio_output_path?: string | null;
  generated_audio_output_paths?: Record<string, string | null>;
  generation_manifest_uri?: string | null;
  model_ids?: string[];
}

interface EvaluationReadiness {
  format: string;
  benchmark_spec?: BenchmarkSpec;
  model_lanes?: ModelLane[];
  registry: {
    pool_count: number;
    family_count: number;
  };
  suites: SuiteOption[];
  repairability: {
    tiers: RepairTier[];
  };
  latest_results: {
    metrics_available: boolean;
    metrics_path: string;
  };
  launch_guard: {
    dry_run_default: boolean;
    cost_policy: string;
  };
  latest_evaluation_job?: EvaluationJobState | null;
  active_generated_audio_job?: EvaluationJobState | null;
  latest_generated_audio_result?: EvaluationJobState | null;
  latest_generated_audio_smoke_result?: EvaluationJobState | null;
  latest_generated_audio_full_result?: EvaluationJobState | null;
  active_attribution_scoring_job?: EvaluationJobState | null;
  latest_attribution_scoring_result?: EvaluationJobState | null;
  benchmark_prompt_set?: BenchmarkPromptSet;
}

interface AudioBenchmarkPlanResponse {
  status: string;
  dry_run?: boolean;
  message?: string;
  job?: {
    name?: string | null;
    status?: string | null;
    studio_url?: string | null;
    output_path?: string | null;
  };
  plan?: {
    scope: string;
    model_ids: string[];
    model_groups?: Record<string, string[]>;
    suite_ids: string[];
    seed_ids: number[];
    max_prompts: number;
    estimated_generations: number | string;
    live_ready: boolean;
    live_ready_reason: string;
    prompt_manifest_uri?: string | null;
    output_prefix?: string | Record<string, string>;
    audio_output_policy?: string;
    metrics_policy?: string;
    cost_policy?: string;
  };
  jobs?: Array<{
    family?: string;
    name?: string | null;
    status?: string | null;
    studio_url?: string | null;
    output_path?: string | null;
  }>;
}

interface AttributionScoringPlanResponse {
  status: string;
  dry_run?: boolean;
  message?: string;
  job?: {
    name?: string | null;
    status?: string | null;
    studio_url?: string | null;
    output_path?: string | null;
    metrics_uri?: string | null;
  };
  plan?: {
    audio_job_name?: string | null;
    model_ids?: string[];
    source_model_ids?: string[];
    selected_families?: string[];
    generated_audio_output_path?: string | null;
    generated_audio_output_paths?: Record<string, string | null>;
    pending_score_output_paths?: Record<string, string | null>;
    stable_audio_trained_model_data?: string | null;
    context_trained_model_data?: string | null;
    generation_manifest_uri?: string | null;
    live_ready: boolean;
    live_ready_reason: string;
    force_rescore?: boolean;
    metrics_policy: string;
    cost_policy?: string;
  };
}

interface AudioProgressRow {
  model_id?: string;
  suite_id?: string;
  completed: number;
  planned?: number | null;
  percent?: number | null;
}

interface AudioBenchmarkProgress {
  checked_at: string;
  method: string;
  job: EvaluationJobState;
  scope?: string | null;
  model_ids: string[];
  suite_ids: string[];
  seed_ids: Array<number | string>;
  max_prompts: number;
  planned_generations?: number | null;
  completed_generations: number;
  progress_percent: number;
  model_progress: AudioProgressRow[];
  suite_progress: AudioProgressRow[];
  by_model_suite: Record<string, Record<string, number>>;
  manifest_available: boolean;
  metrics_available: boolean;
  report_available: boolean;
  latest_completed_item?: {
    model_id?: string | null;
    suite_id?: string | null;
    file?: string | null;
    blob?: string | null;
    completed_at?: string | null;
  } | null;
  latest_blob_at?: string | null;
  blob_error?: string | null;
  note?: string;
}

const LIVE_WAVE_1_MODEL_IDS = [
  'diffusion_cara_strong_full_modest_arch',
  'context_diffusion_cara_strong_full',
  'musicgen_cara_strong_full',
];
const LIVE_WAVE_1_SUITE_IDS = [
  'heldout_audio_attribution',
  'known_pool_prompt_recall',
  'control_token_confound',
  'baseline_negative_control',
];
const AUDIO_BENCHMARK_CONFIRMATION = 'LAUNCH AUDIO BENCHMARK';
const ATTRIBUTION_SCORING_CONFIRMATION = 'LAUNCH ATTRIBUTION SCORING';

const fetchJson = async <T,>(url: string, init?: RequestInit): Promise<T> => {
  const response = await fetch(url, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail || body));
  return body as T;
};

const statusClass = (status?: string | null) => {
  const normalized = (status ?? '').toLowerCase();
  if (normalized.includes('ready') || normalized.includes('complete') || normalized.includes('scored') || normalized.includes('available')) {
    return 'status-done';
  }
  if (normalized.includes('blocked') || normalized.includes('missing') || normalized.includes('failed')) return 'status-error';
  if (normalized.includes('pending') || normalized.includes('not run')) return 'status-queued';
  return 'status-running';
};

const stageStatus = (done: boolean, running: boolean, blocked: boolean) => {
  if (running) return 'Running';
  if (done) return 'Ready';
  if (blocked) return 'Blocked';
  return 'Pending';
};

const gateLabel = (job?: EvaluationJobState | null) => {
  const normalized = String(job?.status ?? '').toLowerCase();
  if (!job) return 'Pending';
  if (normalized === 'completed' || normalized === 'succeeded') return 'Passed';
  if (normalized === 'recorded') return 'Recorded';
  return job.status ?? 'Recorded';
};

const formatList = (values?: Array<string | number> | string | number | null) => {
  if (Array.isArray(values)) return values.length ? values.join(', ') : 'none';
  if (values === null || values === undefined || values === '') return 'none';
  return String(values);
};

const formatOutputPrefix = (value?: string | Record<string, string | null>) => {
  if (!value) return 'assigned on live submit';
  if (typeof value === 'string') return value;
  return Object.entries(value)
    .filter(([, uri]) => Boolean(uri))
    .map(([family, uri]) => `${family}: ${uri}`)
    .join(' | ');
};

const generatedAudioOutputPath = (job?: EvaluationJobState | null) => {
  if (!job) return null;
  const paths = job.generated_audio_output_paths ?? {};
  return job.output_path ?? paths.stable_audio ?? paths.musicgen ?? null;
};

const generationManifestText = (job?: EvaluationJobState | null) => {
  if (!job) return null;
  if (job.generation_manifest_uri) return job.generation_manifest_uri;
  const paths = job.generated_audio_output_paths ?? {};
  const manifestUris = Object.entries(paths)
    .filter(([, uri]) => Boolean(uri))
    .map(([family, uri]) => `${family}: ${String(uri).replace(/\/$/, '')}/generation_manifest.jsonl`);
  if (manifestUris.length) return manifestUris.join(' | ');
  const outputPath = generatedAudioOutputPath(job);
  return outputPath ? `${outputPath.replace(/\/$/, '')}/generation_manifest.jsonl` : null;
};

const PreflightCheck: React.FC<{ passed: boolean; label: string; detail?: React.ReactNode }> = ({ passed, label, detail }) => (
  <div className={`preflight-check ${passed ? 'is-passed' : 'is-blocked'}`}>
    <span className="preflight-icon">{passed ? <CheckCircle2 size={15} /> : <AlertTriangle size={15} />}</span>
    <span>
      <strong>{label}</strong>
      {detail ? <span className="dim">{detail}</span> : null}
    </span>
  </div>
);

const laneOrder = [
  'diffusion_cara_strong_full_modest_arch',
  'context_diffusion_cara_strong_full',
  'musicgen_cara_strong_full',
  'retrieval_baseline',
  'base_stable_audio_open_small',
  'stable_audio_no_cara_baseline',
  'base_musicgen_small',
  'musicgen_no_cara_baseline',
];

export const TestingPage: React.FC = () => {
  const [readiness, setReadiness] = useState<EvaluationReadiness | null>(null);
  const [audioScope, setAudioScope] = useState<'smoke' | 'full'>('smoke');
  const [audioModelIds, setAudioModelIds] = useState<string[]>(LIVE_WAVE_1_MODEL_IDS);
  const [audioSuiteIds, setAudioSuiteIds] = useState<string[]>(['known_pool_prompt_recall', 'control_token_confound']);
  const [audioMaxPrompts, setAudioMaxPrompts] = useState<number>(20);
  const [audioDryRun, setAudioDryRun] = useState<boolean>(true);
  const [audioConfirmation, setAudioConfirmation] = useState<string>('');
  const [scoreDryRun, setScoreDryRun] = useState<boolean>(true);
  const [scoreForceRescore, setScoreForceRescore] = useState<boolean>(false);
  const [scoreConfirmation, setScoreConfirmation] = useState<string>('');
  const [scoreModelIds, setScoreModelIds] = useState<string[]>(LIVE_WAVE_1_MODEL_IDS);
  const [loading, setLoading] = useState<boolean>(false);
  const [lockingPromptSet, setLockingPromptSet] = useState<boolean>(false);
  const [planningAudio, setPlanningAudio] = useState<boolean>(false);
  const [planningScore, setPlanningScore] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [audioRunResponse, setAudioRunResponse] = useState<AudioBenchmarkPlanResponse | null>(null);
  const [scoreRunResponse, setScoreRunResponse] = useState<AttributionScoringPlanResponse | null>(null);
  const [progressOpen, setProgressOpen] = useState<boolean>(false);
  const [progressJobName, setProgressJobName] = useState<string | null>(null);
  const [progress, setProgress] = useState<AudioBenchmarkProgress | null>(null);
  const [progressLoading, setProgressLoading] = useState<boolean>(false);
  const [progressError, setProgressError] = useState<string | null>(null);
  const [retryingMusicGen, setRetryingMusicGen] = useState<boolean>(false);

  const loadReadiness = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = await fetchJson<EvaluationReadiness>('/api/evaluation/readiness');
      setReadiness(payload);
      setAudioDryRun(payload.launch_guard.dry_run_default);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadReadiness();
  }, []);

  const loadAudioProgress = async (jobName?: string | null) => {
    setProgressLoading(true);
    setProgressError(null);
    try {
      const query = jobName ? `?job_name=${encodeURIComponent(jobName)}` : '';
      setProgress(await fetchJson<AudioBenchmarkProgress>(`/api/evaluation/audio-benchmark/progress${query}`));
    } catch (err) {
      setProgressError(err instanceof Error ? err.message : String(err));
    } finally {
      setProgressLoading(false);
    }
  };

  useEffect(() => {
    if (!progressOpen) return undefined;
    void loadAudioProgress(progressJobName);
    const timer = window.setInterval(() => {
      void loadAudioProgress(progressJobName);
    }, 15000);
    return () => window.clearInterval(timer);
  }, [progressOpen, progressJobName]);

  const promptSetLocked = Boolean(readiness?.benchmark_prompt_set?.locked);
  const activeAudioJob = readiness?.active_generated_audio_job ?? null;
  const activeAudioJobRunning = Boolean(activeAudioJob?.active);
  const latestAudioResult = readiness?.latest_generated_audio_result ?? null;
  const latestSmokeResult = readiness?.latest_generated_audio_smoke_result ?? (latestAudioResult?.scope === 'smoke' ? latestAudioResult : null);
  const latestFullResult = readiness?.latest_generated_audio_full_result ?? (latestAudioResult?.scope === 'full' ? latestAudioResult : null);
  const currentFullResult = latestFullResult;
  const activeScoreJob = readiness?.active_attribution_scoring_job ?? null;
  const activeScoreJobRunning = Boolean(activeScoreJob?.active);
  const latestScoreResult = readiness?.latest_attribution_scoring_result ?? null;
  const latestGenerationManifest = generationManifestText(currentFullResult);
  const audioSeedIds = [0];
  const fullScoreResult = currentFullResult
    ? currentFullResult.job_name && latestScoreResult?.source_audio_job_name
      ? latestScoreResult.source_audio_job_name === currentFullResult.job_name
        ? latestScoreResult
        : null
      : latestScoreResult
    : null;
  const scoreReady = Boolean(currentFullResult);
  const latestProgressJobName = activeAudioJob?.job_name ?? latestFullResult?.job_name ?? latestAudioResult?.job_name ?? latestSmokeResult?.job_name ?? null;

  const visibleLanes = useMemo(() => {
    const lanes = readiness?.model_lanes ?? [];
    return [...lanes].sort((a, b) => {
      const ai = laneOrder.indexOf(a.model_id);
      const bi = laneOrder.indexOf(b.model_id);
      return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
    });
  }, [readiness?.model_lanes]);
  const selectableAudioLaneIdSet = new Set(LIVE_WAVE_1_MODEL_IDS);
  const selectableAudioLanes = visibleLanes.filter((lane) => selectableAudioLaneIdSet.has(lane.model_id));
  const selectedAudioLaneLabels = selectableAudioLanes.filter((lane) => audioModelIds.includes(lane.model_id)).map((lane) => lane.label);
  const sourceScoreModelIds = (currentFullResult?.model_ids ?? []).filter((modelId) => selectableAudioLaneIdSet.has(modelId));
  const scoreSelectableLanes = sourceScoreModelIds.length
    ? selectableAudioLanes.filter((lane) => sourceScoreModelIds.includes(lane.model_id))
    : selectableAudioLanes;
  const selectedScoreModelIds = scoreModelIds.filter((modelId) => scoreSelectableLanes.some((lane) => lane.model_id === modelId));
  const selectedScoreLaneLabels = scoreSelectableLanes.filter((lane) => selectedScoreModelIds.includes(lane.model_id)).map((lane) => lane.label);
  const generatedAudioLaneIds = new Set(audioModelIds);
  const excludedGeneratedAudioLanes = visibleLanes.filter(
    (lane) => !generatedAudioLaneIds.has(lane.model_id) && lane.generation_adapter !== 'post_hoc',
  );
  const excludedGeneratedAudioText = excludedGeneratedAudioLanes.length
    ? excludedGeneratedAudioLanes.map((lane) => `${lane.label} (${lane.status})`).join(', ')
    : 'none';

  const readyLanes = visibleLanes.filter((lane) => statusClass(lane.status) === 'status-done').length;
  const blockedLanes = visibleLanes.filter((lane) => statusClass(lane.status) === 'status-error').length;
  const runway = [
    {
      label: '1. Register',
      status: stageStatus(readyLanes > 0 && blockedLanes === 0, false, blockedLanes > 0),
      note: `${readyLanes}/${visibleLanes.length} lanes ready`,
    },
    {
      label: '2. Lock prompts',
      status: stageStatus(promptSetLocked, false, !readiness?.benchmark_prompt_set?.can_lock && !promptSetLocked),
      note: readiness?.benchmark_spec?.prompt_set_version ?? 'Prompt Set v2',
    },
    {
      label: '3. Smoke',
      status: activeAudioJobRunning && activeAudioJob?.scope === 'smoke' ? 'Running' : latestSmokeResult ? gateLabel(latestSmokeResult) : stageStatus(false, false, !promptSetLocked),
      note: latestSmokeResult?.job_name ?? '20 prompt check',
    },
    {
      label: '4. Full audio',
      status: activeAudioJobRunning && activeAudioJob?.scope === 'full' ? 'Running' : currentFullResult ? gateLabel(currentFullResult) : stageStatus(false, false, !latestSmokeResult),
      note: currentFullResult?.job_name ?? (latestSmokeResult ? 'ready to launch' : 'smoke first'),
    },
    {
      label: '5. Score',
      status: stageStatus(Boolean(fullScoreResult), activeScoreJobRunning, !scoreReady),
      note: fullScoreResult?.job_name ?? 'native + probe predictions',
    },
    {
      label: '6. Compare',
      status: stageStatus(Boolean(readiness?.latest_results.metrics_available), false, !fullScoreResult),
      note: readiness?.latest_results.metrics_available ? 'matrix ready' : 'awaiting metrics',
    },
  ];

  const toggleSuite = (suiteId: string) => {
    setAudioConfirmation('');
    setAudioSuiteIds((current) => (current.includes(suiteId) ? current.filter((id) => id !== suiteId) : [...current, suiteId]));
  };
  const toggleAudioModel = (modelId: string) => {
    setAudioConfirmation('');
    setAudioModelIds((current) => (current.includes(modelId) ? current.filter((id) => id !== modelId) : [...current, modelId]));
  };
  const selectAudioModels = (modelIds: string[]) => {
    const allowed = new Set(LIVE_WAVE_1_MODEL_IDS);
    const deduped = Array.from(new Set(modelIds.filter((modelId) => allowed.has(modelId))));
    setAudioConfirmation('');
    setAudioModelIds(deduped);
  };
  const toggleScoreModel = (modelId: string) => {
    setScoreConfirmation('');
    setScoreModelIds((current) => (current.includes(modelId) ? current.filter((id) => id !== modelId) : [...current, modelId]));
  };
  const selectScoreModels = (modelIds: string[]) => {
    const allowed = new Set(scoreSelectableLanes.map((lane) => lane.model_id));
    const deduped = Array.from(new Set(modelIds.filter((modelId) => allowed.has(modelId))));
    setScoreConfirmation('');
    setScoreModelIds(deduped);
  };
  const audioConfirmationValid = audioConfirmation.trim() === AUDIO_BENCHMARK_CONFIRMATION;

  const audioLaunchProblems = !audioDryRun
    ? [
        ...(activeAudioJobRunning ? [`Generated-audio benchmark already running: ${activeAudioJob?.job_name ?? 'unknown job'}.`] : []),
        ...(!promptSetLocked ? ['Lock Benchmark Prompt Set v2 before generated-audio scoring.'] : []),
        ...(audioScope === 'smoke' && audioMaxPrompts <= 0 ? ['Audio smoke requires a positive prompt limit.'] : []),
        ...(audioModelIds.length === 0 ? ['Select at least one model lane.'] : []),
        ...(audioSuiteIds.length === 0 ? ['Select at least one locked benchmark suite.'] : []),
        ...(!audioConfirmationValid ? [`Type ${AUDIO_BENCHMARK_CONFIRMATION} before live audio launch.`] : []),
      ]
    : [];
  const audioLaunchDisabled =
    planningAudio ||
    !promptSetLocked ||
    audioModelIds.length === 0 ||
    audioSuiteIds.length === 0 ||
    (audioScope === 'smoke' && audioMaxPrompts <= 0) ||
    (!audioDryRun && (activeAudioJobRunning || audioLaunchProblems.length > 0));
  const scoreLaunchProblems = !scoreDryRun
    ? [
        ...(activeScoreJobRunning ? [`Attribution scoring already running: ${activeScoreJob?.job_name ?? 'unknown job'}.`] : []),
        ...(!currentFullResult ? ['Complete the full generated-audio benchmark after the latest smoke gate before attribution scoring.'] : []),
        ...(selectedScoreModelIds.length === 0 ? ['Select at least one model lane to score.'] : []),
        ...(scoreConfirmation.trim() !== ATTRIBUTION_SCORING_CONFIRMATION ? [`Type ${ATTRIBUTION_SCORING_CONFIRMATION} before live scoring launch.`] : []),
      ]
    : [];
  const scoreLaunchDisabled = planningScore || !currentFullResult || selectedScoreModelIds.length === 0 || (!scoreDryRun && (activeScoreJobRunning || scoreLaunchProblems.length > 0));

  const handleLockPromptSet = async () => {
    setLockingPromptSet(true);
    setError(null);
    try {
      const response = await fetchJson<{ readiness: EvaluationReadiness }>('/api/evaluation/prompt-set/lock', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_name: readiness?.latest_evaluation_job?.job_name ?? null, confirmed: true }),
      });
      setReadiness(response.readiness);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLockingPromptSet(false);
    }
  };

  const handleAudioBenchmarkPlan = async () => {
    setPlanningAudio(true);
    setError(null);
    setAudioRunResponse(null);
    try {
      const response = await fetchJson<AudioBenchmarkPlanResponse>('/api/evaluation/audio-benchmark/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_ids: audioModelIds,
          suite_ids: audioScope === 'full' ? LIVE_WAVE_1_SUITE_IDS : audioSuiteIds,
          scope: audioScope,
          seed_ids: audioSeedIds,
          max_prompts: audioScope === 'full' ? 0 : audioMaxPrompts,
          dry_run: audioDryRun,
          launch_confirmation: audioConfirmation,
        }),
      });
      setAudioRunResponse(response);
      if (response.job?.name) {
        setProgressJobName(response.job.name);
        setProgressOpen(true);
      }
      if (!audioDryRun) {
        setAudioConfirmation('');
        void loadReadiness();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPlanningAudio(false);
    }
  };

  const canRetryMusicGenOnly = useMemo(() => {
    const byModel = Object.fromEntries((progress?.model_progress ?? []).map((row) => [row.model_id ?? '', row]));
    const stableDone = ['diffusion_cara_strong_full_modest_arch', 'context_diffusion_cara_strong_full'].every(
      (modelId) => Number(byModel[modelId]?.percent ?? 0) >= 99.5,
    );
    const musicGenDone = ['musicgen_cara_strong_full'].every(
      (modelId) => Number(byModel[modelId]?.percent ?? 0) >= 99.5,
    );
    const hasMusicGen = ['musicgen_cara_strong_full'].some((modelId) => modelId in byModel);
    return Boolean(progress?.job?.job_name && progress?.scope === 'full' && stableDone && hasMusicGen && !musicGenDone);
  }, [progress]);

  const handleRetryMusicGenOnly = async () => {
    if (!progress?.job?.job_name) return;
    setRetryingMusicGen(true);
    setProgressError(null);
    setError(null);
    try {
      const response = await fetchJson<AudioBenchmarkPlanResponse>('/api/evaluation/audio-benchmark/retry-missing', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_job_name: progress.job.job_name,
          dry_run: false,
          launch_confirmation: 'RETRY MUSICGEN AUDIO ONLY',
        }),
      });
      if (response.job?.name) {
        setProgressJobName(response.job.name);
        setAudioRunResponse(response);
      }
      void loadReadiness();
    } catch (err) {
      setProgressError(err instanceof Error ? err.message : String(err));
    } finally {
      setRetryingMusicGen(false);
    }
  };

  const handleAttributionScoringPlan = async () => {
    setPlanningScore(true);
    setError(null);
    setScoreRunResponse(null);
    try {
      const response = await fetchJson<AttributionScoringPlanResponse>('/api/evaluation/attribution-scoring/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          audio_job_name: currentFullResult?.job_name ?? null,
          model_ids: selectedScoreModelIds,
          dry_run: scoreDryRun,
          force_rescore: scoreForceRescore,
          launch_confirmation: scoreConfirmation,
        }),
      });
      setScoreRunResponse(response);
      if (!scoreDryRun) void loadReadiness();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPlanningScore(false);
    }
  };

  return (
    <div style={{ display: 'grid', gap: 20 }}>
      <PageHeader
        kicker="Evaluation · Testing"
        title={
          <>
            CARA-Strong <em>benchmark runway</em>
          </>
        }
        description={
          <>
            A locked prep ladder for defensible comparisons: verify lanes, reuse the manifest, generate audio,
            score native/probe predictions, then publish the matrix.
          </>
        }
        actions={
          <button className="btn btn-ghost" onClick={loadReadiness} disabled={loading} type="button">
            <RefreshCw size={16} /> Refresh
          </button>
        }
      />

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="pool-summary-grid" aria-label="Benchmark runway">
        {runway.map((step) => (
          <div
            className={`pool-metric-card tone-${
              step.status === 'Ready' ? 'good' : step.status === 'Blocked' ? 'bad' : step.status === 'Running' ? 'good' : 'warn'
            }`}
            key={step.label}
          >
            <div className="pool-metric-top">{step.label}</div>
            <div className="pool-metric-value" style={{ fontSize: 24 }}>{step.status}</div>
            <div className="pool-metric-meta">{step.note}</div>
          </div>
        ))}
      </section>

      <section className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Model readiness</div>
            <div className="dim" style={{ marginTop: 4, fontSize: 13 }}>
              Every lane is visible here; adding another model should add a row, not a new dashboard schema.
            </div>
          </div>
          <div className="card-meta">{readiness?.format ?? 'loading'}</div>
        </div>
        <div className="table-scroll">
          <div className="run-table" style={{ minWidth: 920 }}>
            <div className="run-row run-head" style={{ gridTemplateColumns: '1.25fr 0.7fr 0.85fr 0.9fr 1.05fr' }}>
              <span>Lane</span>
              <span>Family</span>
              <span>Variant</span>
              <span>Scoring</span>
              <span>Outcome</span>
            </div>
            {visibleLanes.map((lane) => (
              <div className="run-row" key={lane.model_id} style={{ gridTemplateColumns: '1.25fr 0.7fr 0.85fr 0.9fr 1.05fr' }}>
                <span>{lane.label}</span>
                <span>{lane.family}</span>
                <span>{lane.variant.replace(/_/g, ' ')}</span>
                <span>{lane.native_prediction_adapter?.replace(/_/g, ' ') ?? 'external probe'}</span>
                <span>
                  <span className={`status-pill ${statusClass(lane.status)}`}>{lane.status}</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Prompt lock</div>
            <div className="card-meta">{readiness?.benchmark_prompt_set?.locked ? 'locked' : 'needs lock'}</div>
          </div>
          <div className="metric-list">
            <div>
              <span>Prompt set</span>
              <strong>{readiness?.benchmark_spec?.prompt_set_version ?? 'v2'}</strong>
            </div>
            <div>
              <span>Purpose</span>
              <strong>{readiness?.benchmark_spec?.claim_language ?? 'Pool-level attribution under codeword-withheld evaluation.'}</strong>
            </div>
            <div>
              <span>Reuse rule</span>
              <strong>{readiness?.benchmark_prompt_set?.reason ?? 'All model lanes must score the same locked prompt rows.'}</strong>
            </div>
          </div>
          <button
            className="btn"
            type="button"
            onClick={handleLockPromptSet}
            disabled={loading || lockingPromptSet || readiness?.benchmark_prompt_set?.locked || !readiness?.benchmark_prompt_set?.can_lock}
          >
            <LockKeyhole size={16} /> {lockingPromptSet ? 'Locking...' : readiness?.benchmark_prompt_set?.locked ? 'Prompt Set Locked' : 'Lock Prompt Set'}
          </button>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Research guardrails</div>
            <div className="card-meta">{readiness?.registry.pool_count ?? 0} pools</div>
          </div>
          <div className="metric-list">
            <div>
              <span>Azure policy</span>
              <strong>{readiness?.launch_guard.cost_policy ?? 'Use approved Azure ML compute, datastores, environments, and command jobs only.'}</strong>
            </div>
            <div>
              <span>Coverage</span>
              <strong>{readiness?.registry.pool_count ?? 0} pools · {readiness?.registry.family_count ?? 0} families</strong>
            </div>
            <div>
              <span>Launcher coverage</span>
              <strong>Released Stable Audio, Diffusion CARA-Strong, released MusicGen, and MusicGen CARA-Strong run from this page.</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div>
            <div className="card-title">Generated audio</div>
            <div className="dim" style={{ marginTop: 4, fontSize: 13 }}>
              This launcher submits one benchmark run with architecture-specific Azure jobs: Stable Audio and MusicGen use their approved environments.
            </div>
          </div>
          <div className="row" style={{ alignItems: 'center' }}>
            <div className="card-meta">{activeAudioJobRunning ? 'running' : currentFullResult ? 'full recorded' : latestSmokeResult ? 'smoke recorded' : audioScope}</div>
            {latestProgressJobName ? (
              <button
                className="btn btn-ghost"
                type="button"
                onClick={() => {
                  setProgressJobName(latestProgressJobName);
                  setProgressOpen(true);
                }}
              >
                View progress
              </button>
            ) : null}
          </div>
        </div>
        <div className="controls">
          <div className="metric-list">
            <div>
              <span>Smoke gate</span>
              <strong>
                {latestSmokeResult?.job_name
                  ? `${gateLabel(latestSmokeResult)} · ${latestSmokeResult.job_name}`
                  : 'not run yet'}
                <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                  Passing smoke means the launch path works; it is the go/no-go gate for the full locked set.
                </span>
              </strong>
            </div>
            <div>
              <span>Full benchmark</span>
              <strong>
                {currentFullResult?.job_name
                  ? `${gateLabel(currentFullResult)} · ${currentFullResult.job_name}`
                  : latestSmokeResult
                    ? 'ready to launch'
                    : 'blocked until smoke is recorded'}
                <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                  Full audio is the evidence source for attribution scoring and the benchmark matrix.
                </span>
              </strong>
            </div>
          </div>
          <div className="row" style={{ alignItems: 'center' }}>
            <button
              className={`btn ${audioScope === 'smoke' ? '' : 'btn-ghost'}`}
              type="button"
              onClick={() => {
                setAudioConfirmation('');
                setAudioScope('smoke');
                setAudioMaxPrompts(20);
                setAudioSuiteIds(['known_pool_prompt_recall', 'control_token_confound']);
              }}
              disabled={planningAudio}
            >
              Smoke
            </button>
            <button
              className={`btn ${audioScope === 'full' ? '' : 'btn-ghost'}`}
              type="button"
              onClick={() => {
                setAudioConfirmation('');
                setAudioScope('full');
                setAudioSuiteIds(LIVE_WAVE_1_SUITE_IDS);
                if (audioModelIds.length === 0) setAudioModelIds(LIVE_WAVE_1_MODEL_IDS);
              }}
              disabled={planningAudio}
            >
              Full locked set
            </button>
            <label className="toggle">
              <input
                type="checkbox"
                checked={audioDryRun}
                onChange={(event) => {
                  setAudioConfirmation('');
                  setAudioDryRun(event.target.checked);
                }}
              />
              Dry-run
            </label>
          </div>
          <div className="metric-list">
            <div>
              <span>Models</span>
              <strong>{selectedAudioLaneLabels.length ? formatList(selectedAudioLaneLabels) : 'none selected'}
                <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                  Choose one, several, or all model lanes. The same locked prompt rows are reused for every selected lane.
                </span>
              </strong>
            </div>
            <div>
              <span>Not selected / unavailable</span>
              <strong>{excludedGeneratedAudioText}</strong>
            </div>
            <div>
              <span>Seed</span>
              <strong className="mono">0</strong>
            </div>
            <div>
              <span>Prompt limit</span>
              <strong>
                {audioScope === 'full' ? (
                  'all selected locked rows'
                ) : (
                  <input
                    className="input"
                    type="number"
                    min={1}
                    max={80}
                    value={audioMaxPrompts}
                    onChange={(event) => {
                      setAudioConfirmation('');
                      setAudioMaxPrompts(Number(event.target.value));
                    }}
                    disabled={planningAudio}
                    style={{ width: 110 }}
                  />
                )}
              </strong>
            </div>
          </div>
          <div className="row" style={{ gap: 10, flexWrap: 'wrap' }}>
            <button className="btn btn-ghost" type="button" disabled={planningAudio} onClick={() => selectAudioModels(LIVE_WAVE_1_MODEL_IDS)}>
              All models
            </button>
            <button className="btn btn-ghost" type="button" disabled={planningAudio} onClick={() => selectAudioModels(['diffusion_cara_strong_full_modest_arch'])}>
              Diffusion only
            </button>
            <button className="btn btn-ghost" type="button" disabled={planningAudio} onClick={() => selectAudioModels(['context_diffusion_cara_strong_full'])}>
              Context only
            </button>
            <button className="btn btn-ghost" type="button" disabled={planningAudio} onClick={() => selectAudioModels(['musicgen_cara_strong_full'])}>
              MusicGen only
            </button>
          </div>
          <div className="check-list">
            {selectableAudioLanes.map((lane) => {
              const checked = audioModelIds.includes(lane.model_id);
              return (
                <label key={lane.model_id} className={`check-item${checked ? ' is-checked' : ''}`}>
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={planningAudio}
                    onChange={() => toggleAudioModel(lane.model_id)}
                  />
                  <span className="check-label">
                    <span className="check-icon">
                      <ListChecks size={12} />
                    </span>
                    <span>
                      {lane.label}
                      <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                        {lane.family.replace(/_/g, ' ')} · {lane.variant.replace(/_/g, ' ')} · {lane.status}
                      </span>
                    </span>
                  </span>
                  <span className="dim mono">{lane.native_prediction_adapter?.replace(/_/g, ' ') ?? 'probe'}</span>
                </label>
              );
            })}
          </div>
          <div className="check-list">
            {LIVE_WAVE_1_SUITE_IDS.map((suiteId) => {
              const suite = readiness?.suites.find((item) => item.id === suiteId);
              const checked = audioScope === 'full' || audioSuiteIds.includes(suiteId);
              return (
                <label key={suiteId} className={`check-item${checked ? ' is-checked' : ''}`}>
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={planningAudio || audioScope === 'full'}
                    onChange={() => toggleSuite(suiteId)}
                  />
                  <span className="check-label">
                    <span className="check-icon">
                      <ListChecks size={12} />
                    </span>
                    <span>
                      {suite?.label ?? suiteId}
                      <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                        {suite?.evidence_type?.replace(/_/g, ' ') ?? 'locked evidence'}
                      </span>
                    </span>
                  </span>
                  <span className="dim mono">{suite?.prompt_count || 'held-out'}</span>
                </label>
              );
            })}
          </div>
          <div className="row" style={{ alignItems: 'center' }}>
            <input
              className="input"
              value={audioConfirmation}
              onChange={(event) => setAudioConfirmation(event.target.value)}
              disabled={planningAudio || audioDryRun}
              placeholder={AUDIO_BENCHMARK_CONFIRMATION}
              style={{ flex: 1, minWidth: 260 }}
            />
            {!audioDryRun ? (
              <span className="dim" style={{ fontSize: 12 }}>
                Type after choosing scope and model lanes.
              </span>
            ) : null}
            <button className="btn" onClick={handleAudioBenchmarkPlan} disabled={audioLaunchDisabled} type="button">
              <Play size={16} /> {planningAudio ? 'Checking...' : audioDryRun ? 'Plan audio run' : 'Submit audio run'}
            </button>
            {latestProgressJobName ? (
              <button
                className="btn btn-ghost"
                type="button"
                onClick={() => {
                  setProgressJobName(latestProgressJobName);
                  setProgressOpen(true);
                }}
              >
                View progress
              </button>
            ) : null}
          </div>
          {!audioDryRun && audioLaunchProblems.length ? (
            <div className="error-banner">
              {audioLaunchProblems.map((problem) => (
                <div key={problem}>{problem}</div>
              ))}
            </div>
          ) : null}
          {audioRunResponse ? (
            <div className="preflight-panel" role="status" aria-live="polite">
              <div className="preflight-header">
                <div>
                  <div className="card-title">{audioRunResponse.dry_run ? 'Audio preflight complete' : 'Audio job submitted'}</div>
                  <div className="dim" style={{ marginTop: 4 }}>{audioRunResponse.message ?? 'Plan returned from benchmark launcher.'}</div>
                </div>
                <button className="btn btn-ghost" type="button" onClick={() => setAudioRunResponse(null)}>Dismiss</button>
              </div>
              <div className="preflight-grid">
                <PreflightCheck passed={Boolean(audioRunResponse.plan)} label="Plan generated" detail={audioRunResponse.plan?.scope ?? audioScope} />
                <PreflightCheck passed={Boolean(audioRunResponse.plan?.prompt_manifest_uri)} label="Locked prompt manifest" detail={audioRunResponse.plan?.prompt_manifest_uri ? 'available' : 'missing'} />
                <PreflightCheck passed={Boolean(audioRunResponse.plan?.model_ids?.length)} label="Model lanes selected" detail={formatList(audioRunResponse.plan?.model_ids)} />
                <PreflightCheck passed={Boolean(audioRunResponse.plan?.suite_ids?.length)} label="Suites selected" detail={formatList(audioRunResponse.plan?.suite_ids)} />
                <PreflightCheck passed={Boolean(audioRunResponse.plan?.live_ready)} label="Azure command job ready" detail={audioRunResponse.plan?.live_ready_reason} />
                <PreflightCheck passed={Boolean(audioRunResponse.plan?.cost_policy?.includes('no Marketplace'))} label="Cost guardrail" detail={audioRunResponse.plan?.cost_policy} />
              </div>
              <div className="metric-list">
                <div>
                  <span>Estimated generations</span>
                  <strong>{audioRunResponse.plan?.estimated_generations ?? 'pending'}</strong>
                </div>
                <div>
                  <span>Output prefix</span>
                  <strong className="mono">{formatOutputPrefix(audioRunResponse.plan?.output_prefix) ?? audioRunResponse.job?.output_path ?? 'assigned on live submit'}</strong>
                </div>
                <div>
                  <span>Azure child jobs</span>
                  <strong>{audioRunResponse.jobs?.length ? audioRunResponse.jobs.map((job) => `${job.family}: ${job.name}`).join(', ') : formatList(Object.keys(audioRunResponse.plan?.model_groups ?? {}))}</strong>
                </div>
                <div>
                  <span>Metrics policy</span>
                  <strong>{audioRunResponse.plan?.metrics_policy ?? 'Native and probe lanes are scored from real prediction fields.'}</strong>
                </div>
              </div>
            </div>
          ) : null}
          {activeAudioJobRunning || latestAudioResult || audioRunResponse ? (
            <div className="metric-list">
              <div>
                <span>Current result</span>
                <strong>{activeAudioJob?.job_name ?? latestAudioResult?.job_name ?? audioRunResponse?.job?.name ?? 'planned'}</strong>
              </div>
              <div>
                <span>Status</span>
                <strong>{activeAudioJob?.status ?? latestAudioResult?.status ?? audioRunResponse?.status ?? 'pending'}</strong>
              </div>
              <div>
                <span>Output</span>
                <strong className="mono">{activeAudioJob?.output_path ?? latestAudioResult?.output_path ?? audioRunResponse?.job?.output_path ?? 'pending'}</strong>
              </div>
            </div>
          ) : null}
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">Attribution scoring</div>
          <div className="card-meta">{activeScoreJobRunning ? 'running' : fullScoreResult ? 'complete' : 'after full audio'}</div>
        </div>
        <div className="controls">
          <div className="metric-list">
            <div>
              <span>Source audio</span>
              <strong>{currentFullResult?.job_name ?? 'complete the full generated-audio benchmark after the latest smoke gate first'}</strong>
            </div>
            <div>
              <span>Manifest</span>
              <strong className="mono">{latestGenerationManifest ?? 'pending'}</strong>
            </div>
            <div>
              <span>Model lanes</span>
              <strong>
                {selectedScoreLaneLabels.length ? formatList(selectedScoreLaneLabels) : 'none selected'}
                <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                  Attribution scoring filters the source manifest to these model IDs; previous score folders are not overwritten.
                </span>
              </strong>
            </div>
            <div>
              <span>Latest metrics</span>
              <strong>{readiness?.latest_results.metrics_available ? 'available for benchmark matrix' : 'not available yet'}</strong>
            </div>
          </div>
          <div className="row" style={{ gap: 10, flexWrap: 'wrap' }}>
            <button className="btn btn-ghost" type="button" disabled={planningScore} onClick={() => selectScoreModels(scoreSelectableLanes.map((lane) => lane.model_id))}>
              All lanes in source run
            </button>
            <button className="btn btn-ghost" type="button" disabled={planningScore} onClick={() => selectScoreModels(['diffusion_cara_strong_full_modest_arch'])}>
              Diffusion only
            </button>
            <button className="btn btn-ghost" type="button" disabled={planningScore} onClick={() => selectScoreModels(['context_diffusion_cara_strong_full'])}>
              Context only
            </button>
            <button className="btn btn-ghost" type="button" disabled={planningScore} onClick={() => selectScoreModels(['musicgen_cara_strong_full'])}>
              MusicGen only
            </button>
          </div>
          <div className="check-list">
            {scoreSelectableLanes.map((lane) => {
              const checked = scoreModelIds.includes(lane.model_id);
              return (
                <label key={lane.model_id} className={`check-item${checked ? ' is-checked' : ''}`}>
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={planningScore}
                    onChange={() => toggleScoreModel(lane.model_id)}
                  />
                  <span className="check-label">
                    <span className="check-icon">
                      <ListChecks size={12} />
                    </span>
                    <span>
                      {lane.label}
                      <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                        {lane.family.replace(/_/g, ' ')} · {lane.native_prediction_adapter?.replace(/_/g, ' ') ?? 'external probe'}
                      </span>
                    </span>
                  </span>
                  <span className="dim mono">{sourceScoreModelIds.length ? 'in source run' : 'candidate'}</span>
                </label>
              );
            })}
          </div>
          <div className="row" style={{ alignItems: 'center' }}>
            <label className="toggle">
              <input
                type="checkbox"
                checked={scoreDryRun}
                onChange={(event) => {
                  setScoreConfirmation('');
                  setScoreDryRun(event.target.checked);
                }}
              />
              Dry-run
            </label>
            <label className="toggle">
              <input
                type="checkbox"
                checked={scoreForceRescore}
                onChange={(event) => {
                  setScoreConfirmation('');
                  setScoreForceRescore(event.target.checked);
                }}
              />
              Force re-score current full run
            </label>
            <input
              className="input"
              value={scoreConfirmation}
              onChange={(event) => setScoreConfirmation(event.target.value)}
              disabled={planningScore || scoreDryRun}
              placeholder={ATTRIBUTION_SCORING_CONFIRMATION}
              style={{ flex: 1, minWidth: 260 }}
            />
            <button className="btn" onClick={handleAttributionScoringPlan} disabled={scoreLaunchDisabled} type="button">
              <Play size={16} /> {planningScore ? 'Checking...' : scoreDryRun ? 'Plan scoring run' : 'Submit scoring run'}
            </button>
          </div>
          {!scoreDryRun && scoreLaunchProblems.length ? (
            <div className="error-banner">
              {scoreLaunchProblems.map((problem) => (
                <div key={problem}>{problem}</div>
              ))}
            </div>
          ) : null}
          {scoreRunResponse ? (
            <div className="preflight-panel" role="status" aria-live="polite">
              <div className="preflight-header">
                <div>
                  <div className="card-title">{scoreRunResponse.dry_run ? 'Scoring preflight complete' : 'Scoring job submitted'}</div>
                  <div className="dim" style={{ marginTop: 4 }}>{scoreRunResponse.message ?? 'Plan returned from attribution scorer.'}</div>
                </div>
                <button className="btn btn-ghost" type="button" onClick={() => setScoreRunResponse(null)}>Dismiss</button>
              </div>
              <div className="preflight-grid">
                <PreflightCheck passed={Boolean(scoreRunResponse.plan?.audio_job_name)} label="Full audio selected" detail={scoreRunResponse.plan?.audio_job_name ?? 'missing'} />
                <PreflightCheck passed={Boolean(scoreRunResponse.plan?.model_ids?.length)} label="Model lanes selected" detail={formatList(scoreRunResponse.plan?.model_ids)} />
                <PreflightCheck passed={Boolean(scoreRunResponse.plan?.selected_families?.length)} label="Scoring families" detail={formatList(scoreRunResponse.plan?.selected_families)} />
                <PreflightCheck passed={Boolean(scoreRunResponse.plan?.generation_manifest_uri)} label="Generation manifest" detail={scoreRunResponse.plan?.generation_manifest_uri ? 'available' : 'pending'} />
                <PreflightCheck passed={Boolean(scoreRunResponse.plan?.live_ready)} label="Azure scoring job ready" detail={scoreRunResponse.plan?.live_ready_reason} />
                <PreflightCheck passed={!scoreRunResponse.plan?.force_rescore || Boolean(currentFullResult)} label="Re-score mode" detail={scoreRunResponse.plan?.force_rescore ? 'fresh scorer jobs will be submitted for this generated-audio run' : 'duplicate scorer protection is active'} />
                <PreflightCheck passed={Boolean(scoreRunResponse.plan?.metrics_policy)} label="No label leakage" detail={scoreRunResponse.plan?.metrics_policy} />
                <PreflightCheck passed={Boolean(scoreRunResponse.plan?.cost_policy?.includes('no Marketplace'))} label="Cost guardrail" detail={scoreRunResponse.plan?.cost_policy} />
              </div>
              <div className="metric-list">
                <div>
                  <span>Source output</span>
                  <strong className="mono">{formatOutputPrefix(scoreRunResponse.plan?.pending_score_output_paths ?? scoreRunResponse.plan?.generated_audio_output_paths) ?? scoreRunResponse.plan?.generated_audio_output_path ?? 'pending full audio'}</strong>
                </div>
                <div>
                  <span>Metrics target</span>
                  <strong className="mono">{scoreRunResponse.job?.metrics_uri ?? 'assigned on live submit'}</strong>
                </div>
                <div>
                  <span>Context model artifact</span>
                  <strong className="mono">{scoreRunResponse.plan?.context_trained_model_data ?? 'not selected'}</strong>
                </div>
              </div>
            </div>
          ) : null}
          {activeScoreJobRunning || fullScoreResult || scoreRunResponse ? (
            <div className="metric-list">
              <div>
                <span>Current result</span>
                <strong>{activeScoreJob?.job_name ?? fullScoreResult?.job_name ?? scoreRunResponse?.job?.name ?? 'planned'}</strong>
              </div>
              <div>
                <span>Status</span>
                <strong>{activeScoreJob?.status ?? fullScoreResult?.status ?? scoreRunResponse?.status ?? 'pending'}</strong>
              </div>
              <div>
                <span>Metrics</span>
                <strong className="mono">{activeScoreJob?.metrics_uri ?? fullScoreResult?.metrics_uri ?? scoreRunResponse?.job?.metrics_uri ?? 'pending'}</strong>
              </div>
            </div>
          ) : null}
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">Repairability ladder</div>
          <div className="card-meta">reported separately from exact accuracy</div>
        </div>
        <div className="metric-list">
          <div>
            <span>
              <ShieldCheck size={13} style={{ marginRight: 6, verticalAlign: -2 }} />
              Scoring output
            </span>
            <strong>
              Step 16 writes native predictions, scored manifests, and the separate repairability matrix used on the Benchmarks page.
              <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                Missing predictions means the extractor/probe did not produce auditable CARA rows; it should not be treated as a zero score.
              </span>
            </strong>
          </div>
          {(readiness?.repairability.tiers ?? []).map((tier) => (
            <div key={tier.id}>
              <span>
                <ShieldCheck size={13} style={{ marginRight: 6, verticalAlign: -2 }} />
                {tier.label}
              </span>
              <strong>
                {tier.description}
                <span className="dim" style={{ display: 'block', fontSize: 12 }}>
                  {tier.counts_as_pool_success ? 'counts as pool-level success' : 'supporting evidence only'}
                </span>
              </strong>
            </div>
          ))}
        </div>
      </section>

      {progressOpen ? (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Generated audio benchmark progress">
          <div className="benchmark-progress-modal">
            <div className="metadata-modal-header">
              <div>
                <h2>Generated-audio progress</h2>
                <div className="dim" style={{ marginTop: 4 }}>
                  {progress?.job?.job_name ?? progressJobName ?? 'latest run'} · {progress?.scope ?? 'benchmark'} · refreshes every 15 seconds
                </div>
              </div>
              <div className="row" style={{ alignItems: 'center' }}>
                {canRetryMusicGenOnly ? (
                  <button className="btn btn-ghost" type="button" onClick={handleRetryMusicGenOnly} disabled={retryingMusicGen || progressLoading}>
                    <RefreshCw size={16} /> {retryingMusicGen ? 'Retrying MusicGen...' : 'Retry MusicGen only'}
                  </button>
                ) : null}
                <button className="btn btn-ghost" type="button" onClick={() => void loadAudioProgress(progressJobName)} disabled={progressLoading}>
                  <RefreshCw size={16} /> {progressLoading ? 'Checking...' : 'Refresh'}
                </button>
                <button className="btn" type="button" onClick={() => setProgressOpen(false)}>Close</button>
              </div>
            </div>

            {progressError ? <div className="error-banner">{progressError}</div> : null}
            {canRetryMusicGenOnly ? (
              <div className="info-banner">
                Stable Audio is complete for this run. Use Retry MusicGen only to create a new aggregate run that reuses the completed Stable Audio outputs and submits only the MusicGen child job.
              </div>
            ) : null}

            <div className="progress-hero">
              <div>
                <div className="pool-metric-top">Overall</div>
                <div className="pool-metric-value" style={{ fontSize: 32 }}>
                  {Math.round(progress?.progress_percent ?? 0)}%
                </div>
                <div className="pool-metric-meta">
                  {progress?.completed_generations ?? 0}
                  {progress?.planned_generations ? ` / ${progress.planned_generations}` : ''} WAV outputs observed
                </div>
              </div>
              <div className="progress-track" aria-label="Overall generated audio progress">
                <div className="progress-fill" style={{ width: `${Math.max(0, Math.min(100, progress?.progress_percent ?? 0))}%` }} />
              </div>
            </div>

            <div className="metric-list">
              <div>
                <span>Models run in this job</span>
                <strong>{formatList(progress?.model_ids)}</strong>
              </div>
              <div>
                <span>Blocked or excluded</span>
                <strong>{excludedGeneratedAudioText}</strong>
              </div>
              <div>
                <span>Included suites</span>
                <strong>{formatList(progress?.suite_ids)}</strong>
              </div>
              <div>
                <span>Latest completed</span>
                <strong>
                  {progress?.latest_completed_item?.model_id
                    ? `${progress.latest_completed_item.model_id} · ${progress.latest_completed_item.suite_id ?? 'suite'} · ${progress.latest_completed_item.file ?? 'WAV'}`
                    : progress?.completed_generations
                      ? 'WAV output observed'
                      : 'no generated WAVs visible yet'}
                </strong>
              </div>
              <div>
                <span>Scope note</span>
                <strong>{progress?.note ?? 'This monitor counts generated Stable Audio WAV artifacts in the approved Azure ML datastore.'}</strong>
              </div>
            </div>

            <div className="progress-columns">
              <div>
                <div className="card-title" style={{ marginBottom: 10 }}>Model lanes</div>
                <div className="progress-list">
                  {(progress?.model_progress ?? []).map((row) => {
                    const pct = row.percent ?? (progress?.planned_generations ? Math.min(100, row.completed / progress.planned_generations * 100) : 0);
                    return (
                      <div className="progress-row" key={row.model_id}>
                        <div className="progress-row-top">
                          <span>{row.model_id}</span>
                          <span className="mono">{row.completed}{row.planned ? ` / ${row.planned}` : ''}</span>
                        </div>
                        <div className="progress-track">
                          <div className="progress-fill" style={{ width: `${Math.max(0, Math.min(100, pct ?? 0))}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
              <div>
                <div className="card-title" style={{ marginBottom: 10 }}>Suites</div>
                <div className="progress-list">
                  {(progress?.suite_progress ?? []).map((row) => {
                    const pct = row.percent ?? 0;
                    return (
                      <div className="progress-row" key={row.suite_id}>
                        <div className="progress-row-top">
                          <span>{row.suite_id}</span>
                          <span className="mono">{row.completed}{row.planned ? ` / ${row.planned}` : ''}</span>
                        </div>
                        <div className="progress-track">
                          <div className="progress-fill" style={{ width: `${Math.max(0, Math.min(100, pct ?? 0))}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            <div className="table-scroll">
              <div className="run-table" style={{ minWidth: 720 }}>
                <div className="run-row run-head" style={{ gridTemplateColumns: `1.2fr repeat(${Math.max(1, progress?.suite_ids.length ?? 1)}, minmax(110px, 1fr))` }}>
                  <span>Model / suite</span>
                  {(progress?.suite_ids ?? []).map((suiteId) => <span key={suiteId}>{suiteId}</span>)}
                </div>
                {(progress?.model_ids ?? []).map((modelId) => (
                  <div className="run-row" key={modelId} style={{ gridTemplateColumns: `1.2fr repeat(${Math.max(1, progress?.suite_ids.length ?? 1)}, minmax(110px, 1fr))` }}>
                    <span>{modelId}</span>
                    {(progress?.suite_ids ?? []).map((suiteId) => (
                      <span className="mono" key={`${modelId}-${suiteId}`}>{progress?.by_model_suite?.[modelId]?.[suiteId] ?? 0}</span>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
