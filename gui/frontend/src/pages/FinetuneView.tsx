import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Cloud,
  FileLock2,
  UploadCloud,
  Play,
  RefreshCw,
  Terminal,
  Activity,
  Square,
  DatabaseZap,
} from 'lucide-react';
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { PageHeader, PlaceholderBadge } from './PageHeader';

export type FinetuneVariant = 'diffusion' | 'autoregressive';
type StableAudioSmokeVariant = 'no_cara_baseline' | 'cara_lite' | 'cara_head' | 'cara_strong';
type MusicGenSmokeVariant = 'no_cara_baseline' | 'cara_lite' | 'cara_probe' | 'cara_strong';
type MusicGenSequenceVariant = MusicGenSmokeVariant | 'full_cara_strong';

interface FieldDef {
  key: string;
  label: string;
  type: 'text' | 'number' | 'select';
  options?: string[];
  defaultValue: string | number;
  hint?: string;
}

interface FinetuneViewProps {
  variant: FinetuneVariant;
  kicker: string;
  title: React.ReactNode;
  description: React.ReactNode;
  extraFields: FieldDef[];
}

interface ReadinessGate {
  id: string;
  label: string;
  passed: boolean;
}

interface ActiveTrainingJob {
  name?: string;
  display_name?: string;
  status?: string;
  created_at?: string;
  compute?: string;
  environment?: string;
  studio_url?: string;
  run_name?: string;
  variant?: string;
  training_scope?: string;
  output_path?: string;
}

interface TrainingReadiness {
  status: string;
  training_launch_enabled: boolean;
  training_launch_reason: string;
  gates: ReadinessGate[];
  lock: {
    locked: boolean;
    output_dir: string;
    summary?: {
      accepted_count?: number;
      rejected_count?: number;
      pool_count?: number;
      family_count?: number;
      split_counts?: Record<string, number>;
      tir_id?: string;
    } | null;
    paths?: Record<string, string>;
  };
  azure_upload?: {
    confirmed: boolean;
    confirmed_at?: string | null;
    source_root: string;
    expected_manifest: string;
    expected_audio_root: string;
  };
  data_locations: Record<string, string>;
  preprocess_jobs?: Record<string, {
    passed: boolean;
    active: boolean;
    reason: string;
    latest_job?: (ActiveTrainingJob & {
      model_family?: string;
      output_path?: string;
      compute_strategy?: string;
      compute_reason?: string;
    }) | null;
  }>;
  active_training_jobs?: ActiveTrainingJob[];
  active_stable_audio_smoke_job?: ActiveTrainingJob | null;
  active_musicgen_trainer_job?: ActiveTrainingJob | null;
  musicgen_token_cache?: {
    stage: number;
    label: string;
    passed: boolean;
    active: boolean;
    reason: string;
    latest_job?: ActiveTrainingJob | null;
  };
  musicgen_preflight?: {
    stage: number;
    label: string;
    passed: boolean;
    active: boolean;
    required_environment: string;
    reason: string;
    latest_job?: (ActiveTrainingJob & {
      checkpoint?: string;
      output_path?: string;
    }) | null;
  };
  musicgen_smoke_sequence?: {
    variants: Record<string, {
      stage: number;
      label: string;
      passed: boolean;
      active: boolean;
      reason: string;
      latest_job?: (ActiveTrainingJob & {
        max_steps?: number;
        batch_size?: number;
        learning_rate?: number;
      }) | null;
      latest_passed_job?: (ActiveTrainingJob & {
        max_steps?: number;
        batch_size?: number;
        learning_rate?: number;
      }) | null;
    }>;
    next_variant: string;
    next_stage: number;
    next_label: string;
    reason: string;
  };
  stable_audio_preflight?: {
    passed: boolean;
    active: boolean;
    required_environment: string;
    reason: string;
    latest_job?: (ActiveTrainingJob & {
      checkpoint?: string;
      wrapper_check?: boolean;
      output_path?: string;
    }) | null;
  };
  stable_audio_smoke_sequence?: {
    variants: Record<string, {
      stage: number;
      label: string;
      passed: boolean;
      active: boolean;
      implemented?: boolean;
      reason: string;
      latest_job?: (ActiveTrainingJob & {
        max_steps?: number;
        batch_size?: number;
        num_workers?: number;
        learning_rate?: number;
      }) | null;
      latest_passed_job?: (ActiveTrainingJob & {
        max_steps?: number;
        batch_size?: number;
        num_workers?: number;
        learning_rate?: number;
      }) | null;
    }>;
    next_variant: string;
    next_stage: number;
    next_label: string;
    reason: string;
  };
  stable_audio_full_training?: {
    stage: number;
    label: string;
    passed: boolean;
    active: boolean;
    reason: string;
    latest_job?: (ActiveTrainingJob & {
      max_steps?: number;
      batch_size?: number;
      learning_rate?: number;
      training_scope?: string;
    }) | null;
  };
  stable_audio_training_progress?: TrainingRunProgress | null;
  cloud_job_policy?: {
    durable_submission: boolean;
    browser_close_cancels_job: boolean;
    stop_behavior: string;
    checkpoint_resume: string;
    preprocess_compute_strategy: string;
    musicgen_token_cache?: string;
  };
  submitted_training_jobs?: Array<{
    created_at?: string;
    action?: string;
    job_name?: string;
    studio_url?: string;
    compute?: string;
    compute_strategy?: string;
    compute_reason?: string;
    model_family?: string;
    output_path?: string;
    dry_run?: boolean;
  }>;
  audio_window_policy: Record<string, { max_window_seconds: number; sample_rate_hz: number; channels: string; pre_chunk_required: boolean; note: string }>;
}

interface TrainingRunProgress {
  checked_at: string;
  job_name?: string | null;
  run_name?: string | null;
  studio_url?: string | null;
  status?: string | null;
  variant?: string | null;
  training_scope?: string | null;
  max_steps?: number | null;
  observed_step?: number | null;
  step_percent?: number | null;
  batch_size?: number | null;
  chunks_seen_estimate?: number | null;
  train_chunks?: number | null;
  effective_train_chunks?: number | null;
  completed_epochs_estimate?: number | null;
  epoch_percent?: number | null;
  latest_loss?: number | null;
  elapsed_seconds?: number | null;
  estimated_remaining_seconds?: number | null;
  metrics_available?: boolean;
  metrics_error?: string | null;
  chunk_count_error?: string | null;
  note?: string;
}

interface PreprocessProgress {
  model: string;
  checked_at: string;
  method: string;
  datastore_prefix: string;
  completed_chunks: number;
  completed_duration_seconds: number;
  completed_duration_hours: number;
  chunk_percent: number;
  duration_percent: number;
  remaining_chunks_estimate: number;
  latest_blob_modified?: string | null;
  elapsed_seconds?: number | null;
  estimated_total_seconds?: number | null;
  estimated_remaining_seconds?: number | null;
  job?: {
    job_name?: string | null;
    studio_url?: string | null;
    compute?: string | null;
    submitted_at?: string | null;
  };
  note: string;
  expected: {
    expected_chunks: number;
    expected_duration_seconds: number;
    expected_duration_hours: number;
    valid_source_rows: number;
    rejected_source_rows: number;
    chunk_seconds: number;
    sample_rate: number;
    channels: number;
  };
}

interface ReadinessCachePayload {
  checkedAt: string;
  readiness: TrainingReadiness;
}

interface PreprocessProgressCachePayload {
  checkedAt: string;
  progress: PreprocessProgress;
}

const preprocessModelForVariant = (variant: FinetuneVariant) => variant === 'diffusion' ? 'stable_audio_open_small' : 'musicgen';
const readinessCacheKey = (variant: FinetuneVariant) => `cara:finetune:${variant}:readiness:v1`;
const preprocessProgressCacheKey = (model: string) => `cara:finetune:${model}:preprocess-progress:v1`;

const readReadinessCache = (variant: FinetuneVariant): ReadinessCachePayload | null => {
  try {
    const raw = window.localStorage.getItem(readinessCacheKey(variant));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<ReadinessCachePayload>;
    if (!parsed.checkedAt || !parsed.readiness) return null;
    return { checkedAt: parsed.checkedAt, readiness: parsed.readiness };
  } catch {
    return null;
  }
};

const readPreprocessProgressCache = (model: string): PreprocessProgressCachePayload | null => {
  try {
    const raw = window.localStorage.getItem(preprocessProgressCacheKey(model));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PreprocessProgressCachePayload>;
    if (!parsed.checkedAt || !parsed.progress || parsed.progress.model !== model) return null;
    return { checkedAt: parsed.checkedAt, progress: parsed.progress };
  } catch {
    return null;
  }
};

const writeReadinessCache = (variant: FinetuneVariant, readiness: TrainingReadiness, checkedAt: string) => {
  try {
    window.localStorage.setItem(readinessCacheKey(variant), JSON.stringify({ checkedAt, readiness }));
  } catch {
    // Cache failures should never block Azure state refreshes.
  }
};

const writePreprocessProgressCache = (model: string, progress: PreprocessProgress) => {
  try {
    window.localStorage.setItem(
      preprocessProgressCacheKey(model),
      JSON.stringify({ checkedAt: progress.checked_at, progress }),
    );
  } catch {
    // Cache failures should never block Azure progress checks.
  }
};

const clearPreprocessProgressCache = (model: string) => {
  try {
    window.localStorage.removeItem(preprocessProgressCacheKey(model));
  } catch {
    // Cache failures should never block Azure preprocessing submissions.
  }
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

const formatCloudEvent = (job: NonNullable<TrainingReadiness['submitted_training_jobs']>[number]): string => {
  const action = job.action === 'training_preprocess_submitted'
    ? 'preprocess'
    : job.action === 'musicgen_encodec_cache_submitted'
      ? 'encodec'
      : job.action === 'azure_upload_confirmed'
        ? 'upload'
        : job.action ?? 'cloud';
  const jobName = job.job_name ? ` job=${job.job_name}` : '';
  const model = job.model_family ? ` model=${job.model_family}` : '';
  const compute = job.compute ? ` compute=${job.compute}` : '';
  return `[cloud:${action}]${jobName}${model}${compute}`.trim();
};

const SHARED_FIELDS: FieldDef[] = [
  {
    key: 'run_name',
    label: 'Run name',
    type: 'text',
    defaultValue: 'cara-finetune-001',
    hint: 'Used for log directory and W&B run id',
  },
  {
    key: 'subset_role',
    label: 'Dataset subset',
    type: 'select',
    options: ['music_train_candidate', 'confirmed_only', 'all_freesound'],
    defaultValue: 'music_train_candidate',
    hint: 'Manifest subset_role to feed the trainer',
  },
  {
    key: 'base_checkpoint',
    label: 'Base checkpoint',
    type: 'text',
    defaultValue: 'stabilityai/stable-audio-open-small',
    hint: 'HF id or local path resolvable on the VM',
  },
  {
    key: 'compute_target',
    label: 'Preprocess compute route',
    type: 'select',
    options: ['auto: H100 unless busy, then CPU', 'gpu-smoke-h100', 'cpu-prep-cluster'],
    defaultValue: 'auto: H100 unless busy, then CPU',
    hint: 'Preprocessing submits to H100 only when no active jobs are found on either H100-backed Azure ML compute target.',
  },
  {
    key: 'trainer_compute_target',
    label: 'Trainer compute target',
    type: 'select',
    options: ['gpu-smoke-h100', 'gpu-fulltraining-h100'],
    defaultValue: 'gpu-smoke-h100',
    hint: 'Training is GPU-only. If the H100 is busy, smoke training should wait or block, never fall back to CPU.',
  },
  {
    key: 'batch_size',
    label: 'Batch size',
    type: 'number',
    defaultValue: 8,
  },
  {
    key: 'num_workers',
    label: 'DataLoader workers',
    type: 'number',
    defaultValue: 0,
    hint: 'Use 0 for Stable Audio training on Azure to avoid DataLoader worker RAM blow-up.',
  },
  {
    key: 'checkpoint_keep_last_n',
    label: 'Checkpoint keep last N',
    type: 'number',
    defaultValue: 1,
    hint: 'Keeps only this many periodic checkpoints plus last.ckpt to protect Azure node disk during long runs.',
  },
  {
    key: 'lr',
    label: 'Learning rate',
    type: 'text',
    defaultValue: '1e-5',
    hint: 'Conservative first smoke default for diffusion fine-tuning; raise only after plumbing is proven.',
  },
  {
    key: 'max_steps',
    label: 'Smoke max steps',
    type: 'number',
    defaultValue: 250,
    hint: 'First smoke should validate trainer plumbing without spending much H100 time.',
  },
];

export const FinetuneView: React.FC<FinetuneViewProps> = ({
  variant,
  kicker,
  title,
  description,
  extraFields,
}) => {
  const fields = useMemo(
    () =>
      [...SHARED_FIELDS, ...extraFields].map((field) => {
        if (variant === 'autoregressive' && field.key === 'batch_size') {
          return {
            ...field,
            defaultValue: 2,
            hint: 'Real MusicGen LM smoke/full trainer is clamped to effective batch size 2 and float32 after the Step 08 dtype fix.',
          };
        }
        return field;
      }),
    [extraFields, variant],
  );
  const [values, setValues] = useState<Record<string, string | number>>(() => {
    const out: Record<string, string | number> = {};
    for (const f of fields) out[f.key] = f.defaultValue;
    return out;
  });

  const [logs, setLogs] = useState<string[]>(() => {
    const cached = readReadinessCache(variant);
    const cachedProgress = readPreprocessProgressCache(preprocessModelForVariant(variant));
    return [
      `[boot] ${variant} configuration ready. Azure ML launch remains disabled until the trainer command job is implemented.`,
      ...(cached ? [`[readiness:cache] Restored last gate check from ${new Date(cached.checkedAt).toLocaleString()}.`] : []),
      ...(cachedProgress ? [`[progress:cache] Restored ${cachedProgress.progress.model} prep progress from ${new Date(cachedProgress.checkedAt).toLocaleString()}.`] : []),
    ];
  });
  const [loss] = useState<{ t: number; v: number }[]>([]);
  const [readiness, setReadiness] = useState<TrainingReadiness | null>(() => readReadinessCache(variant)?.readiness ?? null);
  const [readinessError, setReadinessError] = useState<string | null>(null);
  const [readinessLoading, setReadinessLoading] = useState<boolean>(false);
  const [readinessCheckedAt, setReadinessCheckedAt] = useState<string | null>(() => readReadinessCache(variant)?.checkedAt ?? null);
  const [locking, setLocking] = useState<boolean>(false);
  const [confirmingUpload, setConfirmingUpload] = useState<boolean>(false);
  const [preparing, setPreparing] = useState<boolean>(false);
  const [cachingTokens, setCachingTokens] = useState<boolean>(false);
  const [preflighting, setPreflighting] = useState<boolean>(false);
  const [launching, setLaunching] = useState<boolean>(false);
  const [launchingFull, setLaunchingFull] = useState<boolean>(false);
  const [fullTrainingRun, setFullTrainingRun] = useState<boolean>(false);
  const [progress, setProgress] = useState<PreprocessProgress | null>(() => readPreprocessProgressCache(preprocessModelForVariant(variant))?.progress ?? null);
  const [progressLoading, setProgressLoading] = useState<boolean>(false);
  const [progressError, setProgressError] = useState<string | null>(null);

  const setField = (key: string, value: string | number) => {
    setValues((prev) => ({ ...prev, [key]: value }));
  };

  const commitReadiness = (payload: TrainingReadiness, checkedAt: string = new Date().toISOString()) => {
    setReadiness(payload);
    setReadinessCheckedAt(checkedAt);
    writeReadinessCache(variant, payload, checkedAt);
  };

  const refreshReadiness = async () => {
    setReadinessError(null);
    setReadinessLoading(true);
    try {
      const res = await fetch(`/api/training/readiness?variant=${encodeURIComponent(variant)}`);
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Training readiness is unavailable');
      commitReadiness(json);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Training readiness is unavailable';
      setReadinessError(message);
      setLogs((prev) => [...prev, `[readiness:error] ${message}`]);
    } finally {
      setReadinessLoading(false);
    }
  };

  useEffect(() => {
    const cached = readReadinessCache(variant);
    if (cached) {
      setReadiness(cached.readiness);
      setReadinessCheckedAt(cached.checkedAt);
    }
    const cachedProgress = readPreprocessProgressCache(preprocessModelForVariant(variant));
    setProgress(cachedProgress?.progress ?? null);
    setProgressError(null);
    refreshReadiness();
  }, [variant]);

  useEffect(() => {
    const trainingStatus = String(readiness?.stable_audio_training_progress?.status ?? '').toLowerCase();
    const isTrainerActive = ['running', 'starting', 'preparing', 'provisioning', 'queued', 'notstarted', 'finalizing'].includes(trainingStatus);
    if (variant !== 'diffusion' || !isTrainerActive) return;
    const interval = window.setInterval(() => {
      void refreshReadiness();
    }, 30000);
    return () => window.clearInterval(interval);
  }, [readiness?.stable_audio_training_progress?.status, variant]);

  useEffect(() => {
    const cloudEvents = readiness?.submitted_training_jobs ?? [];
    if (!cloudEvents.length) return;
    setLogs((prev) => {
      const hasOnlyBoot = prev.length === 1 && prev[0].startsWith('[boot]');
      if (!hasOnlyBoot) return prev;
      return [
        prev[0],
        '[cloud:history] Restored recent durable Azure submissions after page reload.',
        ...cloudEvents.slice(-6).map(formatCloudEvent),
      ];
    });
  }, [readiness?.submitted_training_jobs]);

  const handleLockManifest = async () => {
    setLocking(true);
    setReadinessError(null);
    setLogs((prev) => [...prev, '[lock] Locking CARA-Strong v0.4 manifest and registry...']);
    try {
      const res = await fetch('/api/training/lock-manifest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ require_audio_exists: false }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Manifest lock failed');
      commitReadiness(json.readiness);
      setLogs((prev) => [
        ...prev,
        `[lock:ok] accepted=${json.summary.accepted_count} · rejected=${json.summary.rejected_count} · pools=${json.summary.pool_count}`,
        `[lock:tir] ${json.summary.tir_id}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Manifest lock failed';
      setReadinessError(message);
      setLogs((prev) => [...prev, `[lock:error] ${message}`]);
    } finally {
      setLocking(false);
    }
  };

  const handlePrepareDatasets = async () => {
    setPreparing(true);
    setReadinessError(null);
    const modelFamily = preprocessModelForVariant(variant);
    const computeValue = String(values.compute_target);
    const computeStrategy = computeValue.startsWith('auto') ? 'prefer_h100_else_cpu' : computeValue === 'cpu-prep-cluster' ? 'cpu_only' : 'h100_only';
    clearPreprocessProgressCache(modelFamily);
    setProgress(null);
    setProgressError(null);
    setLogs((prev) => [...prev, `[preprocess] Submitting ${modelFamily} Azure preprocessing job...`]);
    try {
      const res = await fetch('/api/training/preprocess-model-datasets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dry_run: false, models: modelFamily, compute_strategy: computeStrategy }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Preprocessing submission failed');
      commitReadiness(json.readiness);
      const selected = json.job.compute_selected;
      setLogs((prev) => [
        ...prev,
        `[preprocess:submitted] job=${json.job.name ?? 'unknown'} · model=${json.job.model_family ?? modelFamily} · compute=${selected?.compute ?? json.job.compute ?? 'unknown'}`,
        selected?.reason ? `[preprocess:route] ${selected.reason}` : `[preprocess:route] Open Operations / Azure Runs to inspect live compute state.`,
        `[preprocess:output] ${json.job.output_path}`,
        `[preprocess:studio] ${json.job.studio_url ?? 'Open Operations / Azure Runs for job details.'}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Preprocessing submission failed';
      setReadinessError(message);
      setLogs((prev) => [...prev, `[preprocess:error] ${message}`]);
    } finally {
      setPreparing(false);
    }
  };

  const handleConfirmAzureUpload = async () => {
    if (!window.confirm('Confirm the full finetune-subset upload is complete in Azure ML? This unlocks model preprocessing.')) return;
    setConfirmingUpload(true);
    setReadinessError(null);
    setLogs((prev) => [...prev, '[upload] Confirming full Azure dataset upload is complete...']);
    try {
      const res = await fetch('/api/training/confirm-azure-upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confirmed: true }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Azure upload confirmation failed');
      commitReadiness(json.readiness);
      setLogs((prev) => [...prev, '[upload:ok] Azure upload confirmed. Model preprocessing is now unlocked.']);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Azure upload confirmation failed';
      setReadinessError(message);
      setLogs((prev) => [...prev, `[upload:error] ${message}`]);
    } finally {
      setConfirmingUpload(false);
    }
  };

  const handleCacheMusicGenTokens = async () => {
    setCachingTokens(true);
    setReadinessError(null);
    const computeValue = String(values.compute_target);
    const computeStrategy = computeValue.startsWith('auto') ? 'prefer_h100_else_cpu' : computeValue === 'cpu-prep-cluster' ? 'cpu_only' : 'h100_only';
    setLogs((prev) => [...prev, '[encodec] Submitting MusicGen EnCodec token-cache job...']);
    try {
      const res = await fetch('/api/training/cache-musicgen-tokens', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dry_run: false, compute_strategy: computeStrategy }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'MusicGen token-cache submission failed');
      commitReadiness(json.readiness);
      const selected = json.job.compute_selected;
      setLogs((prev) => [
        ...prev,
        `[encodec:submitted] job=${json.job.name ?? 'unknown'} · compute=${selected?.compute ?? json.job.compute ?? 'unknown'}`,
        selected?.reason ? `[encodec:route] ${selected.reason}` : '[encodec:route] Open Operations / Azure Runs to inspect live compute state.',
        `[encodec:output] ${json.job.output_path}`,
        `[encodec:studio] ${json.job.studio_url ?? 'Open Operations / Azure Runs for job details.'}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'MusicGen token-cache submission failed';
      setReadinessError(message);
      setLogs((prev) => [...prev, `[encodec:error] ${message}`]);
    } finally {
      setCachingTokens(false);
    }
  };

  const handleStableAudioPreflight = async () => {
    if (variant !== 'diffusion') return;
    setPreflighting(true);
    setLogs((prev) => [...prev, `[preflight] Submitting Stable Audio trainer preflight · checkpoint=${values.base_checkpoint}`]);
    try {
      const res = await fetch('/api/training/stable-audio-preflight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          checkpoint: String(values.base_checkpoint ?? 'stabilityai/stable-audio-open-small'),
          wrapper_check: true,
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Stable Audio trainer preflight submission failed');
      commitReadiness(json.readiness);
      setLogs((prev) => [
        ...prev,
        `[preflight:submitted] ${json.job?.name ?? 'azure-job'} · compute=${json.job?.compute ?? 'cpu-prep-cluster'}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Stable Audio trainer preflight submission failed';
      setLogs((prev) => [...prev, `[preflight:error] ${message}`]);
    } finally {
      setPreflighting(false);
      refreshReadiness();
    }
  };

  const handleMusicGenPreflight = async () => {
    if (variant !== 'autoregressive') return;
    setPreflighting(true);
    const musicGenCheckpoint = String(values.tokenizer_ckpt || values.base_checkpoint || 'facebook/musicgen-small');
    setLogs((prev) => [...prev, `[preflight] Submitting MusicGen trainer preflight · checkpoint=${musicGenCheckpoint}`]);
    try {
      const res = await fetch('/api/training/musicgen-preflight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          checkpoint: musicGenCheckpoint,
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'MusicGen trainer preflight submission failed');
      commitReadiness(json.readiness);
      setLogs((prev) => [
        ...prev,
        `[preflight:submitted] ${json.job?.name ?? 'azure-job'} · compute=${json.job?.compute ?? 'gpu-smoke-h100'}`,
        `[preflight:output] ${json.job?.output_path ?? 'Azure output path pending'}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'MusicGen trainer preflight submission failed';
      setReadinessError(message);
      setLogs((prev) => [...prev, `[preflight:error] ${message}`]);
    } finally {
      setPreflighting(false);
      refreshReadiness();
    }
  };

  const handleLaunchSmoke = async (stableAudioVariant: StableAudioSmokeVariant) => {
    if (variant !== 'diffusion') {
      setLogs((prev) => [...prev, '[blocked] MusicGen smoke trainer launch is still pending its autoregressive training command job.']);
      return;
    }
    const confirmationPhrases: Record<StableAudioSmokeVariant, string> = {
      no_cara_baseline: 'LAUNCH BASELINE SMOKE',
      cara_lite: 'LAUNCH CARA-LITE SMOKE',
      cara_head: 'LAUNCH CARA ATTRIBUTION-HEAD SMOKE',
      cara_strong: 'LAUNCH CARA-STRONG SMOKE',
    };
    const confirmationPhrase = confirmationPhrases[stableAudioVariant];
    const enteredPhrase = window.prompt(`This starts H100 smoke training in Azure ML. Type exactly: ${confirmationPhrase}`);
    if (enteredPhrase !== confirmationPhrase) {
      setLogs((prev) => [...prev, `[launch:blocked] Smoke launch cancelled; typed confirmation did not match ${confirmationPhrase}.`]);
      return;
    }
    const requestedRunName = String(values.run_name || 'cara-finetune-001');
    const suffixByVariant: Partial<Record<StableAudioSmokeVariant, string>> = {
      cara_lite: 'cara-lite',
      cara_head: 'cara-head',
      cara_strong: 'cara-strong',
    };
    const suffix = suffixByVariant[stableAudioVariant];
    const stableAudioRunName = suffix && !requestedRunName.match(new RegExp(suffix, 'i'))
      ? `${requestedRunName}-${suffix}`
      : requestedRunName;
    setLaunching(true);
    setReadinessError(null);
    setLogs((prev) => [
      ...prev,
      `[launch] Submitting Stable Audio smoke trainer · variant=${stableAudioVariant} · trainer_compute=${values.trainer_compute_target} · max_steps=${values.max_steps}`,
    ]);
    try {
      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_family: 'stable_audio_open_small',
          variant: stableAudioVariant,
          run_name: stableAudioRunName,
          max_steps: Number(values.max_steps),
          batch_size: Number(values.batch_size),
          num_workers: Number(values.num_workers ?? 0),
          learning_rate: Number(values.lr),
          attribution_loss_weight: Number(values.attribution_loss_weight ?? 0.05),
          checkpoint_keep_last_n: Number(values.checkpoint_keep_last_n ?? 1),
          checkpoint: values.base_checkpoint,
          trainer_compute_target: values.trainer_compute_target,
          dry_run: false,
          launch_confirmation: confirmationPhrase,
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Stable Audio smoke trainer submission failed');
      commitReadiness(json.readiness);
      setLogs((prev) => [
        ...prev,
        `[launch:submitted] job=${json.job.name ?? 'unknown'} · compute=${json.job.compute ?? values.trainer_compute_target}`,
        `[launch:output] ${json.job.output_path ?? 'Azure output path pending'}`,
        `[launch:studio] ${json.job.studio_url ?? 'Open Operations / Azure Runs for job details.'}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Stable Audio smoke trainer submission failed';
      setReadinessError(message);
      setLogs((prev) => [...prev, `[launch:error] ${message}`]);
    } finally {
      setLaunching(false);
    }
  };

  const handleLaunchMusicGenSmoke = async (musicGenVariant: MusicGenSmokeVariant) => {
    if (variant !== 'autoregressive') return;
    const confirmationPhrases: Record<MusicGenSmokeVariant, string> = {
      no_cara_baseline: 'LAUNCH MUSICGEN BASELINE SMOKE',
      cara_lite: 'LAUNCH MUSICGEN CARA-LITE SMOKE',
      cara_probe: 'LAUNCH MUSICGEN CARA PROBE SMOKE',
      cara_strong: 'LAUNCH MUSICGEN CARA-STRONG SMOKE',
    };
    const confirmationPhrase = confirmationPhrases[musicGenVariant];
    const enteredPhrase = window.prompt(`This starts H100 MusicGen smoke training in Azure ML. Type exactly: ${confirmationPhrase}`);
    if (enteredPhrase !== confirmationPhrase) {
      setLogs((prev) => [...prev, `[launch:blocked] MusicGen smoke launch cancelled; typed confirmation did not match ${confirmationPhrase}.`]);
      return;
    }
    const requestedRunName = String(values.run_name || 'cara-finetune-001');
    const suffixByVariant: Partial<Record<MusicGenSmokeVariant, string>> = {
      cara_lite: 'musicgen-cara-lite',
      cara_probe: 'musicgen-cara-probe',
      cara_strong: 'musicgen-cara-strong',
    };
    const suffix = suffixByVariant[musicGenVariant] ?? 'musicgen-baseline';
    const musicGenRunName = requestedRunName.match(new RegExp(suffix, 'i'))
      ? requestedRunName
      : `${requestedRunName}-${suffix}`;
    const musicGenBatchSize = Math.max(1, Math.min(Number(values.batch_size) || 2, 2));
    setLaunching(true);
    setReadinessError(null);
    setLogs((prev) => [
      ...prev,
      `[launch] Submitting real MusicGen LM smoke trainer · variant=${musicGenVariant} · trainer_compute=${values.trainer_compute_target} · max_steps=${values.max_steps} · batch_size=${musicGenBatchSize}`,
    ]);
    try {
      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_family: 'musicgen',
          variant: musicGenVariant,
          run_name: musicGenRunName,
          max_steps: Number(values.max_steps),
          batch_size: musicGenBatchSize,
          learning_rate: Number(values.lr),
          attribution_loss_weight: Number(values.attribution_loss_weight ?? 0.05),
          max_train_files: 2048,
          checkpoint: values.tokenizer_ckpt || values.base_checkpoint || 'facebook/musicgen-small',
          trainer_compute_target: values.trainer_compute_target,
          dry_run: false,
          launch_confirmation: confirmationPhrase,
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'MusicGen smoke trainer submission failed');
      commitReadiness(json.readiness);
      setLogs((prev) => [
        ...prev,
        `[launch:submitted] musicgen job=${json.job.name ?? 'unknown'} · compute=${json.job.compute ?? values.trainer_compute_target}`,
        `[launch:output] ${json.job.output_path ?? 'Azure output path pending'}`,
        `[launch:studio] ${json.job.studio_url ?? 'Open Operations / Azure Runs for job details.'}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'MusicGen smoke trainer submission failed';
      setReadinessError(message);
      setLogs((prev) => [...prev, `[launch:error] ${message}`]);
    } finally {
      setLaunching(false);
    }
  };

  const handleLaunchFullStableAudio = async () => {
    if (variant !== 'diffusion') return;
    const confirmationPhrase = 'LAUNCH FULL CARA-STRONG FINETUNE';
    const enteredPhrase = window.prompt(`This starts the full H100 CARA-Strong fine-tune in Azure ML. Type exactly: ${confirmationPhrase}`);
    if (enteredPhrase !== confirmationPhrase) {
      setLogs((prev) => [...prev, `[launch:blocked] Full fine-tune cancelled; typed confirmation did not match ${confirmationPhrase}.`]);
      return;
    }
    const requestedRunName = String(values.run_name || 'cara-finetune-001');
    const fullRunName = requestedRunName.match(/cara-strong-full/i)
      ? requestedRunName
      : `${requestedRunName}-cara-strong-full`;
    const requestedSteps = Number(values.max_steps);
    const fullMaxSteps = fullTrainingRun ? 0 : Number.isFinite(requestedSteps) && requestedSteps > 2000 ? requestedSteps : 20000;
    const fullComputeTarget = String(values.trainer_compute_target) === 'gpu-smoke-h100'
      ? 'gpu-fulltraining-h100'
      : String(values.trainer_compute_target);
    setLaunchingFull(true);
    setReadinessError(null);
    setLogs((prev) => [
      ...prev,
      `[launch] Submitting full Stable Audio CARA-Strong fine-tune · trainer_compute=${fullComputeTarget} · ${fullTrainingRun ? 'full_training_run=true' : `max_steps=${fullMaxSteps}`}`,
    ]);
    try {
      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_family: 'stable_audio_open_small',
          variant: 'cara_strong',
          training_scope: 'full',
          run_name: fullRunName,
          max_steps: fullMaxSteps,
          full_training_run: fullTrainingRun,
          batch_size: Number(values.batch_size),
          num_workers: Number(values.num_workers ?? 0),
          learning_rate: Number(values.lr),
          attribution_loss_weight: Number(values.attribution_loss_weight ?? 0.05),
          checkpoint_keep_last_n: Number(values.checkpoint_keep_last_n ?? 1),
          max_train_files: 0,
          max_eval_files: 0,
          max_eval_batches: 0,
          run_eval: true,
          checkpoint: values.base_checkpoint,
          trainer_compute_target: fullComputeTarget,
          dry_run: false,
          launch_confirmation: confirmationPhrase,
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Full Stable Audio fine-tune submission failed');
      commitReadiness(json.readiness);
      setLogs((prev) => [
        ...prev,
        `[launch:submitted] full job=${json.job.name ?? 'unknown'} · compute=${json.job.compute ?? fullComputeTarget}`,
        `[launch:output] ${json.job.output_path ?? 'Azure output path pending'}`,
        `[launch:studio] ${json.job.studio_url ?? 'Open Operations / Azure Runs for job details.'}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Full Stable Audio fine-tune submission failed';
      setReadinessError(message);
      setLogs((prev) => [...prev, `[launch:error] ${message}`]);
    } finally {
      setLaunchingFull(false);
    }
  };

  const handleLaunchFullMusicGen = async () => {
    if (variant !== 'autoregressive') return;
    const confirmationPhrase = 'LAUNCH FULL MUSICGEN CARA-STRONG FINETUNE';
    const enteredPhrase = window.prompt(`This starts the full H100 real MusicGen LM CARA-Strong fine-tune in Azure ML. Type exactly: ${confirmationPhrase}`);
    if (enteredPhrase !== confirmationPhrase) {
      setLogs((prev) => [...prev, `[launch:blocked] Full MusicGen fine-tune cancelled; typed confirmation did not match ${confirmationPhrase}.`]);
      return;
    }
    const requestedRunName = String(values.run_name || 'cara-finetune-001');
    const fullRunName = requestedRunName.match(/musicgen-cara-strong-full/i)
      ? requestedRunName
      : `${requestedRunName}-musicgen-cara-strong-full`;
    const requestedSteps = Number(values.max_steps);
    const fullMaxSteps = Number.isFinite(requestedSteps) && requestedSteps > 2000 ? requestedSteps : 20000;
    const musicGenBatchSize = Math.max(1, Math.min(Number(values.batch_size) || 2, 2));
    const fullComputeTarget = String(values.trainer_compute_target) === 'gpu-smoke-h100'
      ? 'gpu-fulltraining-h100'
      : String(values.trainer_compute_target);
    setLaunchingFull(true);
    setReadinessError(null);
    setLogs((prev) => [
      ...prev,
      `[launch] Submitting full real MusicGen LM CARA-Strong fine-tune · trainer_compute=${fullComputeTarget} · max_steps=${fullMaxSteps} · batch_size=${musicGenBatchSize}`,
    ]);
    try {
      const res = await fetch('/api/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_family: 'musicgen',
          variant: 'cara_strong',
          training_scope: 'full',
          run_name: fullRunName,
          max_steps: fullMaxSteps,
          full_training_run: false,
          batch_size: musicGenBatchSize,
          learning_rate: Number(values.lr),
          attribution_loss_weight: Number(values.attribution_loss_weight ?? 0.05),
          max_train_files: 0,
          max_eval_files: 0,
          checkpoint: values.tokenizer_ckpt || values.base_checkpoint || 'facebook/musicgen-small',
          trainer_compute_target: fullComputeTarget,
          dry_run: false,
          launch_confirmation: confirmationPhrase,
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Full MusicGen fine-tune submission failed');
      commitReadiness(json.readiness);
      setLogs((prev) => [
        ...prev,
        `[launch:submitted] full musicgen job=${json.job.name ?? 'unknown'} · compute=${json.job.compute ?? fullComputeTarget}`,
        `[launch:output] ${json.job.output_path ?? 'Azure output path pending'}`,
        `[launch:studio] ${json.job.studio_url ?? 'Open Operations / Azure Runs for job details.'}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Full MusicGen fine-tune submission failed';
      setReadinessError(message);
      setLogs((prev) => [...prev, `[launch:error] ${message}`]);
    } finally {
      setLaunchingFull(false);
    }
  };

  const handleLaunch = async () => {
    const nextVariant = String(readiness?.stable_audio_smoke_sequence?.next_variant ?? 'no_cara_baseline') as StableAudioSmokeVariant;
    if (!['no_cara_baseline', 'cara_lite', 'cara_head', 'cara_strong'].includes(nextVariant)) {
      setLogs((prev) => [...prev, `[blocked] ${readiness?.stable_audio_smoke_sequence?.reason ?? 'The next smoke stage is not implemented yet.'}`]);
      return;
    }
    await handleLaunchSmoke(nextVariant);
  };

  const handleRefresh = () => {
    refreshReadiness();
    setLogs((prev) => [...prev, `[monitor] Refreshed fine-tuning readiness. Azure job history remains under Operations / Azure Runs.`]);
  };

  const handleCheckPreprocessProgress = async () => {
    const modelFamily = preprocessModelForVariant(variant);
    setProgressLoading(true);
    setProgressError(null);
    setLogs((prev) => [...prev, `[progress] Checking ${modelFamily} prepared blobs in Azure datastore...`]);
    try {
      const res = await fetch(`/api/training/preprocess-progress?model=${encodeURIComponent(modelFamily)}`);
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Preprocessing progress check failed');
      setProgress(json);
      writePreprocessProgressCache(modelFamily, json);
      setLogs((prev) => [
        ...prev,
        `[progress:ok] chunks=${json.completed_chunks}/${json.expected.expected_chunks} · duration=${json.duration_percent}%`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Preprocessing progress check failed';
      setProgressError(message);
      setLogs((prev) => [...prev, `[progress:error] ${message}`]);
    } finally {
      setProgressLoading(false);
    }
  };

  useEffect(() => {
    if (!readiness || progress || progressLoading || progressError) return;
    const modelFamily = variant === 'diffusion' ? 'stable_audio_open_small' : 'musicgen';
    const hasSubmittedPrep = (readiness.submitted_training_jobs ?? []).some(
      (job) => job.action === 'training_preprocess_submitted' && job.model_family === modelFamily,
    );
    if (!hasSubmittedPrep) return;
    const timer = window.setTimeout(() => {
      void handleCheckPreprocessProgress();
    }, 250);
    return () => window.clearTimeout(timer);
  }, [readiness, progress, progressError, progressLoading, variant]);

  const lastLoss = loss[loss.length - 1]?.v ?? null;
  const splitCounts = readiness?.lock.summary?.split_counts ?? {};
  const windowPolicy = variant === 'diffusion'
    ? readiness?.audio_window_policy?.stable_audio_open_small
    : readiness?.audio_window_policy?.musicgen;
  const recentTrainingJobs = readiness?.submitted_training_jobs ?? [];
  const modelFamilyForPage = variant === 'diffusion' ? 'stable_audio_open_small' : 'musicgen';
  const latestModelPreprocessJob = recentTrainingJobs
    .slice()
    .reverse()
    .find((job) => job.action === 'training_preprocess_submitted' && job.model_family === modelFamilyForPage);
  const latestModelPreprocessState = readiness?.preprocess_jobs?.[modelFamilyForPage] ?? null;
  const hasModelPreprocessJob = Boolean(latestModelPreprocessJob || latestModelPreprocessState?.latest_job);
  const modelPreprocessComplete = Boolean(
    latestModelPreprocessState?.passed
    || (
      progress
      && progress.model === modelFamilyForPage
      && (progress.duration_percent >= 99.5 || progress.chunk_percent >= 99.5)
    ),
  );
  const modelPreprocessRunningRaw = (Boolean(latestModelPreprocessState?.active) || hasModelPreprocessJob) && !modelPreprocessComplete;
  const hasModelPreprocessReadyRaw = modelPreprocessComplete;
  const musicGenTokenCache = variant === 'autoregressive' ? readiness?.musicgen_token_cache ?? null : null;
  const hasMusicGenTokenCacheJob = Boolean(musicGenTokenCache?.latest_job) || recentTrainingJobs.some((job) => job.action === 'musicgen_encodec_cache_submitted');
  const musicGenTokenCacheRunning = variant === 'autoregressive' && Boolean(musicGenTokenCache?.active);
  const musicGenTokenCacheReady = variant !== 'autoregressive' || Boolean(musicGenTokenCache?.passed);
  const activeStableAudioSmokeJob = variant === 'diffusion' ? readiness?.active_stable_audio_smoke_job ?? null : null;
  const activeMusicGenTrainerJob = variant === 'autoregressive' ? readiness?.active_musicgen_trainer_job ?? null : null;
  const activeTrainerJob = variant === 'diffusion' ? activeStableAudioSmokeJob : activeMusicGenTrainerJob;
  const hasActiveTrainerJob = Boolean(activeTrainerJob);
  const stableAudioSmokeSequence = variant === 'diffusion' ? readiness?.stable_audio_smoke_sequence ?? null : null;
  const stableAudioSmokeVariants = stableAudioSmokeSequence?.variants ?? {};
  const baselineSmoke = stableAudioSmokeVariants.no_cara_baseline ?? null;
  const caraLiteSmoke = stableAudioSmokeVariants.cara_lite ?? null;
  const caraHeadSmoke = stableAudioSmokeVariants.cara_head ?? null;
  const caraStrongSmoke = stableAudioSmokeVariants.cara_strong ?? null;
  const baselineSmokePassed = Boolean(baselineSmoke?.passed);
  const baselineSmokeActive = Boolean(baselineSmoke?.active);
  const caraLiteSmokePassed = Boolean(caraLiteSmoke?.passed);
  const caraLiteSmokeActive = Boolean(caraLiteSmoke?.active);
  const caraHeadSmokePassed = Boolean(caraHeadSmoke?.passed);
  const caraHeadSmokeActive = Boolean(caraHeadSmoke?.active);
  const caraStrongSmokePassed = Boolean(caraStrongSmoke?.passed);
  const caraStrongSmokeActive = Boolean(caraStrongSmoke?.active);
  const stableAudioFullTraining = variant === 'diffusion' ? readiness?.stable_audio_full_training ?? null : null;
  const stableAudioFullPassed = Boolean(stableAudioFullTraining?.passed);
  const stableAudioFullActive = Boolean(stableAudioFullTraining?.active);
  const stableAudioTrainingProgress = variant === 'diffusion' ? readiness?.stable_audio_training_progress ?? null : null;
  const nextStableAudioSmokeVariant = String(stableAudioSmokeSequence?.next_variant ?? 'no_cara_baseline');
  const stableAudioPreflight = variant === 'diffusion' ? readiness?.stable_audio_preflight ?? null : null;
  const stableAudioPreflightPassed = variant !== 'diffusion' || Boolean(stableAudioPreflight?.passed);
  const stableAudioPreflightActive = variant === 'diffusion' && Boolean(stableAudioPreflight?.active);
  const stableAudioPreflightLatestJob = stableAudioPreflight?.latest_job ?? null;
  const stableAudioPreflightStatus = String(stableAudioPreflightLatestJob?.status ?? '').toLowerCase();
  const stableAudioPreflightCreatedAtMs = stableAudioPreflightLatestJob?.created_at ? Date.parse(stableAudioPreflightLatestJob.created_at) : 0;
  const caraStrongLatestCreatedAtMs = caraStrongSmoke?.latest_job?.created_at ? Date.parse(caraStrongSmoke.latest_job.created_at) : 0;
  const caraStrongLatestPassedCreatedAtMs = caraStrongSmoke?.latest_passed_job?.created_at ? Date.parse(caraStrongSmoke.latest_passed_job.created_at) : 0;
  const caraStrongLatestStatus = String(caraStrongSmoke?.latest_job?.status ?? '').toLowerCase();
  const caraStrongLatestAttemptFailed = ['failed', 'canceled', 'cancelled'].includes(caraStrongLatestStatus)
    && caraStrongLatestCreatedAtMs >= caraStrongLatestPassedCreatedAtMs;
  const caraStrongRepeatRecommended = variant === 'diffusion'
    && caraHeadSmokePassed
    && stableAudioPreflightPassed
    && (
      caraStrongLatestAttemptFailed
      || (stableAudioPreflightCreatedAtMs > 0 && stableAudioPreflightCreatedAtMs > caraStrongLatestCreatedAtMs)
    )
    && !stableAudioFullActive
    && !stableAudioFullPassed;
  const stableAudioDownstreamDatasetEvidence = variant === 'diffusion' && (
    Boolean(stableAudioPreflight?.passed)
    || baselineSmokePassed
    || caraLiteSmokePassed
    || caraHeadSmokePassed
    || caraStrongSmokePassed
    || stableAudioFullPassed
    || stableAudioFullActive
  );
  const musicGenPreflight = variant === 'autoregressive' ? readiness?.musicgen_preflight ?? null : null;
  const musicGenPreflightPassed = variant !== 'autoregressive' || Boolean(musicGenPreflight?.passed);
  const musicGenPreflightActive = variant === 'autoregressive' && Boolean(musicGenPreflight?.active);
  const musicGenSmokeSequence = variant === 'autoregressive' ? readiness?.musicgen_smoke_sequence ?? null : null;
  const musicGenSmokeVariants = musicGenSmokeSequence?.variants ?? {};
  const musicGenBaselineSmoke = musicGenSmokeVariants.no_cara_baseline ?? null;
  const musicGenCaraLiteSmoke = musicGenSmokeVariants.cara_lite ?? null;
  const musicGenCaraProbeSmoke = musicGenSmokeVariants.cara_probe ?? null;
  const musicGenCaraStrongSmoke = musicGenSmokeVariants.cara_strong ?? null;
  const musicGenBaselinePassed = Boolean(musicGenBaselineSmoke?.passed);
  const musicGenBaselineActive = Boolean(musicGenBaselineSmoke?.active);
  const musicGenCaraLitePassed = Boolean(musicGenCaraLiteSmoke?.passed);
  const musicGenCaraLiteActive = Boolean(musicGenCaraLiteSmoke?.active);
  const musicGenCaraProbePassed = Boolean(musicGenCaraProbeSmoke?.passed);
  const musicGenCaraProbeActive = Boolean(musicGenCaraProbeSmoke?.active);
  const musicGenCaraStrongPassed = Boolean(musicGenCaraStrongSmoke?.passed);
  const musicGenCaraStrongActive = Boolean(musicGenCaraStrongSmoke?.active);
  const nextMusicGenSmokeVariant = String(musicGenSmokeSequence?.next_variant ?? 'no_cara_baseline') as MusicGenSequenceVariant;
  const musicGenDownstreamDatasetEvidence = variant === 'autoregressive' && (
    musicGenTokenCacheReady
    || Boolean(musicGenPreflight?.passed)
    || musicGenBaselinePassed
    || musicGenCaraLitePassed
    || musicGenCaraProbePassed
    || musicGenCaraStrongPassed
  );
  const downstreamDatasetEvidence = stableAudioDownstreamDatasetEvidence || musicGenDownstreamDatasetEvidence;
  const hasModelPreprocessSubmittedForPage = hasModelPreprocessJob || downstreamDatasetEvidence;
  const hasModelPreprocessReady = hasModelPreprocessReadyRaw || downstreamDatasetEvidence;
  const modelPreprocessRunning = modelPreprocessRunningRaw && !downstreamDatasetEvidence;
  const hasTrainerInputsReady = variant === 'autoregressive'
    ? hasModelPreprocessReady && musicGenTokenCacheReady
    : hasModelPreprocessReady;
  const stableAudioPreflightCanRun = variant === 'diffusion'
    && hasModelPreprocessReady
    && !preflighting
    && !stableAudioPreflightActive
    && !stableAudioFullActive
    && !stableAudioFullPassed;
  const musicGenPreflightCanRun = variant === 'autoregressive'
    && musicGenTokenCacheReady
    && !preflighting
    && !musicGenPreflightActive
    && !hasActiveTrainerJob;
  const trainerLaunchEnabled = Boolean(readiness?.training_launch_enabled);
  const trainerLaunchBlockedReason = readiness?.training_launch_reason ?? 'Trainer command job is not implemented yet.';
  const manifestLocked = Boolean(readiness?.lock.locked);
  const azureUploadConfirmed = Boolean(readiness?.azure_upload?.confirmed);
  const stageTotal = variant === 'diffusion' ? 9 : 12;
  const currentStage = !manifestLocked
    ? 1
    : !azureUploadConfirmed
      ? 2
      : !hasModelPreprocessSubmittedForPage || modelPreprocessRunning
        ? 3
        : variant === 'diffusion' && !stableAudioPreflightPassed
          ? 4
          : variant === 'autoregressive' && (!hasMusicGenTokenCacheJob || !musicGenTokenCacheReady)
            ? 4
          : variant === 'autoregressive' && !musicGenPreflightPassed
            ? 7
          : variant === 'autoregressive' && (musicGenBaselineActive || (!musicGenBaselinePassed && nextMusicGenSmokeVariant === 'no_cara_baseline'))
            ? 8
          : variant === 'autoregressive' && (musicGenCaraLiteActive || (musicGenBaselinePassed && !musicGenCaraLitePassed && nextMusicGenSmokeVariant === 'cara_lite'))
            ? 9
          : variant === 'autoregressive' && (musicGenCaraProbeActive || (musicGenCaraLitePassed && !musicGenCaraProbePassed && nextMusicGenSmokeVariant === 'cara_probe'))
            ? 10
          : variant === 'autoregressive' && (musicGenCaraStrongActive || (musicGenCaraProbePassed && !musicGenCaraStrongPassed && nextMusicGenSmokeVariant === 'cara_strong'))
            ? 11
          : variant === 'diffusion' && (baselineSmokeActive || (!baselineSmokePassed && nextStableAudioSmokeVariant === 'no_cara_baseline'))
            ? 5
          : variant === 'diffusion' && (caraLiteSmokeActive || (baselineSmokePassed && !caraLiteSmokePassed && nextStableAudioSmokeVariant === 'cara_lite'))
            ? 6
          : variant === 'diffusion' && (caraHeadSmokeActive || (caraLiteSmokePassed && !caraHeadSmokePassed && nextStableAudioSmokeVariant === 'cara_head'))
            ? 7
          : variant === 'diffusion' && (caraStrongSmokeActive || caraStrongRepeatRecommended || (caraHeadSmokePassed && !caraStrongSmokePassed && nextStableAudioSmokeVariant === 'cara_strong'))
            ? 8
          : variant === 'diffusion' && (stableAudioFullActive || (caraStrongSmokePassed && !stableAudioFullPassed && !caraStrongRepeatRecommended))
            ? 9
          : stageTotal;
  const stageLabel = variant === 'diffusion' && currentStage === 9
    ? stableAudioFullActive
      ? 'full fine-tune running'
      : stableAudioFullPassed
        ? 'full fine-tune passed'
        : 'ready for full fine-tune'
    : variant === 'diffusion' && currentStage === 8
    ? caraStrongSmokeActive
      ? 'CARA-Strong running'
      : caraStrongRepeatRecommended
        ? 're-run CARA-Strong'
        : caraStrongSmokePassed
          ? 'CARA-Strong passed'
          : 'launch CARA-Strong'
    : currentStage === stageTotal
      ? hasActiveTrainerJob
        ? 'smoke running'
        : trainerLaunchEnabled
        ? 'ready to launch'
        : 'trainer pending'
    : currentStage === 1
      ? 'lock manifest'
      : currentStage === 2
        ? 'confirm upload'
        : currentStage === 3
          ? modelPreprocessRunning
            ? 'dataset prep running'
            : 'prepare dataset'
          : variant === 'diffusion' && currentStage === 5
            ? baselineSmokeActive
              ? 'baseline running'
              : baselineSmokePassed
                ? 'baseline passed'
                : 'launch baseline'
          : variant === 'diffusion' && currentStage === 6
            ? caraLiteSmokeActive
              ? 'CARA-lite running'
              : caraLiteSmokePassed
                ? 'CARA-lite passed'
                : 'launch CARA-lite'
          : variant === 'diffusion' && currentStage === 7
            ? caraHeadSmokeActive
              ? 'CARA head running'
              : caraHeadSmokePassed
                ? 'CARA head passed'
                : 'launch CARA head'
          : variant === 'diffusion' && currentStage === 8
            ? caraStrongSmokeActive
              ? 'CARA-Strong running'
              : caraStrongRepeatRecommended
                ? 're-run CARA-Strong'
                : caraStrongSmokePassed
                  ? 'CARA-Strong passed'
                  : 'launch CARA-Strong'
          : variant === 'diffusion'
            ? stableAudioPreflightActive
              ? 'preflight running'
              : stableAudioPreflightStatus === 'failed'
                ? 'preflight failed'
                : 'run preflight'
            : musicGenPreflightActive
              ? 'preflight running'
            : musicGenTokenCacheRunning
              ? 'token cache submitted'
              : variant === 'autoregressive' && currentStage === 8
                ? musicGenBaselineActive ? 'baseline running' : 'launch baseline'
              : variant === 'autoregressive' && currentStage === 9
                ? musicGenCaraLiteActive ? 'CARA-lite running' : 'launch CARA-lite'
              : variant === 'autoregressive' && currentStage === 10
                ? musicGenCaraProbeActive ? 'probe running' : 'launch CARA probe'
              : variant === 'autoregressive' && currentStage === 11
                ? musicGenCaraStrongActive ? 'CARA-Strong running' : 'launch CARA-Strong'
              : 'cache tokens';
  const status = !readiness
    ? 'Loading gates'
    : variant === 'diffusion' && stableAudioFullActive
      ? 'Full fine-tune running'
    : hasActiveTrainerJob
        ? `Trainer running (${activeStableAudioSmokeJob?.status ?? 'active'})`
    : variant === 'diffusion' && stableAudioPreflightActive
    ? 'Preflight running'
    : variant === 'diffusion' && stableAudioFullActive
      ? 'Full fine-tune running'
    : variant === 'diffusion' && hasTrainerInputsReady && !stableAudioPreflightPassed
      ? stableAudioPreflightStatus === 'failed'
        ? 'Preflight failed'
        : 'Preflight required'
    : variant === 'autoregressive' && musicGenPreflightActive
      ? 'MusicGen preflight running'
    : variant === 'autoregressive' && hasTrainerInputsReady && !musicGenPreflightPassed
      ? 'MusicGen preflight required'
    : variant === 'autoregressive' && musicGenBaselinePassed && !musicGenCaraLitePassed
      ? 'Ready for MusicGen CARA-lite smoke'
    : variant === 'autoregressive' && musicGenCaraLitePassed && !musicGenCaraProbePassed
      ? 'Ready for MusicGen CARA probe smoke'
    : variant === 'autoregressive' && musicGenCaraProbePassed && !musicGenCaraStrongPassed
      ? 'Ready for MusicGen CARA-Strong smoke'
    : variant === 'autoregressive' && musicGenCaraStrongPassed
      ? 'MusicGen CARA-Strong smoke complete'
    : readiness.status !== 'ready_for_smoke_training'
      ? 'Blocked'
      : variant === 'diffusion' && baselineSmokePassed && !caraLiteSmokePassed
        ? 'Ready for CARA-lite smoke'
      : variant === 'diffusion' && caraLiteSmokePassed && !caraHeadSmokePassed
        ? 'Ready for CARA attribution-head smoke'
      : variant === 'diffusion' && caraHeadSmokePassed && !caraStrongSmokePassed
        ? 'Ready for CARA-Strong smoke'
      : variant === 'diffusion' && caraStrongRepeatRecommended
        ? 'Ready to re-run CARA-Strong smoke'
      : variant === 'diffusion' && caraStrongSmokePassed && !stableAudioFullPassed
        ? 'Ready for full CARA-Strong fine-tune'
      : variant === 'diffusion' && stableAudioFullPassed
        ? 'Full CARA-Strong fine-tune complete'
      : hasTrainerInputsReady && stableAudioPreflightPassed
        ? trainerLaunchEnabled
          ? 'Ready to launch smoke'
          : 'Smoke trainer pending'
        : modelPreprocessRunning
          ? 'Dataset prep running'
          : musicGenTokenCacheRunning
            ? 'Token cache running'
            : 'Prep gates active';
  const prepareLabel = variant === 'diffusion' ? 'Prepare Stable Audio Dataset' : 'Prepare MusicGen Dataset';
  const prepareStepLabel = variant === 'autoregressive' ? `03 ${prepareLabel}` : `03 ${prepareLabel}`;
  const smokeStepByVariant: Record<string, string> = {
    no_cara_baseline: '05',
    cara_lite: '06',
    cara_head: '07',
    cara_strong: '08',
  };
  const smokeNameByVariant: Record<string, string> = {
    no_cara_baseline: 'Baseline Smoke',
    cara_lite: 'CARA-Lite Smoke',
    cara_head: 'CARA Attribution-Head Smoke',
    cara_strong: 'CARA-Strong Smoke',
  };
  const nextLaunchStepNumber = smokeStepByVariant[nextStableAudioSmokeVariant] ?? '05';
  const nextLaunchName = smokeNameByVariant[nextStableAudioSmokeVariant] ?? 'Baseline Smoke';
  const nextStableAudioSmokeLaunchable = ['no_cara_baseline', 'cara_lite', 'cara_head', 'cara_strong'].includes(nextStableAudioSmokeVariant);
  const launchStepLabel = hasActiveTrainerJob
    ? `${nextLaunchStepNumber} Smoke Running`
    : trainerLaunchEnabled && nextStableAudioSmokeLaunchable
      ? `${nextLaunchStepNumber} Launch ${nextLaunchName}`
      : variant === 'diffusion' && !nextStableAudioSmokeLaunchable
        ? `${stableAudioSmokeSequence?.next_stage ?? '07'} Future Smoke Pending`
        : `${nextLaunchStepNumber} Smoke Trainer Pending`;
  const stageButtonClass = (stage: number) => `btn stage-action${currentStage === stage ? ' is-current' : ' btn-ghost is-muted'}`;
  const baselineLaunchEnabled = variant === 'diffusion'
    && nextStableAudioSmokeVariant === 'no_cara_baseline'
    && !baselineSmokePassed
    && !hasActiveTrainerJob
    && hasTrainerInputsReady
    && stableAudioPreflightPassed
    && trainerLaunchEnabled;
  const caraLiteLaunchEnabled = variant === 'diffusion'
    && nextStableAudioSmokeVariant === 'cara_lite'
    && baselineSmokePassed
    && !caraLiteSmokePassed
    && !hasActiveTrainerJob
    && hasTrainerInputsReady
    && stableAudioPreflightPassed
    && trainerLaunchEnabled;
  const caraHeadLaunchEnabled = variant === 'diffusion'
    && nextStableAudioSmokeVariant === 'cara_head'
    && caraLiteSmokePassed
    && !caraHeadSmokePassed
    && !hasActiveTrainerJob
    && hasTrainerInputsReady
    && stableAudioPreflightPassed
    && trainerLaunchEnabled;
  const caraStrongLaunchEnabled = variant === 'diffusion'
    && caraHeadSmokePassed
    && (nextStableAudioSmokeVariant === 'cara_strong' || caraStrongSmokePassed)
    && !stableAudioFullActive
    && !stableAudioFullPassed
    && !hasActiveTrainerJob
    && hasTrainerInputsReady
    && stableAudioPreflightPassed
    && trainerLaunchEnabled;
  const fullStableAudioLaunchEnabled = variant === 'diffusion'
    && caraStrongSmokePassed
    && !caraStrongRepeatRecommended
    && !stableAudioFullPassed
    && !stableAudioFullActive
    && !hasActiveTrainerJob
    && hasTrainerInputsReady
    && stableAudioPreflightPassed
    && trainerLaunchEnabled;
  const musicGenTrainerReady = variant === 'autoregressive'
    && hasTrainerInputsReady
    && musicGenPreflightPassed
    && !hasActiveTrainerJob;
  const musicGenBaselineLaunchEnabled = musicGenTrainerReady
    && nextMusicGenSmokeVariant === 'no_cara_baseline'
    && !musicGenBaselinePassed;
  const musicGenCaraLiteLaunchEnabled = musicGenTrainerReady
    && nextMusicGenSmokeVariant === 'cara_lite'
    && musicGenBaselinePassed
    && !musicGenCaraLitePassed;
  const musicGenCaraProbeLaunchEnabled = musicGenTrainerReady
    && nextMusicGenSmokeVariant === 'cara_probe'
    && musicGenCaraLitePassed
    && !musicGenCaraProbePassed;
  const musicGenCaraStrongLaunchEnabled = musicGenTrainerReady
    && nextMusicGenSmokeVariant === 'cara_strong'
    && musicGenCaraProbePassed
    && !musicGenCaraStrongPassed;
  const musicGenFullLaunchEnabled = variant === 'autoregressive'
    && musicGenCaraStrongPassed
    && !hasActiveTrainerJob
    && hasTrainerInputsReady
    && musicGenPreflightPassed
    && trainerLaunchEnabled;
  const stableAudioSmokeLabel = (
    step: string,
    label: string,
    state?: { passed?: boolean; active?: boolean; latest_job?: ActiveTrainingJob | null } | null,
  ) => {
    if (state?.active) return `${step} ${label} Running`;
    if (state?.passed) return `${step} ${label} Passed`;
    return `${step} ${label}`;
  };

  return (
    <div style={{ display: 'grid', gap: 20 }}>
      <PageHeader
        kicker={kicker}
        title={title}
        description={description}
        actions={<PlaceholderBadge />}
      />

      <section className="kpi-grid" aria-label="Run status">
        <div className="kpi">
          <div className="kpi-label">Status</div>
          <div className="kpi-value" style={{ fontSize: 22 }}>
            <Activity size={16} style={{ marginRight: 8, opacity: 0.8 }} />
            {status}
          </div>
          <div className="kpi-trend">{values.run_name as string}</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Prep stage</div>
          <div className="kpi-value">
            {currentStage}
            <span className="kpi-unit">/ {stageTotal}</span>
          </div>
          <div className="kpi-trend">{stageLabel}</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Last loss</div>
          <div className="kpi-value">{lastLoss !== null ? lastLoss.toFixed(4) : '—'}</div>
          <div className="kpi-trend">not started</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Compute target</div>
          <div className="kpi-value" style={{ fontSize: 18 }}>
            <Cloud size={14} style={{ marginRight: 6, opacity: 0.8 }} />
            {values.compute_target as string}
          </div>
          <div className="kpi-trend">preprocessing route</div>
        </div>
      </section>

      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Readiness Gates</div>
            <div className="card-meta">
              {readinessLoading
                ? 'refreshing Azure state'
                : readinessCheckedAt
                  ? `checked ${new Date(readinessCheckedAt).toLocaleString()}`
                  : hasActiveTrainerJob
                    ? 'cloud run active'
                    : readiness?.training_launch_enabled
                      ? 'launch enabled'
                      : 'launch blocked'}
            </div>
          </div>
          {readinessError ? (
            <div className="pool-empty-state">
              <AlertTriangle size={18} /> {readinessError}
            </div>
          ) : null}
          <div className="controls">
            <div className="control-row">
              {(readiness?.gates ?? []).map((gate) => (
                <div className="field" key={gate.id}>
                  <label>{gate.label}</label>
                  <div className={`live-chip ${gate.passed ? '' : 'warn'}`}>
                    {gate.passed ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                    {gate.passed ? 'Pass' : 'Required'}
                  </div>
                </div>
              ))}
            </div>
            <div className="row" style={{ justifyContent: 'flex-end', gap: 10 }}>
              <button className="btn btn-ghost" onClick={refreshReadiness} type="button" disabled={readinessLoading}>
                <RefreshCw size={16} className={readinessLoading ? 'spin' : ''} /> {readinessLoading ? 'Refreshing Gates...' : 'Refresh Gates'}
              </button>
              <button className="btn btn-ghost" type="button" onClick={handleCheckPreprocessProgress} disabled={progressLoading}>
                <RefreshCw size={16} className={progressLoading ? 'spin' : ''} /> {progressLoading ? 'Checking...' : 'Check Prep Progress'}
              </button>
            </div>
            <div className="stage-action-list" aria-label="Fine-tuning preparation steps">
              <button
                className={stageButtonClass(1)}
                onClick={handleLockManifest}
                type="button"
                disabled={locking || manifestLocked || !readiness}
                title={manifestLocked ? '01 is complete. Unlocking/relocking should be a deliberate registry operation.' : 'First lock the manifest and pool registry.'}
              >
                <FileLock2 size={16} /> {locking ? '01 Locking...' : '01 Lock Manifest'}
              </button>
              <button
                className={stageButtonClass(2)}
                onClick={handleConfirmAzureUpload}
                type="button"
                disabled={confirmingUpload || !manifestLocked || azureUploadConfirmed}
                title={!manifestLocked ? '02 unlocks after 01 Lock Manifest.' : azureUploadConfirmed ? '02 is complete. Azure upload has been confirmed.' : 'Confirm the full finetune-subset upload is complete before Azure preprocessing.'}
              >
                <UploadCloud size={16} /> {confirmingUpload ? '02 Confirming...' : '02 Confirm Azure Upload'}
              </button>
              <button
                className={stageButtonClass(3)}
                onClick={handlePrepareDatasets}
                type="button"
                disabled={preparing || !azureUploadConfirmed || hasModelPreprocessSubmittedForPage || hasModelPreprocessReady}
                title={!azureUploadConfirmed ? '03 unlocks after 02 Confirm Azure Upload.' : modelPreprocessRunning ? '03 has been submitted and is still running. Use Check Prep Progress.' : hasModelPreprocessReady ? '03 is complete.' : 'Prepare the model-specific Azure dataset.'}
              >
                <Play size={16} /> {preparing ? '03 Submitting...' : prepareStepLabel}
              </button>
              {variant === 'autoregressive' ? (
                <button
                  className={stageButtonClass(4)}
                  onClick={handleCacheMusicGenTokens}
                  type="button"
                  disabled={cachingTokens || !hasModelPreprocessReady || musicGenTokenCacheReady || musicGenTokenCacheRunning}
                  title={!hasModelPreprocessReady ? '04 unlocks after 03 Prepare MusicGen Dataset is complete.' : musicGenTokenCacheReady ? '04 EnCodec token cache has completed.' : musicGenTokenCacheRunning ? musicGenTokenCache?.reason ?? '04 EnCodec token cache is running.' : 'Cache EnCodec audio-token targets for prepared MusicGen chunks.'}
                >
                  <DatabaseZap size={16} /> {cachingTokens ? '04 Submitting...' : musicGenTokenCacheReady ? '04 EnCodec Tokens Cached' : musicGenTokenCacheRunning ? '04 Token Cache Running' : '04 Cache EnCodec Tokens'}
                </button>
              ) : null}
              {variant === 'diffusion' ? (
                <button
                  className={stageButtonClass(4)}
                  onClick={handleStableAudioPreflight}
                  type="button"
                  disabled={!stableAudioPreflightCanRun}
                  title={
                    !hasModelPreprocessReady
                      ? '04 unlocks after 03 Prepare Stable Audio Dataset is complete.'
                      : stableAudioPreflightActive
                        ? '04 is running in Azure ML. Refresh gates to monitor completion.'
                        : stableAudioPreflightPassed
                          ? `04 has passed already. Re-run now to validate the updated trainer before step 09.`
                          : stableAudioPreflight?.reason ?? 'Run the Stable Audio trainer preflight before smoke training.'
                  }
                >
                  <Terminal size={16} /> {preflighting ? '04 Submitting...' : stableAudioPreflightActive ? '04 Preflight Running' : stableAudioPreflightPassed ? '04 Re-run Trainer Preflight' : '04 Run Trainer Preflight'}
                </button>
              ) : null}
              {variant === 'diffusion' ? (
                <>
                  <button
                    className={stageButtonClass(5)}
                    onClick={() => handleLaunchSmoke('no_cara_baseline')}
                    type="button"
                    disabled={launching || !baselineLaunchEnabled}
                    title={
                      baselineSmokePassed
                        ? baselineSmoke?.reason ?? '05 baseline smoke has passed.'
                        : baselineSmokeActive
                          ? baselineSmoke?.reason ?? '05 baseline smoke is running.'
                          : !hasTrainerInputsReady || !stableAudioPreflightPassed
                            ? '05 unlocks after the Stable Audio dataset and preflight are complete.'
                            : trainerLaunchBlockedReason
                    }
                  >
                    <Play size={16} /> {launching && nextStableAudioSmokeVariant === 'no_cara_baseline' ? '05 Submitting...' : stableAudioSmokeLabel('05', 'Baseline Smoke', baselineSmoke)}
                  </button>
                  <button
                    className={stageButtonClass(6)}
                    onClick={() => handleLaunchSmoke('cara_lite')}
                    type="button"
                    disabled={launching || !caraLiteLaunchEnabled}
                    title={
                      !baselineSmokePassed
                        ? '06 unlocks after 05 Baseline Smoke has passed.'
                        : caraLiteSmokePassed
                          ? caraLiteSmoke?.reason ?? '06 CARA-lite smoke has passed.'
                          : caraLiteSmokeActive
                            ? caraLiteSmoke?.reason ?? '06 CARA-lite smoke is running.'
                            : caraLiteSmoke?.reason ?? 'Launch the CARA-lite prompt-control smoke.'
                    }
                  >
                    <Play size={16} /> {launching && nextStableAudioSmokeVariant === 'cara_lite' ? '06 Submitting...' : stableAudioSmokeLabel('06', 'CARA-Lite Smoke', caraLiteSmoke)}
                  </button>
                  <button
                    className={stageButtonClass(7)}
                    onClick={() => handleLaunchSmoke('cara_head')}
                    type="button"
                    disabled={launching || !caraHeadLaunchEnabled}
                    title={
                      !caraLiteSmokePassed
                        ? '07 unlocks after 06 CARA-lite Smoke has passed.'
                        : caraHeadSmokePassed
                          ? caraHeadSmoke?.reason ?? '07 CARA attribution-head smoke has passed.'
                          : caraHeadSmokeActive
                            ? caraHeadSmoke?.reason ?? '07 CARA attribution-head smoke is running.'
                            : caraHeadSmoke?.reason ?? 'Launch the detached CARA attribution-head smoke.'
                    }
                  >
                    <Play size={16} /> {launching && nextStableAudioSmokeVariant === 'cara_head' ? '07 Submitting...' : stableAudioSmokeLabel('07', 'CARA Attribution-Head Smoke', caraHeadSmoke)}
                  </button>
                  <button
                    className={stageButtonClass(8)}
                    onClick={() => handleLaunchSmoke('cara_strong')}
                    type="button"
                    disabled={launching || !caraStrongLaunchEnabled}
                    title={
                      !caraHeadSmokePassed
                        ? '08 unlocks after 07 CARA Attribution-Head Smoke has passed.'
                      : caraStrongRepeatRecommended
                          ? caraStrongSmoke?.reason ?? 'Re-run a short updated CARA-Strong smoke before step 09.'
                      : caraStrongSmokePassed
                          ? '08 has passed already. Re-run a short updated CARA-Strong smoke for clean benchmark evidence before step 09.'
                          : caraStrongSmokeActive
                            ? caraStrongSmoke?.reason ?? '08 CARA-Strong smoke is running.'
                            : caraStrongSmoke?.reason ?? 'Launch the non-detached CARA-Strong smoke.'
                    }
                  >
                    <Play size={16} /> {launching && nextStableAudioSmokeVariant === 'cara_strong' ? '08 Submitting...' : caraStrongSmokePassed ? '08 Re-run CARA-Strong Smoke' : stableAudioSmokeLabel('08', 'CARA-Strong Smoke', caraStrongSmoke)}
                  </button>
                  <button
                    className={stageButtonClass(9)}
                    onClick={handleLaunchFullStableAudio}
                    type="button"
                    disabled={launchingFull || !fullStableAudioLaunchEnabled}
                    title={
                      !caraStrongSmokePassed
                        ? '09 unlocks after 08 CARA-Strong Smoke has passed.'
                        : stableAudioFullPassed
                          ? stableAudioFullTraining?.reason ?? '09 full CARA-Strong fine-tune has completed.'
                          : stableAudioFullActive
                            ? stableAudioFullTraining?.reason ?? '09 full CARA-Strong fine-tune is running.'
                            : stableAudioFullTraining?.reason ?? 'Launch the full CARA-Strong fine-tune with held-out validation/test CARA metrics.'
                    }
                  >
                    <Play size={16} /> {launchingFull ? '09 Submitting...' : stableAudioFullActive ? '09 Full Fine-Tune Running' : stableAudioFullPassed ? '09 Full Fine-Tune Passed' : '09 Full CARA-Strong Fine-Tune'}
                  </button>
                </>
              ) : (
                <>
                  <button
                    className={stageButtonClass(7)}
                    onClick={handleMusicGenPreflight}
                    type="button"
                    disabled={!musicGenPreflightCanRun}
                    title={!musicGenTokenCacheReady ? '07 unlocks after 04 Cache EnCodec Tokens is complete.' : musicGenPreflightActive ? musicGenPreflight?.reason ?? '07 MusicGen preflight is running.' : musicGenPreflightPassed ? '07 MusicGen preflight has passed.' : musicGenPreflight?.reason ?? 'Run MusicGen trainer preflight before smoke training.'}
                  >
                    <Terminal size={16} /> {preflighting ? '07 Submitting...' : musicGenPreflightActive ? '07 Preflight Running' : musicGenPreflightPassed ? '07 MusicGen Preflight Passed' : '07 Run MusicGen Preflight'}
                  </button>
                  <button
                    className={stageButtonClass(8)}
                    onClick={() => handleLaunchMusicGenSmoke('no_cara_baseline')}
                    type="button"
                    disabled={launching || !musicGenBaselineLaunchEnabled}
                    title={!musicGenPreflightPassed ? '08 unlocks after 07 MusicGen Preflight has passed.' : musicGenBaselineSmoke?.reason ?? 'Launch the same-data no-CARA MusicGen baseline smoke.'}
                  >
                    <Play size={16} /> {launching && nextMusicGenSmokeVariant === 'no_cara_baseline' ? '08 Submitting...' : stableAudioSmokeLabel('08', 'MusicGen Baseline Smoke', musicGenBaselineSmoke)}
                  </button>
                  <button
                    className={stageButtonClass(9)}
                    onClick={() => handleLaunchMusicGenSmoke('cara_lite')}
                    type="button"
                    disabled={launching || !musicGenCaraLiteLaunchEnabled}
                    title={!musicGenBaselinePassed ? '09 unlocks after 08 MusicGen Baseline Smoke has passed.' : musicGenCaraLiteSmoke?.reason ?? 'Launch the MusicGen CARA-lite prompt-control smoke.'}
                  >
                    <Play size={16} /> {launching && nextMusicGenSmokeVariant === 'cara_lite' ? '09 Submitting...' : stableAudioSmokeLabel('09', 'MusicGen CARA-Lite Smoke', musicGenCaraLiteSmoke)}
                  </button>
                  <button
                    className={stageButtonClass(10)}
                    onClick={() => handleLaunchMusicGenSmoke('cara_probe')}
                    type="button"
                    disabled={launching || !musicGenCaraProbeLaunchEnabled}
                    title={!musicGenCaraLitePassed ? '10 unlocks after 09 MusicGen CARA-Lite Smoke has passed.' : musicGenCaraProbeSmoke?.reason ?? 'Launch the detached MusicGen CARA suffix-probe smoke.'}
                  >
                    <Play size={16} /> {launching && nextMusicGenSmokeVariant === 'cara_probe' ? '10 Submitting...' : stableAudioSmokeLabel('10', 'MusicGen CARA Suffix-Probe Smoke', musicGenCaraProbeSmoke)}
                  </button>
                  <button
                    className={stageButtonClass(11)}
                    onClick={() => handleLaunchMusicGenSmoke('cara_strong')}
                    type="button"
                    disabled={launching || !musicGenCaraStrongLaunchEnabled}
                    title={!musicGenCaraProbePassed ? '11 unlocks after 10 MusicGen CARA Suffix-Probe Smoke has passed.' : musicGenCaraStrongSmoke?.reason ?? 'Launch the non-detached MusicGen CARA-Strong suffix smoke.'}
                  >
                    <Play size={16} /> {launching && nextMusicGenSmokeVariant === 'cara_strong' ? '11 Submitting...' : stableAudioSmokeLabel('11', 'MusicGen CARA-Strong Smoke', musicGenCaraStrongSmoke)}
                  </button>
                  <button
                    className={stageButtonClass(12)}
                    onClick={handleLaunchFullMusicGen}
                    type="button"
                    disabled={launchingFull || !musicGenFullLaunchEnabled}
                    title={!musicGenCaraStrongPassed ? '12 unlocks after 11 MusicGen CARA-Strong smoke passes.' : 'Launch the full real MusicGen LM CARA-Strong fine-tune on H100 compute.'}
                  >
                    <Play size={16} /> {launchingFull ? '12 Submitting...' : '12 Full MusicGen LM CARA-Strong Fine-Tune'}
                  </button>
                </>
              )}
            </div>
            {hasActiveTrainerJob ? (
              <div className="pool-empty-state" style={{ marginTop: 12 }}>
                Active {variant === 'diffusion' ? 'Stable Audio' : 'MusicGen'} trainer job <span className="mono">{activeTrainerJob?.name ?? 'unknown'}</span> is <span className="mono">{activeTrainerJob?.status ?? 'active'}</span>. Monitor it in Operations / Azure Runs; use Hard stop there before launching another run.
              </div>
            ) : null}
            {variant === 'diffusion' && hasTrainerInputsReady && !stableAudioPreflightPassed ? (
              <div className="pool-empty-state" style={{ marginTop: 12 }}>
                {stableAudioPreflight?.reason ?? 'Run and pass the Stable Audio trainer preflight before launching smoke training.'}
              </div>
            ) : null}
            {hasTrainerInputsReady && stableAudioPreflightPassed && !trainerLaunchEnabled && !hasActiveTrainerJob ? (
              <div className="pool-empty-state" style={{ marginTop: 12 }}>
                Prepared inputs are complete. {trainerLaunchBlockedReason}
              </div>
            ) : null}
            {variant === 'diffusion' && baselineSmokePassed && !caraLiteSmokePassed && !hasActiveTrainerJob ? (
              <div className="pool-empty-state" style={{ marginTop: 12 }}>
                Baseline smoke is preserved as step 05. The next repeatable protocol action is step 06: CARA-lite prompt-control smoke.
              </div>
            ) : null}
            {variant === 'diffusion' && caraStrongSmokePassed && !stableAudioFullPassed && !stableAudioFullActive ? (
              <div className="pool-empty-state" style={{ marginTop: 12 }}>
                Recommended revalidation before step 09: re-run 04 Trainer Preflight, then optionally re-run 08 CARA-Strong Smoke, then launch 09 Full CARA-Strong Fine-Tune.
              </div>
            ) : null}
            {variant === 'autoregressive' && musicGenCaraStrongPassed ? (
              <div className="pool-empty-state" style={{ marginTop: 12 }}>
                MusicGen CARA-Strong smoke is preserved as matched autoregressive evidence. Step 12 now launches the full real MusicGen LM fine-tune after you review the suffix decode report.
              </div>
            ) : null}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Locked Training Set</div>
            <div className="card-meta">{readiness?.lock.locked ? 'ready' : 'not locked'}</div>
          </div>
          <div className="metric-list">
            <div><span className="dim">Accepted rows</span><strong>{readiness?.lock.summary?.accepted_count ?? '—'}</strong></div>
            <div><span className="dim">Rejected rows</span><strong>{readiness?.lock.summary?.rejected_count ?? '—'}</strong></div>
            <div><span className="dim">Pools</span><strong>{readiness?.lock.summary?.pool_count ?? '—'}</strong></div>
            <div><span className="dim">Families</span><strong>{readiness?.lock.summary?.family_count ?? '—'}</strong></div>
            <div><span className="dim">Train / Val / Test</span><strong>{splitCounts.train ?? '—'} / {splitCounts.validation ?? '—'} / {splitCounts.test ?? '—'}</strong></div>
            <div><span className="dim">TIR</span><strong className="mono">{readiness?.lock.summary?.tir_id ?? '—'}</strong></div>
          </div>
          <div className="pool-empty-state" style={{ marginTop: 14 }}>
            <span className="mono">{readiness?.lock.paths?.locked_manifest ?? 'Lock the manifest to create the training file.'}</span>
          </div>
          <div className="pool-empty-state" style={{ marginTop: 10 }}>
            Azure prepared root: <span className="mono">{readiness?.data_locations?.azure_prepared_root ?? '—'}</span>
          </div>
          <div className="pool-empty-state" style={{ marginTop: 10 }}>
            Prepared audio is split into <span className="mono">train</span>, <span className="mono">validation</span>, and <span className="mono">test</span> partitions; these are disjoint chunk sets, not triplicate copies.
          </div>
          {variant === 'autoregressive' ? (
            <div className="pool-empty-state" style={{ marginTop: 10 }}>
              EnCodec cache manifest: <span className="mono">{readiness?.data_locations?.azure_musicgen_encodec_manifest ?? '—'}</span>
            </div>
          ) : null}
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">Cloud Run Lifecycle</div>
          <div className="card-meta">Azure ML owns execution after submission</div>
        </div>
        <div className="metric-list">
          <div><span className="dim">Browser closed</span><strong>{readiness?.cloud_job_policy?.browser_close_cancels_job ? 'Stops job' : 'Job keeps running'}</strong></div>
          <div><span className="dim">Submission</span><strong>{readiness?.cloud_job_policy?.durable_submission ? 'Durable Azure job' : 'Local only'}</strong></div>
          <div><span className="dim">Preprocess route</span><strong>H100 only if both H100 compute targets are idle</strong></div>
          <div><span className="dim">Hard stop</span><strong><Square size={13} style={{ marginRight: 6 }} />Operations / Azure Runs</strong></div>
        </div>
        <div className="pool-empty-state" style={{ marginTop: 14 }}>
          {readiness?.cloud_job_policy?.stop_behavior ?? 'Azure job lifecycle policy will appear after the backend is available.'}
        </div>
        <div className="pool-empty-state" style={{ marginTop: 10 }}>
          {readiness?.cloud_job_policy?.checkpoint_resume ?? 'Checkpoint resume policy will be attached to the trainer command job.'}
        </div>
        {variant === 'autoregressive' ? (
          <div className="pool-empty-state" style={{ marginTop: 10 }}>
            {readiness?.cloud_job_policy?.musicgen_token_cache ?? 'MusicGen token-cache policy will appear after the backend is available.'}
          </div>
        ) : null}
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">Dataset Prep Progress</div>
          <div className="card-meta">{progress ? `checked ${new Date(progress.checked_at).toLocaleString()}` : 'read-only Azure blob estimate'}</div>
        </div>
        {progressError ? (
          <div className="error-banner">{progressError}</div>
        ) : null}
        <div className="metric-list">
          <div><span className="dim">Chunks</span><strong>{progress ? `${progress.completed_chunks.toLocaleString()} / ${progress.expected.expected_chunks.toLocaleString()}` : '—'}</strong></div>
          <div><span className="dim">Chunk estimate</span><strong>{progress ? `${progress.chunk_percent.toFixed(2)}%` : '—'}</strong></div>
          <div><span className="dim">Duration estimate</span><strong>{progress ? `${progress.completed_duration_hours.toFixed(2)}h / ${progress.expected.expected_duration_hours.toFixed(2)}h` : '—'}</strong></div>
          <div><span className="dim">Remaining chunks</span><strong>{progress ? progress.remaining_chunks_estimate.toLocaleString() : '—'}</strong></div>
          <div><span className="dim">Elapsed</span><strong>{progress ? formatDuration(progress.elapsed_seconds) : '—'}</strong></div>
          <div><span className="dim">Estimated time left</span><strong>{progress ? formatDuration(progress.estimated_remaining_seconds) : '—'}</strong></div>
          <div><span className="dim">Azure job</span><strong className="mono">{progress?.job?.job_name ?? '—'}</strong></div>
        </div>
        <div style={{ height: 10, background: '#151a22', borderRadius: 999, overflow: 'hidden', marginTop: 14 }}>
          <div
            style={{
              height: '100%',
              width: `${Math.min(100, progress?.duration_percent ?? 0)}%`,
              background: 'linear-gradient(90deg, #7c5cff, #18d2ff)',
            }}
          />
        </div>
        <div className="row" style={{ justifyContent: 'space-between', gap: 10, marginTop: 14, flexWrap: 'wrap' }}>
          <div className="pool-empty-state" style={{ margin: 0 }}>
            {progress?.note ?? 'Counts prepared WAV blobs already visible in the Azure datastore and compares them to the expected chunk plan.'}
          </div>
          <button className="btn btn-ghost" type="button" onClick={handleCheckPreprocessProgress} disabled={progressLoading}>
            <RefreshCw size={16} className={progressLoading ? 'spin' : ''} /> {progressLoading ? 'Checking...' : 'Check Prep Progress'}
          </button>
        </div>
      </section>

      {variant === 'diffusion' ? (
        <section className="card">
          <div className="card-header">
            <div className="card-title">Training Run Progress</div>
            <div className="card-meta">
              {stableAudioTrainingProgress ? `checked ${new Date(stableAudioTrainingProgress.checked_at).toLocaleString()}` : 'awaiting MLflow steps'}
            </div>
          </div>
          <div className="metric-list">
            <div><span className="dim">Azure job</span><strong className="mono">{stableAudioTrainingProgress?.job_name ?? '—'}</strong></div>
            <div><span className="dim">Status</span><strong>{stableAudioTrainingProgress?.status ?? '—'}</strong></div>
            <div><span className="dim">Variant</span><strong>{stableAudioTrainingProgress?.variant ?? '—'}</strong></div>
            <div><span className="dim">Scope</span><strong>{stableAudioTrainingProgress?.training_scope ?? '—'}</strong></div>
            <div><span className="dim">Step</span><strong>{stableAudioTrainingProgress?.observed_step ? `${stableAudioTrainingProgress.observed_step.toLocaleString()} / ${stableAudioTrainingProgress.max_steps?.toLocaleString() ?? '—'}` : '—'}</strong></div>
            <div><span className="dim">Step progress</span><strong>{stableAudioTrainingProgress?.step_percent !== null && stableAudioTrainingProgress?.step_percent !== undefined ? `${stableAudioTrainingProgress.step_percent.toFixed(2)}%` : '—'}</strong></div>
            <div><span className="dim">Chunks seen</span><strong>{stableAudioTrainingProgress?.chunks_seen_estimate !== null && stableAudioTrainingProgress?.chunks_seen_estimate !== undefined ? `${stableAudioTrainingProgress.chunks_seen_estimate.toLocaleString()} / ${stableAudioTrainingProgress.effective_train_chunks?.toLocaleString() ?? '—'}` : '—'}</strong></div>
            <div><span className="dim">Epoch estimate</span><strong>{stableAudioTrainingProgress?.completed_epochs_estimate !== null && stableAudioTrainingProgress?.completed_epochs_estimate !== undefined ? `${stableAudioTrainingProgress.completed_epochs_estimate.toLocaleString()} + ${stableAudioTrainingProgress.epoch_percent?.toFixed(2) ?? '0.00'}%` : '—'}</strong></div>
            <div><span className="dim">Latest loss</span><strong>{stableAudioTrainingProgress?.latest_loss !== null && stableAudioTrainingProgress?.latest_loss !== undefined ? stableAudioTrainingProgress.latest_loss.toFixed(5) : '—'}</strong></div>
            <div><span className="dim">Elapsed</span><strong>{formatDuration(stableAudioTrainingProgress?.elapsed_seconds)}</strong></div>
            <div><span className="dim">Estimated time left</span><strong>{formatDuration(stableAudioTrainingProgress?.estimated_remaining_seconds)}</strong></div>
            <div><span className="dim">Metrics</span><strong>{stableAudioTrainingProgress?.metrics_available ? 'MLflow live' : stableAudioTrainingProgress?.metrics_error ? 'Unavailable' : '—'}</strong></div>
          </div>
          <div style={{ height: 10, background: '#151a22', borderRadius: 999, overflow: 'hidden', marginTop: 14 }}>
            <div
              style={{
                height: '100%',
                width: `${Math.min(100, stableAudioTrainingProgress?.step_percent ?? 0)}%`,
                background: 'linear-gradient(90deg, #7c5cff, #18d2ff)',
              }}
            />
          </div>
          <div className="row" style={{ justifyContent: 'space-between', gap: 10, marginTop: 14, flexWrap: 'wrap' }}>
            <div className="pool-empty-state" style={{ margin: 0 }}>
              {stableAudioTrainingProgress?.metrics_error
                ? `MLflow metrics unavailable: ${stableAudioTrainingProgress.metrics_error}`
                : stableAudioTrainingProgress?.chunk_count_error
                  ? `Chunk count unavailable: ${stableAudioTrainingProgress.chunk_count_error}`
                  : stableAudioTrainingProgress?.note ?? 'Reads Azure MLflow steps and estimates chunk/epoch progress from the prepared train chunk count.'}
            </div>
            {stableAudioTrainingProgress?.studio_url ? (
              <a className="btn btn-ghost" href={stableAudioTrainingProgress.studio_url} target="_blank" rel="noreferrer">
                <Cloud size={16} /> Open in Studio
              </a>
            ) : null}
          </div>
        </section>
      ) : null}

      <section className="card">
        <div className="card-header">
          <div className="card-title">Audio Window Policy</div>
          <div className="card-meta">{variant}</div>
        </div>
        <div className="metric-list">
          <div><span className="dim">Model window</span><strong>{windowPolicy?.max_window_seconds ?? '—'} sec</strong></div>
          <div><span className="dim">Sample rate</span><strong>{windowPolicy?.sample_rate_hz ?? '—'} Hz</strong></div>
          <div><span className="dim">Channels</span><strong>{windowPolicy?.channels ?? '—'}</strong></div>
          <div><span className="dim">Pre-chunk required</span><strong>{windowPolicy?.pre_chunk_required ? 'Yes' : 'No'}</strong></div>
        </div>
        <div className="pool-empty-state" style={{ marginTop: 14 }}>
          {windowPolicy?.note ?? 'Windowing policy will appear after the backend is available.'}
        </div>
      </section>

      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Run Configuration</div>
            <div className="card-meta">{variant}</div>
          </div>
          <div className="controls">
            {variant === 'diffusion' ? (
              <label className="row" style={{ justifyContent: 'flex-start', gap: 10, marginBottom: 12 }}>
                <input
                  type="checkbox"
                  checked={fullTrainingRun}
                  onChange={(event) => setFullTrainingRun(event.target.checked)}
                  style={{ width: 18, height: 18 }}
                />
                <span>Full training run</span>
                <span className="field-hint" style={{ margin: 0 }}>
                  Applies to step 09 only; uses the full prepared train set and overrides max steps with dataset-pass mode.
                </span>
              </label>
            ) : null}
            <div className="control-row">
              {fields.map((f) => (
                <div className="field" key={f.key}>
                  <label htmlFor={`f-${f.key}`}>{f.label}</label>
                  {f.key === 'max_steps' && fullTrainingRun && variant === 'diffusion' ? (
                    <input
                      id={`f-${f.key}`}
                      type="text"
                      value="Full training run"
                      disabled
                      style={{ opacity: 0.55, cursor: 'not-allowed' }}
                    />
                  ) : f.type === 'select' ? (
                    <select
                      id={`f-${f.key}`}
                      value={String(values[f.key])}
                      onChange={(e) => setField(f.key, e.target.value)}
                    >
                      {(f.options ?? []).map((opt) => (
                        <option key={opt} value={opt}>
                          {opt}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={`f-${f.key}`}
                      type={f.type === 'number' ? 'number' : 'text'}
                      value={String(values[f.key])}
                      onChange={(e) =>
                        setField(f.key, f.type === 'number' ? Number(e.target.value) : e.target.value)
                      }
                    />
                  )}
                  {f.hint ? <div className="field-hint">{f.hint}</div> : null}
                </div>
              ))}
            </div>

            <div className="row" style={{ justifyContent: 'flex-end', gap: 10 }}>
              <button className="btn btn-ghost" onClick={handleRefresh} type="button">
                <RefreshCw size={16} /> Refresh status
              </button>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Live loss</div>
            <div className="card-meta">awaiting Azure MLflow metrics</div>
          </div>
          <div className="chart-wrap" style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={loss} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id={`lossGrad-${variant}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#7c5cff" stopOpacity={0.7} />
                    <stop offset="100%" stopColor="#18d2ff" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="t" hide />
                <YAxis hide domain={[0, 'dataMax']} />
                <Tooltip
                  contentStyle={{
                    background: '#14171d',
                    border: '1px solid #242932',
                    borderRadius: 8,
                    fontSize: 12,
                    color: '#f3efe7',
                  }}
                  labelFormatter={(t: any) => new Date(Number(t)).toLocaleTimeString()}
                  formatter={(v: any) => [Number(v).toFixed(4), 'loss']}
                />
                <Area
                  type="monotone"
                  dataKey="v"
                  stroke="#7c5cff"
                  strokeWidth={2}
                  fill={`url(#lossGrad-${variant})`}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">
            <Terminal size={14} style={{ marginRight: 6, opacity: 0.8 }} />
            Live Log
          </div>
          <div className="card-meta">submission handoff notes</div>
        </div>
        <div className="log-stream" role="log" aria-live="polite">
          {logs.map((line, i) => (
            <div key={i} className="log-line">
              {line}
            </div>
          ))}
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">Recent Runs</div>
          <div className="card-meta">moved to Operations / Azure Runs</div>
        </div>
        {recentTrainingJobs.length ? (
          <div className="mini-list">
            {recentTrainingJobs.slice().reverse().map((job, index) => (
              <div key={`${job.job_name}-${index}`}>
                <span className="k">{job.model_family ?? job.action ?? 'training job'}</span>
                <span className="v mono">{job.job_name ?? 'pending'} · {job.compute ?? 'compute unknown'}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="pool-empty-state">
            Workspace job history now lives in the Azure Runs page so smoke tests and full training runs share one monitor.
          </div>
        )}
      </section>
    </div>
  );
};
