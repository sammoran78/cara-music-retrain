import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  DatabaseZap,
  FlaskConical,
  Layers3,
  Lock,
  RefreshCw,
  Route,
  ShieldCheck,
  Workflow,
} from 'lucide-react';
import { PageHeader, PlaceholderBadge } from './PageHeader';
import type { ViewId } from '../nav';

type BranchStepState = 'complete' | 'active' | 'planned' | 'locked';

interface ActiveTrainingJob {
  name?: string;
  status?: string;
  studio_url?: string;
  display_name?: string;
}

interface StableAudioStep {
  stage?: number;
  label?: string;
  passed?: boolean;
  active?: boolean;
  reason?: string;
  latest_job?: ActiveTrainingJob | null;
}

interface TrainingReadiness {
  lock?: {
    locked?: boolean;
    summary?: {
      accepted_count?: number;
      pool_count?: number;
      family_count?: number;
      split_counts?: Record<string, number>;
      tir_id?: string;
    } | null;
  };
  azure_upload?: {
    confirmed?: boolean;
    source_root?: string;
  };
  data_locations?: Record<string, string>;
  preprocess_jobs?: Record<string, StableAudioStep>;
  stable_audio_preflight?: StableAudioStep | null;
  stable_audio_smoke_sequence?: {
    variants?: Record<string, StableAudioStep>;
    next_stage?: number;
    next_label?: string;
    reason?: string;
  };
  stable_audio_full_training?: StableAudioStep | null;
  context_diffusion_ladder?: {
    context_packs?: StableAudioStep | null;
    context_cache?: StableAudioStep | null;
    context_preflight?: StableAudioStep | null;
    context_smoke?: StableAudioStep | null;
    context_full?: StableAudioStep | null;
    next_stage?: number;
    next_label?: string;
    root_output_path?: string;
    trainer_status?: string;
  };
  context_diffusion_launch?: {
    context_packs_enabled?: boolean;
    context_cache_enabled?: boolean;
    context_preflight_enabled?: boolean;
    context_smoke_enabled?: boolean;
    context_full_enabled?: boolean;
    context_smoke_reason?: string;
    context_full_reason?: string;
  };
  training_launch_reason?: string;
  cloud_job_policy?: Record<string, string | boolean | undefined>;
}

interface TrainingRunProgress {
  checked_at?: string;
  model_key?: string;
  model_label?: string;
  job_name?: string;
  run_name?: string;
  studio_url?: string;
  status?: string;
  variant?: string;
  training_scope?: string;
  action?: string;
  max_steps?: number | null;
  observed_step?: number | null;
  step_percent?: number | null;
  batch_size?: number | null;
  chunks_seen_estimate?: number | null;
  effective_train_chunks?: number | null;
  completed_epochs_estimate?: number | null;
  epoch_percent?: number | null;
  latest_loss?: number | null;
  elapsed_seconds?: number | null;
  estimated_remaining_seconds?: number | null;
  metrics_available?: boolean;
  metrics_source?: string | null;
  metrics_artifact?: string | null;
  metrics_row_count?: number | null;
  metrics_error?: string | null;
  chunk_count_error?: string | null;
  note?: string;
}

interface ContextStep {
  id: string;
  step: number;
  label: string;
  state: BranchStepState;
  evidence: string;
  source: 'reused' | 'new';
}

interface FinetuneContextDiffusionPageProps {
  onNavigate?: (view: ViewId) => void;
}

const smokeLabels: Record<string, string> = {
  no_cara_baseline: 'Baseline smoke',
  cara_lite: 'CARA-lite smoke',
  cara_head: 'Attribution-head smoke',
  cara_strong: 'CARA-Strong smoke',
};

const pct = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
const count = new Intl.NumberFormat();

const formatDuration = (seconds?: number | null): string => {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '-';
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${secs}s`;
  return `${secs}s`;
};

const isActiveStatus = (status?: string | null): boolean =>
  ['notstarted', 'queued', 'preparing', 'starting', 'provisioning', 'running', 'finalizing'].includes(
    String(status || '').toLowerCase(),
  );

const stableStepState = (step?: StableAudioStep | null): BranchStepState => {
  if (!step) return 'locked';
  if (step.active) return 'active';
  if (step.passed) return 'complete';
  return 'locked';
};

const stepIcon = (state: BranchStepState) => {
  if (state === 'complete') return <CheckCircle2 size={16} />;
  if (state === 'active') return <RefreshCw size={16} />;
  if (state === 'planned') return <CircleDashed size={16} />;
  return <Lock size={16} />;
};

const stepClass = (state: BranchStepState) => {
  if (state === 'complete') return 'is-complete';
  if (state === 'active' || state === 'planned') return 'is-current';
  return 'is-muted';
};

const sourceBadge = (source: ContextStep['source']) => (
  <span className={`status-pill ${source === 'reused' ? 'status-done' : 'status-running'}`}>
    {source === 'reused' ? 'reused diffusion evidence' : 'new context branch'}
  </span>
);

export const FinetuneContextDiffusionPage: React.FC<FinetuneContextDiffusionPageProps> = ({ onNavigate }) => {
  const [readiness, setReadiness] = useState<TrainingReadiness | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [checkedAt, setCheckedAt] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [submittingStage, setSubmittingStage] = useState<number | null>(null);
  const [runProgress, setRunProgress] = useState<TrainingRunProgress | null>(null);
  const [runProgressLoading, setRunProgressLoading] = useState(false);
  const [runProgressError, setRunProgressError] = useState<string | null>(null);

  const refreshRunProgress = async () => {
    setRunProgressLoading(true);
    setRunProgressError(null);
    try {
      const res = await fetch('/api/training/run-progress?model=context_diffusion');
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Context Diffusion run-progress check failed');
      setRunProgress(json.progress ?? null);
    } catch (err) {
      setRunProgressError(err instanceof Error ? err.message : 'Context Diffusion run-progress check failed');
    } finally {
      setRunProgressLoading(false);
    }
  };

  const refreshReadiness = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/training/context-diffusion-readiness');
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Context Diffusion readiness check failed');
      const now = new Date().toISOString();
      setReadiness(json);
      setCheckedAt(now);
      setLogs((prev) => [
        ...prev.slice(-7),
        `[monitor] Refreshed Context Diffusion gates · next ${json.context_diffusion_ladder?.next_stage ?? 'n/a'} ${json.context_diffusion_ladder?.next_label ?? ''}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Context Diffusion readiness check failed';
      setError(message);
      setLogs((prev) => [...prev.slice(-7), `[error] ${message}`]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refreshReadiness();
    void refreshRunProgress();
  }, []);

  useEffect(() => {
    if (!isActiveStatus(runProgress?.status)) return undefined;
    const interval = window.setInterval(() => {
      void refreshRunProgress();
    }, 60000);
    return () => window.clearInterval(interval);
  }, [runProgress?.status]);

  const submitContextStage = async (stage: 10 | 11 | 12 | 13 | 14) => {
    const fullConfirmation = 'LAUNCH CONTEXT FULL FINE-TUNE';
    if (stage === 14) {
      const typed = window.prompt(`Type ${fullConfirmation} to launch the full Context Diffusion H100 fine-tune.`);
      if (typed !== fullConfirmation) {
        setLogs((prev) => [...prev.slice(-7), '[submit] Step 14 launch cancelled; confirmation phrase did not match.']);
        return;
      }
    }
    const endpoint =
      stage === 10
        ? '/api/training/context-diffusion/packs'
        : stage === 11
          ? '/api/training/context-diffusion/cache'
          : stage === 12
            ? '/api/training/context-diffusion/preflight'
            : stage === 13
              ? '/api/training/context-diffusion/smoke'
              : '/api/training/context-diffusion/full';
    setSubmittingStage(stage);
    setError(null);
    try {
      const body =
        stage === 10
          ? { dry_run: false, max_contexts: 3, selection_seed: 'cara-context-v1' }
          : stage === 13
            ? { dry_run: false, max_steps: 250, batch_size: 64, learning_rate: 0.001, max_train_rows: 4096, max_eval_rows: 1024 }
            : stage === 14
              ? {
                  dry_run: false,
                  confirmation_phrase: fullConfirmation,
                  max_steps: 20000,
                  batch_size: 8,
                  learning_rate: 0.00001,
                  num_workers: 0,
                  precision: '16-mixed',
                  checkpoint_every: 1000,
                  checkpoint_keep_last_n: 1,
                  max_train_files: 0,
                  max_eval_files: 0,
                  max_eval_batches: 0,
                  attribution_loss_weight: 0.05,
                }
            : { dry_run: false };
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? `Context Diffusion step ${stage} submission failed`);
      setReadiness(json.readiness ?? null);
      setCheckedAt(new Date().toISOString());
      setLogs((prev) => [
        ...prev.slice(-7),
        `[submit] Step ${stage} submitted as ${json.job?.name ?? 'Azure ML job'} · ${json.job?.status ?? 'submitted'}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : `Context Diffusion step ${stage} submission failed`;
      setError(message);
      setLogs((prev) => [...prev.slice(-7), `[error] ${message}`]);
    } finally {
      setSubmittingStage(null);
    }
  };

  const steps = useMemo<ContextStep[]>(() => {
    const prep = readiness?.preprocess_jobs?.stable_audio_open_small;
    const smoke = readiness?.stable_audio_smoke_sequence?.variants ?? {};
    const context = readiness?.context_diffusion_ladder ?? {};
    const inherited: ContextStep[] = [
      {
        id: 'lock',
        step: 1,
        label: 'Lock manifest',
        state: readiness?.lock?.locked ? 'complete' : 'locked',
        evidence: readiness?.lock?.summary?.tir_id
          ? `Locked CARA registry ${readiness.lock.summary.tir_id}`
          : 'Reuses the original CARA-Strong manifest and registry lock.',
        source: 'reused',
      },
      {
        id: 'upload',
        step: 2,
        label: 'Confirm Azure upload',
        state: readiness?.azure_upload?.confirmed ? 'complete' : 'locked',
        evidence: readiness?.azure_upload?.source_root
          ? `Source root ${readiness.azure_upload.source_root}`
          : 'Reuses the same datastore source folder and manifest.',
        source: 'reused',
      },
      {
        id: 'prepare',
        step: 3,
        label: 'Prepare Stable Audio dataset',
        state: stableStepState(prep),
        evidence: prep?.reason ?? 'Reuses the 44.1 kHz stereo Stable Audio chunk manifest.',
        source: 'reused',
      },
      {
        id: 'preflight',
        step: 4,
        label: 'Stable Audio trainer preflight',
        state: stableStepState(readiness?.stable_audio_preflight),
        evidence: readiness?.stable_audio_preflight?.reason ?? 'Reuses the same Stable Audio Tools trainer environment gate.',
        source: 'reused',
      },
      ...(['no_cara_baseline', 'cara_lite', 'cara_head', 'cara_strong'] as const).map((variant, index) => {
        const row = smoke[variant];
        return {
          id: variant,
          step: 5 + index,
          label: smokeLabels[variant],
          state: stableStepState(row),
          evidence: row?.reason ?? `Reuses prior ${smokeLabels[variant].toLowerCase()} evidence.`,
          source: 'reused' as const,
        };
      }),
      {
        id: 'full',
        step: 9,
        label: 'Full CARA-Strong fine-tune',
        state: stableStepState(readiness?.stable_audio_full_training),
        evidence: readiness?.stable_audio_full_training?.reason ?? 'Reuses completed Stable Audio CARA-Strong full-run evidence.',
        source: 'reused',
      },
    ];

    const newBranch: ContextStep[] = [
      {
        id: 'context-pack-design',
        step: 10,
        label: 'Design CARA context packs',
        state: stableStepState(context.context_packs) === 'locked' ? 'planned' : stableStepState(context.context_packs),
        evidence: context.context_packs?.reason ?? 'Select 1-3 same-pool/family source-disjoint audio context examples per target, with registry hash and withheld-label policy.',
        source: 'new',
      },
      {
        id: 'context-cache',
        step: 11,
        label: 'Prepare context conditioning cache',
        state: stableStepState(context.context_cache),
        evidence: context.context_cache?.reason ?? 'Validate context WAV references and store context cache rows beside the prepared Stable Audio manifest.',
        source: 'new',
      },
      {
        id: 'context-preflight',
        step: 12,
        label: 'Context conditioner preflight',
        state: stableStepState(context.context_preflight),
        evidence: context.context_preflight?.reason ?? 'Validate context projection shape, cross-attention concat, prompt/context dropout, and source-disjoint batch metadata before training.',
        source: 'new',
      },
      {
        id: 'context-smoke',
        step: 13,
        label: 'Context Diffusion smoke',
        state: stableStepState(context.context_smoke),
        evidence: context.context_smoke?.reason ?? 'Run C+P, C-only, P-only, shuffled-context, and mismatched-context conditioner controls.',
        source: 'new',
      },
      {
        id: 'context-full',
        step: 14,
        label: 'Full Context Diffusion fine-tune',
        state: stableStepState(context.context_full),
        evidence: context.context_full?.reason ?? 'Train the context-conditioned DiT path only after smoke evidence shows context reaches the model and preserves registry-resolved CARA labels.',
        source: 'new',
      },
      {
        id: 'context-benchmark',
        step: 15,
        label: 'Context benchmark scoring',
        state: context.context_full?.passed ? 'active' : 'locked',
        evidence: context.context_full?.passed
          ? 'Full Context Diffusion is complete. Continue on the Testing page: run the generated-audio benchmark, then attribution scoring, using the locked prompt set.'
          : 'Benchmark against original Diffusion, MusicGen, retrieval floor, and baseline using the locked prompt set plus context/no-context lanes.',
        source: 'new',
      },
    ];
    return [...inherited, ...newBranch];
  }, [readiness]);

  const inheritedComplete = steps.filter((step) => step.source === 'reused' && step.state === 'complete').length;
  const inheritedTotal = steps.filter((step) => step.source === 'reused').length;
  const poolCount = readiness?.lock?.summary?.pool_count ?? 0;
  const trainCount = readiness?.lock?.summary?.split_counts?.train ?? 0;
  const inheritedPct = inheritedTotal ? (inheritedComplete / inheritedTotal) * 100 : 0;
  const contextLadder = readiness?.context_diffusion_ladder;
  const contextLaunch = readiness?.context_diffusion_launch;
  const activeStage = contextLadder?.next_stage ?? 10;

  return (
    <>
      <PageHeader
        kicker="Fine-tuning · Context Diffusion"
        title={
          <>
            Context Diffusion <em>CARA</em> comparison branch
          </>
        }
        description={
          <>
            Track a Stable Audio follow-on experiment that tests whether CARA attribution improves when source
            audio context examples are injected as model conditioning beside the ordinary prompt.
          </>
        }
        actions={<PlaceholderBadge label="Runnable prep branch" />}
      />

      <section className="kpi-grid" aria-label="Context Diffusion status">
        <div className="kpi">
          <div className="kpi-label">Inherited diffusion ladder</div>
          <div className="kpi-value">{inheritedComplete} / {inheritedTotal}</div>
          <div className="kpi-trend">{pct.format(inheritedPct)}% reused from Stable Audio branch</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">New branch status</div>
          <div className="kpi-value">{contextLadder?.next_label ?? 'Loading'}</div>
          <div className="kpi-trend">context smoke stays locked until conditioner/trainer code exists</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Context examples</div>
          <div className="kpi-value">1-3</div>
          <div className="kpi-trend">same-pool or same-family, source-disjoint</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Registry scope</div>
          <div className="kpi-value">{poolCount ? count.format(poolCount) : '98'} pools</div>
          <div className="kpi-trend">{trainCount ? `${count.format(trainCount)} training chunks` : 'same locked CARA manifest'}</div>
        </div>
      </section>

      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Context Diffusion Ladder</div>
            <div className="card-meta">{checkedAt ? `checked ${new Date(checkedAt).toLocaleString()}` : 'inherited gates loading'}</div>
          </div>
          <div className="stage-action-list">
            {steps.map((step) => (
              <button
                key={step.id}
                type="button"
                className={`btn stage-action ${stepClass(step.state)}`}
                disabled
                title={step.evidence}
              >
                {stepIcon(step.state)}
                {String(step.step).padStart(2, '0')} {step.label}
              </button>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Why This Branch Exists</div>
            <div className="card-meta">learning from context diffusion</div>
          </div>
          <div className="metric-list">
            <div>
              <span>Original Diffusion CARA signal</span>
              <strong>Structured int conditioning + DiT head</strong>
            </div>
            <div>
              <span>New signal under test</span>
              <strong>Audio context tokens beside text conditioning</strong>
            </div>
            <div>
              <span>Main transfer from Context Diffusion</span>
              <strong>Context is first-class conditioning, not prompt text</strong>
            </div>
            <div>
              <span>Primary risk</span>
              <strong className="v-warn">Context leakage or source overlap</strong>
            </div>
          </div>
          <div className="pool-empty-state" style={{ marginTop: 16 }}>
            <AlertTriangle size={18} />
            This page can now submit context-pack, context-cache, and context-preflight Azure jobs. Context
            smoke is now available after preflight as a context-conditioner control test. Full Stable Audio
            context fine-tuning remains locked until the DiT trainer consumes frozen context latents.
          </div>
          <div className="pool-empty-state" style={{ marginTop: 12 }}>
            <Layers3 size={18} />
            TLDR: the first Diffusion run taught the model CARA labels from each audio chunk. This branch
            also gives it a few related example sounds as model-readable context, then checks whether
            correct context helps and shuffled or wrong-family context hurts.
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">Context Training Run Progress</div>
          <div className="card-meta">
            {runProgress?.checked_at ? `checked ${new Date(runProgress.checked_at).toLocaleString()}` : 'awaiting training metrics'}
          </div>
        </div>
        <div className="metric-list compact">
          <div><span>Model</span><strong>{runProgress?.model_label ?? 'Context Diffusion'}</strong></div>
          <div><span>Azure job</span><strong className="mono">{runProgress?.job_name ?? '-'}</strong></div>
          <div><span>Status</span><strong>{runProgress?.status ?? '-'}</strong></div>
          <div><span>Scope</span><strong>{runProgress?.training_scope ?? '-'}</strong></div>
          <div>
            <span>Step</span>
            <strong>
              {runProgress?.observed_step
                ? `${runProgress.observed_step.toLocaleString()} / ${runProgress.max_steps?.toLocaleString() ?? '-'}`
                : '-'}
            </strong>
          </div>
          <div>
            <span>Step progress</span>
            <strong>{runProgress?.step_percent !== null && runProgress?.step_percent !== undefined ? `${runProgress.step_percent.toFixed(2)}%` : '-'}</strong>
          </div>
          <div>
            <span>Chunks seen</span>
            <strong>
              {runProgress?.chunks_seen_estimate !== null && runProgress?.chunks_seen_estimate !== undefined
                ? `${runProgress.chunks_seen_estimate.toLocaleString()} / ${runProgress.effective_train_chunks?.toLocaleString() ?? '-'}`
                : '-'}
            </strong>
          </div>
          <div>
            <span>Epoch estimate</span>
            <strong>
              {runProgress?.completed_epochs_estimate !== null && runProgress?.completed_epochs_estimate !== undefined
                ? `${runProgress.completed_epochs_estimate.toLocaleString()} + ${runProgress.epoch_percent?.toFixed(2) ?? '0.00'}%`
                : '-'}
            </strong>
          </div>
          <div><span>Latest loss</span><strong>{runProgress?.latest_loss !== null && runProgress?.latest_loss !== undefined ? runProgress.latest_loss.toFixed(5) : '-'}</strong></div>
          <div><span>Elapsed</span><strong>{formatDuration(runProgress?.elapsed_seconds)}</strong></div>
          <div><span>Estimated time left</span><strong>{formatDuration(runProgress?.estimated_remaining_seconds)}</strong></div>
          <div>
            <span>Metrics source</span>
            <strong>{runProgress?.metrics_source === 'azure_datastore_lightning_metrics_csv' ? 'datastore metrics.csv' : runProgress?.metrics_source === 'azure_mlflow_step_metrics' ? 'MLflow live' : runProgress?.metrics_error ? 'unavailable' : '-'}</strong>
          </div>
        </div>
        <div className="bar" aria-label={`Context Diffusion progress ${runProgress?.step_percent ?? 0} percent`} style={{ marginTop: 14 }}>
          <div
            className="bar-fill"
            style={{
              width: `${Math.min(100, Math.max(0, runProgress?.step_percent ?? 0))}%`,
            }}
          />
        </div>
        <div className="pool-empty-state" style={{ marginTop: 14 }}>
          <RefreshCw size={18} />
          {runProgress?.metrics_artifact
            ? `Progress is being read from ${runProgress.metrics_artifact}${runProgress.metrics_row_count ? ` (${runProgress.metrics_row_count.toLocaleString()} rows)` : ''}.`
            : runProgress?.metrics_error
              ? `Metrics unavailable: ${runProgress.metrics_error}`
              : runProgress?.note ?? 'Progress is read from Azure MLflow or datastore artifacts without touching the running job.'}
        </div>
        {runProgressError ? <div className="error-banner" style={{ marginTop: 12 }}>{runProgressError}</div> : null}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 14 }}>
          <button className="btn btn-ghost" type="button" onClick={refreshRunProgress} disabled={runProgressLoading}>
            <RefreshCw size={16} className={runProgressLoading ? 'spin' : ''} /> {runProgressLoading ? 'Checking...' : 'Check Run Progress'}
          </button>
          {runProgress?.studio_url ? (
            <a className="btn btn-ghost" href={runProgress.studio_url} target="_blank" rel="noreferrer">
              Open in Azure Studio
            </a>
          ) : null}
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">Methodology Delta</div>
          <div className="card-meta">append-only comparator</div>
        </div>
        <div className="hybrid-comparison-grid">
          <div>
            <div className="hybrid-comparison-title">
              <Workflow size={16} /> Existing Diffusion
            </div>
            <p className="dim">
              Ordinary prompt text stays unchanged. CARA pool and family indices are native integer conditioners,
              and the attribution head reads DiT hidden states.
            </p>
          </div>
          <div>
            <div className="hybrid-comparison-title">
              <Layers3 size={16} /> Context Diffusion
            </div>
            <p className="dim">
              Add frozen audio context embeddings from source-disjoint examples as cross-attention conditioning
              beside text, then train/evaluate the same attribution head family.
            </p>
          </div>
          <div>
            <div className="hybrid-comparison-title">
              <FlaskConical size={16} /> Controls
            </div>
            <p className="dim">
              Compare context plus prompt, context-only, prompt-only, shuffled context, mismatched family, and
              no-context baseline lanes before making claims.
            </p>
          </div>
        </div>
      </section>

      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">New Evidence Requirements</div>
            <div className="card-meta">must be logged</div>
          </div>
          <div className="metric-list">
            <div>
              <span>Context pack manifest</span>
              <strong>target chunk, context chunk ids, pool/family, split, registry hash</strong>
            </div>
            <div>
              <span>Context dropout</span>
              <strong>C+P / C-only / P-only rates recorded per run</strong>
            </div>
            <div>
              <span>Source disjointness</span>
              <strong>no context example from the target source file</strong>
            </div>
            <div>
              <span>Attribution output</span>
              <strong>exact, correctly repaired, family fallback, unattributable</strong>
            </div>
            <div>
              <span>Negative controls</span>
              <strong>shuffled context and mismatched-family context</strong>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Launch Guardrail</div>
            <div className="card-meta">existing Azure resources only</div>
          </div>
          <div className="pool-empty-state">
            <ShieldCheck size={18} />
            Context Diffusion must reuse the existing Azure ML workspace, approved compute, datastore,
            and Stable Audio environment lineage. No Marketplace endpoints or new paid services are part of
            this branch.
          </div>
          <div className="hybrid-launch-locks">
            <button className="btn stage-action" type="button" onClick={refreshReadiness} disabled={loading}>
              <RefreshCw size={16} className={loading ? 'spin' : ''} /> Refresh Inherited Gates
            </button>
            <button
              className={`btn stage-action ${activeStage === 10 ? 'is-current' : 'is-muted'}`}
              type="button"
              disabled={!contextLaunch?.context_packs_enabled || submittingStage !== null}
              onClick={() => void submitContextStage(10)}
              title={contextLadder?.context_packs?.reason}
            >
              <DatabaseZap size={16} className={submittingStage === 10 ? 'spin' : ''} /> 10 Submit Context Packs
            </button>
            <button
              className={`btn stage-action ${activeStage === 11 ? 'is-current' : 'is-muted'}`}
              type="button"
              disabled={!contextLaunch?.context_cache_enabled || submittingStage !== null}
              onClick={() => void submitContextStage(11)}
              title={contextLadder?.context_cache?.reason}
            >
              <Route size={16} className={submittingStage === 11 ? 'spin' : ''} /> 11 Submit Context Cache
            </button>
            <button
              className={`btn stage-action ${activeStage === 12 ? 'is-current' : 'is-muted'}`}
              type="button"
              disabled={!contextLaunch?.context_preflight_enabled || submittingStage !== null}
              onClick={() => void submitContextStage(12)}
              title={contextLadder?.context_preflight?.reason}
            >
              <FlaskConical size={16} className={submittingStage === 12 ? 'spin' : ''} /> 12 Submit Context Preflight
            </button>
            <button
              className={`btn stage-action ${activeStage === 13 ? 'is-current' : 'is-muted'}`}
              type="button"
              disabled={!contextLaunch?.context_smoke_enabled || submittingStage !== null}
              onClick={() => void submitContextStage(13)}
              title={contextLaunch?.context_smoke_reason}
            >
              <FlaskConical size={16} className={submittingStage === 13 ? 'spin' : ''} /> 13 Submit Context Smoke
            </button>
            <button
              className={`btn stage-action ${activeStage === 14 ? 'is-current' : 'is-muted'}`}
              type="button"
              disabled={!contextLaunch?.context_full_enabled || submittingStage !== null}
              onClick={() => void submitContextStage(14)}
              title={contextLaunch?.context_full_reason}
            >
              <Workflow size={16} className={submittingStage === 14 ? 'spin' : ''} /> 14 Launch Full Context Fine-Tune
            </button>
            <button
              className={`btn stage-action ${activeStage === 15 ? 'is-current' : 'is-muted'}`}
              type="button"
              disabled={!contextLadder?.context_full?.passed}
              onClick={() => onNavigate?.('testing')}
              title={
                contextLadder?.context_full?.passed
                  ? 'Open Testing to run Step 15 generated-audio benchmark and Step 16 attribution scoring with the locked prompt set.'
                  : 'Step 15 unlocks after the full Context Diffusion fine-tune completes.'
              }
            >
              <FlaskConical size={16} /> 15 Open Benchmark Testing
            </button>
          </div>
          {error ? (
            <div className="pool-empty-state" style={{ marginTop: 12 }}>
              <AlertTriangle size={18} /> {error}
            </div>
          ) : null}
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">Inherited And New Evidence</div>
          <div className="card-meta">audit trail</div>
        </div>
        <div className="metric-list">
          {steps.map((step) => (
            <div key={`audit-${step.id}`}>
              <span>
                <strong>{String(step.step).padStart(2, '0')} {step.label}</strong>
                <span className="dim" style={{ display: 'block', marginTop: 4 }}>{step.evidence}</span>
              </span>
              {sourceBadge(step.source)}
            </div>
          ))}
        </div>
        <div className="paths" style={{ marginTop: 18 }}>
          <span>Stable Audio manifest: <span className="mono">{readiness?.data_locations?.azure_stable_audio_manifest ?? 'pending'}</span></span>
          <span>Full output root: <span className="mono">{readiness?.data_locations?.azure_stable_audio_full_output_root ?? 'pending'}</span></span>
          <span>Context branch root: <span className="mono">{readiness?.data_locations?.azure_stable_audio_context_root ?? 'azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/stable_audio_context/'}</span></span>
        </div>
        {logs.length > 0 ? <pre className="log-panel" style={{ marginTop: 18 }}>{logs.join('\n')}</pre> : null}
      </section>
    </>
  );
};
