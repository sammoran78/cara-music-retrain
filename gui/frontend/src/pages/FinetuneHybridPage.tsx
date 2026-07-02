import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Brain,
  CheckCircle2,
  CloudUpload,
  Cpu,
  FileCheck2,
  GitBranch,
  Layers3,
  Lock,
  Microscope,
  Play,
  RefreshCw,
  Route,
  ShieldCheck,
  SlidersHorizontal,
  Terminal,
  Workflow,
} from 'lucide-react';
import { PageHeader, PlaceholderBadge } from './PageHeader';

const ladder = [
  {
    step: '01',
    label: 'ACE source + license review',
    state: 'complete',
    evidence: 'Official checkpoint, training route, license/access terms, and cost guardrail documented.',
    comparator: 'Pre-study gate, equivalent to choosing Stable Audio and MusicGen base checkpoints.',
  },
  {
    step: '02',
    label: 'ACE environment preflight',
    state: 'blocked',
    evidence: 'CUDA, ACE-Step imports, Side-Step/LoRA dependencies, checkpoint access, and datastore access.',
    comparator: 'Matches Stable Audio and MusicGen environment preflight gates.',
  },
  {
    step: '03',
    label: 'Prepare ACE tensors',
    state: 'blocked',
    evidence: 'ACE tensor manifests preserve chunk lineage, CARA pool/family labels, and registry resolver hash.',
    comparator: 'Matches Stable Audio chunks and MusicGen EnCodec-cache manifest binding.',
  },
  {
    step: '04',
    label: 'Planner survival probe',
    state: 'blocked',
    evidence: 'CARA exact/repairable/lost rates after LM planning, caption rewriting, and structured metadata output.',
    comparator: 'Unique ACE bottleneck test; no training claim yet.',
  },
  {
    step: '05',
    label: 'DiT tap discovery',
    state: 'blocked',
    evidence: 'Named mid/late DiT hidden states, tensor shapes, hook stability, and detached head dry-run.',
    comparator: 'Matches Stable Audio hidden-state attribution-head tap discovery.',
  },
  {
    step: '06',
    label: 'Baseline LoRA smoke',
    state: 'blocked',
    evidence: 'Same data, no CARA signal, LoRA checkpoint, loss curve, and resume/checkpoint behavior.',
    comparator: 'Matches no-CARA baseline branches in diffusion and AR ladders.',
  },
  {
    step: '07',
    label: 'CARA-lite planner smoke',
    state: 'blocked',
    evidence: 'Prompt-only CARA survival and leakage metrics through planner text/caption fields.',
    comparator: 'Matches CARA-lite prompt-control branches.',
  },
  {
    step: '08',
    label: 'Detached DiT head smoke',
    state: 'blocked',
    evidence: 'Frozen/detached DiT hidden states predict registry CARA labels with valid top-k decoding.',
    comparator: 'Matches Stable Audio cara_head and MusicGen detached suffix-probe evidence.',
  },
  {
    step: '09',
    label: 'Planner-preserved CARA smoke',
    state: 'blocked',
    evidence: 'Structured CARA passes through LM planner and remains recoverable at DiT taps.',
    comparator: 'Tests the hybrid claim: attribution through planner bottleneck.',
  },
  {
    step: '10',
    label: 'Planner-bypass CARA smoke',
    state: 'blocked',
    evidence: 'Direct/constrained CARA conditioning reaches DiT with planner rewrite reduced or bypassed.',
    comparator: 'Separates planner failure from DiT attribution failure.',
  },
  {
    step: '11',
    label: 'Hybrid CARA-Strong smoke',
    state: 'blocked',
    evidence: 'Non-detached CARA auxiliary loss plus DiT attribution metrics and planner-survival metrics.',
    comparator: 'First ACE stage eligible for CARA-Strong-style evidence.',
  },
  {
    step: '12',
    label: 'Full ACE Side-Step LoRA fine-tune',
    state: 'blocked',
    evidence: 'Deployable Side-Step LoRA adapter delta, held-out planner/DiT attribution, registry decoding, and baseline-vs-CARA comparison report.',
    comparator: 'Final third-arm comparison against diffusion and autoregressive results.',
  },
];

const sources = [
  {
    label: 'ACE-Step v1.5 model card',
    href: 'https://huggingface.co/ACE-Step/Ace-Step1.5',
  },
  {
    label: 'ACE-Step v1.5 GitHub',
    href: 'https://github.com/ace-step/ACE-Step-1.5',
  },
  {
    label: 'ACE-Step v1.5 paper',
    href: 'https://arxiv.org/abs/2602.00744',
  },
  {
    label: 'Side-Step training guide',
    href: 'https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/sidestep/Training%20Guide.md',
  },
  {
    label: 'Side-Step dataset preparation',
    href: 'https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/sidestep/Dataset%20Preparation.md',
  },
];

const evidenceMetrics = [
  'planner_survival_exact',
  'planner_survival_repairable',
  'planner_cara_lost',
  'planner_cara_hallucinated',
  'dit_family_top1',
  'dit_pool_top1',
  'dit_pool_top5',
  'registry_valid_rate',
  'shuffled_label_baseline',
  'source_disjoint_eval',
];

const plannerOutcomes = [
  {
    title: 'Survives Planner',
    status: 'strong positive',
    detail: 'CARA structure remains present or registry-repairable after the LM planner and is recoverable from DiT hidden states.',
  },
  {
    title: 'Lost In Planner',
    status: 'useful negative',
    detail: 'CARA is normalized away before DiT synthesis; report as a planner-mediated attribution-loss mode.',
  },
  {
    title: 'Recoverable At DiT Only',
    status: 'mixed signal',
    detail: 'Planner text loses CARA, but DiT states still recover pool/family labels from conditioned audio evidence.',
  },
];

interface HybridLadderStep {
  stage: number;
  label: string;
  passed?: boolean;
  active?: boolean;
  locked?: boolean;
  required_environment?: string;
  reason?: string;
  latest_job?: {
    name?: string;
    status?: string;
    studio_url?: string;
    output_path?: string;
  } | null;
}

interface HybridReadiness {
  status: string;
  training_launch_enabled: boolean;
  training_launch_reason: string;
  active_ladder_reason?: string;
  target_model?: {
    model_family?: string;
    architecture?: string;
    base_checkpoint?: string;
    planner_checkpoint?: string;
    planner_size?: string;
    dit_variant?: string;
    comparison_role?: string;
  };
  ace_source_review?: HybridLadderStep | null;
  ace_preflight?: HybridLadderStep | null;
  ace_ladder?: {
    steps: HybridLadderStep[];
    next_stage: number;
    next_label: string;
    reason?: string;
  };
  ace_sidestep_inputs?: HybridLadderStep | null;
  ace_launch?: Record<string, boolean>;
  ace_full_prerequisites?: {
    ready?: boolean;
    reason?: string;
    errors?: string[];
    checkpoint_uri?: string;
    sidestep_tensor_uri?: string;
    checks?: Record<string, {
      configured?: boolean;
      verified?: boolean;
      uri?: string;
      prefix?: string;
      reason?: string;
      probe?: {
        exists?: boolean;
        example_blob?: {
          name?: string;
          size?: number;
          last_modified?: string;
        } | null;
      };
      probe_error?: string;
    }>;
  };
  data_locations?: Record<string, string>;
  cloud_job_policy?: Record<string, unknown>;
  evidence_contract?: {
    sample_rate_hz?: number;
    latent_rate_hz?: number;
    metrics?: string[];
    label_fields_required?: string[];
  };
  active_h100_jobs?: Array<{ name?: string; status?: string; display_name?: string; h100_compute_target?: string }>;
  submitted_training_jobs?: Array<Record<string, unknown>>;
}

const fallbackSteps: HybridLadderStep[] = ladder.map((item) => ({
  stage: Number(item.step),
  label: item.label,
  passed: item.state === 'complete',
  active: false,
  locked: item.state !== 'complete',
  reason: item.evidence,
}));

export const FinetuneHybridPage: React.FC = () => {
  const [readiness, setReadiness] = useState<HybridReadiness | null>(null);
  const [loading, setLoading] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [loadCheckpoint, setLoadCheckpoint] = useState(false);
  const [confirmation, setConfirmation] = useState('');
  const [fullConfirmation, setFullConfirmation] = useState('');
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refreshReadiness = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/training/hybrid-readiness');
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'Hybrid readiness check failed');
      setReadiness(json);
      setLogs((prev) => [
        ...prev.slice(-7),
        `[monitor] ${json.ace_ladder?.next_label ?? 'Hybrid readiness'} · ${json.training_launch_reason ?? json.status}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Hybrid readiness check failed';
      setError(message);
      setLogs((prev) => [...prev.slice(-7), `[error] ${message}`]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refreshReadiness();
  }, []);

  const steps = useMemo(() => readiness?.ace_ladder?.steps ?? fallbackSteps, [readiness]);
  const preflightActive = Boolean(readiness?.ace_preflight?.active);
  const preflightReady = Boolean(readiness?.ace_launch?.preflight_enabled ?? readiness?.training_launch_enabled) && !preflightActive;
  const preflightConfirmed = confirmation.trim() === 'LAUNCH ACE PREFLIGHT';
  const canLaunchPreflight = preflightReady && !launching;

  const launchFlag = (key: string) => Boolean(readiness?.ace_launch?.[key]) && !launching;
  const canRecordSourceReview = launchFlag('source_review_enabled');

  const stageLaunchEnabled = (stage: number) => {
    if (stage === 1) return canRecordSourceReview;
    if (stage === 2) return canLaunchPreflight;
    if (stage === 3) return launchFlag('tensor_prepare_enabled');
    if (stage === 4) return launchFlag('planner_probe_enabled');
    if (stage === 5) return launchFlag('dit_tap_enabled');
    if (stage === 6) return launchFlag('baseline_smoke_enabled');
    if (stage === 7) return launchFlag('cara_lite_smoke_enabled');
    if (stage === 8) return launchFlag('cara_head_smoke_enabled');
    if (stage === 9) return launchFlag('planner_preserved_smoke_enabled');
    if (stage === 10) return launchFlag('planner_bypass_smoke_enabled');
    if (stage === 11) return launchFlag('cara_strong_smoke_enabled');
    if (stage === 12) return launchFlag('full_enabled');
    return false;
  };

  const stageLaunchVariant = (stage: number) => {
    if (stage === 6) return 'baseline_lora';
    if (stage === 7) return 'cara_lite';
    if (stage === 8) return 'cara_head';
    if (stage === 9) return 'planner_preserved';
    if (stage === 10) return 'planner_bypass';
    if (stage === 11) return 'cara_strong';
    return undefined;
  };

  const recordSourceReview = async () => {
    if (!canRecordSourceReview) return;
    setLaunching(true);
    setError(null);
    setLogs((prev) => [...prev.slice(-7), '[record] Recording ACE-Step source/license review...']);
    try {
      const res = await fetch('/api/training/ace/source-review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          confirmed: true,
          notes: 'Dashboard-recorded review of ACE-Step v1.5 source, Side-Step training route, and existing-Azure-resource cost guardrail.',
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'ACE-Step source/license review failed');
      setReadiness(json.readiness);
      setLogs((prev) => [
        ...prev.slice(-7),
        `[record:complete] ACE source/license review · artifact=${json.artifact ?? 'registry/cara_strong/ace_step_source_license_review.json'}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'ACE-Step source/license review failed';
      setError(message);
      setLogs((prev) => [...prev.slice(-7), `[record:error] ${message}`]);
    } finally {
      setLaunching(false);
    }
  };

  const runAcePreflight = async () => {
    if (!canLaunchPreflight) return;
    if (!preflightConfirmed) {
      const message = 'Type LAUNCH ACE PREFLIGHT before submitting Step 02.';
      setError(message);
      setLogs((prev) => [...prev.slice(-7), `[launch:blocked] ${message}`]);
      return;
    }
    setLaunching(true);
    setError(null);
    setLogs((prev) => [...prev.slice(-7), '[launch] Submitting ACE-Step v1.5 environment preflight...']);
    try {
      const res = await fetch('/api/training/ace-preflight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          checkpoint: 'ACE-Step/Ace-Step1.5',
          planner_checkpoint: 'ACE-Step/acestep-5Hz-lm-0.6B',
          dit_variant: 'turbo_dit',
          load_checkpoint: loadCheckpoint,
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'ACE-Step preflight submission failed');
      setReadiness(json.readiness);
      setConfirmation('');
      setLogs((prev) => [
        ...prev.slice(-7),
        `[launch:submitted] ACE preflight job=${json.job?.name ?? 'unknown'} · output=${json.job?.output_path ?? 'pending'}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'ACE-Step preflight submission failed';
      setError(message);
      setLogs((prev) => [...prev.slice(-7), `[launch:error] ${message}`]);
    } finally {
      setLaunching(false);
    }
  };

  const submitAceStage = async (stage: number, variant?: string) => {
    const endpoint =
      stage === 3
        ? '/api/training/ace/tensors'
        : stage === 4
          ? '/api/training/ace/planner-probe'
          : stage === 5
            ? '/api/training/ace/dit-taps'
            : stage === 12
              ? '/api/training/ace/full'
              : '/api/training/ace/smoke';
    const fullPhrase = 'LAUNCH ACE FULL FINE-TUNE';
    if (stage === 12 && fullConfirmation.trim() !== fullPhrase) {
      setError(`Type ${fullPhrase} before launching the full ACE stage.`);
      return;
    }
    setLaunching(true);
    setError(null);
    setLogs((prev) => [...prev.slice(-7), `[launch] Submitting ACE Step ${stage}${variant ? ` · ${variant}` : ''}...`]);
    try {
      const body =
        stage === 3
          ? { dry_run: false, max_rows: 0, compute_strategy: 'prefer_h100_else_cpu' }
          : stage === 4
            ? { dry_run: false, max_rows: 0, compute_strategy: 'prefer_h100_else_cpu' }
            : stage === 5
              ? { dry_run: false, load_checkpoint: loadCheckpoint, max_rows: 1024 }
              : stage === 12
                ? {
                    dry_run: false,
                    confirmation_phrase: fullPhrase,
                    run_sidestep: true,
                    max_steps: 20000,
                    batch_size: 4,
                    learning_rate: 0.0001,
                    max_train_rows: 0,
                    max_eval_rows: 2048,
                    checkpoint_dir:
                      readiness?.ace_full_prerequisites?.checkpoint_uri ??
                      readiness?.data_locations?.azure_ace_checkpoint_root ??
                      'azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/ace_step/checkpoints/',
                    sidestep_tensor_dir:
                      readiness?.ace_full_prerequisites?.sidestep_tensor_uri ??
                      readiness?.data_locations?.azure_ace_sidestep_tensor_root ??
                      'azureml://datastores/ds_cara_raw_audio/paths/training-runs/cara-strong-v0.4/ace_step/tensors/sidestep_tensors/',
                    model_variant: 'turbo',
                    adapter_type: 'lora',
                    rank: 64,
                    alpha: 128,
                    num_workers: 0,
                    timestep_mode: 'continuous',
                  }
                : {
                    dry_run: false,
                    variant,
                    max_steps: 250,
                    batch_size: 64,
                    learning_rate: 0.001,
                    max_train_rows: 4096,
                    max_eval_rows: 1024,
                  };
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? `ACE Step ${stage} submission failed`);
      setReadiness(json.readiness);
      setLogs((prev) => [
        ...prev.slice(-7),
        `[launch:submitted] ACE Step ${stage} job=${json.job?.name ?? 'unknown'} · output=${json.job?.output_path ?? 'pending'}`,
      ]);
      if (stage === 12) setFullConfirmation('');
    } catch (err) {
      const message = err instanceof Error ? err.message : `ACE Step ${stage} submission failed`;
      setError(message);
      setLogs((prev) => [...prev.slice(-7), `[launch:error] ${message}`]);
    } finally {
      setLaunching(false);
    }
  };

  const runSidestepInputPrep = async () => {
    if (!launchFlag('sidestep_inputs_enabled')) return;
    setLaunching(true);
    setError(null);
    setLogs((prev) => [...prev.slice(-7), '[launch] Submitting ACE Step 12a Side-Step input preparation...']);
    try {
      const res = await fetch('/api/training/ace/sidestep-inputs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dry_run: false,
          max_rows: 0,
          allow_checkpoint_download: true,
          model_variant: 'turbo',
        }),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.detail ?? 'ACE Side-Step input preparation submission failed');
      setReadiness(json.readiness);
      setLogs((prev) => [
        ...prev.slice(-7),
        `[launch:submitted] ACE Side-Step inputs job=${json.job?.name ?? 'unknown'} · output=${json.job?.output_path ?? 'pending'}`,
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'ACE Side-Step input preparation submission failed';
      setError(message);
      setLogs((prev) => [...prev.slice(-7), `[launch:error] ${message}`]);
    } finally {
      setLaunching(false);
    }
  };

  return (
    <>
      <PageHeader
        kicker="Fine-tuning · Hybrid"
        title={
          <>
            ACE-Step v1.5 <em>hybrid</em> CARA comparison
          </>
        }
        description={
          <>
            0.6B-planner Hybrid arm for testing whether CARA attribution survives ACE-Step v1.5's LM planner
            and remains recoverable at the DiT synthesis stage beside the Diffusion, Context Diffusion, and MusicGen results.
          </>
        }
        actions={<PlaceholderBadge label="0.6B target branch" />}
      />

      <section className="kpi-grid">
        <div className="kpi">
          <div className="kpi-label">Target model</div>
          <div className="kpi-value">ACE 0.6B</div>
          <div className="kpi-trend">LM planner plus DiT synthesis</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Candidate path</div>
          <div className="kpi-value">LoRA</div>
          <div className="kpi-trend">Side-Step corrected mode first</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">DiT tap viability</div>
          <div className="kpi-value">High</div>
          <div className="kpi-trend">same synthesis family as the diffusion branch</div>
        </div>
        <div className="kpi">
          <div className="kpi-label">Planner risk</div>
          <div className="kpi-value">Medium</div>
          <div className="kpi-trend">CARA may be rewritten out before DiT</div>
        </div>
      </section>

      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Hybrid Stage Ladder</div>
            <div className="card-meta">{loading ? 'refreshing' : `next · ${readiness?.ace_ladder?.next_label ?? 'ACE preflight'}`}</div>
          </div>
          <div className="stage-action-list">
            {steps.map((item) => {
              const enabled = stageLaunchEnabled(item.stage);
              const onClick =
                item.stage === 1
                  ? recordSourceReview
                  : item.stage === 2
                    ? runAcePreflight
                    : () => submitAceStage(item.stage, stageLaunchVariant(item.stage));
              return (
                <button
                  key={item.stage}
                  type="button"
                  className={`btn stage-action ${enabled ? 'is-current' : item.active ? 'is-current' : item.passed ? 'is-complete' : 'is-muted'}`}
                  disabled={!enabled}
                  onClick={onClick}
                  title={
                    item.stage === 2 && enabled && !preflightConfirmed
                      ? 'Type LAUNCH ACE PREFLIGHT in the launch box before submitting Step 02.'
                      : item.reason ?? readiness?.training_launch_reason ?? ''
                  }
                >
                  {item.passed ? <CheckCircle2 size={16} /> : item.active ? <RefreshCw size={16} /> : enabled ? <Play size={16} /> : <Lock size={16} />}
                  {String(item.stage).padStart(2, '0')} {item.label}
                </button>
              );
            })}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">How Likely To Work</div>
            <div className="card-meta">ACE-Step v1.5 · 0.6B planner</div>
          </div>
          <div className="metric-list">
            <div>
              <span>DiT hidden-state attribution</span>
              <strong className="v-good">Plausible / high</strong>
            </div>
            <div>
              <span>Structured CARA conditioning through planner</span>
              <strong className="v-warn">Uncertain / medium</strong>
            </div>
            <div>
              <span>Prompt-only CARA leakage control</span>
              <strong className="v-good">Required</strong>
            </div>
            <div>
              <span>Negative result value</span>
              <strong className="v-good">Publishable</strong>
            </div>
          </div>
          <div className="pool-empty-state" style={{ marginTop: 16 }}>
            <AlertTriangle size={18} />
            The live risk is the LM planner: if it preserves CARA structure into DiT conditioning, this is
            strong evidence through a CoT bottleneck. If it rewrites CARA away, that is still a useful
            regulatory finding about planner-mediated attribution loss.
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">Comparative Design</div>
          <div className="card-meta">same registry · architecture-native signals</div>
        </div>
        <div className="hybrid-comparison-grid">
          <div>
            <div className="hybrid-comparison-title">
              <Layers3 size={16} /> Stable Audio
            </div>
            <p className="dim">
              Diffusion lead branch: structured CARA conditioners plus DiT hidden-state attribution head.
            </p>
          </div>
          <div>
            <div className="hybrid-comparison-title">
              <GitBranch size={16} /> MusicGen
            </div>
            <p className="dim">
              Autoregressive branch: CARA suffix tokens tied to cached EnCodec audio-token targets.
            </p>
          </div>
          <div>
            <div className="hybrid-comparison-title">
              <Workflow size={16} /> ACE-Step v1.5
            </div>
            <p className="dim">
              Hybrid branch: 0.6B LM planner survival first, then recover CARA at the DiT synthesis stage.
            </p>
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
          <div className="card-title">Selected Hybrid Target</div>
          <div className="card-meta">side-by-side comparator</div>
        </div>
        <div className="metric-list">
          <div>
            <span>Base checkpoint</span>
            <strong className="mono">{readiness?.target_model?.base_checkpoint ?? 'ACE-Step/Ace-Step1.5'}</strong>
          </div>
          <div>
            <span>Planner checkpoint</span>
            <strong className="mono">{readiness?.target_model?.planner_checkpoint ?? 'ACE-Step/acestep-5Hz-lm-0.6B'}</strong>
          </div>
          <div>
            <span>DiT variant</span>
            <strong>{readiness?.target_model?.dit_variant ?? 'turbo_dit'}</strong>
          </div>
          <div>
            <span>Comparison role</span>
            <strong>{readiness?.target_model?.comparison_role ?? 'Comparable-size Hybrid CARA-Strong arm beside existing model lanes.'}</strong>
          </div>
        </div>
      </section>

      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Planner Survival Outcomes</div>
            <div className="card-meta">all outcomes are evidence</div>
          </div>
          <div className="metric-list">
            {plannerOutcomes.map((outcome) => (
              <div key={outcome.title}>
                <span>
                  <strong>{outcome.title}</strong>
                  <span className="dim" style={{ display: 'block', marginTop: 4 }}>{outcome.detail}</span>
                </span>
                <strong className={outcome.status === 'useful negative' ? 'v-warn' : 'v-good'}>
                  {outcome.status}
                </strong>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Evidence Contract</div>
            <div className="card-meta">logged every smoke</div>
          </div>
          <div className="hybrid-metric-tags">
            {evidenceMetrics.map((metric) => (
              <span key={metric} className="hybrid-metric-tag">{metric}</span>
            ))}
          </div>
          <div className="pool-empty-state" style={{ marginTop: 16 }}>
            <FileCheck2 size={18} />
            A passing ACE smoke must include manifest lock id, registry hash, source-disjoint split summary,
            checkpoint/environment id, and explicit wording about whether it is planner-only, head-only, or
            CARA-Strong evidence.
          </div>
        </div>
      </section>

      <section className="split-2">
        <div className="card">
          <div className="card-header">
            <div className="card-title">Implementation Requirements</div>
            <div className="card-meta">Azure preflight wired</div>
          </div>
          <div className="metric-list">
            <div>
              <span>Checkpoint source</span>
              <strong>ACE-Step/Ace-Step1.5 with 0.6B planner target</strong>
            </div>
            <div>
              <span>Training toolkit</span>
              <strong>Side-Step corrected mode, LoRA first</strong>
            </div>
            <div>
              <span>Preprocess output</span>
              <strong>Side-Step JSON mode, ACE tensors, CARA registry resolver</strong>
            </div>
            <div>
              <span>Evidence controls</span>
              <strong>planner-only, DiT-only, shuffled-label, source-disjoint eval</strong>
            </div>
            <div>
              <span>Launch policy</span>
              <strong>GPU-only after typed confirmation; no CPU fallback for ACE CUDA gates</strong>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">Sources</div>
            <div className="card-meta">primary references</div>
          </div>
          <div className="paths">
            {sources.map((source) => (
              <a key={source.href} href={source.href} target="_blank" rel="noreferrer">
                <Microscope size={14} style={{ verticalAlign: 'text-bottom', marginRight: 6 }} />
                {source.label}
              </a>
            ))}
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-header">
            <div className="card-title">Live Launch Controls</div>
            <div className="card-meta">typed gates · cost guardrail</div>
          </div>
          <div className="pool-empty-state">
            <ShieldCheck size={18} />
          {readiness?.active_ladder_reason ?? readiness?.training_launch_reason ?? 'Hybrid readiness has not been loaded yet.'}
          </div>
        {error && (
          <div className="pool-empty-state" style={{ marginTop: 12 }}>
            <AlertTriangle size={18} />
            {error}
          </div>
        )}
        {readiness?.ace_full_prerequisites && (
          <div className="paths" style={{ marginTop: 18 }}>
            <span>
              ACE checkpoint bundle:{' '}
              <span className="mono">
                {readiness.ace_full_prerequisites.checks?.checkpoint_dir?.verified ? 'ready' : 'missing'}
              </span>
              {' · '}
              <span className="mono">{readiness.ace_full_prerequisites.checkpoint_uri}</span>
            </span>
            <span>
              Side-Step tensors:{' '}
              <span className="mono">
                {readiness.ace_full_prerequisites.checks?.sidestep_tensor_dir?.verified ? 'ready' : 'missing'}
              </span>
              {' · '}
              <span className="mono">{readiness.ace_full_prerequisites.sidestep_tensor_uri}</span>
            </span>
          {readiness.ace_full_prerequisites.errors?.map((item) => (
              <span key={item} className="mono">blocked: {item}</span>
            ))}
            {readiness.ace_sidestep_inputs?.latest_job?.name && (
              <span>
                Side-Step input job:{' '}
                <span className="mono">
                  {readiness.ace_sidestep_inputs.latest_job.name} · {readiness.ace_sidestep_inputs.latest_job.status ?? 'unknown'}
                </span>
              </span>
            )}
          </div>
        )}
        <div className="hybrid-launch-locks">
          <button className="btn stage-action" type="button" onClick={refreshReadiness} disabled={loading}>
            <RefreshCw size={16} /> Refresh Hybrid Gates
          </button>
          {preflightReady && (
            <>
              <label className="toggle-row" style={{ margin: 0 }}>
                <input
                  type="checkbox"
                  checked={loadCheckpoint}
                  onChange={(event) => setLoadCheckpoint(event.target.checked)}
                  disabled={preflightActive || launching}
                />
                Load ACE checkpoint during preflight
              </label>
              <input
                className="input"
                value={confirmation}
                onChange={(event) => setConfirmation(event.target.value)}
                placeholder="Type LAUNCH ACE PREFLIGHT"
                disabled={launching}
              />
              <button
                className={`btn stage-action ${preflightReady && preflightConfirmed ? 'is-current' : 'is-muted'}`}
                type="button"
                onClick={runAcePreflight}
                disabled={!preflightReady || launching}
                title={preflightReady && !preflightConfirmed ? 'Type LAUNCH ACE PREFLIGHT before live submission.' : readiness?.training_launch_reason}
              >
                <CloudUpload size={16} /> 02 Run ACE-Step Preflight
              </button>
            </>
          )}
          <button
            className={`btn stage-action ${launchFlag('sidestep_inputs_enabled') ? 'is-current' : 'is-muted'}`}
            type="button"
            onClick={runSidestepInputPrep}
            disabled={!launchFlag('sidestep_inputs_enabled')}
            title={readiness?.ace_full_prerequisites?.reason ?? readiness?.ace_sidestep_inputs?.reason}
          >
            <CloudUpload size={16} /> 12a Prepare Side-Step Inputs
          </button>
          <input
            className="input"
            value={fullConfirmation}
            onChange={(event) => setFullConfirmation(event.target.value)}
            placeholder="Type LAUNCH ACE FULL FINE-TUNE"
            disabled={!launchFlag('full_enabled') || launching}
          />
          <button className={`btn stage-action ${launchFlag('full_enabled') && fullConfirmation.trim() === 'LAUNCH ACE FULL FINE-TUNE' ? 'is-current' : 'is-muted'}`} type="button" onClick={() => submitAceStage(12)} disabled={!launchFlag('full_enabled') || fullConfirmation.trim() !== 'LAUNCH ACE FULL FINE-TUNE'}>
            <Terminal size={16} /> 12 Full Hybrid Stage
          </button>
        </div>
        <div className="paths" style={{ marginTop: 18 }}>
          <span>Source manifest: <span className="mono">{readiness?.data_locations?.azure_datastore_manifest ?? 'pending'}</span></span>
          <span>Preflight output: <span className="mono">{readiness?.data_locations?.azure_ace_preflight_output_root ?? 'pending'}</span></span>
          <span>ACE environment: <span className="mono">{readiness?.ace_preflight?.required_environment ?? 'azureml:env-ace-step:4'}</span></span>
          <span>Planner target: <span className="mono">{readiness?.target_model?.planner_checkpoint ?? 'ACE-Step/acestep-5Hz-lm-0.6B'}</span></span>
        </div>
        {logs.length > 0 && (
          <pre className="log-panel" style={{ marginTop: 18 }}>{logs.join('\n')}</pre>
        )}
      </section>
    </>
  );
};
