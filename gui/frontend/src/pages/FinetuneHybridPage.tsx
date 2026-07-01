import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Brain,
  CheckCircle2,
  CloudUpload,
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
    label: 'Full hybrid comparison',
    state: 'blocked',
    evidence: 'Held-out planner, DiT attribution, registry decoding, and baseline-vs-CARA comparison report.',
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
  ace_preflight?: HybridLadderStep | null;
  ace_ladder?: {
    steps: HybridLadderStep[];
    next_stage: number;
    next_label: string;
    reason?: string;
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
  const canLaunchPreflight = Boolean(readiness?.training_launch_enabled) && !launching && confirmation.trim() === 'LAUNCH ACE PREFLIGHT';

  const runAcePreflight = async () => {
    if (!canLaunchPreflight) return;
    setLaunching(true);
    setError(null);
    setLogs((prev) => [...prev.slice(-7), '[launch] Submitting ACE-Step v1.5 environment preflight...']);
    try {
      const res = await fetch('/api/training/ace-preflight', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          checkpoint: 'ACE-Step/Ace-Step1.5',
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
            Research scaffold for testing whether CARA attribution survives ACE-Step v1.5's LM planner
            and remains recoverable at the DiT synthesis stage.
          </>
        }
        actions={<PlaceholderBadge label="Research scaffold" />}
      />

      <section className="kpi-grid">
        <div className="kpi">
          <div className="kpi-label">Model family</div>
          <div className="kpi-value">LM + DiT</div>
          <div className="kpi-trend">planner bottleneck plus diffusion synthesis</div>
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
            <div className="card-title">Benchmark Testing Ladder</div>
            <div className="card-meta">{loading ? 'refreshing' : `next · ${readiness?.ace_ladder?.next_label ?? 'ACE preflight'}`}</div>
          </div>
          <div className="stage-action-list">
            {steps.map((item) => (
              <button
                key={item.stage}
                type="button"
                className={`btn stage-action ${item.active ? 'is-current' : item.passed ? 'is-complete' : 'is-muted'}`}
                disabled
                title={item.reason ?? ''}
              >
                {item.passed ? <CheckCircle2 size={16} /> : item.active ? <RefreshCw size={16} /> : <Lock size={16} />}
                {String(item.stage).padStart(2, '0')} {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div className="card-title">How Likely To Work</div>
            <div className="card-meta">ACE-Step v1.5</div>
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
              Hybrid branch: test planner survival first, then recover CARA at the DiT synthesis stage.
            </p>
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
              <strong>ACE-Step/Ace-Step1.5 or official model zoo variant</strong>
            </div>
            <div>
              <span>Training toolkit</span>
              <strong>Side-Step corrected mode, LoRA first</strong>
            </div>
            <div>
              <span>Preprocess output</span>
              <strong>ACE tensors plus CARA registry resolver</strong>
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
          <div className="card-title">Azure Launch Policy</div>
          <div className="card-meta">cost guardrail</div>
        </div>
        <div className="pool-empty-state">
          <ShieldCheck size={18} />
          {readiness?.training_launch_reason ?? 'Hybrid readiness has not been loaded yet.'}
        </div>
        {error && (
          <div className="pool-empty-state" style={{ marginTop: 12 }}>
            <AlertTriangle size={18} />
            {error}
          </div>
        )}
        <div className="hybrid-launch-locks">
          <button className="btn stage-action" type="button" onClick={refreshReadiness} disabled={loading}>
            <RefreshCw size={16} /> Refresh Hybrid Gates
          </button>
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
            disabled={preflightActive || launching}
          />
          <button className={`btn stage-action ${canLaunchPreflight ? 'is-current' : 'is-muted'}`} type="button" onClick={runAcePreflight} disabled={!canLaunchPreflight}>
            <CloudUpload size={16} /> 02 Run ACE-Step Preflight
          </button>
          <button className="btn stage-action is-muted" type="button" disabled>
            <Route size={16} /> Planner Survival Probe Locked
          </button>
          <button className="btn stage-action is-muted" type="button" disabled>
            <SlidersHorizontal size={16} /> Planner-Bypass Probe Locked
          </button>
          <button className="btn stage-action is-muted" type="button" disabled>
            <Play size={16} /> Hybrid Smoke Launch Locked
          </button>
        </div>
        <div className="paths" style={{ marginTop: 18 }}>
          <span>Source manifest: <span className="mono">{readiness?.data_locations?.azure_datastore_manifest ?? 'pending'}</span></span>
          <span>Preflight output: <span className="mono">{readiness?.data_locations?.azure_ace_preflight_output_root ?? 'pending'}</span></span>
          <span>ACE environment: <span className="mono">{readiness?.ace_preflight?.required_environment ?? 'azureml:env-ace-step:1'}</span></span>
        </div>
        {logs.length > 0 && (
          <pre className="log-panel" style={{ marginTop: 18 }}>{logs.join('\n')}</pre>
        )}
      </section>
    </>
  );
};
