import {
  Database,
  FolderOpen,
  Workflow,
  Wand2,
  Cpu,
  GitBranch,
  FlaskConical,
  BarChart3,
  Cloud,
  BookOpenText,
  Layers3,
  ShieldCheck,
  type LucideIcon,
} from 'lucide-react';

export type ViewId =
  | 'dataset'
  | 'pool-creator'
  | 'pool-viewer'
  | 'finetune-diffusion'
  | 'finetune-context-diffusion'
  | 'finetune-autoregressive'
  | 'finetune-hybrid'
  | 'azure-runs'
  | 'test-prep'
  | 'testing'
  | 'benchmarks'
  | 'documentation';

export interface NavItem {
  id: ViewId;
  label: string;
  group: 'data' | 'finetune' | 'ops' | 'eval' | 'docs';
  groupLabel: string;
  icon: LucideIcon;
  description: string;
}

export const NAV_ITEMS: NavItem[] = [
  {
    id: 'dataset',
    label: 'Dataset',
    group: 'data',
    groupLabel: 'Data',
    icon: Database,
    description: 'Freesound music attribution pool · download orchestration',
  },
  {
    id: 'pool-creator',
    label: 'Pool Creator',
    group: 'data',
    groupLabel: 'Data',
    icon: Workflow,
    description: 'Pre-training CaRA source pool allocation and registry',
  },
  {
    id: 'pool-viewer',
    label: 'Pool Viewer',
    group: 'data',
    groupLabel: 'Data',
    icon: FolderOpen,
    description: 'Browse registered CaRA pools and allocated source files',
  },
  {
    id: 'azure-runs',
    label: 'Azure Runs',
    group: 'ops',
    groupLabel: 'Operations',
    icon: Cloud,
    description: 'Monitor Azure ML jobs, compute, environments, metrics, and logs',
  },
  {
    id: 'test-prep',
    label: 'Test Prep',
    group: 'ops',
    groupLabel: 'Operations',
    icon: ShieldCheck,
    description: 'Run and audit Azure ML environment validation phases',
  },
  {
    id: 'finetune-diffusion',
    label: 'Finetune: Diffusion',
    group: 'finetune',
    groupLabel: 'Fine-tuning',
    icon: Wand2,
    description: 'Configure & monitor diffusion fine-tuning runs',
  },
  {
    id: 'finetune-context-diffusion',
    label: 'Finetune: Context Diffusion',
    group: 'finetune',
    groupLabel: 'Fine-tuning',
    icon: Layers3,
    description: 'Track the context-conditioned diffusion comparison branch',
  },
  {
    id: 'finetune-autoregressive',
    label: 'Finetune: Autoregressive',
    group: 'finetune',
    groupLabel: 'Fine-tuning',
    icon: Cpu,
    description: 'Configure & monitor autoregressive fine-tuning runs',
  },
  {
    id: 'finetune-hybrid',
    label: 'Finetune: Hybrid',
    group: 'finetune',
    groupLabel: 'Fine-tuning',
    icon: GitBranch,
    description: 'Plan ACE-Step v1.5 hybrid LM plus DiT comparison runs',
  },
  {
    id: 'testing',
    label: 'Testing',
    group: 'eval',
    groupLabel: 'Evaluation',
    icon: FlaskConical,
    description: 'Run prompt suites against base & fine-tuned checkpoints',
  },
  {
    id: 'benchmarks',
    label: 'Benchmarks',
    group: 'eval',
    groupLabel: 'Evaluation',
    icon: BarChart3,
    description: 'Compare base vs diffusion vs autoregressive results',
  },
  {
    id: 'documentation',
    label: 'Runbook & Logs',
    group: 'docs',
    groupLabel: 'Documentation',
    icon: BookOpenText,
    description: 'Read the live fine-tuning runbook and experiment log',
  },
];
