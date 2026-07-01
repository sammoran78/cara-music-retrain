import React from 'react';
import { FinetuneView } from './FinetuneView';

export const FinetuneDiffusionPage: React.FC = () => {
  return (
    <FinetuneView
      variant="diffusion"
      kicker="Fine-tuning · Diffusion"
      title={
        <>
          Diffusion <em>fine-tune</em> with CARA attribution head
        </>
      }
      description={
        <>
          Prepare the Stable Audio Open Small lead branch: structured pool conditioning,
          generation-side DiT feature taps, and an attribution head resolved through the
          locked CARA registry.
        </>
      }
      extraFields={[
        {
          key: 'pretransform_ckpt',
          label: 'Pretransform checkpoint',
          type: 'text',
          defaultValue: 'stabilityai/stable-audio-open-small/pretransform.safetensors',
          hint: 'Frozen VAE / latent encoder used to pre-encode audio',
        },
        {
          key: 'noise_schedule',
          label: 'Noise schedule',
          type: 'select',
          options: ['cosine', 'linear', 'edm'],
          defaultValue: 'cosine',
        },
        {
          key: 'ema_decay',
          label: 'EMA decay',
          type: 'text',
          defaultValue: '0.999',
        },
        {
          key: 'attribution_loss_weight',
          label: 'Attribution loss weight',
          type: 'text',
          defaultValue: '0.05',
          hint: 'Smoke default for CARA head / CARA-Strong auxiliary pool-family loss.',
        },
        {
          key: 'use_cara_sidecars',
          label: 'CARA sidecars',
          type: 'select',
          options: ['disabled', 'enabled'],
          defaultValue: 'enabled',
          hint: 'Baseline smoke has passed. Keep enabled for the CARA-lite smoke control run.',
        },
      ]}
    />
  );
};
