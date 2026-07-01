import React from 'react';
import { FinetuneView } from './FinetuneView';

export const FinetuneAutoregressivePage: React.FC = () => {
  return (
    <FinetuneView
      variant="autoregressive"
      kicker="Fine-tuning · Autoregressive"
      title={
        <>
          Autoregressive <em>fine-tune</em> with CARA suffix tokens
        </>
      }
      description={
        <>
          Prepare the matched real MusicGen LM branch: same-data baseline, prompt-only
          CARA-lite, detached LM-hidden-state probe, and non-detached CARA-Strong
          suffix prediction resolved through the locked CARA registry.
        </>
      }
      extraFields={[
        {
          key: 'tokenizer_ckpt',
          label: 'Audio tokenizer checkpoint',
          type: 'text',
          defaultValue: 'facebook/musicgen-small',
          hint: 'MusicGen / AudioCraft checkpoint loaded for real LM fine-tuning',
        },
        {
          key: 'sequence_length',
          label: 'Sequence length',
          type: 'number',
          defaultValue: 2048,
        },
        {
          key: 'codebook_size',
          label: 'Codebook size',
          type: 'number',
          defaultValue: 1024,
        },
        {
          key: 'attribution_loss_weight',
          label: 'Attribution loss weight',
          type: 'text',
          defaultValue: '0.25',
          hint: 'Weight of CARA attribution head loss vs LM loss',
        },
        {
          key: 'constrained_decoding',
          label: 'Constrained decoding',
          type: 'select',
          options: ['enabled', 'disabled'],
          defaultValue: 'enabled',
          hint: 'Enforce four-state validation/repair during eval',
        },
      ]}
    />
  );
};
