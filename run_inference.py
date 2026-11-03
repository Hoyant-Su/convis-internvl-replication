import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, GenerationConfig, InternVLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--smoke-image', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    design = json.loads((root / 'data/design.json').read_text())
    config = json.loads((root / 'study.json').read_text())
    torch.set_num_threads(config['cpu_threads_per_worker'])
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = InternVLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True, device_map={'': 'cuda:0'},
    ).eval()
    generation = GenerationConfig(
        do_sample=True, temperature=design['temperature'], top_p=1.0, top_k=0,
        max_new_tokens=design['max_tokens'],
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    selected = set(design['images'][args.shard::args.shards])
    schedule = [r for r in design['schedule'] if r['image'] in selected]
    if args.smoke_image:
        schedule = [dict(schedule[0], image=args.smoke_image.name, run_id='smoke')]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = []
    if args.output.exists():
        completed = [json.loads(line)['run_id'] for line in args.output.read_text().splitlines()]
    assert len(completed) == len(set(completed))
    metadata = {
        'model': config['model'], 'python': platform.python_version(),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'gpu': torch.cuda.get_device_name(), 'dtype': 'bfloat16',
        'attention': 'sdpa',
        'generation': {k: getattr(generation, k) for k in (
            'do_sample', 'temperature', 'top_p', 'top_k', 'max_new_tokens',
            'eos_token_id', 'pad_token_id')},
        'image_processor': {k: v for k, v in processor.image_processor.to_dict().items()
                            if k in ('size', 'min_patches', 'max_patches', 'crop_to_patches',
                                     'image_mean', 'image_std', 'do_resize', 'image_processor_type')},
        'shard': args.shard, 'shards': args.shards,
        'cpu_threads': torch.get_num_threads(),
        'started_at': datetime.now(timezone.utc).isoformat(),
    }
    print(f'Loaded model; {len(schedule) - len(completed)} requests remaining', flush=True)
    started = time.perf_counter()
    with args.output.open('a', buffering=1) as stream:
        for row in schedule:
            if row['run_id'] in completed:
                continue
            image = args.smoke_image or root / 'data/images' / row['image']
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'path': str(image)},
                {'type': 'text', 'text': design['prompts'][row['prompt_id']]},
            ]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors='pt',
            ).to('cuda:0', dtype=torch.bfloat16)
            torch.manual_seed(row['seed'])
            torch.cuda.manual_seed_all(row['seed'])
            torch.cuda.synchronize()
            before = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, generation_config=generation)
            torch.cuda.synchronize()
            tokens = output[0, inputs['input_ids'].shape[1]:]
            result = dict(
                row, model=config['model'],
                raw_output=processor.decode(tokens, skip_special_tokens=True),
                output_tokens=len(tokens),
                finish_reason='stop' if int(tokens[-1]) == generation.eos_token_id else 'length',
                elapsed_seconds=time.perf_counter() - before,
            )
            stream.write(json.dumps(result) + '\n')
            completed.append(row['run_id'])
            if len(completed) % 16 == 0 or args.smoke_image:
                print(f'{len(completed)}/{len(schedule)} requests recorded', flush=True)
    metadata.update(
        finished_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter() - started,
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
