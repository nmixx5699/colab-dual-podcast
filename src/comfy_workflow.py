"""API-format graph; dynamic input names verified against ComfyUI v0.43.0."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def model_names():
    entries = json.loads((ROOT / 'configs/models.lock.json').read_text(encoding='utf-8'))['models']
    return {entry['folder']: Path(entry['file']).name for entry in entries}


def build_graph(config, image, audio_a, audio_b, mask_a, mask_b, prefix, previous=(), seed=42):
    if previous and len(previous) != config['motion_frames']:
        raise ValueError('Exactly motion_frames lossless reference PNGs are required')
    m, graph = model_names(), {}

    def add(class_type, **inputs):
        node = str(len(graph) + 1)
        graph[node] = {'class_type': class_type, 'inputs': inputs}
        return [node, 0]

    unet = add('UNETLoader', unet_name=m['diffusion_models'], weight_dtype='default')
    model = add('LoraLoaderModelOnly', model=unet, lora_name=m['loras'], strength_model=1.0)
    clip = add('CLIPLoader', clip_name=m['text_encoders'], type='wan', device='default')
    vae = add('VAELoader', vae_name=m['vae'])
    patch = add('ModelPatchLoader', name=m['model_patches'])
    encoder = add('AudioEncoderLoader', audio_encoder_name=m['audio_encoders'])
    image_node = add('LoadImage', image=image)
    audio_nodes = [add('LoadAudio', audio=a) for a in (audio_a, audio_b)]
    embeddings = [add('AudioEncoderEncode', audio_encoder=encoder, audio=a) for a in audio_nodes]
    masks = [add('LoadImageMask', image=p, channel='red') for p in (mask_a, mask_b)]
    positive = add('CLIPTextEncode', clip=clip, text=config['prompt'])
    negative = add('ConditioningZeroOut', conditioning=positive)
    inputs = {'mode': 'two_speakers', 'model': model, 'model_patch': patch,
              'positive': positive, 'negative': negative, 'vae': vae,
              'width': config['width'], 'height': config['height'], 'length': config['chunk_frames'],
              'start_image': image_node, 'audio_encoder_output_1': embeddings[0],
              'mode.audio_encoder_output_2': embeddings[1],
              'mode.mask_1': masks[0], 'mode.mask_2': masks[1],
              'motion_frame_count': config['motion_frames'], 'audio_scale': 1.0}
    prev = None
    for filename in previous:
        frame = add('LoadImage', image=filename)
        prev = frame if prev is None else add('ImageBatch', image1=prev, image2=frame)
    if prev:
        inputs['previous_frames'] = prev
    talk = add('PodcastInfiniteTalkToVideo', **inputs)
    guider = add('CFGGuider', model=talk, positive=[talk[0], 1], negative=[talk[0], 2], cfg=1.0)
    sigmas = add('BasicScheduler', model=talk, scheduler='normal', steps=config['steps'], denoise=1.0)
    sampler = add('KSamplerSelect', sampler_name='euler')
    noise = add('RandomNoise', noise_seed=seed)
    samples = add('SamplerCustomAdvanced', noise=noise, guider=guider, sampler=sampler,
                  sigmas=sigmas, latent_image=[talk[0], 3])
    decoded = add('VAEDecode', samples=samples, vae=vae)
    tail = add('ImageFromBatch', image=decoded, batch_index=config['chunk_frames'] - config['motion_frames'],
               length=config['motion_frames'])
    save_tail = add('SaveImage', images=tail, filename_prefix=prefix + '/motion')
    video = add('CreateVideo', images=decoded, fps=25.0)
    save = add('SaveVideo', video=video, filename_prefix=prefix + '/raw', format='mp4', **{'format.codec': 'h264'})
    return graph, save[0], save_tail[0]


def expand_inputs(schema, inputs, prefix=''):
    """Expand v3 DynamicCombo into dotted prompt inputs, matching the pinned parser."""
    result = {}
    for kind in ('required', 'optional'):
        for name, spec in schema.get(kind, {}).items():
            key = prefix + name
            result[key] = (kind, spec)
            if spec[0] == 'COMFY_DYNAMICCOMBO_V3':
                options = spec[1]['options']
                for option in options:
                    if option['key'] == inputs.get(key):
                        result.update(expand_inputs(option['inputs'], inputs, key + '.'))
    return result


def validate_graph(graph, object_info):
    """Fail before sampling on missing nodes, wrong dynamic keys, types or enum values."""
    for node_id, node in graph.items():
        name, inputs = node['class_type'], node['inputs']
        if name not in object_info:
            raise ValueError(f'Missing node {name}; rerun setup and restart ComfyUI')
        specs = expand_inputs(object_info[name]['input'], inputs)
        unknown = set(inputs) - set(specs)
        if unknown:
            raise ValueError(f'{name}: unknown input(s) {sorted(unknown)}')
        for key, (kind, spec) in specs.items():
            if key not in inputs:
                if kind == 'required':
                    raise ValueError(f'{name}: missing {key}')
                continue
            value = inputs[key]
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int):
                source, slot = value
                if source not in graph:
                    raise ValueError(f'Dangling connection {source}')
                outputs = object_info[graph[source]['class_type']]['output']
                if slot < 0 or slot >= len(outputs) or outputs[slot] != spec[0]:
                    raise ValueError(f'{name}.{key}: connection type mismatch')
            elif isinstance(spec[0], list) and value not in spec[0]:
                raise ValueError(f'{name}.{key}: unavailable value {value!r}')
            elif spec[0] == 'COMFY_DYNAMICCOMBO_V3':
                if value not in [option['key'] for option in spec[1]['options']]:
                    raise ValueError(f'{name}.{key}: invalid mode')
    return True
