"""Lossless in-memory transport for native Human single-view caption inputs.

The pinned upstream planner, sample validator, prompt, processor, and generator
remain authoritative. This removes only PNG serialization and readback. Native
TempSeg JPEG transport is deliberately unchanged. One optional worker prepares
the next CPU request while the caller runs the current request on the GPU.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from threading import RLock
from unittest.mock import patch

from .native_profile import NativeProfile

DEFAULT_PREPARED_BYTES = 1024 ** 3


@dataclass(frozen=True)
class PreparedCaption:
    inputs: dict
    text: str
    tensor_bytes: int


@contextmanager
def serialized_processor_access(processor):
    """Serialize tokenizer reads with processor calls that can change its state.

    The pinned native generator calls tokenizer.decode and reads EOS/PAD IDs
    through convert_tokens_to_ids. Processor.batch_decode is guarded too, and
    can re-enter tokenizer.decode through the same RLock.
    Model generation, frame decode, and RGB resize do not hold this lock.
    """
    lock = RLock()
    def guarded(method):
        def call(*args, **kwargs):
            with lock:
                return method(*args, **kwargs)
        return call
    with ExitStack() as hooks:
        for owner, name in ((processor.tokenizer, 'decode'),
                            (processor.tokenizer, 'convert_tokens_to_ids'),
                            (processor, 'batch_decode')):
            hooks.enter_context(patch.object(owner, name, side_effect=guarded(getattr(owner, name))))
        yield lock


def direct_processor_inputs(pipeline, plan, images, processor, instruction='', *,
                            profile=None, identity=None, processor_lock=None):
    """Use exactly the pixels produced by native PNG decode plus native resize.

Memory references are not claimed to be PNG files or PNG hash receipts. The
unchanged sample validator still checks all temporal roles, indices and clocks.
"""
    from flora_unified.unified_input import prompt, validate
    from transformers.video_utils import VideoMetadata
    profile = profile or NativeProfile()
    identity = identity or {}
    receipts = {}
    for index, image in images.items():
        if image.mode != 'RGB':
            raise ValueError('Direct native transport requires decoded RGB images')
        receipts[index] = dict(path=f'memory-rgb:{index}', size=list(image.size),
                               transport='decoded_native_rgb')
    sample = pipeline.a.caption_sample(plan, receipts, ['main'])
    validate(sample)
    with profile.span('caption_rgb_copy_resize', **identity):
        videos = []
        for block, descriptor in zip(sample['blocks'], plan['blocks'], strict=True):
            frames = []
            for index, receipt in zip(descriptor['frame_indices'], block['frames'], strict=True):
                frame = images[index].convert('RGB')
                if list(frame.size) != receipt['size']:
                    raise ValueError('Direct native frame dimensions changed')
                area = frame.width * frame.height
                if area > 262144:
                    scale = math.sqrt(262144 / area)
                    # Keep Pillow's exact native RGB resize defaults and rounding.
                    frame = frame.resize((int(frame.width * scale), int(frame.height * scale)))
                frames.append(frame)
            videos.append(frames)
    metadata = [VideoMetadata(total_num_frames=len(frames),
        fps=float(Fraction(block['clock_fps_num'], block['clock_fps_den'])),
        frames_indices=list(block['block_local_timestamp_indices']))
        for frames, block in zip(videos, sample['blocks'], strict=True)]
    with processor_lock if processor_lock is not None else nullcontext():
        with profile.span('caption_prompt', **identity):
            content = prompt(sample, processor.video_token) + pipeline.instruction_suffix(instruction)
            text = processor.apply_chat_template([dict(role='user', content=content)],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
        with profile.span('caption_processor', **identity):
            inputs = processor(text=[text], videos=videos, video_metadata=metadata,
                do_sample_frames=False, return_metadata=True, return_tensors='pt')
    if any(getattr(getattr(value, 'device', None), 'type', 'cpu') != 'cpu' for value in inputs.values()):
        raise RuntimeError('Native preparation worker must produce CPU tensors only')
    size = sum(value.numel() * value.element_size() for value in inputs.values()
               if hasattr(value, 'numel') and hasattr(value, 'element_size'))
    return PreparedCaption(dict(inputs), text, size)


def caption_single_direct(backend, prepared, segments, *, prefetch=1,
                          max_prepared_bytes=DEFAULT_PREPARED_BYTES):
    """Native caption semantics, with at most one pending CPU preparation.

The byte limit bounds retained source RGB plus final tensor payload per request,
not PyTorch/processor temporary allocations or total host RAM. The active and
one ready request can coexist with the independently bounded decoded cache.
"""
    if prepared.profile not in ('Human', 'Human actions') or set(prepared.videos) != {'main'}:
        raise ValueError('Direct native preparation currently requires Human single-view input')
    if type(prefetch) is not int or prefetch not in (0, 1):
        raise ValueError('Native preparation permits zero or one pending request')
    if type(max_prepared_bytes) is not int or max_prepared_bytes <= 0:
        raise ValueError('Native preparation requires a positive retained-payload byte limit')
    pipeline = backend.pipeline
    profiler = getattr(backend, 'native_profile', NativeProfile())
    episode_id = prepared.job.get('id', prepared.job.get('episode_id', 'unknown'))
    identity = dict(episode_id=episode_id, stage='caption')
    with profiler.span('caption_planning', **identity):
        plans, dispositions = pipeline.single_plans(prepared.videos, segments, prepared.profile)
    if not plans:
        return ([dict(d, caption='', status=d['status']) for d in dispositions], [])
    model, processor = backend.session.load('captioner')
    rows, audit = {}, []

    def prepare_one(index):
        plan = plans[index]
        request_identity = dict(identity, request_ordinal=index, event_id=plan['event_id'])
        with profiler.span('caption_prepare_total', **request_identity):
            wanted = {i for block in plan['blocks'] for i in block['frame_indices']}
            width, height = prepared.videos['main'].size
            expected_source_bytes = len(wanted) * width * height * 3
            if expected_source_bytes > max_prepared_bytes:
                raise ValueError('Native source RGB payload exceeds the bounded direct preparation limit')
            with profiler.span('caption_decode_and_cache', frames=len(wanted), **request_identity):
                images = pipeline.frames_for(prepared.videos, wanted)
            result = direct_processor_inputs(pipeline, plan, images, processor,
                prepared.job.get('instruction', ''), profile=profiler, identity=request_identity,
                processor_lock=processor_lock)
            source_bytes = sum(image.width * image.height * len(image.getbands()) for image in images.values())
            if source_bytes + result.tensor_bytes > max_prepared_bytes:
                raise ValueError('Native RGB and tensor payload exceed the bounded direct preparation limit')
            return result

    def consume(index, item):
        plan = plans[index]
        backend.progress(0.55 + 0.4 * index / len(plans),
                         desc=f'Writing captions · {index + 1}/{len(plans)}')
        with profiler.span('caption_native_generate_total', request_ordinal=index,
                           event_id=plan['event_id'], **identity):
            caption = pipeline.generate(model, processor, item.inputs, 512)
        rows[plan['event_id']] = dict(start_s=plan['start_s'], end_s=plan['end_s'],
                                    caption=caption, status='complete')
        audit.append(dict(event_id=plan['event_id'], blocks=plan['blocks'],
                          prompt_sha256=hashlib.sha256(item.text.encode()).hexdigest()))

    with serialized_processor_access(processor) as processor_lock:
        if prefetch:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='bir-caption-prepare') as executor:
                future = executor.submit(prepare_one, 0)
                for index in range(len(plans)):
                    item = future.result()
                    future = executor.submit(prepare_one, index + 1) if index + 1 < len(plans) else None
                    consume(index, item)
        else:
            for index in range(len(plans)):
                consume(index, prepare_one(index))
    return ([rows.get(d['event_id'], dict(start_s=d['start_s'], end_s=d['end_s'],
             caption='', status=d['status'])) for d in dispositions], audit)
