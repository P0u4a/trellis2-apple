"""
MLX pipeline factory — creates an upstream Trellis2ImageTo3DPipeline
with MLX-backed model adapters.

The upstream PT pipeline handles all orchestration, sampling, and mesh
extraction. MLX models are injected via thin adapters that convert
torch→mlx→torch at model boundaries.
"""
import os
import gc
import json
import time
import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

from . import load_safetensors, remap_flow_model_weights, remap_vae_decoder_weights
from .flow_models import MlxSparseStructureFlowModel, MlxSLatFlowModel
from .vae_decoders import MlxSparseUnetVaeDecoder, MlxFlexiDualGridVaeDecoder
from .structure_decoder import load_structure_decoder
from .dinov3 import load_dinov3_from_hf
from .adapters import (
    MlxFlowModelAdapter,
    MlxStructureDecoderAdapter,
    MlxFlexiDualGridAdapter,
    MlxTexVaeDecoderAdapter,
    MlxImageCondAdapter,
)

_PIPELINE_MODELS = {
    '512': {
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_decoder',
    },
    '1024': {
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    },
    '1024_cascade': {
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    },
    '1536_cascade': {
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    },
}


class _LazyBiRefNet:
    """Load background removal only if the source image has no alpha.

    BiRefNet is a PyTorch model and otherwise occupies unified memory for the
    entire generation. A transparent input never needs it.
    """

    def __init__(self, model_args: dict):
        self._model_args = model_args
        self._model = None
        self._device = 'cpu'

    def _load(self):
        if self._model is None:
            from trellis2.pipelines.rembg import BiRefNet
            print("[MLX] Loading background remover...")
            try:
                self._model = BiRefNet(**self._model_args)
            except OSError as exc:
                configured = self._model_args.get('model_name', '')
                if configured != 'briaai/RMBG-2.0':
                    raise
                print(
                    "[MLX] BRIA RMBG-2.0 is gated; falling back to "
                    "the public ZhengPeng7/BiRefNet checkpoint."
                )
                self._model = BiRefNet(model_name='ZhengPeng7/BiRefNet')
            self._model.to(self._device)
        return self._model

    def to(self, device):
        self._device = device
        if self._model is not None:
            self._model.to(device)
        return self

    def cpu(self):
        return self.to('cpu')

    def __call__(self, image):
        result = self._load()(image)
        # Background removal is a one-shot preprocessing stage. Releasing its
        # PyTorch weights here returns unified memory before diffusion starts.
        self._model = None
        gc.collect()
        try:
            import torch
            if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
        except (ImportError, RuntimeError):
            pass
        return result


def _resolve_hf_path(rel_path: str) -> str:
    """Resolve 'org/repo/path/to/file' to local HF cache path."""
    from huggingface_hub import hf_hub_download
    parts = rel_path.split('/')
    repo_id = f"{parts[0]}/{parts[1]}"
    file_base = '/'.join(parts[2:])
    json_path = hf_hub_download(repo_id, f"{file_base}.json")
    hf_hub_download(repo_id, f"{file_base}.safetensors")
    return json_path.rsplit('.json', 1)[0]


def _resolve_model_path(weights_path: str, rel_path: str) -> str:
    """Resolve model path — local first, then HF Hub."""
    full = os.path.join(weights_path, rel_path)
    if os.path.exists(f"{full}.json"):
        return full
    return _resolve_hf_path(rel_path)


def _load_mlx_flow_model(path: str, config: dict):
    """Load an MLX flow model from config + safetensors."""
    args = config['args']
    if config['name'] == 'SparseStructureFlowModel':
        model = MlxSparseStructureFlowModel(
            resolution=args['resolution'],
            in_channels=args['in_channels'],
            model_channels=args['model_channels'],
            cond_channels=args['cond_channels'],
            out_channels=args['out_channels'],
            num_blocks=args['num_blocks'],
            num_heads=args.get('num_heads', 12),
            mlp_ratio=args.get('mlp_ratio', 5.3334),
            pe_mode=args.get('pe_mode', 'rope'),
            share_mod=args.get('share_mod', True),
            qk_rms_norm=args.get('qk_rms_norm', True),
            qk_rms_norm_cross=args.get('qk_rms_norm_cross', True),
        )
        is_sparse = False
    elif config['name'] in ('SLatFlowModel', 'ElasticSLatFlowModel'):
        model = MlxSLatFlowModel(
            resolution=args['resolution'],
            in_channels=args['in_channels'],
            model_channels=args['model_channels'],
            cond_channels=args['cond_channels'],
            out_channels=args['out_channels'],
            num_blocks=args['num_blocks'],
            num_heads=args.get('num_heads', 12),
            mlp_ratio=args.get('mlp_ratio', 5.3334),
            pe_mode=args.get('pe_mode', 'rope'),
            share_mod=args.get('share_mod', True),
            qk_rms_norm=args.get('qk_rms_norm', True),
            qk_rms_norm_cross=args.get('qk_rms_norm_cross', True),
        )
        is_sparse = True
    else:
        raise ValueError(f"Unknown flow model type: {config['name']}")

    weights = load_safetensors(f"{path}.safetensors")
    weights = remap_flow_model_weights(weights)
    model.load_weights(list(weights.items()))
    return MlxFlowModelAdapter(model, is_sparse=is_sparse)


def _load_mlx_structure_decoder(path: str, config: dict):
    """Load MLX structure decoder and wrap in adapter."""
    model = load_structure_decoder(path)
    return MlxStructureDecoderAdapter(model)


def _load_mlx_shape_decoder(path: str, config: dict):
    """Load MLX FlexiDualGrid shape decoder and wrap in adapter."""
    args = config['args']
    model = MlxFlexiDualGridVaeDecoder(
        resolution=args['resolution'],
        model_channels=args['model_channels'],
        latent_channels=args['latent_channels'],
        num_blocks=args['num_blocks'],
        block_type=args['block_type'],
        up_block_type=args['up_block_type'],
        block_args=args.get('block_args'),
        use_fp16=args.get('use_fp16', False),
    )
    weights = load_safetensors(f"{path}.safetensors")
    weights = remap_vae_decoder_weights(weights)
    weights = {f'decoder.{k}': v for k, v in weights.items()}
    model.load_weights(list(weights.items()))
    return MlxFlexiDualGridAdapter(model)


def _load_mlx_tex_decoder(path: str, config: dict):
    """Load MLX texture VAE decoder and wrap in adapter."""
    args = config['args']
    model = MlxSparseUnetVaeDecoder(
        out_channels=args['out_channels'],
        model_channels=args['model_channels'],
        latent_channels=args['latent_channels'],
        num_blocks=args['num_blocks'],
        block_type=args['block_type'],
        up_block_type=args['up_block_type'],
        block_args=args.get('block_args'),
        use_fp16=args.get('use_fp16', False),
        pred_subdiv=args.get('pred_subdiv', True),
    )
    weights = load_safetensors(f"{path}.safetensors")
    weights = remap_vae_decoder_weights(weights)
    model.load_weights(list(weights.items()))
    return MlxTexVaeDecoderAdapter(model)


# Map model name patterns to loader functions
_LOADER_MAP = {
    'sparse_structure_flow_model': _load_mlx_flow_model,
    'sparse_structure_decoder': _load_mlx_structure_decoder,
    'shape_slat_decoder': _load_mlx_shape_decoder,
    'tex_slat_decoder': _load_mlx_tex_decoder,
}


def _get_loader(name: str, config: dict):
    """Pick the right loader for a model name."""
    # Exact match first
    if name in _LOADER_MAP:
        return _LOADER_MAP[name]
    # Flow models by name pattern
    if 'flow_model' in name:
        return _load_mlx_flow_model
    # VAE decoders by config name
    if config['name'] == 'FlexiDualGridVaeDecoder':
        return _load_mlx_shape_decoder
    if config['name'] == 'SparseUnetVaeDecoder':
        return _load_mlx_tex_decoder
    raise ValueError(f"No loader for model '{name}' (type: {config['name']})")


def create_mlx_pipeline(
    weights_path: str = "weights/TRELLIS.2-4B",
    pipeline_type: str = "512",
    release_models_after_use: bool = True,
):
    """Create upstream Trellis2ImageTo3DPipeline with MLX-backed models.

    All model compute runs in MLX. The upstream PT pipeline handles
    orchestration, sampling (FlowEulerCfgSampler etc.), and mesh extraction.

    Only models needed by ``pipeline_type`` are loaded. This is essential on
    unified-memory Macs: each unused 1.3B flow model costs roughly 2.6 GB in
    bf16, and MLX modules cannot be offloaded to CPU independently.
    """
    import torch
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import samplers

    if pipeline_type not in _PIPELINE_MODELS:
        valid = ', '.join(_PIPELINE_MODELS)
        raise ValueError(f"Invalid pipeline type '{pipeline_type}'. Choose: {valid}")
    required_models = _PIPELINE_MODELS[pipeline_type]

    print(f"[MLX] Loading pipeline config from {weights_path}...")
    config_file = os.path.join(weights_path, "pipeline.json")
    with open(config_file) as f:
        args = json.load(f)['args']

    # Load only models used by the selected resolution.
    models = {}
    for name, rel_path in args['models'].items():
        if name not in required_models:
            print(f"  [MLX] Skipping '{name}' (unused by {pipeline_type})")
            continue
        path = _resolve_model_path(weights_path, rel_path)
        with open(f"{path}.json") as f:
            model_config = json.load(f)

        t0 = time.time()
        loader = _get_loader(name, model_config)
        models[name] = loader(path, model_config)
        dt = time.time() - t0
        print(f"  [MLX] Loaded '{name}' in {dt:.1f}s")

    # Create upstream pipeline with MLX models
    pipeline = Trellis2ImageTo3DPipeline(models)
    pipeline._pretrained_args = args

    # Set up samplers (upstream PT classes — source of truth for sampling)
    pipeline.sparse_structure_sampler = getattr(
        samplers, args['sparse_structure_sampler']['name']
    )(**args['sparse_structure_sampler']['args'])
    pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

    pipeline.shape_slat_sampler = getattr(
        samplers, args['shape_slat_sampler']['name']
    )(**args['shape_slat_sampler']['args'])
    pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

    pipeline.tex_slat_sampler = getattr(
        samplers, args['tex_slat_sampler']['name']
    )(**args['tex_slat_sampler']['args'])
    pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

    # Normalization
    pipeline.shape_slat_normalization = args['shape_slat_normalization']
    pipeline.tex_slat_normalization = args['tex_slat_normalization']

    # Image conditioning (MLX DINOv3)
    pipeline.image_cond_model = MlxImageCondAdapter(
        load_dinov3_from_hf(args['image_cond_model']['args']['model_name'])
    )

    # Background removal is loaded only when a non-transparent input needs it.
    pipeline.rembg_model = _LazyBiRefNet(args['rembg_model']['args'])

    pipeline.low_vram = True
    pipeline.release_models_after_use = release_models_after_use
    pipeline._device = torch.device('cpu')
    pipeline.default_pipeline_type = pipeline_type
    pipeline.pbr_attr_layout = {
        'base_color': slice(0, 3),
        'metallic': slice(3, 4),
        'roughness': slice(4, 5),
        'alpha': slice(5, 6),
    }

    print("[MLX] Pipeline ready.")
    return pipeline


def to_glb(mesh, output_path: str,
           decimation_target: int = 200000,
           texture_size: int = 1024,
           remesh: bool = False,
           verbose: bool = True) -> str:
    """Export MeshWithVoxel to GLB with a Metal-safe face count."""
    import o_voxel
    import torch

    vertices = mesh.vertices.cpu()
    faces = mesh.faces.cpu()
    target_faces = min(decimation_target, faces.shape[0])

    # mtlbvh can fail on the decoder's raw ~800K-face mesh. Simplifying before
    # BVH construction also substantially reduces peak memory during baking.
    if faces.shape[0] > target_faces:
        import fast_simplification
        verts_np, faces_np = fast_simplification.simplify(
            vertices.numpy(),
            faces.numpy(),
            target_count=target_faces,
            agg=10.0,
        )
        vertices = torch.from_numpy(verts_np).float()
        faces = torch.from_numpy(faces_np.astype('int32'))
        print(f"Simplified mesh to {faces.shape[0]:,} faces before texture baking.")

    print(f"Exporting to {output_path}...")
    glb = o_voxel.postprocess.to_glb(
        vertices=vertices,
        faces=faces,
        attr_volume=mesh.attrs.cpu(),
        coords=mesh.coords.cpu(),
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=target_faces,
        texture_size=texture_size,
        remesh=remesh,
        verbose=verbose,
    )
    glb.export(output_path)
    print(f"Exported: {output_path}")
    return output_path


# Backward-compat alias
class MlxTrellis2Pipeline:
    """Deprecated — use create_mlx_pipeline() instead.

    Thin wrapper that creates the upstream pipeline and delegates .run()/.to_glb().
    """

    def __init__(
        self,
        weights_path: str = "weights/TRELLIS.2-4B",
        pipeline_type: str = "512",
    ):
        self._pipeline = create_mlx_pipeline(weights_path, pipeline_type)

    def run(self, image, **kwargs):
        return self._pipeline.run(image, **kwargs)

    def to_glb(self, mesh, output_path, **kwargs):
        return to_glb(mesh, output_path, **kwargs)
