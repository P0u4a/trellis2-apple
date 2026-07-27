"""
Gradio app for Trellis2 with MLX backend (macOS).
Simplified from upstream app.py — no CUDA render preview, direct GLB export.
"""
import gradio as gr
import os
import time
from datetime import datetime
import shutil
import numpy as np
from PIL import Image
import torch

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')


def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def preprocess_image(image: Image.Image) -> Image.Image:
    return pipeline.preprocess_image(image)


def image_to_3d(
    image: Image.Image,
    seed: int,
    resolution: str,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    shape_slat_guidance_strength: float,
    shape_slat_sampling_steps: int,
    tex_slat_guidance_strength: float,
    tex_slat_sampling_steps: int,
    decimation_target: int,
    texture_size: int,
    req: gr.Request,
    progress=gr.Progress(track_tqdm=True),
):
    global pipeline
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)

    pipeline_type = {"512": "512", "1024": "1024"}[resolution]
    if 'sparse_structure_flow_model' not in pipeline.models:
        from mlx_backend.pipeline import create_mlx_pipeline
        pipeline = create_mlx_pipeline(
            weights_path="weights/TRELLIS.2-4B",
            pipeline_type=pipeline_type,
        )

    t0 = time.time()
    meshes = pipeline.run(
        image,
        seed=seed,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "guidance_strength": ss_guidance_strength,
        },
        shape_slat_sampler_params={
            "steps": shape_slat_sampling_steps,
            "guidance_strength": shape_slat_guidance_strength,
        },
        tex_slat_sampler_params={
            "steps": tex_slat_sampling_steps,
            "guidance_strength": tex_slat_guidance_strength,
        },
        pipeline_type=pipeline_type,
    )
    dt_gen = time.time() - t0
    mesh = meshes[0]

    t0 = time.time()
    from mlx_backend.pipeline import to_glb
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
    to_glb(
        mesh,
        glb_path,
        decimation_target=decimation_target,
        texture_size=texture_size,
    )
    dt_post = time.time() - t0

    info = (f"Generation: {dt_gen:.0f}s | Post-processing: {dt_post:.0f}s | "
            f"Verts: {mesh.vertices.shape[0]:,} | Faces: {mesh.faces.shape[0]:,}")
    return glb_path, glb_path, info


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)


with gr.Blocks(title="Trellis2 MLX") as demo:
    gr.Markdown("""
    ## Trellis2 (MLX Backend)
    Upload an image and click Generate to create a 3D asset.
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=400)

            resolution = gr.Radio(["512", "1024"], label="Resolution", value="512")
            seed = gr.Slider(0, MAX_SEED, label="Seed", value=42, step=1)
            randomize_seed = gr.Checkbox(label="Randomize Seed", value=False)
            decimation_target = gr.Slider(100000, 400000, label="Decimation Target", value=200000, step=10000)
            texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)

            generate_btn = gr.Button("Generate", variant="primary")

            with gr.Accordion(label="Advanced Settings", open=False):
                gr.Markdown("### Stage 1: Sparse Structure")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Steps", value=12, step=1)
                gr.Markdown("### Stage 2: Shape")
                with gr.Row():
                    shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="Guidance", value=7.5, step=0.1)
                    shape_slat_sampling_steps = gr.Slider(1, 50, label="Steps", value=12, step=1)
                gr.Markdown("### Stage 3: Texture")
                with gr.Row():
                    tex_slat_guidance_strength = gr.Slider(0.1, 10.0, label="Guidance", value=1.0, step=0.1)
                    tex_slat_sampling_steps = gr.Slider(1, 50, label="Steps", value=12, step=1)

        with gr.Column(scale=2):
            glb_output = gr.Model3D(label="Generated 3D Model", height=700, clear_color=(0.25, 0.25, 0.25, 1.0))
            download_btn = gr.DownloadButton(label="Download GLB")
            info_text = gr.Textbox(label="Info", interactive=False)

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)

    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        image_to_3d,
        inputs=[
            image_prompt, seed, resolution,
            ss_guidance_strength, ss_sampling_steps,
            shape_slat_guidance_strength, shape_slat_sampling_steps,
            tex_slat_guidance_strength, tex_slat_sampling_steps,
            decimation_target, texture_size,
        ],
        outputs=[glb_output, download_btn, info_text],
    )


if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)

    from mlx_backend.pipeline import create_mlx_pipeline
    pipeline = create_mlx_pipeline(
        weights_path="weights/TRELLIS.2-4B",
        pipeline_type="512",
    )

    demo.launch()
