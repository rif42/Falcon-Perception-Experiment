"""Falcon Perception — multi-class single-pass detection/segmentation.

Detect multiple classes in **one batched forward pass** (one prefill + shared
decode). Same image, N different text queries — all tokenized together and
run through the model once. Far cheaper than looping N times.

This mirrors the reference snippet you provided, but uses the real repo
APIs and works on both CUDA and CPU. On CPU (e.g. Ryzen 8845HS) the
paged engine requires CUDA, so the batch engine is used automatically.

Usage
-----
    # default 4 classes on the bundled sample
    python demo/perception_multi.py --image demo/assets/sample.jpg

    # explicit classes (tyro list style)
    python demo/perception_multi.py --image demo/assets/sample.jpg --queries person --queries car --queries "traffic light" --queries dog

    # comma-separated shorthand
    python demo/perception_multi.py --image demo/assets/sample.jpg --queries "person,car,traffic light,dog"
    python demo/perception_multi.py --image demo/assets/sample.jpg --query "person,car"

    # detection (default, CPU-safe) vs segmentation (needs more RAM, OOMs at 1024 on CPU)
    python demo/perception_multi.py --image demo/assets/sample.jpg --task detection
    python demo/perception_multi.py --image demo/assets/sample.jpg --task segmentation --no-compile

    # force paged engine on CUDA
    python demo/perception_multi.py --image demo/assets/sample.jpg --engine-type paged --compile

    # custom output dir
    python demo/perception_multi.py --image demo/assets/sample.jpg --out-dir ./outputs/multi
"""

from pathlib import Path
from typing import Literal

import torch
import tyro

from falcon_perception import (
    PERCEPTION_MODEL_ID,
    build_prompt_for_task,
    cuda_timed,
    load_and_prepare_model,
    setup_torch_config,
)
from falcon_perception.data import load_image, stream_samples_from_hf_dataset
from falcon_perception.nvtx import nvtx_range

setup_torch_config()


@torch.inference_mode()
def main(
    image: str | None = None,
    queries: list[str] | None = None,
    query: str | None = None,
    task: Literal["segmentation", "detection"] = "detection",
    hf_model_id: str | None = None,
    hf_revision: str = "main",
    hf_local_dir: str | None = None,
    device: str | None = None,
    dtype: Literal["bfloat16", "float32", "float"] = "float32",
    engine_type: Literal["batch", "paged"] = "batch",
    flex_attn_safe: bool = False,
    out_dir: str = "./outputs/",
    compile: bool = False,
    cudagraph: bool = False,
):
    """Run Falcon Perception on one image with multiple class queries in one pass.

    Each query becomes its own sequence/prompt. Batch engine packs all
    (image, prompt) pairs into one batch; paged engine creates one Sequence
    per class. Either way it's a single engine.generate() call.

    `--queries` accepts repeated flags or a single comma-separated string:
      --queries person --queries car        -> ["person", "car"]
      --queries "person,car,traffic light" -> ["person", "car", "traffic light"]
    `--query` is a backwards-compat alias for the same.
    """
    # --- normalize queries ---
    raw: list[str] = []
    if queries is not None and len(queries) > 0:
        # expand comma-separated entries (tyro passes ["person,car"] as one element)
        for q in queries:
            if "," in q:
                raw.extend([s.strip() for s in q.split(",") if s.strip()])
            elif q.strip():
                raw.append(q.strip())
    elif query is not None and query.strip():
        raw = [s.strip() for s in query.split(",") if s.strip()]
    else:
        raw = ["person", "car", "traffic light", "dog"]

    # remove empty / dedup preserve order
    seen: set[str] = set()
    classes: list[str] = []
    for q in raw:
        if q and q not in seen:
            seen.add(q)
            classes.append(q)

    if not classes:
        print("No queries provided.")
        return

    kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1} if flex_attn_safe else {}

    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=hf_model_id or PERCEPTION_MODEL_ID,
        hf_revision=hf_revision,
        hf_local_dir=hf_local_dir,
        device=device,
        dtype=dtype,
        compile=compile,
    )
    resolved_device = model.device

    if task == "segmentation" and not model_args.do_segmentation:
        print("Model does not support segmentation (do_segmentation=False), falling back to detection.")
        task = "detection"

    if image is not None:
        pil_image = load_image(image).convert("RGB")
    else:
        print("No --image provided, loading a demo sample ...")
        sample = stream_samples_from_hf_dataset("tiiuae/PBench", split="level_1")[0]
        pil_image = sample["image"]
        sample_query = sample.get("expression") or sample.get("expressions") or "all objects"
        if isinstance(sample_query, list):
            sample_query = ", ".join(str(q) for q in sample_query) if sample_query else "all objects"
        print(f"  Sample query: {sample_query!r}")
        # if no explicit classes, use sample query as single class
        if queries is None and query is None:
            classes = [str(sample_query)]

    w, h = pil_image.size
    print(f"  Task    : {task}")
    print(f"  Classes : {classes}")
    print(f"  Image   : {w} x {h}  ({w*h/1e6:.1f} MP)")
    print(f"  Engine  : {engine_type}  device={resolved_device} dtype={dtype} compile={compile}")
    print()

    from falcon_perception.data import ImageProcessor

    image_processor = ImageProcessor(patch_size=16, merge_size=1)
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.end_of_query_token_id]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "perception_input.jpg").parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(out_path / "perception_input.jpg")

    # --- CPU fallback: paged engine requires CUDA ---
    if engine_type == "paged" and not torch.cuda.is_available():
        print("[warn] --engine-type paged requires CUDA; falling back to batch on CPU.")
        engine_type = "batch"

    if engine_type == "paged":
        # Reference snippet path, corrected to real API
        from falcon_perception.paged_inference import PagedInferenceEngine, SamplingParams, Sequence
        from falcon_perception.visualization_utils import render_paged_inference_outputs

        # Use batch size = num classes (paged engine batches Sequences)
        engine = PagedInferenceEngine(
            model, tokenizer, image_processor,
            max_batch_size=max(2, len(classes)),
            max_seq_length=8192,
            n_pages=128,
            page_size=128,
            prefill_length_limit=8192,
            enable_hr_cache=False,
            capture_cudagraph=cudagraph and torch.cuda.is_available(),
            kernel_options=kernel_options or None,
        )

        sampling_params = SamplingParams(stop_token_ids=stop_token_ids)

        sequences = [
            Sequence(
                text=build_prompt_for_task(cls_name, task),
                image=pil_image,
                min_image_size=256,
                max_image_size=1024,
                task=task,
                request_idx=i,
            )
            for i, cls_name in enumerate(classes)
        ]

        # Single generate() for all classes
        print(f"Running paged inference for {len(classes)} classes in one pass ...")
        with nvtx_range("Generate"):
            with cuda_timed() as t:
                engine.generate(sequences, sampling_params=sampling_params, use_tqdm=True, print_stats=True)
        print(f"Done in {t.elapsed:.1f}s")

        from falcon_perception.visualization_utils import pair_bbox_entries
        print(f"\n{'='*60}")
        print("Results (per class)")
        print("="*60)
        for cls_name, seq in zip(classes, sequences):
            aux = seq.output_aux
            n = 0
            if aux is not None:
                if task == "segmentation":
                    n = len(aux.masks_rle) if hasattr(aux, "masks_rle") else 0
                else:
                    n = len(pair_bbox_entries(aux.bboxes_raw)) if hasattr(aux, "bboxes_raw") else 0
            print(f"  {cls_name:20s} : {n} {'masks' if task=='segmentation' else 'boxes'}")

        render_paged_inference_outputs(sequences, image_processor, output_dir=out_dir, task=task)
        sub = "masks" if task == "segmentation" else "boxes"
        print(f"\n  Input image : {out_path / 'perception_input.jpg'}")
        print(f"  Output dir  : {out_path / sub}")

    else:  # batch
        from falcon_perception.batch_inference import BatchInferenceEngine, process_batch_and_generate
        from falcon_perception.visualization_utils import render_batch_inference_outputs

        prompts = [build_prompt_for_task(q, task) for q in classes]

        engine = BatchInferenceEngine(model, tokenizer, kernel_options=kernel_options or None)

        # One call packs all (image, prompt) pairs into a single batch
        batch_inputs = process_batch_and_generate(
            tokenizer,
            [(pil_image, p) for p in prompts],
            max_length=4096,
            min_dimension=256,
            max_dimension=1024,
        )
        batch_inputs = {
            k: (v.to(resolved_device) if torch.is_tensor(v) else v)
            for k, v in batch_inputs.items()
        }
        print(f"Batch tokens: {batch_inputs['tokens'].shape}  pixel_values: {batch_inputs['pixel_values'].shape}")
        print(f"Running batch inference for {len(classes)} classes in one pass ...")

        with cuda_timed() as t:
            _, aux_out = engine.generate(
                **batch_inputs,
                max_new_tokens=2048,
                temperature=0.0,
                stop_token_ids=stop_token_ids,
                seed=42,
                task=task,
            )
        print(f"Done in {t.elapsed:.1f}s")

        from falcon_perception.aux_output import AuxOutput
        from falcon_perception.visualization_utils import pair_bbox_entries

        print(f"\n{'='*60}")
        print("Results (per class)")
        print("="*60)
        for cls_name, aux in zip(classes, aux_out):
            if isinstance(aux, AuxOutput):
                if task == "segmentation":
                    # detection+seg share bboxes_raw length; masks are in auxiliary
                    n = len(pair_bbox_entries(aux.bboxes_raw))
                    kind = "masks"
                else:
                    n = len(pair_bbox_entries(aux.bboxes_raw))
                    kind = "boxes"
                print(f"  {cls_name:20s} : {n} {kind}")
            else:
                # fallback for raw list form
                n = len(aux) // (3 if task == "segmentation" else 2) if aux else 0
                print(f"  {cls_name:20s} : {n} {'masks' if task=='segmentation' else 'boxes'}")

        batch_inputs["__orig_images__"] = [pil_image] * len(classes)
        render_batch_inference_outputs(
            "BATCH", batch_inputs, aux_out, [], task, out_dir=out_dir, queries=classes,
        )

        print(f"\n  Input image : {out_path / 'perception_input.jpg'}")
        print(f"  Output dir  : {out_path / 'masks'}")


if __name__ == "__main__":
    tyro.cli(main)
