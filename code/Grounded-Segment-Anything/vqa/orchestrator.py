"""Simple orchestration for VQA using GroundingDINO + SAM + local VLM + Groq.

This is an orchestration layer only — it calls into `gsam` for
segmentation utilities, `local_vlm` for local visual-language
capabilities and `groq_client` for LLM-style text generation.

Usage (programmatic):
  from vqa.orchestrator import Orchestrator
  orch = Orchestrator(...)
  answer = orch.run(image_path, question)

There's also a minimal CLI at the bottom.
"""
import os
from typing import Optional, List
import numpy as np
from PIL import Image
import cv2

import groq_client
from local_vlm import LocalVLM
import gsam


class Orchestrator:
    def __init__(
        self,
        grounding_config: str,
        grounding_checkpoint: str,
        sam_encoder: str,
        sam_checkpoint: str,
        device: Optional[str] = None,
    ):
        self.grounding_config = grounding_config
        self.grounding_checkpoint = grounding_checkpoint
        self.sam_encoder = sam_encoder
        self.sam_checkpoint = sam_checkpoint
        self.device = device

        # load models lazily
        self._grounding = None
        self._sam_predictor = None
        self.vlm = None

    def _ensure_models(self):
        if self._grounding is None:
            self._grounding = gsam.load_grounding_model(self.grounding_config, self.grounding_checkpoint)
        if self._sam_predictor is None:
            self._sam_predictor = gsam.load_sam_predictor(self.sam_encoder, self.sam_checkpoint)
        if self.vlm is None:
            try:
                self.vlm = LocalVLM(device=self.device)
            except Exception:
                self.vlm = None

    def run(
        self,
        image_path: str,
        question: str,
        max_candidates: int = 8,
        classes: Optional[List[str]] = None,
        score_threshold: float = 0.35,
    ) -> str:
        """Run the orchestration: produce a short answer string.

        Steps implemented:
        1. Caption the image with local VLM (if available).
        2. Use Groq to produce a short grounding query from caption.
        3. Run GroundingDINO to get boxes & phrases.
        4. Ask Groq which phrase is most relevant to the question (index).
        5. Run SAM to create a mask & crop for that box.
        6. Ask local VLM the question on the crop to get a local answer.
        7. Optionally call Groq to synthesize final answer from caption/context.
        """
        self._ensure_models()

        # load image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 1) caption
        caption = None
        if self.vlm is not None:
            try:
                caption = self.vlm.caption_image(Image.fromarray(img_rgb))
            except Exception:
                caption = None
        caption = caption or "an image"

        # 2) if user provided explicit classes, use them; otherwise ask Groq to produce a grounding query
        if classes is not None and len(classes) > 0:
            # allow classes to be provided as list of strings
            candidates = [c.strip() for c in classes if c.strip()][:max_candidates]
            print("[DEBUG] Using user-supplied classes as candidates:", candidates, flush=True)
            if len(candidates) == 0:
                candidates = [caption]
        else:
            prompt = (
                f"You are a helpful assistant that extracts concise object phrases from an image caption.\n"
                f"Caption: {caption}\n"
                "Return a short list of object phrases separated by the pipe character ' | ', no explanation."
            )
            # show prompt sent to Groq for debugging
            print("[DEBUG] Groq grounding-query prompt:\n", prompt, flush=True)
            try:
                query = groq_client.generate(prompt)
                print("[DEBUG] Groq grounding-query reply:\n", query, flush=True)
            except Exception as e:
                # fallback: use the caption tokens as single phrase
                print(f"[DEBUG] Groq grounding-query failed: {e}", flush=True)
                query = caption

            # normalize into a list
            candidates = [p.strip() for p in query.split("|") if p.strip()][:max_candidates]
            print("[DEBUG] Candidates passed to gsam.predict_with_classes:", candidates, flush=True)
            if len(candidates) == 0:
                candidates = [caption]

        # 3) run grounding
        grounding_out = gsam.predict_with_classes(self._grounding, img_bgr, candidates)
        detections = gsam.detections_to_list(grounding_out, img_bgr)
        # print grounding output for debugging
        try:
            print("[DEBUG] gsam grounding output (raw):", grounding_out, flush=True)
            print("[DEBUG] gsam detections (list):", detections, flush=True)
        except Exception:
            pass
        if len(detections) == 0:
            # no detection -> ask VLM directly
            if self.vlm is not None:
                return self.vlm.answer_question(Image.fromarray(img_rgb), question)
            return "I couldn't find objects to answer the question."

        # collect all detections above the score_threshold
        selected_indices = [i for i, d in enumerate(detections) if d.get("score", 0.0) >= score_threshold]
        if len(selected_indices) == 0:
            # if no detection passes threshold, fall back to best detection
            selected_indices = [0]
        print(f"[DEBUG] Selected detection indices (score >= {score_threshold}):", selected_indices, flush=True)

        # For each selected detection: segment with SAM, create crop, save crop/mask and ask local VLM
        all_local_observations = []
        saved_items = []
        for idx in selected_indices:
            d = detections[idx]
            box = np.array(d["box"])  # xyxy
            masks = gsam.segment_with_sam(self._sam_predictor, img_rgb, np.expand_dims(box, axis=0))
            mask = masks[0] if masks.shape[0] > 0 else np.zeros(img_rgb.shape[:2], dtype=bool)
            crop_bgr = gsam.crop_by_mask(img_bgr, mask)

            # save crop and mask
            try:
                os.makedirs("vqa_outputs", exist_ok=True)
                crop_path = os.path.join("vqa_outputs", f"selected_crop_{idx}.jpg")
                mask_path = os.path.join("vqa_outputs", f"selected_mask_{idx}.png")
                cv2.imwrite(crop_path, crop_bgr)
                mask_vis = (mask.astype("uint8") * 255)
                cv2.imwrite(mask_path, mask_vis)
            except Exception:
                crop_path = None
                mask_path = None

            # ask local VLM about this crop (if available)
            local_answer = None
            if self.vlm is not None and crop_bgr.size != 0:
                try:
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    local_answer = self.vlm.answer_question(Image.fromarray(crop_rgb), question)
                except Exception:
                    local_answer = None

            saved_items.append({
                "index": idx,
                "box": d["box"],
                "score": d.get("score", 0.0),
                "phrase": d.get("phrase", ""),
                "crop_path": crop_path,
                "mask_path": mask_path,
                "local_answer": local_answer,
            })
            all_local_observations.append((idx, d.get("phrase", ""), d.get("score", 0.0), local_answer))

        print("[DEBUG] All saved items:", saved_items, flush=True)

        # 4) synthesize final answer using Groq with full context (include all selected detections)
        # Build a summary of selected detections and local observations
        detections_summary_lines = []
        for item in saved_items:
            detections_summary_lines.append(
                f"Index {item['index']}: phrase='{item['phrase']}', score={item['score']}, local_obs='{item['local_answer']}', crop={item['crop_path']}, mask={item['mask_path']}"
            )

        detections_summary = "\n".join(detections_summary_lines)
        synthesis_prompt = (
            f"You are a concise assistant.\nCaption: {caption}\n"
            f"Detections (selected):\n{detections_summary}\n"
            f"Question: {question}\n"
            "Provide a one-sentence answer using available context (if unsure, say 'I don't know')."
        )
        # show synthesis prompt and Groq reply
        print("[DEBUG] Groq synthesis prompt:\n", synthesis_prompt, flush=True)
        try:
            final = groq_client.generate(synthesis_prompt)
            print("[DEBUG] Groq synthesis reply:\n", final, flush=True)
        except Exception as e:
            print(f"[DEBUG] Groq synthesis call failed: {e}", flush=True)
            # if Groq unavailable, call local VLM on the full image with context
            if self.vlm is not None:
                # build a prompt that includes detection count and caption
                vlm_prompt = (
                    f"Context: {caption}. "
                    f"Detected {len(saved_items)} object(s) in the image. "
                    f"Question: {question}"
                )
                print(f"[DEBUG] Calling local VLM on full image with prompt: {vlm_prompt}", flush=True)
                try:
                    final = self.vlm.answer_question(Image.fromarray(img_rgb), vlm_prompt)
                    print(f"[DEBUG] Local VLM final answer: {final}", flush=True)
                except Exception as vlm_exc:
                    print(f"[DEBUG] Local VLM call failed: {vlm_exc}", flush=True)
                    final = f"Detected {len(saved_items)} object(s) but unable to answer."
            else:
                final = f"Detected {len(saved_items)} object(s) but no VLM available to answer."

        return final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("vqa orchestrator")
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--grounding_config", default="../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounding_checkpoint", default="../groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_encoder", default="vit_h")
    parser.add_argument("--sam_checkpoint", default="../sam_vit_h_4b8939.pth")
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--classes", default=None, help="Optional pipe-separated list of class phrases to use instead of Groq, e.g. 'bus|car|truck'")
    parser.add_argument("--score_threshold", type=float, default=0.35, help="Minimum detection score to include a detection in context")

    args = parser.parse_args()
    orch = Orchestrator(
        grounding_config=args.grounding_config,
        grounding_checkpoint=args.grounding_checkpoint,
        sam_encoder=args.sam_encoder,
        sam_checkpoint=args.sam_checkpoint,
        device=args.device,
    )

    classes_list = args.classes.split("|") if args.classes else None
    ans = orch.run(args.image, args.question, classes=classes_list, score_threshold=args.score_threshold)
    print("FINAL ANSWER:\n", ans)
