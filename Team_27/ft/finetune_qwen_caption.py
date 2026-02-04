#!/usr/bin/env python3
"""
Fine-tune Qwen3-VL model for image captioning on EarthMind dataset.

This is a captioning-focused variant adapted from the grounding finetune script.
It trains with the standard language-modeling loss only (no IoU/box losses).

Usage example:
    python qwen/finetune_qwen_caption_earthmind.py \
        --train-json qwen/../json/caption_all_unmatched.json \
        --output-dir qwen/checkpoints_caption \
        --batch-size 2 \
        --epochs 3
"""
import argparse
import json
import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
from typing import List
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from tqdm import tqdm
import logging
import traceback
import sys
import inspect

# Using plain prints for debug/trace output per user request
from transformers import TrainerCallback
import torch.nn.functional as F

try:
    from unsloth import FastVisionModel
    UNSLOTH_AVAILABLE = True
    print("helll yeahhhhhhhhhhhhhhhhhhhhh")
    
except Exception:
    UNSLOTH_AVAILABLE = False
    print("helll nooooooo")

class CustomTrainer(Trainer):
    """Custom Trainer with prediction monitoring and BLEU/BERT evaluation."""
    
    def __init__(self, *args, eval_samples=None, processor=None, 
                 eval_frequency=100, show_predictions_frequency=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_samples = eval_samples or []
        self.processor = processor
        self.eval_frequency = eval_frequency
        self.show_predictions_frequency = show_predictions_frequency
        self.metrics_history = {'bleu': [], 'bert': [], 'steps': []}
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to print per-item losses and predictions."""
        image_paths = inputs.pop('image_path', None)
        image_sizes = inputs.pop('image_size', None)

        outputs = model(**inputs)
        loss = outputs.loss
        
        # Compute per-sample predictions and losses
        try:
            logits = outputs.logits
            labels = inputs.get('labels')
            
            if logits is not None and labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                batch_size = shift_logits.size(0)
                vocab_size = shift_logits.size(-1)
                
                # Get predictions
                predicted_ids = torch.argmax(shift_logits, dim=-1)
                
                # Compute losses
                flat_logits = shift_logits.view(-1, vocab_size)
                flat_labels = shift_labels.view(-1)
                per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none')
                per_token_loss = per_token_loss.view(batch_size, -1)
                
                for i in range(batch_size):
                    sample_labels = shift_labels[i]
                    sample_preds = predicted_ids[i]
                    valid_mask = (sample_labels != -100)
                    
                    if valid_mask.sum() > 0:
                        sample_loss = per_token_loss[i][valid_mask].mean().item()
                        
                        # Extract valid tokens
                        pred_tokens = sample_preds[valid_mask]
                        label_tokens = sample_labels[valid_mask]
                        
                        # Token accuracy
                        correct = (pred_tokens == label_tokens).sum().item()
                        total = valid_mask.sum().item()
                        accuracy = 100 * correct / total if total > 0 else 0
                        
                        # Basic loss logging (always)
                        img_info = f" (img: {image_paths[i] if image_paths else 'N/A'})"
                        # print(f"{i} loss: {sample_loss:.4f} | acc: {accuracy:.1f}")
                        
                        # Detailed prediction logging (periodic)
                        if i == 0 and self.state.global_step % self.show_predictions_frequency == 0:
                            pred_text = self.processor.decode(pred_tokens, skip_special_tokens=True)
                            label_text = self.processor.decode(label_tokens, skip_special_tokens=True)
                            
                            # print(f"\n{'='*80}")
                            # print(f"STEP {self.state.global_step} - Sample {i}")
                            # print(f"Loss: {sample_loss:.4f} | Token Accuracy: {accuracy:.1f}%")
                            # print(f"{'='*80}")
                            # print(f" PREDICTED:\n{pred_text[:250]}\n")
                            # print(f" REFERENCE:\n{label_text[:250]}")
                            # print(f"{'='*80}\n")
                        
        except Exception as e:
            print(f"Failed to compute per-item loss/predictions: {e}")
            import traceback
            traceback.print_exc()
        
        # Periodic BLEU/BERT evaluation
        if (self.state.global_step > 0 and 
            self.state.global_step % self.eval_frequency == 0 and 
            len(self.eval_samples) > 0):
            self._evaluate_metrics(model)
        
        return (loss, outputs) if return_outputs else loss
        
    def _evaluate_metrics(self, model):
        """Evaluate BLEU and BERT scores on eval samples."""
        # print(f"\n{'='*80}")
        # print(f"GENERATION METRICS - Step {self.state.global_step}")
        # print(f"{'='*80}")
        
        model.eval()
        generated_texts = []
        reference_texts = []
        
        with torch.no_grad():
            for i, sample in enumerate(self.eval_samples[:2]):  # Evaluate 5 samples
                try:
                    conversation = [
                        {"role": "system", "content": [{"type": "text", "text": "You are an assistant that generates detailed, accurate descriptions of satellite and aerial imagery."}]},
                        {"role": "user", "content": [
                            {"type": "image", "image": sample['image']},
                            {"type": "text", "text": "Please describe this image in detail."}
                        ]},
                    ]
                    
                    prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(text=prompt, images=[sample['image']], return_tensors='pt')
                    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                    
                    # Generate with greedy decoding for consistency
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        do_sample=False,
                        num_beams=4,
                    )
                    
                    generated_text = self.processor.decode(
                        output_ids[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    ).strip()
                    
                    generated_texts.append(generated_text)
                    reference_texts.append(sample['caption'])
                    
                    # if i < 2:  # Print first 2 for inspection
                    #     print(f"\n[Sample {i+1}]")
                    #     print(f"Generated: {generated_text[:120]}...")
                    #     print(f"Reference: {sample['caption'][:120]}...")
                
                except Exception as e:
                    print(f"Error generating sample {i}: {e}")
                    continue
        
        # Compute BLEU
        if generated_texts:
            try:
                from nltk.translate.bleu_score import sentence_bleu
                
                bleu_scores = []
                for gen, ref in zip(generated_texts, reference_texts):
                    gen_tokens = gen.lower().split()
                    ref_tokens = ref.lower().split()
                    
                    # BLEU-4 with smoothing
                    bleu = sentence_bleu(
                        [ref_tokens], 
                        gen_tokens, 
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=lambda p, *args, **kwargs: [
                            (p_i.numerator + 1) / (p_i.denominator + 1) for p_i in p
                        ] if hasattr(p[0], 'numerator') else p
                    )
                    bleu_scores.append(bleu)
                
                avg_bleu = sum(bleu_scores) / len(bleu_scores)
                self.metrics_history['bleu'].append(avg_bleu)
                self.metrics_history['steps'].append(self.state.global_step)
                
                # print(f"\n✅ BLEU-4: {avg_bleu:.4f}")
                
            except ImportError:
                print("\n⚠️  NLTK not installed. Run: pip install nltk")
            except Exception as e:
                print(f"\n❌ BLEU computation failed: {e}")
            
            # Compute BERT Score
            try:
                from bert_score import score as bert_score_fn
                
                P, R, F1 = bert_score_fn(
                    generated_texts, 
                    reference_texts, 
                    lang='en', 
                    verbose=False,
                    device=model.device
                )
                avg_bert_f1 = F1.mean().item()
                self.metrics_history['bert'].append(avg_bert_f1)
                
                # print(f"✅ BERT-F1: {avg_bert_f1:.4f}")
                
            except ImportError:
                print("⚠️  bert-score not installed. Run: pip install bert-score")
            except Exception as e:
                print(f"❌ BERT Score computation failed: {e}")
        
        # print(f"{'='*80}\n")
        model.train()
    
    def on_train_end(self, args, state, control, **kwargs):
        """Save metrics history at end of training."""
        if self.metrics_history['steps']:
            metrics_file = os.path.join(args.output_dir, 'metrics_history.json')
            try:
                with open(metrics_file, 'w') as f:
                    json.dump(self.metrics_history, f, indent=2)
                print(f"\n✅ Metrics history saved to: {metrics_file}")
            except Exception as e:
                print(f"⚠️  Failed to save metrics history: {e}")


class CaptionDataset(Dataset):
    """Simple dataset for captioning entries.

    Expects a JSON list where each entry contains at least:
      - 'image': path to image, OR 'file_name': JSON filename to construct image path
      - 'caption' (or 'captions' or 'answer'): reference caption string (used as assistant output)
    """

    def __init__(self, json_path: str, processor, image_dir='EarthMing-Bench/img/test/rgb'):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.image_dir = image_dir
        print(f"CaptionDataset initialized: json_path={json_path}, entries={len(self.data)}, image_dir={self.image_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # Handle different field names: 'image' or construct from 'file_name'
        img_npath = entry.get('image')
        if not img_npath:
            file_name = entry.get('file_name')
            if file_name:
                # Convert 'dfc2023_test_P_0601.json' -> 'dfc2023_test_P_0601.png'
                img_name = file_name.replace('.json', '.png')
                img_path = os.path.join(self.image_dir, img_name)
                
        img_path = os.path.join(self.image_dir, img_npath)
        
        
        if not img_path:
            print(f"No image path found in JSON entry {idx}; skipping to next")
            return self.__getitem__((idx + 1) % len(self.data))
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path} at entry {idx}: {e}\n{traceback.format_exc()}")
            return self.__getitem__((idx + 1) % len(self.data))

        # Extract question and answer from conversations array
        conversations = entry.get('conversations', [])
        human_question = ""
        gpt_answer = ""
        
        for conv in conversations:
            if conv.get('from') == 'human':
                # Remove <image> token if present
                human_question = conv.get('value', '').replace('<image>\n', '').replace('<image>', '').strip()
            elif conv.get('from') == 'gpt':
                gpt_answer = conv.get('value', '').strip()
        
        # Fallback to old caption fields if conversations not found
        if not human_question:
            human_question = "Please describe this image in detail."
        if not gpt_answer:
            gpt_answer = entry.get('caption') or entry.get('captions') or entry.get('answer') or ""
            if isinstance(gpt_answer, list):
                gpt_answer = gpt_answer[0] if gpt_answer else ""

        system_instruction = (
            "You are an assistant that generates detailed, accurate descriptions "
            "of satellite and aerial imagery."
        )

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": human_question}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": gpt_answer}]}
        ]

        # Prepare model inputs using processor's chat template
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=prompt, images=[image], return_tensors='pt')
        # ✅ CORRECT MASKING: Mask everything EXCEPT assistant response
        labels = inputs['input_ids'].clone()
        
        # Method 1: Find assistant response by building prefix without it
        conversation_without_assistant = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": human_question}
            ]},
        ]
        
        # Get prompt up to (and including) the assistant token
        prefix_prompt = self.processor.apply_chat_template(
            conversation_without_assistant,
            tokenize=False,
            add_generation_prompt=True  # This adds "<|im_start|>assistant\n"
        )
        
        # Tokenize the prefix (everything we want to MASK)
        prefix_inputs = self.processor(
            text=prefix_prompt,
            images=[image],  # ⚠️ CRITICAL: Include image here too!
            return_tensors='pt'
        )
        
        # Mask everything up to the assistant response
        prefix_length = prefix_inputs['input_ids'].shape[1]
        labels[0, :prefix_length] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0) if 'pixel_values' in inputs else None,
            'image_grid_thw': inputs['image_grid_thw'].squeeze(0) if 'image_grid_thw' in inputs else None,
            'labels': labels.squeeze(0),
            'image_path': img_path,
            'image_size': image.size,
        }


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    max_len = max(seq.size(0) for seq in input_ids)

    input_ids_padded = torch.stack([
        torch.cat([seq, torch.zeros(max_len - seq.size(0), dtype=seq.dtype)])
        for seq in input_ids
    ])
    attention_mask_padded = torch.stack([
        torch.cat([seq, torch.zeros(max_len - seq.size(0), dtype=seq.dtype)])
        for seq in attention_mask
    ])
    labels_padded = torch.stack([
        torch.cat([seq, torch.full((max_len - seq.size(0),), -100, dtype=seq.dtype)])
        for seq in labels
    ])

    # For Qwen3-VL with dynamic resolution, concatenate pixel_values along spatial dim
    # Each image creates variable number of tokens, model handles this with image_grid_thw
    pixel_values = None
    image_grid_thw = None
    if batch[0]['pixel_values'] is not None:
        # Concatenate all images' pixel values along the spatial token dimension
        pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    if batch[0]['image_grid_thw'] is not None:
        # Stack image_grid_thw to track which tokens belong to which image
        image_grid_thw = torch.stack([item['image_grid_thw'] for item in batch])


    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'labels': labels_padded,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'image_path': [item['image_path'] for item in batch],
        'image_size': [item['image_size'] for item in batch],
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen3-VL for image captioning')
    parser.add_argument('--train-json', required=True, help='Path to training JSON file')
    parser.add_argument('--model', default='unsloth/Qwen3-VL-8B-Instruct', help='Base model to fine-tune')
    parser.add_argument('--output-dir', default='qwen/checkpoints_caption', help='Output directory for checkpoints')
    parser.add_argument('--batch-size', type=int, default=1, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--save-steps', type=int, default=50, help='Save checkpoint every N steps')
    parser.add_argument('--logging-steps', type=int, default=10, help='Log every N steps')
    parser.add_argument('--use-lora', action='store_true', default=False, help='Use LoRA for efficient fine-tuning (optional)')
    parser.add_argument('--load-in-4bit', action='store_true', default=False, help='Use 4-bit quantization')
    parser.add_argument('--image-dir', default='../EarthMind-Bench/img/test/sar/img', help='Directory containing image files')
    parser.add_argument('--local-model-dir', default=None, help='Local directory to load/save the pretrained model')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # If debug requested, prefer simple prints
    global USE_PRINT
    USE_PRINT = bool(getattr(args, 'debug', False))

    # Configure logging; when using print mode, raise logging to WARNING to avoid duplicate output
    log_level = logging.DEBUG if not USE_PRINT and getattr(args, 'debug', False) else logging.WARNING if USE_PRINT else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s %(levelname)s: %(message)s')

    os.environ['TORCH_COMPILE_DISABLE'] = '1'

    print(f'Loading model: {args.model} (local dir: {args.local_model_dir})')

    # Prefer loading from a local model directory if provided and present
    local_dir = args.local_model_dir
    local_exists = bool(local_dir and os.path.isdir(local_dir))

    if UNSLOTH_AVAILABLE and args.use_lora:
        # unsloth path
        if local_exists:
            print(f'Loading unsloth model from local dir: {local_dir}')
            model, tokenizer = FastVisionModel.from_pretrained(local_dir)
            processor = tokenizer
        else:
            print('Using unsloth FastVisionModel with LoRA (downloading)')
            model, tokenizer = FastVisionModel.from_pretrained(
                args.model,
                load_in_4bit=args.load_in_4bit,
                use_gradient_checkpointing='unsloth',
            )
            processor = tokenizer
            if local_dir:
                try:
                    os.makedirs(local_dir, exist_ok=True)
                    model.save_pretrained(local_dir)
                    tokenizer.save_pretrained(local_dir)
                    print(f'Saved unsloth model to local dir: {local_dir}')
                except Exception as e:
                    print(f'Warning: failed to save unsloth model to {local_dir}: {e}')

        # Apply LoRA adapters if requested (keeps original behavior)
        try:
            print(f"Applying LoRA with default config (if not already applied)")
            model = FastVisionModel.get_peft_model(
                model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,
                finetune_mlp_modules=True,
                r=16,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                random_state=3407,
                use_rslora=True,
                loftq_config=None,
            )
        except Exception:
            # If the local model is already adapted or get_peft_model is not desired, ignore
            pass

        # Ensure model dtype compatibility
        try:
            model = model.to(dtype=torch.bfloat16)
        except Exception:
            pass

        # Enable gradient checkpointing to reduce memory (trade compute for memory)
        try:
            model.gradient_checkpointing_enable()
            print('Enabled gradient checkpointing on model to reduce memory usage')
        except Exception:
            pass

    else:
        # standard transformers path
        from transformers import Qwen3VLForConditionalGeneration
        if local_exists:
            print(f'Loading model and processor from local dir: {local_dir}')
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                local_dir,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            processor = AutoProcessor.from_pretrained(local_dir)
            tokenizer = processor.tokenizer
        else:
            model_name = args.model.replace('unsloth/', 'Qwen/')
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            processor = AutoProcessor.from_pretrained(model_name)
            tokenizer = processor.tokenizer
            if local_dir:
                try:
                    os.makedirs(local_dir, exist_ok=True)
                    model.save_pretrained(local_dir)
                    processor.save_pretrained(local_dir)
                    print(f'Saved model and processor to local dir: {local_dir}')
                except Exception as e:
                    print(f'Warning: failed to save model to {local_dir}: {e}')

        # Try enabling gradient checkpointing to reduce GPU memory use
        try:
            model.gradient_checkpointing_enable()
            print('Enabled gradient checkpointing on model to reduce memory usage')
        except Exception:
            pass

    # Log model size / trainable parameters
    try:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total ({100 * trainable_params / total_params:.2f}%)")
    except Exception:
        print('Failed to compute model parameter counts')

    train_dataset = CaptionDataset(args.train_json, processor, image_dir=args.image_dir)
    print(f'Training samples: {len(train_dataset)}')
    
    # ✨ Prepare evaluation samples
    print("Preparing evaluation samples...")
    eval_samples = []
    for i in range(min(5, len(train_dataset))):
        try:
            entry = train_dataset.data[i]
            
            # Get image path
            img_npath = entry.get('image')
            if not img_npath:
                file_name = entry.get('file_name')
                if file_name:
                    img_name = file_name.replace('.json', '.png')
                    img_path = os.path.join(args.image_dir, img_name)
                    
            img_path = os.path.join(args.image_dir, img_npath)
            
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            
            # Get caption from conversations
            conversations = entry.get('conversations', [])
            caption = ""
            for conv in conversations:
                if conv.get('from') == 'gpt':
                    caption = conv.get('value', '').strip()
                    break
            # Fallback
            if not caption:
                caption = entry.get('caption') or entry.get('answer') or ""
                if isinstance(caption, list):
                    caption = caption[0] if caption else ""
            
            eval_samples.append({
                'image': image,
                'caption': caption,
                'image_path': img_path
            })
            print(f"  Loaded eval sample {i+1}: {img_path}")
            
        except Exception as e:
            print(f"  Failed to load eval sample {i}: {e}")
    
    print(f"✅ Prepared {len(eval_samples)} evaluation samples\n")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=False,
        bf16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to='none',
        optim='adamw_8bit' if args.load_in_4bit else 'adamw_torch',
        deepspeed=None,  # Disable DeepSpeed to avoid compatibility issues
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        eval_samples=eval_samples,     # ✨ Add eval samples
        processor=processor,            # ✨ Add processor
        eval_frequency=1000,             # ✨ Evaluate every 100 steps
    )

    print('Starting captioning training...')
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed with exception: {e}\n{traceback.format_exc()}")
        raise

    final_output_dir = os.path.join(args.output_dir, 'final')
    print(f'Saving final model to: {final_output_dir}')
    trainer.save_model(final_output_dir)
    try:
        processor.save_pretrained(final_output_dir)
        model.save_pretrained(final_output_dir)
        print(f"Saved LoRA/adapter weights to: {final_output_dir}")
    except Exception:
        # processor might be tokenizer in unsloth path; save tokenizer if present
        try:
            tokenizer.save_pretrained(final_output_dir)
        except Exception:
            pass

    print('Captioning training complete!')


if __name__ == '__main__':
    main()

