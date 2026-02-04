from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-8B-Instruct",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = False, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


import os
import json
from PIL import Image
from datasets import Dataset as HFDataset
from tqdm import tqdm

def create_hf_dataset(images_dir, annotations_dir, max_samples=None):
    """Create HuggingFace Dataset from VRSBench data"""
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    image_files.sort()
    
    # Limit to max_samples if specified
    if max_samples:
        image_files = image_files[:max_samples]
    
    data = {
        'image': [],
        'text': []  # Changed from 'caption' to 'text' to match your conversion function
    }
    
    for img_filename in tqdm(image_files):
        # Load image
        img_path = os.path.join(images_dir, img_filename)
        image = Image.open(img_path).convert('RGB')
        
        # Load caption
        json_filename = img_filename.replace('.png', '.json')
        json_path = os.path.join(annotations_dir, json_filename)
        
        with open(json_path, 'r') as f:
            annotation = json.load(f)
        
        data['image'].append(image)
        data['text'].append(annotation['caption'])  # Store as 'text'
    
    return HFDataset.from_dict(data)



images_dir = "/home/samyak/scratch/interiit/samyak/GeoPixel/VRSBench/Images_train"
annotations_dir = "/home/samyak/scratch/interiit/samyak/GeoPixel/VRSBench/Annotations_train"

# Load dataset (first 100 images)
dataset = create_hf_dataset(images_dir, annotations_dir, max_samples=None)


instruction = "Describe this satellite image in detail."  # Changed to be more relevant for satellite images

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content": [
              {"type": "text",  "text": instruction},
              {"type": "image", "image": sample["image"]}
          ]
        },
        { "role": "assistant",
          "content": [
              {"type": "text",  "text": sample["text"]}
          ]
        },
    ]
    return {"messages": conversation}


converted_dataset = [convert_to_conversation(sample) for sample in tqdm(dataset)]


from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = converted_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # max_steps = None,
        num_train_epochs = 1, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",     # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        # remove_unused_columns = False,
        # dataset_text_field = "",
        # dataset_kwargs = {"skip_prepare_dataset": True},
        # max_length = 2048,
    ),
)


trainer_stats = trainer.train()




# # Add training + validation flow
# from unsloth import FastVisionModel
# import torch
# from PIL import Image
# from datasets import Dataset as HFDataset
# import os
# import json
# from bert_score import score as bert_score
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from torch.optim import AdamW

# # -----------------------------
# # Configuration (edit as needed)
# # -----------------------------
# train_images_dir = "/home/samyak/scratch/interiit/samyak/GeoPixel/VRSBench/Images_train"
# train_annotations_dir = "/home/samyak/scratch/interiit/samyak/GeoPixel/VRSBench/Annotations_train"
# images_dir_val = "/home/samyak/scratch/interiit/samyak/GeoPixel/VRSBench/Images_val"
# annotations_dir_val = "/home/samyak/scratch/interiit/samyak/GeoPixel/VRSBench/Annotations_val"

# max_train_samples = None   # set to an int to limit training size for quick runs
# train_epochs = 1
# train_batch_size = 1      # per-sample processing; keep 1 unless a collator is provided
# learning_rate = 2e-4
# max_train_steps = None    # optional: set to an int to override epoch-based stopping
# save_dir = "outputs/checkpoint-train"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # PEFT/LoRA settings
# use_peft = True
# lora_r = 16
# lora_alpha = 16
# lora_dropout = 0.0


# # ============================================
# # 1. LOAD THE MODEL (for training)
# # ============================================
# model, tokenizer = FastVisionModel.from_pretrained(
#     model_name = "outputs/checkpoint-30",  # Base or previous checkpoint
#     load_in_4bit = True,
#     use_gradient_checkpointing = "unsloth",
# )

# # Apply LoRA / PEFT adapters if requested
# if use_peft:
#     try:
#         model = FastVisionModel.get_peft_model(
#             model,
#             finetune_vision_layers     = True,
#             finetune_language_layers   = True,
#             finetune_attention_modules = True,
#             finetune_mlp_modules       = True,

#             r = lora_r,
#             lora_alpha = lora_alpha,
#             lora_dropout = lora_dropout,
#             bias = "none",
#             random_state = 3407,
#             use_rslora = False,
#             loftq_config = None,
#         )
#     except RuntimeError as e:
#         msg = str(e)
#         # If adapters already added, continue using the model as-is
#         if "already added LoRA adapters" in msg or "already added lora" in msg.lower():
#             print("LoRA adapters already present on the model; skipping get_peft_model().")
#         else:
#             # Re-raise unexpected RuntimeErrors
#             raise

# FastVisionModel.for_training(model)  # prepare model for training
# model.to(device)
# # Determine model parameter dtype and device for consistent casting
# try:
#     _model_param = next(model.parameters())
#     model_param_dtype = _model_param.dtype
#     model_param_device = _model_param.device
# except StopIteration:
#     model_param_dtype = torch.float32
#     model_param_device = device

# # ============================================
# # 2. LOAD VALIDATION DATASET
# # ============================================
# def create_hf_dataset(images_dir, annotations_dir, max_samples=None):
#     """Create HuggingFace Dataset from VRSBench data"""
#     image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
#     image_files.sort()
    
#     if max_samples:
#         image_files = image_files[:max_samples]
    
#     data = {
#         'image': [],
#         'text': [],
#         'image_filename': []
#     }
    
#     for img_filename in image_files:
#         img_path = os.path.join(images_dir, img_filename)
#         image = Image.open(img_path).convert('RGB')
        
#         json_filename = img_filename.replace('.png', '.json')
#         json_path = os.path.join(annotations_dir, json_filename)
        
#         with open(json_path, 'r') as f:
#             annotation = json.load(f)
        
#         data['image'].append(image)
#         data['text'].append(annotation['caption'])
#         data['image_filename'].append(img_filename)
    
#     return HFDataset.from_dict(data)


# # Simple collate that returns a single-sample dict (we keep per-sample processing
# # because the model/processor may expect image objects and tokenizer.apply_chat_template
# # handles internal batching; implementing a fully correct multi-sample collator
# # requires access to the internal processor/collator from the model repo.)
# def make_dataloader(hf_dataset, batch_size=1, shuffle=False, collate_fn=None):
#     """Create a DataLoader. By default we return the raw list of samples
#     (collate_fn=lambda x: x) so PIL image objects are preserved and not
#     passed into torch's default_collate which expects tensors.
#     """
#     if collate_fn is None:
#         # return raw list of samples
#         collate_fn = lambda x: x
#     return DataLoader(hf_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# # Load validation dataset
# # ============================================
# # 2. LOAD TRAIN + VALIDATION DATASETS
# # ============================================
# train_dataset = create_hf_dataset(train_images_dir, train_annotations_dir, max_samples=max_train_samples)
# val_dataset = create_hf_dataset(images_dir_val, annotations_dir_val, max_samples=None)

# print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

# # DataLoaders (per-sample batching)
# train_loader = make_dataloader(train_dataset, batch_size=train_batch_size, shuffle=True)

# # --------------------------------------------
# # 3. TRAINING LOOP (simple teacher-forcing using labels=input_ids)
# # --------------------------------------------
# # print("Starting training...")
# # optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
# # global_step = 0
# # model.train()

# # # Iterate epochs until max_train_steps or epochs completed
# # for epoch in range(train_epochs):
# #     for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
# #         # Our collate returns a list of samples. Normalize to a list of sample dicts.
# #         if isinstance(batch, list):
# #             samples = batch
# #         elif isinstance(batch, dict):
# #             samples = [batch]
# #         else:
# #             # fallback
# #             samples = list(batch)

# #         # Zero grads at start of batch
# #         optimizer.zero_grad()

# #         # Process each sample in the batch sequentially (accumulate gradients).
# #         for i, sample in enumerate(samples):
# #             messages = [
# #                 {
# #                     "role": "user",
# #                     "content": [
# #                         {"type": "text", "text": "Describe this satellite image in detail."},
# #                         {"type": "image", "image": sample["image"]}
# #                     ]
# #                 },
# #                 {
# #                     "role": "assistant",
# #                     "content": [
# #                         {"type": "text", "text": sample["text"]}
# #                     ]
# #                 }
# #             ]

# #             inputs = tokenizer.apply_chat_template(
# #                 messages,
# #                 tokenize=True,
# #                 add_generation_prompt=True,
# #                 return_dict=True,
# #                 return_tensors="pt"
# #             )

# #             inputs.pop("token_type_ids", None)
# #             # Move all tensor inputs to model device and cast floating tensors to model dtype
# #             inputs = inputs.to(model_param_device)
# #             for k, v in list(inputs.items()):
# #                 if torch.is_floating_point(v):
# #                     inputs[k] = v.to(dtype=model_param_dtype)

# #             labels = inputs["input_ids"].clone()
# #             labels = labels.to(model_param_device)

# #             # Ensure dtype/device compat — retry cast on mismatched dtype errors
# #             try:
# #                 outputs = model(**inputs, labels=labels)
# #             except RuntimeError as e:
# #                 msg = str(e)
# #                 if "must have the same dtype" in msg or "mat1 and mat2 must have the same dtype" in msg:
# #                     # Cast all floating tensors in inputs to model dtype and retry
# #                     for k, v in list(inputs.items()):
# #                         if torch.is_floating_point(v):
# #                             inputs[k] = v.to(dtype=model_param_dtype, device=model_param_device)
# #                     labels = labels.to(device=model_param_device)
# #                     outputs = model(**inputs, labels=labels)
# #                 else:
# #                     raise

# #             loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

# #             # Average loss across batch samples
# #             loss = loss / len(samples)
# #             loss.backward()

# #         # step once per batch
# #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# #         optimizer.step()

# #         global_step += 1

# #         if max_train_steps is not None and global_step >= max_train_steps:
# #             break
# #     if max_train_steps is not None and global_step >= max_train_steps:
# #         break

# # print(f"Training finished. Performed {global_step} steps.")

# # # Save checkpoint
# # os.makedirs(save_dir, exist_ok=True)
# # try:
# #     model.save_pretrained(save_dir)
# #     tokenizer.save_pretrained(save_dir)
# #     print(f"Saved trained model and tokenizer to {save_dir}")
# # except Exception as e:
# #     print(f"Warning: failed to save model with error: {e}")

# # # Switch to inference
# # FastVisionModel.for_inference(model)
# # model.to(device)

# # ============================================
# # 4. GENERATE PREDICTIONS (validation) -- original code follows
# # ============================================

# # ============================================
# # 3. GENERATE PREDICTIONS
# # ============================================
# instruction = "Describe this satellite image in detail."

# predictions = []
# references = []

# print("Generating predictions...")
# for sample in tqdm(val_dataset):
#     # Prepare input messages following Qwen3VL format
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": instruction},
#                 {"type": "image", "image": sample["image"]}
#             ]
#         }
#     ]
    
#     # Use processor.apply_chat_template with tokenize=True.
#     # This is the correct way for Qwen3VL: the processor handles both text
#     # tokenization and image processing internally when you pass messages
#     # with image objects. No need to call tokenizer separately.
#     # Reference: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
#     inputs = tokenizer.apply_chat_template(
#         messages,
#         tokenize=True,
#         add_generation_prompt=True,
#         return_dict=True,
#         return_tensors="pt"
#     )
    
#     # Remove token_type_ids if present (not needed for generation)
#     inputs.pop("token_type_ids", None)
    
#     # Move all tensor inputs to model device and cast floats to model dtype
#     inputs = inputs.to(model_param_device)
#     for k, v in list(inputs.items()):
#         if torch.is_floating_point(v):
#             inputs[k] = v.to(dtype=model_param_dtype, device=model_param_device)
    
#     # Generate
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=512,
#             temperature=0.7,
#             top_p=0.9,
#             do_sample=False,  # For reproducible evaluation
#         )
    
#     # Decode prediction
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     # Extract only the assistant's response (after the prompt)
#     # This depends on how your model formats responses
#     # You might need to adjust this based on your model's output format
#     if "assistant" in generated_text.lower():
#         prediction = generated_text.split("assistant")[-1].strip()
#     else:
#         prediction = generated_text
    
#     predictions.append(prediction)
#     references.append(sample["text"])

# # ============================================
# # 4. COMPUTE BERTSCORE
# # ============================================
# print("\nComputing BERTScore...")

# # Compute BERTScore
# # P = Precision, R = Recall, F1 = F1-score
# P, R, F1 = bert_score(
#     predictions,
#     references,
#     lang="en",
#     verbose=True,
#     model_type="bert-base-uncased"  # You can use other models like "roberta-large"
# )

# # ============================================
# # 5. PRINT RESULTS
# # ============================================
# print("\n" + "="*60)
# print("BERTSCORE EVALUATION RESULTS")
# print("="*60)
# print(f"Number of samples: {len(predictions)}")
# print(f"\nAverage Precision:  {P.mean():.4f} (±{P.std():.4f})")
# print(f"Average Recall:     {R.mean():.4f} (±{R.std():.4f})")
# print(f"Average F1:         {F1.mean():.4f} (±{F1.std():.4f})")
# print("="*60)

# # ============================================
# # 6. SAVE DETAILED RESULTS
# # ============================================
# results = []
# for i in range(len(predictions)):
#     results.append({
#         "image_filename": val_dataset[i]["image_filename"],
#         "reference": references[i],
#         "prediction": predictions[i],
#         "bert_precision": P[i].item(),
#         "bert_recall": R[i].item(),
#         "bert_f1": F1[i].item()
#     })

# # Save to JSON
# with open("evaluation_results.json", "w") as f:
#     json.dump({
#         "summary": {
#             "num_samples": len(predictions),
#             "avg_precision": P.mean().item(),
#             "avg_recall": R.mean().item(),
#             "avg_f1": F1.mean().item(),
#             "std_precision": P.std().item(),
#             "std_recall": R.std().item(),
#             "std_f1": F1.std().item()
#         },
#         "detailed_results": results
#     }, f, indent=2)

# print("\nDetailed results saved to 'evaluation_results.json'")

# # ============================================
# # 7. OPTIONAL: SHOW SOME EXAMPLES
# # ============================================
# print("\n" + "="*60)
# print("SAMPLE PREDICTIONS (First 3)")
# print("="*60)
# for i in range(min(3, len(predictions))):
#     print(f"\nImage: {val_dataset[i]['image_filename']}")
#     print(f"Reference: {references[i][:200]}...")
#     print(f"Prediction: {predictions[i][:200]}...")
#     print(f"BERTScore F1: {F1[i]:.4f}")
#     print("-"*60)
