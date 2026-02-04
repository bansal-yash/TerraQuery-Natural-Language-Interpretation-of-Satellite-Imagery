# VQA System with Local LLM + GroundingDINO + SAM + VLM

A complete Visual Question Answering system that combines:
- **Local Small LLM**: Extracts relevant object classes from user queries
- **GroundingDINO**: Detects objects based on extracted classes
- **SAM (Segment Anything Model)**: Segments detected objects
- **Geometric Analysis**: Computes properties (area, orientation, aspect ratio, distances)
- **Local VLM**: Answers questions with reasoning and evidence

## Architecture Flow

```
User Question + Image
        ↓
[1] Local LLM extracts object classes
    (e.g., "count red buses" → ["red bus", "bus"])
        ↓
[2] GroundingDINO detects objects
    (finds all instances of extracted classes)
        ↓
[3] SAM segments each detection
    (creates precise masks for each object)
        ↓
[4] Compute geometric properties
    - Area (square pixels)
    - Centroid position (x, y)
    - Orientation (degrees)
    - Aspect ratio
    - Bounding box size
    - Perimeter
    - Distance between objects
        ↓
[5] VLM analyzes with full context
    - Original image + all crops
    - All geometric measurements
    - System prompt guidance
        ↓
[6] Structured output
    [THINKING] - Step-by-step reasoning
    [FINAL ANSWER] - Concise answer + relevant image path
```

## Supported Question Types

### 1. **Counting Questions**
- Counts only **directly visible** objects
- Example: "How many red buses are there?"
- Output: `3`

### 2. **Binary Yes/No Questions**
- Simple presence/absence queries
- Example: "Is there a person in the image?"
- Output: `Yes` or `No`

### 3. **Area/Distance Questions**
- Uses segmentation masks for precise measurements
- Reports in pixel units
- Example: "What is the area of the largest building?"
- Output: `15600 square pixels`

### 4. **Attribute Questions**
- Single-word or short phrase attributes
- Example: "What color is the largest vehicle?"
- Output: `red`

## Files

- **`orchestrator.py`**: Main orchestration pipeline
- **`local_llm.py`**: Small LLM for query → classes extraction
- **`local_vlm.py`**: Vision-Language Model for visual reasoning
- **`geometric_utils.py`**: Mask property computation functions
- **`systemprompt.txt`**: System prompt guiding VLM behavior
- **`gsam.py`**: GroundingDINO + SAM utilities (external)

## Usage

### Command Line

```bash
python orchestrator.py \
  --image /path/to/image.jpg \
  --question "How many red buses are visible?" \
  --device cuda \
  --score_threshold 0.35
```

### Python API

```python
from vqa.orchestrator import Orchestrator

orch = Orchestrator(
    grounding_config="path/to/grounding_config.py",
    grounding_checkpoint="path/to/grounding.pth",
    sam_encoder="vit_h",
    sam_checkpoint="path/to/sam.pth",
    device="cuda",
    system_prompt_path="systemprompt.txt"
)

answer = orch.run(
    image_path="/path/to/image.jpg",
    question="How many cars are there?",
    score_threshold=0.35
)

print(answer)
```

### Custom Classes

Override automatic class extraction:

```bash
python orchestrator.py \
  --image /path/to/image.jpg \
  --question "Count the vehicles" \
  --classes "red bus|yellow bus|car|truck"
```

## Output Format

The VLM returns a structured response:

```
[THINKING]
I can see 2 objects classified as "red bus" with high confidence scores (0.89 and 0.82). 
These are both directly visible in the image. Their combined area is approximately 25000 
square pixels. The car detection is not relevant to this question.

[FINAL ANSWER]
2
/path/to/vqa_outputs/crop_0.jpg
```

## Output Artifacts

All intermediate results are saved to `vqa_outputs/`:
- `crop_0.jpg`, `crop_1.jpg`, ... : Cropped object images
- `mask_0.png`, `mask_1.png`, ... : Binary segmentation masks

## Geometric Properties Computed

For each detected object:

| Property | Description | Unit |
|----------|-------------|------|
| Area | Total pixels in mask | square pixels |
| Centroid | Center of mass position | (x, y) pixels |
| Orientation | Angle of major axis | degrees (-90 to 90) |
| Aspect Ratio | Major axis / minor axis | ratio |
| Bounding Box | Smallest enclosing rectangle | width × height pixels |
| Perimeter | Boundary length | pixels |

Between objects:
- **Distance**: Euclidean distance between centroids (pixels)
- **Relative Position**: Directional relationship (e.g., "above-left")

## System Prompt

The `systemprompt.txt` file guides the VLM to:
1. Think step-by-step before answering
2. Use provided geometric measurements
3. Count only directly visible objects
4. Format output with [THINKING] and [FINAL ANSWER] sections
5. Always include the most relevant image path

Edit this file to customize VLM behavior.

## Dependencies

```
torch
transformers
opencv-python (cv2)
numpy
pillow
groundingdino
segment-anything
```

## Model Requirements

### Local LLM (for class extraction)
- Default: `microsoft/Phi-3-mini-4k-instruct`
- ~8GB GPU memory
- Can use CPU (slower)

### Local VLM (for visual reasoning)
- Default: `Qwen/Qwen3-VL-4B-Thinking`
- ~16GB GPU memory
- Streams reasoning tokens in real-time

### GroundingDINO
- Config: `GroundingDINO_SwinT_OGC.py`
- Checkpoint: `groundingdino_swint_ogc.pth`

### SAM
- Encoder: `vit_h`
- Checkpoint: `sam_vit_h_4b8939.pth`

## Example Scenarios

### Scenario 1: Counting colored objects
```
Question: "Count all red and yellow buses"
LLM extracts: ["red bus", "yellow bus"]
GSAM detects: 3 red buses, 2 yellow buses
VLM answers: "5" (with thinking showing breakdown)
```

### Scenario 2: Area measurement
```
Question: "What is the area of the largest building?"
LLM extracts: ["building"]
GSAM detects: building (15600 sq px), building (8900 sq px)
VLM answers: "15600 square pixels"
```

### Scenario 3: Distance query
```
Question: "How far apart are the two cars?"
LLM extracts: ["car"]
GSAM detects: car 1, car 2
Geometric util computes: 450 pixels distance
VLM answers: "450 pixels"
```

## Troubleshooting

### No objects detected
- Lower `--score_threshold` (try 0.2 or 0.25)
- Check if classes match actual objects in image
- Use `--classes` to provide explicit object names

### VLM gives wrong count
- Review [THINKING] section to see reasoning
- Check if low-confidence detections are included
- Adjust system prompt to emphasize "directly visible"

### Out of memory
- Use smaller models
- Process fewer detections at once
- Reduce image resolution
- Switch to CPU (add `--device cpu`)

## Future Improvements

- [ ] Support for video/multi-frame analysis
- [ ] Spatial relationship reasoning (left/right/above/below)
- [ ] Integration with knowledge bases
- [ ] Fine-tuning VLM on specific domains
- [ ] Batch processing multiple images
- [ ] Web UI for interactive queries
