# 📊 Pipeline Evaluation Benchmarks

Quantitative evaluation suite for the Multi-Modal Document Intelligence Pipeline.

## Overview

This benchmark suite measures pipeline accuracy across four dimensions:

| Metric Category | What It Measures |
|:---|:---|
| **Text Extraction** | Word-level precision, recall, F1, and character accuracy |
| **Table Structure** | Cell-level accuracy, row/column correctness, structure fidelity |
| **Layout Detection** | IoU (Intersection over Union) for region bounding boxes |
| **Conflict Resolution** | Detection rate, resolution accuracy, confidence improvement |

## Quick Start

```bash
# Run evaluation with sample dataset
python -m benchmarks.evaluate_pipeline

# Custom dataset
python -m benchmarks.evaluate_pipeline --dataset path/to/ground_truth.json

# Custom output
python -m benchmarks.evaluate_pipeline --output reports/my_report.json
```

## Ground Truth Format

Create a JSON file following the schema in `sample_dataset/ground_truth.json`:

```json
{
  "dataset_name": "My Benchmark",
  "documents": [
    {
      "document_id": "doc_001",
      "expected_text": "...",
      "expected_tables": [[["Header1", "Header2"], ["Cell1", "Cell2"]]],
      "expected_regions": [{"type": "table", "bbox": [x1, y1, x2, y2]}]
    }
  ]
}
```

## Output

Reports are saved as JSON to `benchmarks/reports/` and include per-category metrics plus an overall F1 score.
