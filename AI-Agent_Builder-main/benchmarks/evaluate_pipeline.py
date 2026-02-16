"""
Pipeline Evaluation Suite
=========================

Quantitative evaluation harness for the Multi-Modal Document Intelligence Pipeline.
Computes precision, recall, F1 score, and table structure accuracy by comparing
pipeline output against labeled ground-truth data.

Usage:
    python -m benchmarks.evaluate_pipeline
    python -m benchmarks.evaluate_pipeline --dataset benchmarks/sample_dataset/ground_truth.json
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from difflib import SequenceMatcher
from collections import Counter
import re

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class TextExtractionMetrics:
    """Metrics for text extraction accuracy."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    character_accuracy: float = 0.0
    word_accuracy: float = 0.0
    sample_count: int = 0


@dataclass
class TableExtractionMetrics:
    """Metrics for table structure accuracy."""
    cell_precision: float = 0.0
    cell_recall: float = 0.0
    cell_f1: float = 0.0
    structure_accuracy: float = 0.0
    row_count_accuracy: float = 0.0
    column_count_accuracy: float = 0.0
    sample_count: int = 0


@dataclass
class LayoutDetectionMetrics:
    """Metrics for layout/region detection accuracy."""
    iou_mean: float = 0.0
    iou_median: float = 0.0
    region_precision: float = 0.0
    region_recall: float = 0.0
    region_f1: float = 0.0
    sample_count: int = 0


@dataclass
class ConflictResolutionMetrics:
    """Metrics for the conflict resolution engine."""
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    resolution_accuracy: float = 0.0
    avg_confidence_improvement: float = 0.0


@dataclass
class PipelineEvaluationReport:
    """Complete evaluation report for the document processing pipeline."""
    timestamp: str = ""
    dataset_name: str = ""
    total_documents: int = 0
    total_processing_time_seconds: float = 0.0
    avg_processing_time_seconds: float = 0.0

    text_metrics: TextExtractionMetrics = field(default_factory=TextExtractionMetrics)
    table_metrics: TableExtractionMetrics = field(default_factory=TableExtractionMetrics)
    layout_metrics: LayoutDetectionMetrics = field(default_factory=LayoutDetectionMetrics)
    conflict_metrics: ConflictResolutionMetrics = field(default_factory=ConflictResolutionMetrics)

    overall_accuracy: float = 0.0
    overall_f1: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ─────────────────────────────────────────────
# Evaluation Functions
# ─────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def compute_character_accuracy(predicted: str, ground_truth: str) -> float:
    """Compute character-level accuracy using sequence matching."""
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    matcher = SequenceMatcher(None, predicted, ground_truth)
    return matcher.ratio()


def compute_word_metrics(predicted: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Compute word-level precision, recall, and F1 score.
    
    Returns:
        (precision, recall, f1_score)
    """
    pred_words = set(normalize_text(predicted).split())
    gt_words = set(normalize_text(ground_truth).split())

    if not gt_words and not pred_words:
        return 1.0, 1.0, 1.0
    if not gt_words:
        return 0.0, 1.0, 0.0
    if not pred_words:
        return 1.0, 0.0, 0.0

    true_positives = len(pred_words & gt_words)
    precision = true_positives / len(pred_words) if pred_words else 0.0
    recall = true_positives / len(gt_words) if gt_words else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_table_metrics(
    predicted_table: List[List[str]],
    ground_truth_table: List[List[str]]
) -> Dict[str, float]:
    """
    Compute table extraction metrics.
    
    Compares predicted table cells against ground truth, measuring:
    - Cell-level precision/recall/F1
    - Row count accuracy
    - Column count accuracy
    - Overall structure accuracy
    """
    gt_rows = len(ground_truth_table)
    gt_cols = max(len(row) for row in ground_truth_table) if ground_truth_table else 0
    pred_rows = len(predicted_table)
    pred_cols = max(len(row) for row in predicted_table) if predicted_table else 0

    # Row/Column count accuracy
    row_acc = 1.0 - abs(gt_rows - pred_rows) / max(gt_rows, 1)
    col_acc = 1.0 - abs(gt_cols - pred_cols) / max(gt_cols, 1)
    row_acc = max(0.0, row_acc)
    col_acc = max(0.0, col_acc)

    # Cell-level matching
    gt_cells = set()
    for i, row in enumerate(ground_truth_table):
        for j, cell in enumerate(row):
            gt_cells.add((i, j, normalize_text(str(cell))))

    pred_cells = set()
    for i, row in enumerate(predicted_table):
        for j, cell in enumerate(row):
            pred_cells.add((i, j, normalize_text(str(cell))))

    true_positives = len(gt_cells & pred_cells)
    cell_precision = true_positives / len(pred_cells) if pred_cells else 0.0
    cell_recall = true_positives / len(gt_cells) if gt_cells else 0.0
    cell_f1 = (
        2 * cell_precision * cell_recall / (cell_precision + cell_recall)
        if (cell_precision + cell_recall) > 0 else 0.0
    )

    # Structure accuracy = geometric mean of row, col, and cell accuracy
    structure_acc = (row_acc * col_acc * cell_f1) ** (1 / 3) if cell_f1 > 0 else 0.0

    return {
        "cell_precision": round(cell_precision, 4),
        "cell_recall": round(cell_recall, 4),
        "cell_f1": round(cell_f1, 4),
        "row_count_accuracy": round(row_acc, 4),
        "column_count_accuracy": round(col_acc, 4),
        "structure_accuracy": round(structure_acc, 4),
    }


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection over Union for two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


# ─────────────────────────────────────────────
# Pipeline Evaluator
# ─────────────────────────────────────────────

class PipelineEvaluator:
    """
    Evaluator for the Multi-Modal Document Intelligence Pipeline.
    
    Compares pipeline output against ground-truth labels to produce
    quantitative metrics for text extraction, table structure,
    layout detection, and conflict resolution.
    
    Usage:
        evaluator = PipelineEvaluator()
        evaluator.load_ground_truth("benchmarks/sample_dataset/ground_truth.json")
        report = evaluator.evaluate(pipeline_outputs)
        report_json = report.to_json()
    """

    def __init__(self):
        self.ground_truth: List[Dict[str, Any]] = []
        self.report = PipelineEvaluationReport()

    def load_ground_truth(self, path: str) -> None:
        """Load ground truth dataset from JSON file."""
        gt_path = Path(path)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {path}")

        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.ground_truth = data.get("documents", [])
        self.report.dataset_name = data.get("dataset_name", gt_path.stem)
        self.report.total_documents = len(self.ground_truth)
        logger.info(f"Loaded {len(self.ground_truth)} ground truth documents from {path}")

    def evaluate(
        self,
        pipeline_outputs: List[Dict[str, Any]],
    ) -> PipelineEvaluationReport:
        """
        Evaluate pipeline outputs against ground truth.
        
        Args:
            pipeline_outputs: List of pipeline output dicts, each containing:
                - document_id: str
                - extracted_text: str
                - tables: List[List[List[str]]]
                - regions: List[Dict] with bounding boxes
                - conflicts: List[Dict] with resolution info
                - processing_time: float (seconds)
        
        Returns:
            PipelineEvaluationReport with all metrics
        """
        import datetime
        self.report.timestamp = datetime.datetime.now().isoformat()

        # Index pipeline outputs by document_id
        output_map = {o["document_id"]: o for o in pipeline_outputs}

        text_precisions, text_recalls, text_f1s = [], [], []
        char_accuracies, word_accuracies = [], []
        all_table_metrics = []
        all_ious = []
        region_precisions, region_recalls = [], []
        total_time = 0.0

        for gt_doc in self.ground_truth:
            doc_id = gt_doc["document_id"]
            pred = output_map.get(doc_id)
            if not pred:
                logger.warning(f"Missing pipeline output for document: {doc_id}")
                continue

            total_time += pred.get("processing_time", 0.0)

            # ── Text Metrics ──
            if "expected_text" in gt_doc and "extracted_text" in pred:
                gt_text = gt_doc["expected_text"]
                pred_text = pred["extracted_text"]

                p, r, f1 = compute_word_metrics(pred_text, gt_text)
                text_precisions.append(p)
                text_recalls.append(r)
                text_f1s.append(f1)
                char_accuracies.append(compute_character_accuracy(
                    normalize_text(pred_text), normalize_text(gt_text)
                ))
                word_accuracies.append(f1)

            # ── Table Metrics ──
            gt_tables = gt_doc.get("expected_tables", [])
            pred_tables = pred.get("tables", [])
            for gt_tbl, pred_tbl in zip(gt_tables, pred_tables):
                tbl_m = compute_table_metrics(pred_tbl, gt_tbl)
                all_table_metrics.append(tbl_m)

            # ── Layout/Region Metrics ──
            gt_regions = gt_doc.get("expected_regions", [])
            pred_regions = pred.get("regions", [])
            if gt_regions and pred_regions:
                matched = 0
                for gt_r in gt_regions:
                    best_iou = 0.0
                    for pred_r in pred_regions:
                        iou = compute_iou(gt_r["bbox"], pred_r["bbox"])
                        best_iou = max(best_iou, iou)
                    all_ious.append(best_iou)
                    if best_iou >= 0.5:
                        matched += 1
                region_recalls.append(matched / len(gt_regions) if gt_regions else 0.0)
                region_precisions.append(matched / len(pred_regions) if pred_regions else 0.0)

            # ── Conflict Resolution Metrics ──
            conflicts = pred.get("conflicts", [])
            for c in conflicts:
                self.report.conflict_metrics.conflicts_detected += 1
                if c.get("resolved", False):
                    self.report.conflict_metrics.conflicts_resolved += 1

        # ── Aggregate Text Metrics ──
        if text_precisions:
            self.report.text_metrics = TextExtractionMetrics(
                precision=round(sum(text_precisions) / len(text_precisions), 4),
                recall=round(sum(text_recalls) / len(text_recalls), 4),
                f1_score=round(sum(text_f1s) / len(text_f1s), 4),
                character_accuracy=round(sum(char_accuracies) / len(char_accuracies), 4),
                word_accuracy=round(sum(word_accuracies) / len(word_accuracies), 4),
                sample_count=len(text_precisions),
            )

        # ── Aggregate Table Metrics ──
        if all_table_metrics:
            n = len(all_table_metrics)
            self.report.table_metrics = TableExtractionMetrics(
                cell_precision=round(sum(m["cell_precision"] for m in all_table_metrics) / n, 4),
                cell_recall=round(sum(m["cell_recall"] for m in all_table_metrics) / n, 4),
                cell_f1=round(sum(m["cell_f1"] for m in all_table_metrics) / n, 4),
                structure_accuracy=round(sum(m["structure_accuracy"] for m in all_table_metrics) / n, 4),
                row_count_accuracy=round(sum(m["row_count_accuracy"] for m in all_table_metrics) / n, 4),
                column_count_accuracy=round(sum(m["column_count_accuracy"] for m in all_table_metrics) / n, 4),
                sample_count=n,
            )

        # ── Aggregate Layout Metrics ──
        if all_ious:
            sorted_ious = sorted(all_ious)
            mid = len(sorted_ious) // 2
            median_iou = (
                sorted_ious[mid]
                if len(sorted_ious) % 2 != 0
                else (sorted_ious[mid - 1] + sorted_ious[mid]) / 2
            )
            avg_rp = sum(region_precisions) / len(region_precisions) if region_precisions else 0.0
            avg_rr = sum(region_recalls) / len(region_recalls) if region_recalls else 0.0
            r_f1 = 2 * avg_rp * avg_rr / (avg_rp + avg_rr) if (avg_rp + avg_rr) > 0 else 0.0

            self.report.layout_metrics = LayoutDetectionMetrics(
                iou_mean=round(sum(all_ious) / len(all_ious), 4),
                iou_median=round(median_iou, 4),
                region_precision=round(avg_rp, 4),
                region_recall=round(avg_rr, 4),
                region_f1=round(r_f1, 4),
                sample_count=len(all_ious),
            )

        # ── Conflict Resolution Metrics ──
        cm = self.report.conflict_metrics
        if cm.conflicts_detected > 0:
            cm.resolution_accuracy = round(cm.conflicts_resolved / cm.conflicts_detected, 4)

        # ── Overall Metrics ──
        self.report.total_processing_time_seconds = round(total_time, 2)
        n_docs = max(len(pipeline_outputs), 1)
        self.report.avg_processing_time_seconds = round(total_time / n_docs, 2)

        component_f1s = []
        if self.report.text_metrics.f1_score > 0:
            component_f1s.append(self.report.text_metrics.f1_score)
        if self.report.table_metrics.cell_f1 > 0:
            component_f1s.append(self.report.table_metrics.cell_f1)
        if self.report.layout_metrics.region_f1 > 0:
            component_f1s.append(self.report.layout_metrics.region_f1)

        if component_f1s:
            self.report.overall_f1 = round(sum(component_f1s) / len(component_f1s), 4)
            self.report.overall_accuracy = self.report.overall_f1

        return self.report

    def save_report(self, output_path: str) -> None:
        """Save evaluation report to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(self.report.to_json())
        logger.info(f"Evaluation report saved to {output_path}")

    def print_summary(self) -> None:
        """Print a human-readable evaluation summary."""
        r = self.report
        print("\n" + "=" * 60)
        print("  📊 PIPELINE EVALUATION REPORT")
        print("=" * 60)
        print(f"  Dataset:     {r.dataset_name}")
        print(f"  Documents:   {r.total_documents}")
        print(f"  Total Time:  {r.total_processing_time_seconds}s")
        print(f"  Avg Time:    {r.avg_processing_time_seconds}s/doc")
        print("-" * 60)

        print("\n  📝 Text Extraction:")
        m = r.text_metrics
        print(f"     Precision:    {m.precision:.1%}")
        print(f"     Recall:       {m.recall:.1%}")
        print(f"     F1 Score:     {m.f1_score:.1%}")
        print(f"     Char Acc:     {m.character_accuracy:.1%}")

        print("\n  📊 Table Extraction:")
        t = r.table_metrics
        print(f"     Cell F1:      {t.cell_f1:.1%}")
        print(f"     Structure:    {t.structure_accuracy:.1%}")
        print(f"     Row Acc:      {t.row_count_accuracy:.1%}")
        print(f"     Col Acc:      {t.column_count_accuracy:.1%}")

        print("\n  📐 Layout Detection:")
        l = r.layout_metrics
        print(f"     Mean IoU:     {l.iou_mean:.1%}")
        print(f"     Region F1:    {l.region_f1:.1%}")

        print("\n  ⚡ Conflict Resolution:")
        c = r.conflict_metrics
        print(f"     Detected:     {c.conflicts_detected}")
        print(f"     Resolved:     {c.conflicts_resolved}")
        print(f"     Accuracy:     {c.resolution_accuracy:.1%}")

        print("\n" + "-" * 60)
        print(f"  🏆 OVERALL F1 SCORE:  {r.overall_f1:.1%}")
        print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def main():
    """Run evaluation with sample data for demonstration."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Document Intelligence Pipeline")
    parser.add_argument(
        "--dataset",
        default="benchmarks/sample_dataset/ground_truth.json",
        help="Path to ground truth dataset JSON",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/reports/evaluation_report.json",
        help="Path to save the evaluation report",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    evaluator = PipelineEvaluator()
    evaluator.load_ground_truth(args.dataset)

    # Simulated pipeline outputs for demonstration
    # In production, these would come from running the actual pipeline
    sample_outputs = [
        {
            "document_id": "sample_001",
            "extracted_text": "Annual Financial Report 2025. Revenue increased by 15% to $2.4 billion. Net income was $340 million, up from $290 million in the prior year.",
            "tables": [
                [
                    ["Metric", "2024", "2025"],
                    ["Revenue", "$2.1B", "$2.4B"],
                    ["Net Income", "$290M", "$340M"],
                    ["Employees", "12,500", "14,200"],
                ]
            ],
            "regions": [
                {"type": "title", "bbox": [50, 20, 550, 70]},
                {"type": "paragraph", "bbox": [50, 80, 550, 200]},
                {"type": "table", "bbox": [50, 220, 550, 420]},
            ],
            "conflicts": [
                {"type": "ocr_vs_vision", "resolved": True, "confidence_before": 0.65, "confidence_after": 0.92}
            ],
            "processing_time": 3.2,
        },
        {
            "document_id": "sample_002",
            "extracted_text": "Patient intake form. Name: Jane Smith. Date of Birth: 1985-03-15. Insurance ID: BCB-9912-XY.",
            "tables": [],
            "regions": [
                {"type": "header", "bbox": [30, 10, 570, 50]},
                {"type": "paragraph", "bbox": [30, 60, 570, 300]},
            ],
            "conflicts": [],
            "processing_time": 1.8,
        },
        {
            "document_id": "sample_003",
            "extracted_text": "Technical Specification v2.1. The system operates at 99.9% uptime with auto-scaling from 2 to 16 cores.",
            "tables": [
                [
                    ["Component", "Specification"],
                    ["CPU", "16 cores"],
                    ["Memory", "64 GB"],
                    ["Storage", "2 TB NVMe"],
                ]
            ],
            "regions": [
                {"type": "title", "bbox": [40, 15, 560, 55]},
                {"type": "paragraph", "bbox": [40, 65, 560, 180]},
                {"type": "table", "bbox": [40, 200, 560, 380]},
            ],
            "conflicts": [
                {"type": "table_structure", "resolved": True, "confidence_before": 0.58, "confidence_after": 0.89}
            ],
            "processing_time": 2.5,
        },
    ]

    report = evaluator.evaluate(sample_outputs)
    evaluator.print_summary()
    evaluator.save_report(args.output)
    print(f"  ✅ Report saved to: {args.output}\n")


if __name__ == "__main__":
    main()
