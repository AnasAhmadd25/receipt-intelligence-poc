from app.providers.ocr import group_detection_rows


def test_row_grouping_pairs_prices_despite_skew_and_preserves_boxes():
    boxes = [
        {"text": "TOTAL", "bbox": [10, 51, 80, 72], "confidence": 98},
        {"text": "19.000", "bbox": [110, 13, 160, 33], "confidence": 96},
        {"text": "Coffee", "bbox": [10, 10, 80, 30], "confidence": 99},
        {"text": "20.000", "bbox": [110, 50, 160, 70], "confidence": 97},
    ]
    rows = group_detection_rows(boxes)
    assert [r["text"] for r in rows] == ["Coffee | 19.000", "TOTAL | 20.000"]
    assert sum(len(r["boxes"]) for r in rows) == len(boxes)
    assert boxes[1]["text"] == "19.000"


def test_arabic_row_order_and_adjacent_rows_remain_distinct():
    boxes = [
        {"text": "١٢", "bbox": [10, 10, 25, 20], "confidence": 90},
        {"text": "حليب", "bbox": [80, 11, 100, 21], "confidence": 90},
        {"text": "خبز", "bbox": [80, 23, 100, 33], "confidence": 90},
    ]
    rows = group_detection_rows(boxes)
    assert [r["text"] for r in rows] == ["حليب | ١٢", "خبز"]
