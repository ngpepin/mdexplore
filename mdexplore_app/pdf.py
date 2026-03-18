"""PDF utilities and error formatting helpers for mdexplore."""

from __future__ import annotations

import re
from io import BytesIO


def extract_plantuml_error_details(stderr_text: str) -> str:
    """Parse PlantUML stderr into a readable, more detailed message."""
    raw = (stderr_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    if not lines:
        return "unknown error"

    if len(lines) >= 3 and lines[0].upper() == "ERROR" and lines[1].isdigit():
        return f"line {lines[1]}: {lines[2]}"

    return "\n".join(lines[:8])


def stamp_pdf_page_numbers(
    pdf_bytes: bytes, layout_hints: dict[str, object] | None = None
) -> bytes:
    """Overlay centered `N of M` footers on every page of a PDF payload."""
    if not pdf_bytes:
        raise ValueError("Empty PDF payload")

    try:
        from pypdf import PageObject, PdfReader, PdfWriter, Transformation
    except Exception as exc:
        raise RuntimeError("Missing dependency 'pypdf' for PDF page numbering") from exc

    try:
        from reportlab.pdfgen import canvas
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'reportlab' for PDF page numbering"
        ) from exc

    reader = PdfReader(BytesIO(pdf_bytes))
    layout_hints = layout_hints if isinstance(layout_hints, dict) else {}

    def normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip().casefold()

    landscape_headings = {
        normalize_text(item)
        for item in (layout_hints.get("landscapeHeadings") or [])
        if str(item or "").strip()
    }
    diagram_headings = {
        normalize_text(item)
        for item in (layout_hints.get("diagramHeadings") or [])
        if str(item or "").strip()
    }

    def page_text(page) -> str:
        try:
            return page.extract_text() or ""
        except Exception:
            return ""

    raw_page_texts = [page_text(page) for page in reader.pages]
    normalized_page_texts = [normalize_text(text) for text in raw_page_texts]
    landscape_token_literal = "__MDEXPLORE_LANDSCAPE_PAGE__"
    landscape_token = normalize_text(landscape_token_literal)
    landscape_token_compact = re.sub(r"\s+", "", landscape_token)
    landscape_token_pattern = re.compile(
        "".join(re.escape(char) + r"\s*" for char in landscape_token_literal),
        re.IGNORECASE,
    )

    def contains_landscape_token(raw_text: str) -> bool:
        normalized = normalize_text(raw_text)
        if landscape_token in normalized:
            return True
        collapsed = re.sub(r"\s+", "", normalized)
        return landscape_token_compact in collapsed
    token_page_flags = [
        contains_landscape_token(raw_text) for raw_text in raw_page_texts
    ]
    landscape_heading_page_flags = [
        any(heading and heading in page_text for heading in landscape_headings)
        for page_text in normalized_page_texts
    ]
    diagram_page_flags = [
        any(heading and heading in page_text for heading in diagram_headings)
        for page_text in normalized_page_texts
    ]
    landscape_flags = list(token_page_flags)
    for page_index, heading_hit in enumerate(landscape_heading_page_flags):
        if not heading_hit:
            continue
        landscape_flags[page_index] = True
        previous_index = page_index - 1
        if (
            previous_index >= 0
            and token_page_flags[previous_index]
            and not landscape_heading_page_flags[previous_index]
            and not diagram_page_flags[previous_index]
        ):
            # Chromium can leak the invisible landscape token onto the
            # preceding prose page while the real heading+diagram stays on the
            # next page. Prefer the page that actually contains the landscape
            # heading/diagram block.
            landscape_flags[previous_index] = False

    def raster_page_bounds() -> list[tuple[float, float, float, float] | None]:
        if not reader.pages:
            return []
        try:
            import shutil
            import subprocess
            import tempfile
            from pathlib import Path
            from PIL import Image
        except Exception:
            return [None] * len(reader.pages)

        if shutil.which("pdftoppm") is None:
            return [None] * len(reader.pages)

        try:
            with tempfile.TemporaryDirectory(
                prefix="mdexplore-pdf-bounds-"
            ) as tmpdir_str:
                tmpdir = Path(tmpdir_str)
                pdf_path = tmpdir / "input.pdf"
                prefix = tmpdir / "page"
                pdf_path.write_bytes(pdf_bytes)
                result = subprocess.run(
                    ["pdftoppm", "-png", "-r", "72", str(pdf_path), str(prefix)],
                    capture_output=True,
                    check=False,
                    timeout=60,
                )
                if result.returncode != 0:
                    return [None] * len(reader.pages)

                bounds_by_index: list[tuple[float, float, float, float] | None] = [
                    None
                ] * len(reader.pages)
                image_paths = sorted(tmpdir.glob("page-*.png"))
                for image_path in image_paths:
                    match = re.search(r"-(\d+)\.png$", image_path.name)
                    if not match:
                        continue
                    page_index = int(match.group(1)) - 1
                    if page_index < 0 or page_index >= len(reader.pages):
                        continue
                    image = Image.open(image_path).convert("RGB")
                    width_px, height_px = image.size
                    if width_px <= 0 or height_px <= 0:
                        continue
                    pixels = image.load()
                    minx, miny = width_px, height_px
                    maxx = maxy = -1
                    for y in range(height_px):
                        for x in range(width_px):
                            r, g, b = pixels[x, y]
                            if r < 247 or g < 247 or b < 247:
                                if x < minx:
                                    minx = x
                                if y < miny:
                                    miny = y
                                if x > maxx:
                                    maxx = x
                                if y > maxy:
                                    maxy = y
                    if maxx < 0 or maxy < 0:
                        continue
                    pad_x = max(4, int((maxx - minx + 1) * 0.025))
                    pad_y = max(4, int((maxy - miny + 1) * 0.025))
                    minx = max(0, minx - pad_x)
                    miny = max(0, miny - pad_y)
                    maxx = min(width_px - 1, maxx + pad_x)
                    maxy = min(height_px - 1, maxy + pad_y)
                    page = reader.pages[page_index]
                    page_width = float(page.mediabox.width)
                    page_height = float(page.mediabox.height)
                    crop_left = (minx / width_px) * page_width
                    crop_right = ((maxx + 1) / width_px) * page_width
                    crop_top = (miny / height_px) * page_height
                    crop_bottom = ((maxy + 1) / height_px) * page_height
                    lower_y = max(0.0, page_height - crop_bottom)
                    upper_y = min(page_height, page_height - crop_top)
                    bounds_by_index[page_index] = (
                        crop_left,
                        lower_y,
                        crop_right,
                        upper_y,
                    )
                return bounds_by_index
        except Exception:
            return [None] * len(reader.pages)

    crop_bounds_by_page = raster_page_bounds()

    def page_has_meaningful_content(page, extracted_text: str) -> bool:
        text = extracted_text.strip()
        if text:
            return True

        try:
            annotations = page.get("/Annots")
        except Exception:
            annotations = None
        if annotations:
            return True

        try:
            resources = page.get("/Resources")
        except Exception:
            resources = None
        if resources:
            try:
                xobjects = resources.get("/XObject")
            except Exception:
                xobjects = None
            if xobjects:
                try:
                    if len(xobjects) > 0:
                        return True
                except Exception:
                    return True

        try:
            contents = page.get_contents()
        except Exception:
            contents = None
        if contents is None:
            return False

        streams = contents if isinstance(contents, list) else [contents]
        paint_tokens = (
            " Do",
            " S",
            " s",
            " f",
            " f*",
            " F",
            " B",
            " B*",
            " b",
            " b*",
            " sh",
            " Tj",
            " TJ",
            " '",
            ' "',
        )
        for stream in streams:
            try:
                raw = stream.get_data()
            except Exception:
                continue
            if not raw:
                continue
            text_stream = raw.decode("latin-1", errors="ignore")
            if any(token in text_stream for token in paint_tokens):
                return True

        return False

    base_page_records = [
        {
            "page": page,
            "is_landscape_page": landscape,
            "is_diagram_page": is_diagram_page,
            "crop_bounds": crop_bounds,
            "raw_text": extracted_text,
        }
        for page, landscape, is_diagram_page, crop_bounds, extracted_text in zip(
            reader.pages,
            landscape_flags,
            diagram_page_flags,
            crop_bounds_by_page,
            raw_page_texts,
        )
        if page_has_meaningful_content(page, extracted_text)
    ]
    if not base_page_records:
        base_page_records = [
            {
                "page": page,
                "is_landscape_page": landscape,
                "is_diagram_page": is_diagram_page,
                "crop_bounds": crop_bounds,
                "raw_text": "",
            }
            for page, landscape, is_diagram_page, crop_bounds in zip(
                reader.pages,
                landscape_flags,
                diagram_page_flags,
                crop_bounds_by_page,
            )
        ]

    def extract_orphan_landscape_heading(raw_text: str) -> str:
        def compact_spaced_heading_text(text: str) -> str:
            tokens = text.split()
            compacted: list[str] = []
            run: list[str] = []

            def flush_run() -> None:
                if not run:
                    return
                compacted.append("".join(run))
                run.clear()

            for token in tokens:
                if len(token) == 1 and token.isalnum():
                    run.append(token)
                    continue
                if token == "." and run:
                    run.append(token)
                    flush_run()
                    continue
                flush_run()
                compacted.append(token)

            flush_run()
            compacted_text = " ".join(compacted)
            compacted_text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", compacted_text)
            return re.sub(r"\s+", " ", compacted_text).strip()

        heading_lines: list[str] = []
        for raw_line in str(raw_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            normalized = normalize_text(line)
            if not normalized:
                continue
            if contains_landscape_token(line):
                continue
            if normalized == "mdexplore landscape page":
                continue
            if re.fullmatch(r"\d+\s+of\s+\d+", normalized):
                continue
            cleaned_line = landscape_token_pattern.sub("", line)
            cleaned_line = re.sub(r"\b\d+\s+of\s+\d+\b", "", cleaned_line, flags=re.IGNORECASE)
            cleaned_line = compact_spaced_heading_text(cleaned_line)
            if cleaned_line:
                heading_lines.append(cleaned_line)
        return " ".join(heading_lines).strip()

    kept_page_records: list[dict[str, object]] = []
    page_index = 0
    while page_index < len(base_page_records):
        record = base_page_records[page_index]
        attached_heading_text = ""
        next_index = page_index + 1
        if (
            next_index < len(base_page_records)
            and contains_landscape_token(str(record.get("raw_text") or ""))
        ):
            next_record = base_page_records[next_index]
            heading_text = extract_orphan_landscape_heading(record.get("raw_text") or "")
            heading_word_count = len(re.findall(r"\w+", heading_text))
            if 0 < heading_word_count <= 12 and heading_text:
                merged_record = dict(next_record)
                merged_record["attached_heading_text"] = heading_text
                kept_page_records.append(merged_record)
                page_index += 2
                continue
        merged_record = dict(record)
        merged_record["attached_heading_text"] = attached_heading_text
        kept_page_records.append(merged_record)
        page_index += 1

    page_total = len(kept_page_records)
    if page_total <= 0:
        raise RuntimeError("Generated PDF has no pages")

    def estimate_majority_font_size() -> float:
        size_counts: dict[float, int] = {}
        for record in kept_page_records[
            : min(5, page_total)
        ]:
            page = record["page"]
            try:
                contents = page.get_contents()
            except Exception:
                contents = None
            if contents is None:
                continue

            streams = contents if isinstance(contents, list) else [contents]
            for stream in streams:
                try:
                    raw = stream.get_data()
                except Exception:
                    continue
                if not raw:
                    continue
                text = raw.decode("latin-1", errors="ignore")
                for match in re.finditer(r"([0-9]+(?:\\.[0-9]+)?)\\s+Tf\\b", text):
                    try:
                        size = float(match.group(1))
                    except Exception:
                        continue
                    if 6.0 <= size <= 24.0:
                        bucket = round(size * 2.0) / 2.0
                        size_counts[bucket] = size_counts.get(bucket, 0) + 1

        if not size_counts:
            return 10.5
        return max(
            size_counts.items(), key=lambda item: (item[1], -abs(item[0] - 11.0))
        )[0]

    def estimate_page_diagram_font_size(page) -> float:
        try:
            contents = page.get_contents()
        except Exception:
            contents = None
        if contents is None:
            return 0.0

        candidate_sizes: list[float] = []
        streams = contents if isinstance(contents, list) else [contents]
        for stream in streams:
            try:
                raw = stream.get_data()
            except Exception:
                continue
            if not raw:
                continue
            text = raw.decode("latin-1", errors="ignore")
            for match in re.finditer(r"([0-9]+(?:\\.[0-9]+)?)\\s+Tf\\b", text):
                try:
                    size = float(match.group(1))
                except Exception:
                    continue
                if 6.0 <= size <= 20.0:
                    candidate_sizes.append(size)
        if not candidate_sizes:
            return 0.0
        return max(candidate_sizes)

    base_body_font_size = estimate_majority_font_size()

    def page_layout_bands(
        *, page_width: float, page_height: float, is_landscape_page: bool
    ) -> tuple[float, float, float]:
        if is_landscape_page:
            side_margin = max(18.0, min(page_width * 0.05, base_body_font_size * 2.2))
            top_margin = max(22.0, min(page_height * 0.07, base_body_font_size * 2.7))
            footer_band_height = max(
                34.0, min(page_height * 0.11, base_body_font_size * 3.5)
            )
            return side_margin, top_margin, footer_band_height

        side_margin = max(34.0, min(page_width * 0.12, base_body_font_size * 4.2))
        top_margin = max(30.0, min(page_height * 0.10, base_body_font_size * 3.8))
        footer_band_height = max(
            42.0, min(page_height * 0.16, base_body_font_size * 4.4)
        )
        return side_margin, top_margin, footer_band_height

    def max_page_content_scale(page, *, is_landscape_page: bool, is_diagram_page: bool) -> float:
        if not (is_diagram_page or is_landscape_page):
            return 1.0

        page_diagram_font_size = estimate_page_diagram_font_size(page)
        if page_diagram_font_size <= 0:
            return 2.75 if is_landscape_page else 1.8

        target_font_size = 30.0 if is_landscape_page else 16.0
        return max(1.0, min(2.75, target_font_size / page_diagram_font_size))

    writer = PdfWriter()
    for page_number, (
        page,
        is_landscape_page,
        is_diagram_page,
        crop_bounds,
        attached_heading_text,
    ) in enumerate(
        (
            (
                record["page"],
                bool(record["is_landscape_page"]),
                bool(record["is_diagram_page"]),
                record["crop_bounds"],
                str(record.get("attached_heading_text") or ""),
            )
            for record in kept_page_records
        ),
        start=1,
    ):
        source_width = float(page.mediabox.width)
        source_height = float(page.mediabox.height)
        if source_width <= 0 or source_height <= 0:
            writer.add_page(page)
            continue

        crop_left = crop_bottom = 0.0
        crop_right = source_width
        crop_top = source_height
        if crop_bounds is not None and (is_diagram_page or is_landscape_page):
            crop_left, crop_bottom, crop_right, crop_top = crop_bounds
            crop_left = max(0.0, min(source_width - 1.0, crop_left))
            crop_right = max(crop_left + 1.0, min(source_width, crop_right))
            crop_bottom = max(0.0, min(source_height - 1.0, crop_bottom))
            crop_top = max(crop_bottom + 1.0, min(source_height, crop_top))

        content_source_width = crop_right - crop_left
        content_source_height = crop_top - crop_bottom

        page_width = source_height if is_landscape_page else source_width
        page_height = source_width if is_landscape_page else source_height

        side_margin, top_margin, footer_band_height = page_layout_bands(
            page_width=page_width,
            page_height=page_height,
            is_landscape_page=is_landscape_page,
        )
        attached_heading_font_size = 0.0
        attached_heading_block_height = 0.0
        if attached_heading_text:
            attached_heading_font_size = max(
                16.0, min(24.0, base_body_font_size * 1.95)
            )
            attached_heading_block_height = attached_heading_font_size * 1.85
        content_box_width = max(72.0, page_width - (2.0 * side_margin))
        content_box_height = max(
            72.0,
            page_height
            - top_margin
            - footer_band_height
            - attached_heading_block_height,
        )

        max_safe_scale = max_page_content_scale(
            page,
            is_landscape_page=is_landscape_page,
            is_diagram_page=is_diagram_page,
        )
        content_scale = min(
            max_safe_scale,
            content_box_width / content_source_width,
            content_box_height / content_source_height,
        )
        content_width = content_source_width * content_scale
        content_height = content_source_height * content_scale
        content_translate_x = side_margin + max(
            0.0, (content_box_width - content_width) / 2.0
        )
        content_translate_y = footer_band_height + max(
            0.0, (content_box_height - content_height) / 2.0
        )

        composed_page = PageObject.create_blank_page(
            width=page_width, height=page_height
        )
        transform = (
            Transformation()
            .scale(content_scale, content_scale)
            .translate(
                content_translate_x - (crop_left * content_scale),
                content_translate_y - (crop_bottom * content_scale),
            )
        )
        composed_page.merge_transformed_page(page, transform, over=True)

        footer_font_size = max(8.0, min(14.0, base_body_font_size * content_scale))
        footer_baseline_y = max(12.0, (footer_band_height - footer_font_size) / 2.0)

        overlay_buffer = BytesIO()
        footer_canvas = canvas.Canvas(
            overlay_buffer, pagesize=(page_width, page_height)
        )
        if attached_heading_text:
            footer_canvas.setFont("Helvetica-Bold", attached_heading_font_size)
            heading_baseline_y = max(
                footer_band_height + content_box_height + (attached_heading_font_size * 0.55),
                page_height - top_margin - attached_heading_font_size,
            )
            footer_canvas.drawString(
                side_margin, heading_baseline_y, attached_heading_text
            )
        footer_canvas.setFont("Helvetica", footer_font_size)
        footer_text = f"{page_number} of {page_total}"
        footer_width = footer_canvas.stringWidth(
            footer_text, "Helvetica", footer_font_size
        )
        x = max(0.0, (page_width - footer_width) / 2.0)
        footer_canvas.drawString(x, footer_baseline_y, footer_text)
        footer_canvas.save()

        overlay_buffer.seek(0)
        overlay_pdf = PdfReader(overlay_buffer)
        if overlay_pdf.pages:
            composed_page.merge_page(overlay_pdf.pages[0])
        writer.add_page(composed_page)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()
