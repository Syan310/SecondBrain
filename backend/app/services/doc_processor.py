"""Document processor: parse PDF/Word/Markdown → structured parent/child chunks.

Pipeline:
  1. Parse file to Markdown (MinerU for PDF, python-docx for Word, direct read for .md)
  2. Two-pass chunking:
     - First pass: find the top heading level, split into parent chunks
     - Second pass: within each parent, split into children by sub-headings + char limit
  3. Extract concepts from headings for knowledge_concepts table
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


# ── Data structures ──────────────────────────────────────────


@dataclass
class ChildChunk:
    """A retrieval unit, embedded and stored in Milvus."""

    text: str
    parent_id: str
    heading_path: str
    chunk_index: int


@dataclass
class ParentChunk:
    """A context unit, provided to LLM alongside matched child chunk."""

    id: str
    heading: str
    heading_path: str
    full_text: str
    children: list[ChildChunk] = field(default_factory=list)


@dataclass
class ProcessedDocument:
    """Result of processing a single document."""

    source: str
    parents: list[ParentChunk]


# ── Processor ────────────────────────────────────────────────


class DocProcessor:
    """Parse documents and split into hierarchical chunks."""

    def __init__(self, max_child_chars: int = 750, overlap_chars: int = 75):
        """
        Args:
            max_child_chars: Max characters per child chunk (≈512 tokens for Chinese).
            overlap_chars: Overlap when force-splitting oversized text (≈50 tokens).
        """
        self.max_child_chars = max_child_chars
        self.overlap_chars = overlap_chars

    def process_file(self, file_path: str) -> ProcessedDocument:
        """Main entry: parse file → chunk → extract concepts."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        parsers = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_word,
            ".doc": self._parse_word,
            ".md": self._parse_markdown,
            ".markdown": self._parse_markdown,
        }

        parser = parsers.get(suffix)
        if parser is None:
            raise ValueError(f"Unsupported file type: {suffix}")

        logger.info("Parsing %s ...", path.name)
        markdown = parser(str(path))

        logger.info("Chunking %s ...", path.name)
        parents = self._chunk_markdown(markdown)

        logger.info(
            "Done: %s → %d parents, %d children",
            path.name,
            len(parents),
            sum(len(p.children) for p in parents),
        )
        return ProcessedDocument(source=path.name, parents=parents)

    # ── Parsers ──────────────────────────────────────────────

    def _parse_pdf(self, file_path: str) -> str:
        """Parse PDF to Markdown via MinerU."""
        import tempfile

        from magic_pdf.config.enums import SupportedPdfParseMethod
        from magic_pdf.data.data_reader_writer import (
            FileBasedDataReader,
            FileBasedDataWriter,
        )
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(file_path)
        ds = PymuDocDataset(pdf_bytes)

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_writer = FileBasedDataWriter(tmp_dir)

            if ds.classify() == SupportedPdfParseMethod.OCR:
                infer_result = ds.apply(doc_analyze, ocr=True)
                pipe_result = infer_result.pipe_ocr_mode(image_writer)
            else:
                infer_result = ds.apply(doc_analyze, ocr=False)
                pipe_result = infer_result.pipe_txt_mode(image_writer)

            md_content = pipe_result.pipe_mk_markdown(tmp_dir)

        return md_content

    def _parse_word(self, file_path: str) -> str:
        """Parse Word (.docx) to Markdown via python-docx."""
        from docx import Document
        from docx.oxml.ns import qn
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = Document(file_path)
        lines: list[str] = []

        for element in doc.element.body:
            tag = element.tag

            if tag == qn("w:p"):
                para = Paragraph(element, doc)
                text = para.text.strip()
                if not text:
                    continue
                style_name = para.style.name if para.style else ""
                if style_name.startswith("Heading"):
                    try:
                        level = int(style_name.split()[-1])
                        lines.append(f"{'#' * level} {text}")
                    except (ValueError, IndexError):
                        lines.append(text)
                else:
                    lines.append(text)

            elif tag == qn("w:tbl"):
                table = Table(element, doc)
                lines.append(self._table_to_markdown(table))

        return "\n\n".join(lines)

    @staticmethod
    def _table_to_markdown(table) -> str:
        """Convert a python-docx Table to Markdown table string."""
        rows: list[str] = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append("| " + " | ".join(cells) + " |")
        if rows:
            col_count = len(table.rows[0].cells)
            separator = "| " + " | ".join(["---"] * col_count) + " |"
            rows.insert(1, separator)
        return "\n".join(rows)

    @staticmethod
    def _parse_markdown(file_path: str) -> str:
        """Read Markdown file directly."""
        return Path(file_path).read_text(encoding="utf-8")

    # ── Chunking ─────────────────────────────────────────────

    def _chunk_markdown(self, markdown: str) -> list[ParentChunk]:
        """Two-pass split: top heading → parent, sub-sections + char limit → children."""
        sections = self._parse_sections(markdown)

        if not sections:
            return []

        # Find the top (smallest number) heading level
        heading_levels = [s["level"] for s in sections if s["level"] > 0]

        if not heading_levels:
            # No headings → single parent, split by size
            parent_id = str(uuid.uuid4())
            text = "\n\n".join(s["content"] for s in sections)
            children = self._split_text(text, parent_id, "")
            return [
                ParentChunk(
                    id=parent_id, heading="", heading_path="",
                    full_text=text, children=children,
                )
            ]

        top_level = min(heading_levels)

        # Group sections by top-level heading
        groups: list[tuple[str, list[dict]]] = []
        cur_heading = ""
        cur_sects: list[dict] = []

        for s in sections:
            if s["level"] == top_level:
                if cur_heading or cur_sects:
                    groups.append((cur_heading, cur_sects))
                cur_heading = s["heading"]
                cur_sects = [s]
            else:
                cur_sects.append(s)

        if cur_heading or cur_sects:
            groups.append((cur_heading, cur_sects))

        # Build ParentChunks
        result: list[ParentChunk] = []

        for parent_heading, sects in groups:
            parent_id = str(uuid.uuid4())
            full_parts: list[str] = []
            children: list[ChildChunk] = []

            for s in sects:
                content = s["content"]
                if not content:
                    continue
                full_parts.append(content)

                # Build heading_path for this sub-section
                sub_heading = s["heading"]
                if sub_heading and sub_heading != parent_heading:
                    h_path = (
                        f"{parent_heading}/{sub_heading}" if parent_heading else sub_heading
                    )
                else:
                    h_path = parent_heading

                new_children = self._split_text(
                    content, parent_id, h_path, start_index=len(children),
                )
                children.extend(new_children)

            if not children:
                continue

            result.append(
                ParentChunk(
                    id=parent_id,
                    heading=parent_heading,
                    heading_path=parent_heading,
                    full_text="\n\n".join(full_parts),
                    children=children,
                )
            )

        return result

    @staticmethod
    def _parse_sections(markdown: str) -> list[dict]:
        """Split markdown into flat list of {level, heading, content} sections."""
        lines = markdown.split("\n")
        sections: list[dict] = []
        current: dict = {"level": 0, "heading": "", "lines": []}

        for line in lines:
            m = _HEADING_RE.match(line.strip())
            if m:
                # Flush current section
                content = "\n".join(current["lines"]).strip()
                if content or current["heading"]:
                    sections.append({
                        "level": current["level"],
                        "heading": current["heading"],
                        "content": content,
                    })
                current = {
                    "level": len(m.group(1)),
                    "heading": m.group(2).strip(),
                    "lines": [],
                }
            else:
                current["lines"].append(line)

        # Flush last section
        content = "\n".join(current["lines"]).strip()
        if content or current["heading"]:
            sections.append({
                "level": current["level"],
                "heading": current["heading"],
                "content": content,
            })

        return sections

    def _split_text(
        self,
        text: str,
        parent_id: str,
        heading_path: str,
        start_index: int = 0,
    ) -> list[ChildChunk]:
        """Split text into child chunks by paragraph boundaries + char limit."""
        text = text.strip()
        if not text:
            return []

        # Small enough → single chunk
        if len(text) <= self.max_child_chars:
            return [
                ChildChunk(
                    text=text, parent_id=parent_id,
                    heading_path=heading_path, chunk_index=start_index,
                )
            ]

        # Merge paragraphs up to limit
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        merged: list[str] = []
        buf = ""

        for para in paragraphs:
            if not buf:
                buf = para
            elif len(buf) + len(para) + 2 <= self.max_child_chars:
                buf = f"{buf}\n\n{para}"
            else:
                merged.append(buf)
                buf = para
        if buf:
            merged.append(buf)

        # Force-split any chunk still oversized (with overlap)
        final: list[str] = []
        for chunk in merged:
            if len(chunk) <= self.max_child_chars:
                final.append(chunk)
            else:
                pos = 0
                while pos < len(chunk):
                    end = min(pos + self.max_child_chars, len(chunk))
                    final.append(chunk[pos:end])
                    if end >= len(chunk):
                        break
                    pos = end - self.overlap_chars

        return [
            ChildChunk(
                text=c, parent_id=parent_id,
                heading_path=heading_path, chunk_index=start_index + i,
            )
            for i, c in enumerate(final)
            if c.strip()
        ]



# Module-level singleton
doc_processor = DocProcessor()
