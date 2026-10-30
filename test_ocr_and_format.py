"""Tests for OCR path resolution / loud failures and search-result filenames.

Run:  py -3.12 -m unittest test_ocr_and_format -v   (from the repo root)
OCR tests use the real Tesseract install; no database or network access.
"""
import io, os, tempfile, unittest
from pathlib import Path
from unittest import mock

import ingest_rag_pgvector_v3 as ing
import rag_flask_mcp as mcp

HAVE_TESSERACT = any(Path(c).is_file() for c in ing._TESSERACT_CANDIDATES)


def make_image_pdf(path: Path, text: str):
    """One-page PDF whose only content is a raster image of `text`."""
    import fitz
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (1200, 300), "white")
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except OSError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((40, 100), text, fill="black", font=font)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    doc = fitz.open()
    page = doc.new_page(width=600, height=150)
    page.insert_image(page.rect, stream=buf.getvalue())
    doc.save(str(path)); doc.close()


class ResolveTesseractTests(unittest.TestCase):
    def test_finds_install_when_not_on_path(self):
        if not HAVE_TESSERACT:
            self.skipTest("Tesseract not installed")
        with mock.patch.dict(os.environ, {"PATH": "", "TESSERACT_CMD": ""}), \
             mock.patch("shutil.which", return_value=None):
            cmd = ing.resolve_tesseract_cmd()
        self.assertTrue(Path(cmd).is_file())

    def test_bad_env_override_raises(self):
        with mock.patch.dict(os.environ, {"TESSERACT_CMD": r"C:\nope\tesseract.exe"}):
            with self.assertRaises(ing.OCRError):
                ing.resolve_tesseract_cmd()

    def test_not_found_anywhere_raises(self):
        with mock.patch.dict(os.environ, {"TESSERACT_CMD": ""}), \
             mock.patch("shutil.which", return_value=None), \
             mock.patch.object(ing, "_TESSERACT_CANDIDATES", [r"C:\nope\t.exe"]):
            with self.assertRaises(ing.OCRError):
                ing.resolve_tesseract_cmd()


@unittest.skipUnless(HAVE_TESSERACT, "Tesseract not installed")
class ExtractPdfOcrTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pdf = Path(self.tmp.name) / "scan.pdf"
        make_image_pdf(self.pdf, "Quarterly revenue grew")

    def tearDown(self):
        self.tmp.cleanup()

    def test_image_pdf_is_ocrd_without_path(self):
        self.assertTrue(ing.is_image_based(self.pdf))
        with mock.patch("shutil.which", return_value=None):
            pages = ing.extract_pdf(self.pdf)
        self.assertTrue(pages)
        self.assertIn("revenue", pages[0]["text"].lower())

    def test_missing_tesseract_fails_loudly(self):
        with mock.patch.object(ing, "resolve_tesseract_cmd",
                               side_effect=ing.OCRError("tesseract.exe not found")):
            with self.assertRaises(ing.OCRError):
                ing.extract_pdf_ocr(self.pdf)

    def test_page_error_fails_loudly(self):
        import pytesseract
        with mock.patch.object(pytesseract, "image_to_string",
                               side_effect=RuntimeError("boom")):
            with self.assertRaises(ing.OCRError) as cm:
                ing.extract_pdf_ocr(self.pdf)
        self.assertIn("boom", str(cm.exception))

    def test_process_file_reports_ocr_failure(self):
        with mock.patch.object(ing, "resolve_tesseract_cmd",
                               side_effect=ing.OCRError("tesseract.exe not found")):
            status, n, err = ing.process_file(
                mock.Mock(), self.pdf, "pdf", None, 500, 50)
        self.assertEqual(status, "failed")
        self.assertIn("tesseract.exe not found", err)


class FormatResultsTests(unittest.TestCase):
    def test_hybrid_search_row_shows_filename(self):
        # Shape returned by the hybrid_search() SQL function (rag_search).
        row = {"chunk_id": 1, "source_path": r"C:\vault\20 Wiki\Note A.md",
               "chunk_text": "hello", "heading_path": "Intro",
               "page_number": None, "domain": "ai", "rrf_score": 0.5}
        out = mcp.format_results([row])
        self.assertNotIn("unknown", out)
        self.assertIn("[1] Note A.md | markdown | ", out)
        self.assertIn(r"Path: C:\vault\20 Wiki\Note A.md", out)

    def test_metadata_fallback(self):
        row = {"chunk_text": "x", "rrf_score": 0.1,
               "metadata": {"full_path": r"C:\docs\Book.pdf", "file_type": "pdf"}}
        self.assertIn("[1] Book.pdf | pdf |", mcp.format_results([row]))

    def test_semantic_row_unchanged(self):
        row = {"chunk_text": "x", "rrf_score": 0.1, "file_path": r"C:\d\B.pdf",
               "doc_type": "pdf", "page_number": 3}
        self.assertIn("[1] B.pdf | pdf | p.3 |", mcp.format_results([row]))


if __name__ == "__main__":
    unittest.main()
