"""Document parsing, lifecycle, and project-isolated vector retrieval tests."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from intelligent_agents_chat.database import ChatRepository
from intelligent_agents_chat.documents import (
    BlobStore,
    Chunker,
    DocumentParser,
    DocumentService,
    DocumentStore,
    DocumentValidationError,
    ProjectRAGRetriever,
)
from intelligent_agents_chat.embeddings import EMBEDDING_DIMENSION, EmbeddingError
from intelligent_agents_chat.retrieval import RetrievalQuery


class SemanticFakeEmbeddingGateway:
    """A fake standing in for the real local model -- fast, deterministic,
    no ONNX inference or model download in unit tests. Vectors are chosen so
    unrelated topics land far apart in cosine space, exactly like a real
    embedding model would, just without actually understanding language.
    """

    model_name = "test-embedding"

    async def embed(self, texts):
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        # One-hot in the first 3 dimensions, padded with zeros to
        # document_chunks_vec's fixed EMBEDDING_DIMENSION (see documents.py) --
        # the schema itself only ever accepts that exact dimension now.
        normalized = text.casefold()
        if "orchid" in normalized or "flower" in normalized:
            head = [1.0, 0.0, 0.0]
        elif "finance" in normalized or "budget" in normalized:
            head = [0.0, 1.0, 0.0]
        else:
            head = [0.0, 0.0, 1.0]
        return head + [0.0] * (EMBEDDING_DIMENSION - len(head))


class FlakyEmbeddingGateway(SemanticFakeEmbeddingGateway):
    def __init__(self) -> None:
        self.fail = True

    async def embed(self, texts):
        if self.fail:
            self.fail = False
            raise EmbeddingError("temporary embedding outage")
        return await super().embed(texts)


class DocumentRAGTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.database_path = root / "chats.sqlite3"
        self.repository = ChatRepository(self.database_path)
        self.repository.initialize()
        self.store = DocumentStore(self.database_path)
        self.store.initialize()
        self.blob_store = BlobStore(root / "documents")
        self.embedding_gateway = SemanticFakeEmbeddingGateway()
        self.service = DocumentService(
            self.store,
            self.blob_store,
            DocumentParser(),
            Chunker(chunk_chars=240, overlap_chars=40),
            self.embedding_gateway,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    async def test_upload_duplicate_and_delete_have_deterministic_lifecycle(self) -> None:
        project = self.repository.create_project("Research")
        first = await self.service.upload(
            project_id=project.id,
            display_name="notes.txt",
            media_type="text/plain",
            data=b"The orchid launch code is amber.",
        )

        self.assertFalse(first.duplicate)
        self.assertEqual(first.document.status, "indexed")
        self.assertEqual(first.document.embedding_model, "test-embedding")
        self.assertTrue(self.blob_store.exists(first.document.storage_key))
        self.assertGreater(first.document.chunk_count, 0)

        # Same bytes under a different filename: recognised by content hash
        # (see DocumentService.upload), so it isn't embedded a second time.
        duplicate = await self.service.upload(
            project_id=project.id,
            display_name="renamed.txt",
            media_type="text/plain",
            data=b"The orchid launch code is amber.",
        )
        self.assertTrue(duplicate.duplicate)
        self.assertEqual(duplicate.document.id, first.document.id)
        self.assertEqual(len(self.store.list_documents(project.id)), 1)

        document_id = first.document.id
        self.assertTrue(await self.service.delete(document_id, project.id))
        self.assertIsNone(self.store.get_document(document_id))
        self.assertEqual(self.store.list_chunks(document_id), [])
        self.assertFalse(self.blob_store.exists(first.document.storage_key))

    async def test_retrieval_never_crosses_project_boundaries_and_has_citations(self) -> None:
        public = self.repository.create_project("Public")
        private = self.repository.create_project("Private")
        public_upload = await self.service.upload(
            project_id=public.id,
            display_name="orchids.txt",
            media_type="text/plain",
            data=b"The orchid launch code is amber.",
        )
        await self.service.upload(
            project_id=private.id,
            display_name="secret.txt",
            media_type="text/plain",
            data=b"The orchid launch code is violet.",
        )

        retriever = ProjectRAGRetriever(self.store, self.embedding_gateway)
        results = await retriever.retrieve(
            RetrievalQuery(project_id=public.id, text="What is the flower launch code?")
        )

        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all(result.project_id == public.id for result in results))
        self.assertTrue(all("violet" not in result.text for result in results))
        self.assertEqual(results[0].source_kind, "project_document")
        self.assertEqual(results[0].title, "orchids.txt")
        self.assertEqual(results[0].locator, "document")
        self.assertEqual(results[0].source_conversation_id, None)
        self.assertIn(
            results[0].source_id,
            {
                f"{chunk.document_id}:{chunk.ordinal}"
                for chunk in self.store.list_chunks(public_upload.document.id)
            },
        )

    async def test_failed_embedding_is_visible_and_retryable(self) -> None:
        project = self.repository.create_project("Retries")
        flaky = FlakyEmbeddingGateway()
        service = DocumentService(
            self.store,
            self.blob_store,
            DocumentParser(),
            Chunker(),
            flaky,
        )

        with self.assertRaisesRegex(EmbeddingError, "temporary embedding outage"):
            await service.upload(
                project_id=project.id,
                display_name="retry.txt",
                media_type="text/plain",
                data=b"An orchid fact that should survive a temporary outage.",
            )

        failed = self.store.list_documents(project.id)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].status, "failed")
        self.assertIn("temporary embedding outage", failed[0].error_message or "")
        self.assertTrue(self.blob_store.exists(failed[0].storage_key))

        indexed = await service.reindex(failed[0].id, project.id)
        self.assertEqual(indexed.status, "indexed")
        self.assertIsNone(indexed.error_message)
        self.assertGreater(indexed.chunk_count, 0)


class DocumentParserTests(unittest.TestCase):
    def test_chunking_is_stable_and_overlapping(self) -> None:
        chunker = Chunker(chunk_chars=200, overlap_chars=40)
        segments = DocumentParser().parse(
            ("alpha beta gamma delta " * 40).encode(),
            display_name="long.txt",
            media_type="text/plain",
        )

        first = chunker.chunk(segments)
        second = chunker.chunk(segments)

        self.assertGreater(len(first), 1)
        # Same input, same chunks -- a re-index can't silently reshuffle what
        # was already indexed (see DocumentService.reindex).
        self.assertEqual(
            [(chunk.ordinal, chunk.content, chunk.locator) for chunk in first],
            [(chunk.ordinal, chunk.content, chunk.locator) for chunk in second],
        )
        self.assertTrue(set(first[0].content.split()).intersection(first[1].content.split()))

    def test_markdown_sections_and_pdf_pages_preserve_locators(self) -> None:
        parser = DocumentParser()
        markdown = parser.parse(
            b"# Architecture\nUse SQLite.\n\n## Retrieval\nUse embeddings.",
            display_name="design.md",
            media_type="text/markdown",
        )
        pdf = parser.parse(
            _pdf_with_text("Evidence on page one"),
            display_name="evidence.pdf",
            media_type="application/pdf",
        )

        self.assertEqual(
            [segment.locator for segment in markdown],
            ["section: Architecture", "section: Retrieval"],
        )
        self.assertEqual(pdf[0].locator, "page 1")
        self.assertIn("Evidence on page one", pdf[0].text)

    def test_rejects_mismatched_media_type_for_pdf(self) -> None:
        parser = DocumentParser()
        with self.assertRaisesRegex(DocumentValidationError, "Unsupported media type for PDF"):
            parser.parse(
                _pdf_with_text("irrelevant"),
                display_name="evidence.pdf",
                media_type="application/zip",
            )

    def test_rejects_unsupported_extension_and_empty_scanned_pdf(self) -> None:
        parser = DocumentParser()
        with self.assertRaisesRegex(DocumentValidationError, "Supported document types"):
            parser.parse(b"binary", display_name="image.png", media_type="image/png")

        writer = PdfWriter()
        writer.add_blank_page(width=300, height=300)
        output = BytesIO()
        writer.write(output)
        with self.assertRaisesRegex(DocumentValidationError, "no extractable text"):
            parser.parse(
                output.getvalue(),
                display_name="scan.pdf",
                media_type="application/pdf",
            )


def _pdf_with_text(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    resources = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    page[NameObject("/Resources")] = resources
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 200 Td ({escaped}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


if __name__ == "__main__":
    unittest.main()
