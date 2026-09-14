"""Document parsing, lifecycle, and project-isolated vector retrieval tests."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import sqlite_vec

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from intelligent_agents_chat.database import (
    ChatRepository,
    ContextRunInput,
    ContextSourceInput,
    connect_database,
)
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
from intelligent_agents_chat.memory import ProjectMemoryStore


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
        await self.service.upload(
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
        question = "What is the flower launch code?"
        results = await retriever.retrieve(project_id=public.id, text=question, limit=5)

        # Both projects hold an equally good match for this query, so the
        # only thing keeping them apart is the project scope itself.
        self.assertGreaterEqual(len(results), 1)
        self.assertTrue(all("violet" not in result.text for result in results))
        self.assertIn("amber", results[0].text)
        self.assertEqual(results[0].title, "orchids.txt")
        self.assertEqual(results[0].locator, "document")

        # ... and the other way round, so this can't pass just because the
        # private document was never indexed in the first place.
        private_results = await retriever.retrieve(project_id=private.id, text=question, limit=5)
        self.assertTrue(all("amber" not in result.text for result in private_results))
        self.assertIn("violet", private_results[0].text)
        self.assertEqual(private_results[0].title, "secret.txt")

    async def test_delete_project_removes_its_records_vectors_and_files_only(self) -> None:
        removed = self.repository.create_project("Remove")
        kept = self.repository.create_project("Keep")
        for project, marker in [(removed, "orchid"), (kept, "budget")]:
            conversation = self.repository.create_conversation(project_id=project.id)
            self.repository.add_message(conversation.id, "user", f"Remember the {marker}.")
            answer = self.repository.add_message(conversation.id, "assistant", f"Saved {marker}.")
            ProjectMemoryStore(self.database_path).rebuild_project(project.id)
            self.repository.add_message_context_sources(
                answer.id, [ContextSourceInput(marker, "notes", marker, 1, 5)]
            )
            self.repository.add_message_context_run(
                answer.id, ContextRunInput(32768, 31744, 50, 45, 10, 55)
            )
            self.repository.set_conversation_compaction(
                conversation.id, compacted_through_message_id=answer.id, summary=marker
            )
            await self.service.upload(
                project_id=project.id,
                display_name="notes.txt",
                media_type="text/plain",
                data=f"Facts about the {marker}.".encode(),
            )
        kept_document = self.store.list_documents(kept.id)[0]
        orphan_key = self.blob_store.storage_key(removed.id, "orphan-upload", ".txt")
        self.blob_store.write(orphan_key, b"A file left by an older upload.")
        settings = self.repository.set_app_settings(
            model_profile="qwen3-8b", thinking_enabled=False, memory_enabled=True
        )
        with connect_database(self.database_path) as connection:
            connection.execute(
                "INSERT INTO document_chunks_vec(row_id, project_id, embedding) VALUES (?, ?, ?)",
                (
                    99999,
                    removed.id,
                    sqlite_vec.serialize_float32(self.embedding_gateway._vector("orchid")),
                ),
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0], 2
            )

        self.assertTrue(self.repository.delete_project(removed.id))
        self.blob_store.delete_project(removed.id)

        self.assertIsNone(self.repository.get_project(removed.id))
        self.assertIsNotNone(self.repository.get_project(kept.id))
        self.assertEqual(self.repository.get_app_settings(), settings)
        self.assertEqual(self.store.list_documents(removed.id), [])
        self.assertFalse((self.blob_store.root / removed.id).exists())
        self.assertTrue(self.blob_store.exists(kept_document.storage_key))
        with connect_database(self.database_path) as connection:
            for table in [
                "conversations",
                "memory_entries",
                "message_context_sources",
                "message_context_runs",
                "conversation_compactions",
                "documents",
                "document_chunks",
                "document_chunks_vec",
            ]:
                with self.subTest(table=table):
                    self.assertEqual(
                        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0], 1
                    )
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 2)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            self.assertEqual(
                connection.execute(
                    "SELECT rowid FROM memory_entries_fts WHERE memory_entries_fts MATCH 'orchid'"
                ).fetchall(),
                [],
            )
        hits = await ProjectRAGRetriever(self.store, self.embedding_gateway).retrieve(
            project_id=kept.id, text="budget", limit=5
        )
        self.assertEqual(len(hits), 1)
        self.assertIn("budget", hits[0].text)
        self.assertFalse(self.repository.delete_project(removed.id))
        self.blob_store.delete_project(removed.id)

    async def test_project_deletion_waits_for_indexing(self) -> None:
        project = self.repository.create_project("Uploading")
        upload = await self.service.upload(
            project_id=project.id,
            display_name="notes.txt",
            media_type="text/plain",
            data=b"An orchid fact.",
        )
        self.store.mark_processing(upload.document.id)
        with self.assertRaisesRegex(ValueError, "Wait for document indexing"):
            self.repository.delete_project(project.id)
        self.assertIsNotNone(self.repository.get_project(project.id))
        self.assertTrue(self.blob_store.exists(upload.document.storage_key))
        with connect_database(self.database_path) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM document_chunks_vec").fetchone()[0], 1
            )

    def test_project_file_deletion_rejects_path_traversal_and_symlinks(self) -> None:
        key = self.blob_store.storage_key("keep", "document", ".txt")
        self.blob_store.write(key, b"Keep this file.")
        for invalid in ["..", ".", "../keep", "/keep", ""]:
            with self.subTest(project_id=invalid), self.assertRaises(DocumentValidationError):
                self.blob_store.delete_project(invalid)
        (self.blob_store.root / "alias").symlink_to(
            self.blob_store.root / "keep", target_is_directory=True
        )
        with self.assertRaises(OSError):
            self.blob_store.delete_project("alias")
        self.assertTrue(self.blob_store.exists(key))

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
