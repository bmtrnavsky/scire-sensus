"""Regression tests for the ingest purge-safety guards (2026-10-29).

Run:  py -3.12 -m unittest test_ingest_purge_guard -v   (from the repo root)
No database or network access: the DB connection is mocked.
"""
import tempfile, unittest
from pathlib import Path
from unittest import mock

import ingest_rag_pgvector_v3 as ing


class PlanPurgeTests(unittest.TestCase):
    def setUp(self):
        self.db = ["C:/v/n{}.md".format(i) for i in range(100)]

    def test_subset_never_purges(self):
        to_del, reason = ing.plan_purge(self.db, self.db[:2], subset="Wiki")
        self.assertEqual(to_del, [])
        self.assertIn("subset", reason)

    def test_missing_root_never_purges(self):
        to_del, reason = ing.plan_purge(self.db, self.db[:50],
                                        missing_roots=[Path("C:/dropbox")])
        self.assertEqual(to_del, [])
        self.assertIn("missing", reason)

    def test_small_purge_allowed(self):
        to_del, reason = ing.plan_purge(self.db, self.db[5:])  # 5 of 100 = 5%
        self.assertIsNone(reason)
        self.assertEqual(sorted(to_del), sorted(self.db[:5]))

    def test_mass_purge_aborted(self):
        to_del, reason = ing.plan_purge(self.db, self.db[6:])  # 6 > 5
        self.assertEqual(to_del, [])
        self.assertIn("--force-purge", reason)

    def test_mass_purge_forced(self):
        to_del, reason = ing.plan_purge(self.db, self.db[50:], force=True)
        self.assertIsNone(reason)
        self.assertEqual(len(to_del), 50)

    def test_force_does_not_bypass_subset_or_missing(self):
        self.assertEqual(ing.plan_purge(self.db, [], subset="x", force=True)[0], [])
        self.assertEqual(ing.plan_purge(self.db, [], missing_roots=["r"], force=True)[0], [])

    def test_min_allowed_on_tiny_db(self):
        to_del, reason = ing.plan_purge(["a", "b", "c", "d"], ["d"])  # 3 <= min 3
        self.assertIsNone(reason)
        self.assertEqual(len(to_del), 3)


class CollectDiskFilesTests(unittest.TestCase):
    def test_missing_and_empty_roots_reported(self):
        with tempfile.TemporaryDirectory() as td:
            vault = Path(td, "vault"); vault.mkdir()
            (vault / "a.md").write_text("# hi\nbody")
            empty_pdfs = Path(td, "pdfs"); empty_pdfs.mkdir()
            roots = {"vault": vault, "pdfs": empty_pdfs}
            with mock.patch.object(ing, "SOURCE_ROOTS", roots):
                missing = []
                files = ing.collect_disk_files(None, missing)
                self.assertEqual(len(files), 1)
                self.assertEqual(missing, [empty_pdfs])

            roots = {"vault": vault, "pdfs": Path(td, "gone")}
            with mock.patch.object(ing, "SOURCE_ROOTS", roots):
                missing = []
                ing.collect_disk_files(None, missing)
                self.assertEqual(missing, [Path(td, "gone")])

    def test_subset_does_not_flag_root_empty(self):
        with tempfile.TemporaryDirectory() as td:
            vault = Path(td, "vault"); vault.mkdir()
            (vault / "a.md").write_text("x")
            with mock.patch.object(ing, "SOURCE_ROOTS", {"vault": vault}):
                missing = []
                self.assertEqual(ing.collect_disk_files("nomatch", missing), [])
                self.assertEqual(missing, [])


class IngestEndToEndTests(unittest.TestCase):
    """Drive ingest() with a mocked DB and assert no DELETE on source_files."""

    def _run(self, roots, db_paths, **kw):
        conn = mock.MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = []
        db = {p: {"id": i, "hash": "h", "excluded": False, "chunk_count": 1}
              for i, p in enumerate(db_paths)}
        with mock.patch.object(ing, "SOURCE_ROOTS", roots), \
             mock.patch.object(ing, "get_db_conn", return_value=conn), \
             mock.patch.object(ing, "get_all_db_files", return_value=db), \
             mock.patch.object(ing, "sha256_file", return_value="h"):
            code = ing.ingest(**kw)
        deletes = [c for c in cur.execute.call_args_list
                   if "DELETE FROM source_files" in c.args[0]]
        return code, deletes

    def _vault(self, td, n):
        vault = Path(td, "vault"); vault.mkdir()
        for i in range(n):
            (vault / "n{}.md".format(i)).write_text("x")
        return vault

    def test_subset_run_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            vault = self._vault(td, 3)
            db = [str(vault / "n{}.md".format(i)) for i in range(3)] + ["C:/other/z.md"] * 1
            db += ["C:/other/o{}.md".format(i) for i in range(50)]
            code, deletes = self._run({"vault": vault}, db, subset="n1")
            self.assertEqual(deletes, [])
            self.assertEqual(code, 0)

    def test_missing_pdf_root_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            vault = self._vault(td, 3)
            pdfs = Path(td, "pdfs_unmounted")
            db = [str(vault / "n{}.md".format(i)) for i in range(3)]
            db += [str(pdfs / "p{}.pdf".format(i)) for i in range(40)]
            code, deletes = self._run({"vault": vault, "pdfs": pdfs}, db)
            self.assertEqual(deletes, [])
            self.assertEqual(code, 2)

    def test_mass_deletion_aborted(self):
        with tempfile.TemporaryDirectory() as td:
            vault = self._vault(td, 10)
            db = [str(vault / "n{}.md".format(i)) for i in range(10)]
            db += [str(vault / "gone{}.md".format(i)) for i in range(10)]
            code, deletes = self._run({"vault": vault}, db)
            self.assertEqual(deletes, [])
            self.assertEqual(code, 2)

    def test_normal_small_purge_still_works(self):
        with tempfile.TemporaryDirectory() as td:
            vault = self._vault(td, 100)
            db = [str(vault / "n{}.md".format(i)) for i in range(100)]
            db += [str(vault / "gone{}.md".format(i)) for i in range(2)]
            code, deletes = self._run({"vault": vault}, db)
            self.assertEqual(len(deletes), 2)
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
