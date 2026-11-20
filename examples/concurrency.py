"""Deterministic shared-handle concurrency reproduction and regression checks.

Run from the repository root:  python3 -m examples.concurrency
Exit status is 0 when every scenario verdict passes, 1 otherwise. The full
call/return history, commit outcomes and per-read key/body/document-revision/
read-revision records are printed as JSON.

# What is verified

Two threads share one strataleaf.Database handle (hence one SQLite
connection, the real shared-handle path):

* family "writer_mid_commit": a statement gate pauses the writer batch after
  the first operation's version row and index refresh, before the second
  operation; one single-shot reader query (tag == "old", limit 100) runs
  during that window. In the "conflict" kind the second operation violates
  the unique code index, so the whole batch must roll back and the read must
  still return the complete pre-commit revision (both a and b). In the
  "success" kind the batch commits; the read may legitimately return the
  complete pre-commit view or the complete post-commit view, judged against
  the revision the read itself reports.
* family "reader_holds_snapshot": the reader is paused inside its own read
  transaction; the writer then runs a successful batch. A successful commit
  must not be cancelled by a concurrent read.

Each family runs against the equality-index plan, force_scan, and a
collection without a tag index, so all three read paths are checked.

# Scheduling control points (and whether they alter the path under test)

* ProbedConnection wraps the one real sqlite3.Connection. It delegates every
  attribute and executes the same SQL on the same connection and WAL file;
  the hooks only observe statement text and may pause the calling thread in
  pure Python BEFORE delegating (family W) or AFTER delegating (family R).
  While paused, the gate itself holds no SQLite lock and no handle lock, so
  the other thread genuinely uses the shared connection. No library code is
  modified, mocked or reordered.
* ProbedLock wraps the same threading.RLock instance the handle already
  uses; acquire/release semantics are identical (delegated), it only reports
  contended acquisitions so the orchestrator can release a gate.
* All synchronisation is event-driven (threading.Event rendezvous); no
  arbitrary sleep decides the interleaving, and the two business calls are
  genuinely concurrent: one thread is provably inside its call (mid
  transaction / mid query) while the other runs, as the connection and lock
  activity in the history shows.

# Verdict rule

A read history is accepted iff the returned records equal the COMPLETE
committed view at the revision the read itself reports (Page.revision),
given the commits that exist when the run ends. Overlapping call times are
not treated as an order, and two independent reads may legitimately differ
when a commit lands between them. A result that no complete committed
revision can explain (e.g. only "b" at revision r, where the committed
revision r still matches both a and b) fails the run.

# Interleavings not covered

reader fully before the writer's BEGIN (serialises benignly); reader in the
small window between the writer's last statement and the SQLite commit;
multiple concurrent readers; overlap across two independent handles (those
coordinate through SQLite locks, not the handle lock); crash mid-commit;
checkpoint during a long read; retention compaction during the overlap (the
floor stays 0 here, all history retained).
"""
import json
import sys
import tempfile
import threading
from pathlib import Path

from strataleaf import Conflict, Database

OLD_A = {"tag": "old", "code": "A"}
OLD_B = {"tag": "old", "code": "B"}
NEW_A = {"tag": "new", "code": "A"}
PREDICATE = {"field": "tag", "op": "eq", "value": "old"}
ACCESSES = ("equality_index", "force_scan", "no_tag_index")

NOTES = {
    "control_points": [
        "ProbedConnection: transparent delegate on the one real sqlite3 connection; "
        "pauses the calling thread in pure Python at a chosen statement boundary "
        "(family W: before the writer's second INSERT INTO versions; family R: after "
        "the reader's BEGIN). Same connection, same SQL, same order, same WAL file; "
        "no library code modified.",
        "ProbedLock: transparent delegate on the handle's existing RLock; identical "
        "mutual-exclusion semantics, only reports contended acquisitions.",
        "threading.Event rendezvous orders the threads; no sleep decides any interleaving "
        "and no business call is serialised by the harness itself.",
    ],
    "verdict_depends_on": [
        "the read's own reported revision (Page.revision)",
        "the commits existing at end of run (final head, commit outcome)",
        "the complete committed view per revision (both documents at the seed revision; "
        "empty tag==old view after a successful commit)",
    ],
    "uncovered_interleavings": [
        "reader completing before the writer's BEGIN (benign serialisation)",
        "reader between the writer's last statement and the SQLite commit",
        "multiple concurrent readers",
        "overlap across two independent Database handles (SQLite-lock coordinated)",
        "crash mid-commit, checkpoint during a long read, retention compaction overlap",
    ],
}


class ProbedConnection:
    """Transparent delegate for the one real sqlite3 connection."""

    def __init__(self, real, before=None, after=None, observe=None):
        self._real = real
        self._before = before
        self._after = after
        self._observe = observe

    def execute(self, sql, parameters=(), /):
        tid = threading.get_ident()
        if self._observe is not None:
            self._observe(sql, tid)
        if self._before is not None:
            self._before(sql, tid)
        cursor = self._real.execute(sql, parameters)
        if self._after is not None:
            self._after(sql, tid)
        return cursor

    def __getattr__(self, item):
        return getattr(self._real, item)


class ProbedLock:
    """Transparent delegate for the handle lock that reports contention."""

    def __init__(self, real, on_contention=None):
        self._real = real
        self._on_contention = on_contention

    def acquire(self, blocking=True, timeout=-1):
        if self._real.acquire(blocking=False):
            return True
        if not blocking:
            return False
        if self._on_contention is not None:
            self._on_contention(threading.get_ident())
        if timeout == -1:
            self._real.acquire()
            return True
        return self._real.acquire(timeout=timeout)

    def release(self):
        self._real.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


class Timeline:
    def __init__(self):
        self._lock = threading.Lock()
        self._entries = []

    def add(self, role, event, detail, **extra):
        with self._lock:
            entry = {"seq": len(self._entries), "role": role, "event": event, "detail": detail}
            entry.update(extra)
            self._entries.append(entry)

    @property
    def entries(self):
        with self._lock:
            return list(self._entries)


def wait_first(*events):
    fired = threading.Event()

    def watch(event):
        event.wait()
        fired.set()

    for event in events:
        threading.Thread(target=watch, args=(event,), daemon=True).start()
    fired.wait()


def join_threads(threads, checks):
    hung = []
    for name, thread in threads.items():
        thread.join(30)
        if thread.is_alive():
            hung.append(name)
    if hung:
        checks.append({"name": "orchestration completes without deadlock", "ok": False,
                       "detail": "threads still blocked: " + ",".join(hung)})
        return False
    checks.append({"name": "orchestration completes without deadlock", "ok": True,
                   "detail": "writer and reader joined"})
    return True


def check(checks, name, ok, detail):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def setup_database(path, with_tag_index):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(path)
    db.catalog.create("docs")
    seed = db.batch("seed_docs")
    seed.put("docs", "a", dict(OLD_A))
    seed.put("docs", "b", dict(OLD_B))
    revision = seed.commit().revision
    db.indexes.create("docs", "by_code", "code", unique=True)
    if with_tag_index:
        db.indexes.create("docs", "by_tag", "tag")
    return db, revision


def expected_views(revision, committed):
    views = {revision: {"a": {"document": OLD_A, "revision": revision},
                        "b": {"document": OLD_B, "revision": revision}}}
    if committed:
        views[revision + 1] = {}
    return views


def page_record(page):
    return {"revision": page.revision, "plan": page.plan, "cursor": page.cursor is not None,
            "records": [{"key": rec.key, "document": rec.document, "revision": rec.revision}
                        for rec in page.records]}


def records_view(page):
    return {rec.key: {"document": rec.document, "revision": rec.revision} for rec in page.records}


def feed_revisions(db):
    return [commit["revision"] for commit in db.feed.read(0)]


def verdict_read(checks, reader_outcome, views):
    if "error" in reader_outcome:
        check(checks, "single read returns a complete committed view", False,
              "read raised " + json.dumps(reader_outcome["error"], ensure_ascii=False))
        return
    page = reader_outcome["page"]
    actual = records_view(page)
    if page.revision in views and actual == views[page.revision]:
        check(checks, "single read returns a complete committed view", True,
              f"read revision {page.revision} explains all {len(actual)} record(s)")
        return
    allowed = {revision: sorted(view) for revision, view in views.items()}
    check(checks, "single read returns a complete committed view", False,
          f"read revision {page.revision} returned keys {sorted(actual)}; no complete commit "
          f"explains this result (complete views by revision: {allowed})")


class OverlapRun:
    def __init__(self, folder):
        self.timeline = Timeline()
        self.checks = []
        self.roles = {}
        self.events = {key: threading.Event() for key in
                       ("mid_tx", "release", "reader_in_tx", "reader_release",
                        "reader_blocked", "reader_done", "writer_blocked", "writer_done")}
        self.outcome = {"writer": {}, "reader": {}}
        self.gate_active = threading.Event()
        self.db, self.revision = setup_database(folder / "shared.db", True)
        real_connection, real_lock = self.db._store.connection, self.db._store.lock
        self._real = (real_connection, real_lock)

    def role_of(self, tid):
        return self.roles.get(tid, "main")

    def observe(self, sql, tid):
        self.timeline.add(self.role_of(tid), "sql", sql.split("(", 1)[0][:72],
                          during_uncommitted_window=self.gate_active.is_set())

    def on_contention(self, tid):
        role = self.role_of(tid)
        self.timeline.add(role, "lock", "blocked acquiring the handle lock")
        event = self.events.get(role + "_blocked")
        if event is not None:
            event.set()

    def install(self, before=None, after=None):
        real_connection, real_lock = self._real
        self.db._store.connection = ProbedConnection(real_connection, before=before,
                                                     after=after, observe=self.observe)
        self.db._store.lock = ProbedLock(real_lock, on_contention=self.on_contention)

    def restore(self):
        real_connection, real_lock = self._real
        self.db._store.connection = real_connection
        self.db._store.lock = real_lock
        for event in self.events.values():
            event.set()

    def evidence(self):
        entries = self.timeline.entries
        return {
            "reader_statements_during_uncommitted_window":
                [entry["detail"] for entry in entries
                 if entry["event"] == "sql" and entry["role"] == "reader"
                 and entry["during_uncommitted_window"]],
            "reader_blocked_on_handle_lock":
                any(entry["event"] == "lock" and entry["role"] == "reader" for entry in entries),
            "writer_blocked_on_handle_lock":
                any(entry["event"] == "lock" and entry["role"] == "writer" for entry in entries),
        }


def run_writer_mid_commit(folder, kind, access):
    run = OverlapRun(folder)
    db, r, events, timeline = run.db, run.revision, run.events, run.timeline
    if access == "no_tag_index":
        db.indexes.drop("docs", "by_tag")
    force_scan = access == "force_scan"
    writer_tid = {}
    inserts = {"versions": 0}
    first_page = db.query("docs", PREDICATE, limit=1) if kind == "success" else None

    def before(sql, tid):
        if tid != writer_tid.get("tid"):
            return
        if sql.lstrip().upper().startswith("INSERT INTO VERSIONS"):
            inserts["versions"] += 1
            if inserts["versions"] == 2:
                run.gate_active.set()
                events["mid_tx"].set()
                timeline.add("writer", "gate", "paused before second INSERT INTO versions; "
                             "first operation's version row and index refresh are uncommitted")
                events["release"].wait()
                run.gate_active.clear()
                timeline.add("writer", "gate", "resumed")

    def writer():
        tid = threading.get_ident()
        run.roles[tid] = "writer"
        writer_tid["tid"] = tid
        second_code = "A" if kind == "conflict" else "B"
        call = (f"batch(req_overlap).put(a, {NEW_A}).put(b, tag=new/code={second_code}).commit()")
        timeline.add("writer", "api", "call " + call)
        try:
            result = db.batch("req_overlap").put("docs", "a", dict(NEW_A)).put(
                "docs", "b", {"tag": "new", "code": second_code}).commit()
            run.outcome["writer"] = {"committed": True, "result": result.__dict__}
            timeline.add("writer", "api", f"commit returned revision {result.revision}")
        except Exception as exc:  # recorded for the verdict, not swallowed
            run.outcome["writer"] = {"committed": False, "error": type(exc).__name__,
                                     "message": str(exc)}
            timeline.add("writer", "api", "commit raised " + type(exc).__name__)
        finally:
            events["writer_done"].set()

    def reader():
        tid = threading.get_ident()
        run.roles[tid] = "reader"
        events["mid_tx"].wait()
        timeline.add("reader", "api", f"call query(tag == 'old', limit=100, force_scan={force_scan})")
        try:
            page = db.query("docs", PREDICATE, limit=100, force_scan=force_scan)
            run.outcome["reader"] = {"page": page}
            timeline.add("reader", "api",
                         f"query returned {len(page.records)} record(s) at read revision {page.revision}")
        except Exception as exc:
            run.outcome["reader"] = {"error": {"type": type(exc).__name__, "message": str(exc)}}
            timeline.add("reader", "api", "query raised " + type(exc).__name__)
        finally:
            events["reader_done"].set()

    run.install(before=before)
    threads = {"writer": threading.Thread(target=writer, daemon=True),
               "reader": threading.Thread(target=reader, daemon=True)}
    threads["writer"].start()
    threads["reader"].start()
    wait_first(events["reader_done"], events["reader_blocked"])
    events["release"].set()
    join_threads(threads, run.checks)
    run.restore()

    writer_outcome, reader_outcome = run.outcome["writer"], run.outcome["reader"]
    committed = writer_outcome.get("committed", False)
    head_after = db.revision
    views = expected_views(r, committed)

    if kind == "conflict":
        check(run.checks, "conflicting batch fails with Conflict and stays atomic",
              not committed and writer_outcome.get("error") == "Conflict" and head_after == r,
              f"outcome={writer_outcome.get('error', 'committed')} head={head_after} expected head={r}")
        a_now, b_now = db.get("docs", "a"), db.get("docs", "b")
        check(run.checks, "rolled-back batch leaves data untouched",
              a_now is not None and a_now.document == OLD_A and a_now.revision == r
              and b_now is not None and b_now.document == OLD_B and b_now.revision == r,
              f"a={a_now and a_now.document} b={b_now and b_now.document}")
        retry = db.batch("req_overlap").put("docs", "a", dict(NEW_A)).put(
            "docs", "b", {"tag": "new", "code": "B"}).commit()
        replay = db.batch("req_overlap").put("docs", "a", dict(NEW_A)).put(
            "docs", "b", {"tag": "new", "code": "B"}).commit()
        check(run.checks, "failed batch left no receipt; corrected retry commits and replays",
              not retry.replayed and retry.revision == r + 1 and replay.replayed and replay.revision == r + 1,
              f"retry={retry.__dict__} replay={replay.__dict__}")
        check(run.checks, "change feed holds only complete commits",
              feed_revisions(db) == [r, r + 1], f"feed={feed_revisions(db)}")
    else:
        check(run.checks, "successful commit is not cancelled by the concurrent read",
              committed and writer_outcome["result"]["revision"] == r + 1 and head_after == r + 1,
              f"outcome={writer_outcome} head={head_after}")
        replay = db.batch("req_overlap").put("docs", "a", dict(NEW_A)).put(
            "docs", "b", {"tag": "new", "code": "B"}).commit()
        check(run.checks, "request receipt replays the recorded commit",
              replay.replayed and replay.revision == r + 1, f"replay={replay.__dict__}")
        check(run.checks, "change feed holds only complete commits",
              feed_revisions(db) == [r, r + 1], f"feed={feed_revisions(db)}")
        second = db.query("docs", PREDICATE, limit=100, cursor=first_page.cursor)
        check(run.checks, "pre-commit pagination cursor keeps its read revision",
              second.revision == r and records_view(second) == {"b": {"document": OLD_B, "revision": r}},
              f"second page revision={second.revision} keys={[rec.key for rec in second.records]}")
        past, present = db.get("docs", "a", revision=r), db.get("docs", "a")
        check(run.checks, "explicit old-revision read keeps its revision while retained",
              past.document == OLD_A and past.revision == r and present.document == NEW_A,
              f"past={past.document}@{past.revision} present={present.document}@{present.revision}")

    verdict_read(run.checks, reader_outcome, views)
    if "page" in reader_outcome:
        plan_access = reader_outcome["page"].plan["access"]
        expected_access = "equality_index" if access == "equality_index" else "version_scan"
        check(run.checks, "expected read path exercised", plan_access == expected_access,
              f"plan={reader_outcome['page'].plan}")

    read = (page_record(reader_outcome["page"]) if "page" in reader_outcome
            else {"error": reader_outcome.get("error")})
    scenario = {"name": f"writer_mid_commit/{kind}/{access}", "history": timeline.entries,
                "commit": writer_outcome, "read": read, "head_after_commit": head_after,
                "expected_views_by_revision": {rev: sorted(view) for rev, view in views.items()},
                "overlap_evidence": run.evidence(), "checks": run.checks}
    scenario["ok"] = all(item["ok"] for item in run.checks)
    db.close()
    return scenario


def run_reader_holds_snapshot(folder, access):
    run = OverlapRun(folder)
    db, r, events, timeline = run.db, run.revision, run.events, run.timeline
    if access == "no_tag_index":
        db.indexes.drop("docs", "by_tag")
    force_scan = access == "force_scan"

    def after(sql, tid):
        if run.role_of(tid) == "reader" and sql.strip().upper() == "BEGIN":
            events["reader_in_tx"].set()
            timeline.add("reader", "gate", "paused inside its own read transaction")
            events["reader_release"].wait()
            timeline.add("reader", "gate", "resumed")

    def reader():
        tid = threading.get_ident()
        run.roles[tid] = "reader"
        timeline.add("reader", "api", f"call query(tag == 'old', limit=100, force_scan={force_scan})")
        try:
            page = db.query("docs", PREDICATE, limit=100, force_scan=force_scan)
            run.outcome["reader"] = {"page": page}
            timeline.add("reader", "api",
                         f"query returned {len(page.records)} record(s) at read revision {page.revision}")
        except Exception as exc:
            run.outcome["reader"] = {"error": {"type": type(exc).__name__, "message": str(exc)}}
            timeline.add("reader", "api", "query raised " + type(exc).__name__)
        finally:
            events["reader_done"].set()

    def writer():
        tid = threading.get_ident()
        run.roles[tid] = "writer"
        events["reader_in_tx"].wait()
        timeline.add("writer", "api", "call batch(req_overlap).put(a, new/A).put(b, new/B).commit()")
        try:
            result = db.batch("req_overlap").put("docs", "a", dict(NEW_A)).put(
                "docs", "b", {"tag": "new", "code": "B"}).commit()
            run.outcome["writer"] = {"committed": True, "result": result.__dict__}
            timeline.add("writer", "api", f"commit returned revision {result.revision}")
        except Exception as exc:
            run.outcome["writer"] = {"committed": False, "error": type(exc).__name__,
                                     "message": str(exc)}
            timeline.add("writer", "api", "commit raised " + type(exc).__name__)
        finally:
            events["writer_done"].set()

    run.install(after=after)
    threads = {"reader": threading.Thread(target=reader, daemon=True),
               "writer": threading.Thread(target=writer, daemon=True)}
    threads["reader"].start()
    threads["writer"].start()
    wait_first(events["writer_blocked"], events["writer_done"])
    events["reader_release"].set()
    join_threads(threads, run.checks)
    run.restore()

    writer_outcome, reader_outcome = run.outcome["writer"], run.outcome["reader"]
    committed = writer_outcome.get("committed", False)
    head_after = db.revision
    views = expected_views(r, committed)
    check(run.checks, "successful commit is not cancelled by the concurrent read",
          committed and writer_outcome["result"]["revision"] == r + 1,
          f"outcome={writer_outcome}")
    check(run.checks, "head is consistent with the commit outcome",
          head_after == (r + 1 if committed else r), f"head={head_after}")
    check(run.checks, "change feed holds only complete commits",
          feed_revisions(db) == ([r, r + 1] if committed else [r]), f"feed={feed_revisions(db)}")
    verdict_read(run.checks, reader_outcome, views)
    if "page" in reader_outcome:
        plan_access = reader_outcome["page"].plan["access"]
        expected_access = "equality_index" if access == "equality_index" else "version_scan"
        check(run.checks, "expected read path exercised", plan_access == expected_access,
              f"plan={reader_outcome['page'].plan}")

    read = (page_record(reader_outcome["page"]) if "page" in reader_outcome
            else {"error": reader_outcome.get("error")})
    scenario = {"name": f"reader_holds_snapshot/success/{access}", "history": timeline.entries,
                "commit": writer_outcome, "read": read, "head_after_commit": head_after,
                "expected_views_by_revision": {rev: sorted(view) for rev, view in views.items()},
                "overlap_evidence": run.evidence(), "checks": run.checks}
    scenario["ok"] = all(item["ok"] for item in run.checks)
    db.close()
    return scenario


def preservation_checks(folder):
    checks = []
    db, r = setup_database(folder / "preserve.db", with_tag_index=True)
    try:
        db.batch().put("docs", "a", dict(NEW_A), expected=r + 1).commit()
        check(checks, "conditional version conflict is rejected", False, "no Conflict raised")
    except Conflict as exc:
        unchanged = db.get("docs", "a")
        check(checks, "conditional version conflict is rejected",
              unchanged.document == OLD_A and db.revision == r, str(exc))
    try:
        db.batch().put("docs", "x", {"tag": "new", "code": "X"}).put(
            "docs", "y", {"tag": "new", "code": "X"}).commit()
        check(checks, "unique rejection keeps the batch atomic", False, "no Conflict raised")
    except Conflict as exc:
        check(checks, "unique rejection keeps the batch atomic",
              db.get("docs", "x") is None and db.get("docs", "y") is None and db.revision == r, str(exc))
    first = db.batch("req_p1").put("docs", "c", {"tag": "old", "code": "C"}).commit()
    replay = db.batch("req_p1").put("docs", "c", {"tag": "old", "code": "C"}).commit()
    check(checks, "request receipts replay idempotently",
          not first.replayed and replay.replayed and replay.revision == first.revision,
          f"first={first.__dict__} replay={replay.__dict__}")
    try:
        db.batch("req_p1").put("docs", "d", {"tag": "old", "code": "D"}).commit()
        check(checks, "request key reuse with different operations is rejected", False, "no Conflict raised")
    except Conflict as exc:
        check(checks, "request key reuse with different operations is rejected", True, str(exc))
    check(checks, "change feed replays complete commits in order",
          feed_revisions(db) == [r, r + 1], f"feed={feed_revisions(db)}")
    other = Database(folder / "preserve.db")
    try:
        seen = other.revision
        written = other.batch("req_handle2").put("docs", "h", {"tag": "old", "code": "H"}).commit()
        found = db.get("docs", "h")
        check(checks, "independent handle reads and writes",
              seen == db.revision - 1 and written.revision == seen + 1
              and found is not None and found.document == {"tag": "old", "code": "H"},
              f"seen={seen} written={written.__dict__}")
    finally:
        other.close()
    db.close()
    return {"name": "preservation", "checks": checks,
            "ok": all(item["ok"] for item in checks)}


def main():
    scenarios = []
    with tempfile.TemporaryDirectory(prefix="strataleaf-concurrency-") as directory:
        root = Path(directory)
        for kind in ("conflict", "success"):
            for access in ACCESSES:
                scenarios.append(run_writer_mid_commit(root / f"w-{kind}-{access}", kind, access))
        for access in ACCESSES:
            scenarios.append(run_reader_holds_snapshot(root / f"r-success-{access}", access))
        preservation = preservation_checks(root / "preservation")
    ok = all(scenario["ok"] for scenario in scenarios) and preservation["ok"]
    report = {"scenarios": scenarios, "preservation": preservation, "notes": NOTES, "ok": ok}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
