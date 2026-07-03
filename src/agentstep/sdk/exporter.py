import sqlite3
import json
from typing import Sequence
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan

class ReplayOtelExporter(SpanExporter):
    """
    Exports OpenTelemetry spans directly to a local SQLite database in the `otel_spans` table.
    """
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS otel_spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT,
                parent_span_id TEXT,
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                attributes TEXT,
                events TEXT,
                status_code TEXT,
                thread_id TEXT
            )
        ''')
        # Index on thread_id for fast lookup during branch replay
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_otel_spans_thread_id ON otel_spans (thread_id)')
        self.conn.commit()

    @staticmethod
    def _safe_json(obj, max_len: int = 4096):
        """Serialize to JSON with graceful fallback for non-serializable values.

        LangChain/LangGraph objects (e.g. tool definitions, message classes) are
        not always JSON-native — a single bad attribute would otherwise drop the
        entire batch via an unhandled TypeError. We string-cast and truncate
        instead of crashing.
        """
        try:
            text = json.dumps(obj)
        except (TypeError, ValueError):
            # Fallback: repr + truncation keeps the exporter flowing.
            text = repr(obj)[:max_len]
        return text

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        cursor = self.conn.cursor()

        for span in spans:
            attrs = dict(span.attributes) if span.attributes else {}
            thread_id = attrs.get("lg.thread_id")

            events = []
            for event in span.events:
                try:
                    events.append({
                        "name": event.name,
                        "timestamp": str(event.timestamp),
                        "attributes": dict(event.attributes) if event.attributes else {},
                    })
                except Exception:
                    # Corrupt event — skip rather than dropping the whole batch.
                    pass

            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO otel_spans
                    (span_id, trace_id, parent_span_id, name, start_time, end_time, attributes, events, status_code, thread_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(span.context.span_id),
                    str(span.context.trace_id),
                    str(span.parent.span_id) if span.parent else None,
                    span.name,
                    span.start_time,
                    span.end_time,
                    self._safe_json(attrs),
                    self._safe_json(events),
                    span.status.status_code.name if span.status else "UNSET",
                    str(thread_id) if thread_id else None,
                ))
            except Exception:
                # Corrupt span — skip rather than dropping the whole batch.
                pass

        try:
            self.conn.commit()
        except sqlite3.Error:
            # Transaction conflict or connection issue — best effort.
            pass
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        # NOTE: intentionally a no-op. The tracer's teardown_otel() handles
        # SQLite connection lifecycle directly (commit + close) to avoid leaving
        # stale file handles on Windows where OTel's default shutdown doesn't
        # close the underlying DB connection. Closing it here would double-close
        # if teardown_otel runs after us, and skipping commit would lose data.
        pass
