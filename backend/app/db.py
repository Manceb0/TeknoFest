import json
import threading
from pathlib import Path

import duckdb

from .core.config import settings


class Database:
    """DuckDB-backed store with the VSS extension for vector similarity search.

    Each incident persists the YOLOv8 feature embedding of its frame, enabling
    in-process retrieval of semantically similar past incidents (the capability
    described in the PDR's "DuckDB + VSS" architecture).
    """

    def __init__(self) -> None:
        self.path = Path(settings.database_path)
        self.dim = settings.embedding_dim
        self.lock = threading.Lock()
        self.conn = duckdb.connect(str(self.path))
        self.init()

    def init(self):
        with self.lock:
            self.conn.execute("INSTALL vss")
            self.conn.execute("LOAD vss")
            # Required to build/persist an HNSW index inside a file-backed DB.
            self.conn.execute("SET hnsw_enable_experimental_persistence = true")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                  session_id TEXT PRIMARY KEY, phone TEXT, verified_at TEXT, created_at TEXT
                )
            """)
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS incidents (
                  incident_id TEXT PRIMARY KEY, session_id TEXT, timestamp TEXT, frame_id INTEGER,
                  plate_text TEXT, plate_confidence REAL, behavior_label TEXT,
                  behavior_confidence REAL, risk_score REAL, qod_state TEXT,
                  latency_ms REAL, model_provider TEXT, detection_json TEXT, snapshot_path TEXT,
                  embedding FLOAT[{self.dim}]
                )
            """)
            try:
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS incidents_vec ON incidents "
                    "USING HNSW (embedding) WITH (metric = 'cosine')"
                )
            except duckdb.Error:
                # Index is an optimization; similarity search still works without it.
                pass

    def create_session(self, session_id, phone, timestamp):
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO sessions VALUES (?,?,?,?)",
                [session_id, phone, timestamp, timestamp],
            )

    def add_incident(self, incident: dict):
        d = incident["detection"]
        embedding = incident.get("embedding")
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO incidents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    incident["incident_id"], d["session_id"], d["timestamp"], d["frame_id"],
                    d["plate"]["text"], d["plate"]["confidence"], d["behavior"]["label"],
                    d["behavior"]["confidence"], d["risk"]["score"], d["qod"]["state"],
                    d["latency_ms"], d["model_provider"], json.dumps(d),
                    incident.get("snapshot_path"), embedding,
                ],
            )

    _LIST_COLS = (
        "incident_id, session_id, timestamp, frame_id, plate_text, plate_confidence, "
        "behavior_label, behavior_confidence, risk_score, qod_state, latency_ms, "
        "model_provider, snapshot_path"
    )

    def incidents(self):
        with self.lock:
            cur = self.conn.execute(
                f"SELECT {self._LIST_COLS} FROM incidents ORDER BY timestamp DESC"
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def incident(self, incident_id):
        with self.lock:
            cur = self.conn.execute(
                f"SELECT {self._LIST_COLS}, detection_json FROM incidents WHERE incident_id=?",
                [incident_id],
            )
            cols = [c[0] for c in cur.description]
            row = cur.fetchone()
        if not row:
            return None
        data = dict(zip(cols, row))
        data["detection"] = json.loads(data.pop("detection_json"))
        return data

    def similar(self, incident_id: str, k: int = 5):
        """Return up to ``k`` past incidents most similar to ``incident_id``."""
        with self.lock:
            ref = self.conn.execute(
                "SELECT embedding FROM incidents WHERE incident_id=?", [incident_id]
            ).fetchone()
            if not ref or ref[0] is None:
                return []
            cur = self.conn.execute(
                f"""
                SELECT {self._LIST_COLS},
                       array_cosine_distance(embedding, ?::FLOAT[{self.dim}]) AS distance
                FROM incidents
                WHERE incident_id != ? AND embedding IS NOT NULL
                ORDER BY distance ASC
                LIMIT ?
                """,
                [ref[0], incident_id, k],
            )
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
        results = []
        for row in rows:
            item = dict(zip(cols, row))
            item["similarity"] = round(1 - float(item["distance"]), 4)
            results.append(item)
        return results

    def count(self):
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]


db = Database()
