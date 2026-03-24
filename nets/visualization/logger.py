# import json
# import os
# import uuid
# import threading
# import time


# class Logger:

#     def __init__(self, path="runs.json"):
#         self.path = path
#         self.lock = threading.Lock()   # 🔥 concurrency fix
#         self.runs = {}
#         self.current_run = None
#         self.load()

#     # ---------------- LOAD (PUBLIC FIX)
#     def load(self):
#         with self.lock:
#             if os.path.exists(self.path):
#                 try:
#                     with open(self.path, "r") as f:
#                         self.runs = json.load(f)
#                 except:
#                     self.runs = {}
#             else:
#                 self.runs = {}
#         return self.runs

#     # ---------------- SAFE SAVE (ATOMIC WRITE)
#     def _save(self):
#         with self.lock:
#             tmp_path = self.path + ".tmp"

#             with open(tmp_path, "w") as f:
#                 json.dump(self.runs, f, indent=2)

#             os.replace(tmp_path, self.path)  # 🔥 atomic write

#     # ---------------- START RUN
#     def start_run(self, name, meta=None):
#         rid = str(uuid.uuid4())

#         with self.lock:
#             self.runs[rid] = {
#                 "name": name,
#                 "meta": meta or {},
#                 "logs": [],
#                 "created_at": time.time()
#             }
#             self.current_run = rid

#         self._save()
#         return rid

#     # ---------------- LOG
#     def log(self, data, run_id=None):
#         rid = run_id or self.current_run

#         if rid is None:
#             raise Exception("No active run. Call start_run() first.")

#         with self.lock:
#             if rid in self.runs:
#                 self.runs[rid]["logs"].append(data)

#         self._save()

#     # ---------------- GET (ALWAYS FRESH)
#     def get_runs(self):
#         return self.load()

#     # ---------------- DELETE
#     def delete_run(self, rid):
#         with self.lock:
#             if rid in self.runs:
#                 del self.runs[rid]
#         self._save()

#     def delete_runs(self, rids):
#         with self.lock:
#             for rid in rids:
#                 if rid in self.runs:
#                     del self.runs[rid]
#         self._save()

#     def clear_all(self):
#         with self.lock:
#             self.runs = {}
#             self.current_run = None
#         self._save()

import json
import os
import uuid
import threading
import time


class Logger:
    def __init__(self, path="runs.json"):
        self.path = path
        self.lock = threading.RLock()
        self.runs = {}
        self.current_run = None
        self.load()

    # ---------- LOAD
    def load(self):
        with self.lock:
            if os.path.exists(self.path):
                try:
                    with open(self.path, "r") as f:
                        self.runs = json.load(f)
                except:
                    self.runs = {}
            else:
                self.runs = {}
        return self.runs

    # ---------- SAVE (atomic)
    def _save(self):
        with self.lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.runs, f, indent=2)
            os.replace(tmp, self.path)

    # ---------- RUN CONTROL
    def start_run(self, name, meta=None):
        rid = str(uuid.uuid4())
        with self.lock:
            self.runs[rid] = {
                "name": name,
                "meta": meta or {},
                "logs": [],
                "created_at": time.time()
            }
            self.current_run = rid
        self._save()
        return rid

    def log(self, data, run_id=None):
        rid = run_id or self.current_run
        if rid is None:
            return
        with self.lock:
            self.runs[rid]["logs"].append(data)
        self._save()

    def get_runs(self):
        return self.load()

    def delete_run(self, rid):
        with self.lock:
            self.runs.pop(rid, None)
        self._save()

    def clear_all(self):
        with self.lock:
            self.runs = {}
            self.current_run = None
        self._save()