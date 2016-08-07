/*
 * $Id$
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifdef LFL_SQLCIPHER
#include "sqlcipher/sqlite3.h"
#else
#include "sqlite3.h"
#endif

#include "core/app/db/sqlite.h"

namespace LFL {
void SQLite::Row::GetColumnVal(int col, int *out) {
  *out = sqlite3_column_int(FromVoid<sqlite3_stmt*>(*parent), col);
}

void SQLite::Row::GetColumnVal(int col, StringPiece *out) {
  auto stmt = FromVoid<sqlite3_stmt*>(*parent);
  *out = StringPiece(MakeSigned(sqlite3_column_text(stmt, col)), sqlite3_column_bytes(stmt, col));
}

void SQLite::Row::GetColumnVal(int col, BlobPiece *out) {
  auto stmt = FromVoid<sqlite3_stmt*>(*parent);
  *out = BlobPiece(reinterpret_cast<const char*>(sqlite3_column_blob(stmt, col)), sqlite3_column_bytes(stmt, col));
}

void SQLite::Close(SQLite::Database db) {
  if (db) sqlite3_close_v2(FromVoid<sqlite3*>(db));
}

SQLite::Database SQLite::Open(const string &fn) {
  int ret = 0;
  sqlite3 *db = 0;
  INFO("SQLite::Open opening ", fn);
  if (SQLITE_OK != (ret = sqlite3_open_v2(fn.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0))) {
    if (db) sqlite3_close_v2(db);
    return ERRORv(nullptr, "sqlite3_open_v2: ", fn, " error=", ret, ": ", sqlite3_errmsg(db));
  }
  return db;
}

bool SQLite::Exec(SQLite::Database db, const string &q, const RowVisitor &cb) {
  Statement stmt = Prepare(db, q);
  bool ret = ExecPrepared(stmt, cb);
  Finalize(stmt);
  return ret;
}

bool SQLite::Exec(SQLite::Database db, const string &q, const RowTextVisitor &cb) {
  int ret = 0;
  char *errmsg = 0;
  auto closure = [](void *opaque, int n, char **v, char **k) -> int {
    return (*reinterpret_cast<const RowTextVisitor*>(opaque))(n, v, k);
  };

  if (SQLITE_OK != (ret = sqlite3_exec
                    (FromVoid<sqlite3*>(db), q.c_str(), closure, Void(&cb), &errmsg)))
    return ERRORv(false, "sqlite3_exec: ", q, " error=", sqlite3_errstr(ret), ": ", SpellNull(errmsg));
  return true;
}

SQLite::Statement SQLite::Prepare(Database db, const string &q) {
  int ret = 0;
  sqlite3_stmt *stmt=0;
  if (SQLITE_OK != (ret = sqlite3_prepare_v2(FromVoid<sqlite3*>(db), q.c_str(), q.size()+1, &stmt, 0)))
    return ERRORv(nullptr, "sqlite3_prepare_v2: ", q, " error=", sqlite3_errstr(ret));
  return stmt;
}

void SQLite::Finalize(Statement stmt) {
  int ret = 0;
  if (SQLITE_OK != (ret = sqlite3_finalize(FromVoid<sqlite3_stmt*>(stmt))))
    ERROR("sqlite3_finalize: error=", sqlite3_errstr(ret));
}

void SQLite::Bind(Statement stmt, int ind, int v) {
  int ret = 0;
  if (SQLITE_OK != (ret = sqlite3_bind_int(FromVoid<sqlite3_stmt*>(stmt), ind, v)))
    ERROR("sqlite3_bind_int: error=", sqlite3_errstr(ret));
}

void SQLite::Bind(Statement stmt, int ind, const StringPiece &v) {
  int ret = 0;
  if (SQLITE_OK != (ret = sqlite3_bind_text(FromVoid<sqlite3_stmt*>(stmt), ind, v.buf, v.len, 0)))
    ERROR("sqlite3_bind_text: error=", sqlite3_errstr(ret));
}

void SQLite::Bind(Statement stmt, int ind, const BlobPiece &v) {
  int ret = 0;
  if (SQLITE_OK != (ret = sqlite3_bind_blob(FromVoid<sqlite3_stmt*>(stmt), ind, v.buf, v.len, 0)))
    ERROR("sqlite3_bind_blob: error=", sqlite3_errstr(ret));
}

bool SQLite::ExecPrepared(Statement stmt, const RowVisitor &cb) {
  int ret = 0;
  Row row(&stmt);
  while (SQLITE_ROW == (ret = sqlite3_step(FromVoid<sqlite3_stmt*>(stmt))))
    if (cb && cb(&row)) {
      break;
    }

  if (SQLITE_DONE != ret) return ERRORv(false, "sqlite3_step: error=", sqlite3_errstr(ret));
  return true;
}

void SQLiteIdValueStore::Open(SQLite::Database *d, const string &tn) {
  db = d;
  table_name = tn;
  CHECK(SQLite::Exec(*db, StrCat("CREATE TABLE IF NOT EXISTS ", tn ,"\n"
                                 "(id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
                                 "data BLOB NOT NULL);\n")));
  CHECK(SQLite::Exec(*db, StrCat("SELECT id, data FROM ", tn, ";"),
                     [=](SQLite::Row *r) {
                       data[r->GetColumnValue<int>(0)] = r->GetColumnValue<BlobPiece>(1).str();
                       return 0;
                     }));
}

int SQLiteIdValueStore::Insert(const BlobPiece &val) {
  SQLite::Statement stmt = SQLite::Prepare(*db, StrCat("INSERT INTO ", table_name, " (data) VALUES (?);"));
  SQLite::ExecPrepared(stmt, val);
  SQLite::Finalize(stmt);
  int ret = sqlite3_last_insert_rowid(FromVoid<sqlite3*>(*db));
  data[ret] = val.str();
  return ret;
}

bool SQLiteIdValueStore::Update(int id, const BlobPiece &val) {
  SQLite::Statement stmt = SQLite::Prepare(*db, StrCat("UPDATE ", table_name, " SET data = ? WHERE id = ?;"));
  SQLite::ExecPrepared(stmt, val, id);
  SQLite::Finalize(stmt);
  data[id] = val.str();
  return true;
}

bool SQLiteIdValueStore::Erase(int id) {
  SQLite::Statement stmt = SQLite::Prepare(*db, StrCat("DELETE FROM ", table_name, " WHERE id = ?;"));
  SQLite::ExecPrepared(stmt, id);
  SQLite::Finalize(stmt);
  data.erase(id);
  return true;
}

}; // namespace LFL
