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

#include "sqlite3.h"
#include "core/app/db/sqlite.h"

namespace LFL {
void SQLite::Close(SQLite::Database db) {
  if (db) sqlite3_close_v2(FromVoid<sqlite3*>(db));
}

SQLite::Database SQLite::Open(const string &fn) {
  int ret = 0;
  sqlite3 *db = 0;
  if (SQLITE_OK != (ret = sqlite3_open_v2(fn.c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, 0))) {
    if (db) sqlite3_close_v2(db);
    return ERRORv(nullptr, "sqlite3_open_v2: ", fn, " error=", ret, ": ", sqlite3_errmsg(db));
  }
  return db;
}

bool SQLite::Exec(SQLite::Database db, const string &q) {
  int ret = 0;
  char *errmsg = 0;
  if (SQLITE_OK != (ret = sqlite3_exec
                    (FromVoid<sqlite3*>(db), q.c_str(),
                     [](void*,int,char**,char**){ return int(0); }, 0, &errmsg)))
    return ERRORv(false, "sqlite3_exec: ", q, " error=", sqlite3_errstr(ret), ": ", SpellNull(errmsg));
  return true;
}

}; // namespace LFL
