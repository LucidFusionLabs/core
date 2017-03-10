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

#ifndef LFL_CORE_APP_DB_SQLITE_H__
#define LFL_CORE_APP_DB_SQLITE_H__

namespace LFL {
struct SQLite {
  struct Database  : public VoidPtr { using VoidPtr::VoidPtr; };
  struct Statement : public VoidPtr { using VoidPtr::VoidPtr; };

  struct Row {
    Statement *parent;
    Row(Statement *P) : parent(P) {}
    void GetColumnVal(int col,         int *out);
    void GetColumnVal(int col,     int64_t *out);
    void GetColumnVal(int col, StringPiece *out);
    void GetColumnVal(int col,   BlobPiece *out);
    template <class X> X GetColumnValue(int i);
  };

  typedef function<int(Row*)>                RowVisitor;
  typedef function<int(int, char**, char**)> RowTextVisitor;

  static void Close(Database db);
  static Database Open(const string &fn);
  static bool UsePassphrase(Database db, const string &pw);
  static void ChangePassphrase(Database db, const string &pw);
  static bool Exec(Database db, const string&, const RowVisitor &cb);
  static bool Exec(Database db, const string&, const RowTextVisitor &cb = [](int, char**, char**){ return 0; });

  static Statement Prepare(Database db, const string&);
  static void Bind(Statement stmt, int ind, int                v);
  static void Bind(Statement stmt, int ind, int64_t            v);
  static void Bind(Statement stmt, int ind, const BlobPiece   &v);
  static void Bind(Statement stmt, int ind, const StringPiece &v);
  static void Finalize(Statement);
  /**/                                    static bool ExecPrepared(Statement,                        const RowVisitor &cb = RowVisitor());
  template <class X1>                     static bool ExecPrepared(Statement s, X1 a1,               const RowVisitor &cb = RowVisitor()) { Bind(s, 1, a1);                                 return ExecPrepared(s, cb); }
  template <class X1, class X2>           static bool ExecPrepared(Statement s, X1 a1, X2 a2,        const RowVisitor &cb = RowVisitor()) { Bind(s, 1, a1); Bind(s, 2, a2);                 return ExecPrepared(s, cb); }
  template <class X1, class X2, class X3> static bool ExecPrepared(Statement s, X1 a1, X2 a2, X3 a3, const RowVisitor &cb = RowVisitor()) { Bind(s, 1, a1); Bind(s, 2, a2); Bind(s, 3, a3); return ExecPrepared(s, cb); }
};

template <> inline int         SQLite::Row::GetColumnValue<int>        (int i) { int         v; GetColumnVal(i, &v); return v; }
template <> inline int64_t     SQLite::Row::GetColumnValue<int64_t>    (int i) { int64_t     v; GetColumnVal(i, &v); return v; }
template <> inline StringPiece SQLite::Row::GetColumnValue<StringPiece>(int i) { StringPiece v; GetColumnVal(i, &v); return v; }
template <> inline BlobPiece   SQLite::Row::GetColumnValue<BlobPiece>  (int i) { BlobPiece   v; GetColumnVal(i, &v); return v; }

struct SQLiteIdValueStore {
  struct Entry { string blob; Time date; };
  struct EntryPointer { int id; const string *blob; Time date; };
  SQLite::Database *db;
  string table_name;
  unordered_map<int, Entry> data;
  SQLiteIdValueStore() : db(0) {}
  SQLiteIdValueStore(SQLite::Database *db, const string &tn) { Open(db, tn); }

  void Open(SQLite::Database *db, const string &tn);
  bool Erase(int row_id);
  int Insert(const BlobPiece &val);
  bool Update(int row_id, const BlobPiece &val);
  bool UpdateDate(int row_id, Time val);
#ifdef LFL_FLATBUFFERS
  int Insert(const FlatBufferPiece &blob) { return Insert(MakeBlobPiece(blob)); }
  bool Update(int row_id, const FlatBufferPiece &blob) { return Update(row_id, MakeBlobPiece(blob)); }
#endif
};

}; // namespace LFL
#endif // LFL_CORE_APP_DB_SQLITE_H__
