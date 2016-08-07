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
      void GetColumnVal(int col, StringPiece *out);
      void GetColumnVal(int col,   BlobPiece *out);
      template <class X> X GetColumnValue(int i);
    };

    typedef function<int(Row*)>                RowVisitor;
    typedef function<int(int, char**, char**)> RowTextVisitor;

    static void Close(Database db);
    static Database Open(const string &fn);
    static bool Exec(Database db, const string&, const RowVisitor &cb);
    static bool Exec(Database db, const string&, const RowTextVisitor &cb = [](int, char**, char**){ return 0; });

    static Statement Prepare(Database db, const string&);
    static void Bind(Statement stmt, int ind, int                v);
    static void Bind(Statement stmt, int ind, const BlobPiece   &v);
    static void Bind(Statement stmt, int ind, const StringPiece &v);
    static void Finalize(Statement);
    static bool ExecPrepared(Statement,                                               const RowVisitor &cb = RowVisitor());
    template <class X1>           static bool ExecPrepared(Statement s, X1 a1,        const RowVisitor &cb = RowVisitor()) { Bind(s, 1, a1);                 return ExecPrepared(s, cb); }
    template <class X1, class X2> static bool ExecPrepared(Statement s, X1 a1, X2 a2, const RowVisitor &cb = RowVisitor()) { Bind(s, 1, a1); Bind(s, 2, a2); return ExecPrepared(s, cb); }
  };

  template <> inline int         SQLite::Row::GetColumnValue<int>        (int i) { int         v; GetColumnVal(i, &v); return v; }
  template <> inline StringPiece SQLite::Row::GetColumnValue<StringPiece>(int i) { StringPiece v; GetColumnVal(i, &v); return v; }
  template <> inline BlobPiece   SQLite::Row::GetColumnValue<BlobPiece>  (int i) { BlobPiece   v; GetColumnVal(i, &v); return v; }
                                                                                    
  struct SQLiteIdValueStore {
    SQLite::Database *db;
    string table_name;
    unordered_map<int, string> data;

    void Open(SQLite::Database *, const string &tn);
    int Insert(const BlobPiece &val);
    bool Update(int, const BlobPiece &val);
    bool Erase(int);
  };
}; // namespace LFL
#endif // LFL_CORE_APP_DB_SQLITE_H__
