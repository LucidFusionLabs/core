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
    struct Database : public VoidPtr { using VoidPtr::VoidPtr; };
    static void Close(Database db);
    static Database Open(const string &fn);
    static bool Exec(Database db, const string&);
  };
}; // namespace LFL
#endif // LFL_CORE_APP_DB_SQLITE_H__
