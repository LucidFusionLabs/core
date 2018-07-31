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

#include "json/json.h"

namespace LFL {
  
JSON::Value::Value() : impl(new Json::Value()), owner(true) {}
JSON::Value::Value(Void v) : impl(v), owner(false) {}
JSON::Value::Value(JSON::Value &&v) : impl(v.impl), owner(v.owner) { v.owner = false; }
JSON::Value::~Value() { if (owner) delete FromVoid<Json::Value*>(impl); }

int JSON::Value::Size() const { return FromVoid<const Json::Value*>(impl)->size(); }
string JSON::Value::GetString() const { return FromVoid<const Json::Value*>(impl)->asString(); }
bool JSON::Value::IsMember(const string &n) const { return FromVoid<const Json::Value*>(impl)->isMember(n); }
JSON::Value JSON::Value::operator[](int i) { return JSON::Value(Void(&(*FromVoid<const Json::Value*>(impl))[i])); }
JSON::Value JSON::Value::operator[](const string &n) { return JSON::Value(Void(&(*FromVoid<const Json::Value*>(impl))[n])); }

JSON::Value JSON::Parse(const string &s) {
  JSON::Value ret;
  Json::Reader reader;
  if (!reader.parse(s, *FromVoid<Json::Value*>(ret.impl), false)) return JSON::Value(nullptr);
  return ret;
}

}; // namespace LFL
