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

namespace LFL {
Regex::~Regex() {}
Regex::Regex(const string &patternstr) {}
Regex::Result Regex::MatchOne(const StringPiece&)   { return ERRORv(Regex::Result(), "regex not implemented"); }
Regex::Result Regex::MatchOne(const String16Piece&) { return ERRORv(Regex::Result(), "regex not implemented"); }

}; // namespace LFL
