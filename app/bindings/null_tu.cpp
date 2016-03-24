/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
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
const int TranslationUnit::Token::Punctuation = 0;
const int TranslationUnit::Token::Keyword     = 0;
const int TranslationUnit::Token::Identifier  = 0;
const int TranslationUnit::Token::Literal     = 0;
const int TranslationUnit::Token::Comment     = 0;

TranslationUnit::TranslationUnit(const string &f, const string &cc, const string &wd) {}
TranslationUnit::~TranslationUnit() {}
FileNameAndOffset TranslationUnit::FindDefinition(const string&, int) { return FileNameAndOffset(); }
void TranslationUnit::TokenVisitor::Visit() {}
}; // namespace LFL
