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

#include "core/app/app.h"
#include "clang-c/Index.h"

namespace LFL {
const int TranslationUnit::Token::Punctuation = CXToken_Punctuation;
const int TranslationUnit::Token::Keyword     = CXToken_Keyword;
const int TranslationUnit::Token::Identifier  = CXToken_Identifier;
const int TranslationUnit::Token::Literal     = CXToken_Literal;
const int TranslationUnit::Token::Comment     = CXToken_Comment;

typedef function<CXChildVisitResult(CXCursor)> ClangCursorVisitor;

unsigned ClangVisitChildren(CXCursor p, ClangCursorVisitor f) {
  auto visitor_closure = [](CXCursor c, CXCursor p, CXClientData v) -> CXChildVisitResult {
    return (*reinterpret_cast<const ClangCursorVisitor*>(v))(c);
  };
  return clang_visitChildren(p, visitor_closure, &f);
}

string GetClangString(const CXString &s) {
  string v = BlankNull(clang_getCString(s));
  clang_disposeString(s);
  return v;
}

TranslationUnit::TranslationUnit(const string &f, const string &cc, const string &wd) :
  index(clang_createIndex(0, 0)), filename(f), compile_command(cc), working_directory(wd) {
  vector<string> argv;
  vector<const char*> av = { "-xc++", "-std=c++11" };
  Split(compile_command, isspace, &argv);
  for (int i=1; i<(int)argv.size()-4; i++)
    if (!PrefixMatch(argv[i], "-O") && !PrefixMatch(argv[i], "-m")) av.push_back(argv[i].data());
  INFO("TranslationUnit args ", Join(av, " "));

  chdir(working_directory.c_str());
  CXErrorCode ret = clang_parseTranslationUnit2(index, filename.c_str(), av.data(), av.size(), 0, 0,
                                                CXTranslationUnit_None, &tu);
  if (!tu) ERROR("TranslationUnit ", f, " create failed ", StringPrintf("%d",ret));
}

TranslationUnit::~TranslationUnit() {
  clang_disposeTranslationUnit(tu);
  clang_disposeIndex(index);
}

FileNameAndOffset TranslationUnit::FindDefinition(const string &fn, int offset) {
  CXFile cf = clang_getFile(tu, fn.c_str());
  if (!cf) return FileNameAndOffset();
  CXCursor cursor = clang_getCursor(tu, clang_getLocationForOffset(tu, cf, offset));
  CXCursor cursor_canonical = clang_getCanonicalCursor(cursor), null = clang_getNullCursor();
  CXCursor parent = clang_getCursorSemanticParent(cursor), child = null;
  if (clang_equalCursors(parent, null)) return FileNameAndOffset();
  ClangVisitChildren(parent, [&](CXCursor c){
    if (!clang_equalCursors(clang_getCanonicalCursor(c), cursor_canonical)) return CXChildVisit_Continue;
    else { child = c; return CXChildVisit_Break; }
  });
  if (clang_equalCursors(child, null)) return FileNameAndOffset();
  CXFile rf;
  unsigned ry, rx, ro;
  CXSourceRange sr = clang_getCursorExtent(child);
  clang_getSpellingLocation(clang_getRangeStart(sr), &rf, &ry, &rx, &ro);
  string rfn = GetClangString(clang_getFileName(rf));
  return FileNameAndOffset(GetClangString(clang_getFileName(rf)), ro, ry, rx);
}

void TranslationUnit::TokenVisitor::Visit() {
  CXToken* tokens=0;
  unsigned num_tokens=0, by=0, bx=0, ey=0, ex=0;
  CXFile cf = clang_getFile(tu->tu, tu->filename.c_str());
  CXSourceRange sr = clang_getRange(clang_getLocationForOffset(tu->tu, cf, 0),
                                    clang_getLocationForOffset(tu->tu, cf, LocalFile(tu->filename, "r").Size()));
  clang_tokenize(tu->tu, sr, &tokens, &num_tokens);
  for (int i = 0; i < num_tokens; i++) {
    sr = clang_getTokenExtent(tu->tu, tokens[i]);
    clang_getSpellingLocation(clang_getRangeStart(sr), NULL, &by, &bx, NULL);
    clang_getSpellingLocation(clang_getRangeEnd  (sr), NULL, &ey, &ex, NULL);

    if (1)                  CHECK_LE(last_token.y, (int)by);
    if (by == last_token.y) CHECK_LT(last_token.x, (int)bx);
    cb(this, clang_getTokenKind(tokens[i]), by, bx);
    last_token = point(bx, by);
  }
}
}; // namespace LFL
