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

#include "core/app/gui.h"
#include "core/app/ipc.h"
#include "core/app/bindings/ide.h"
#include "clang-c/Index.h"
#include "clang-c/CXCompilationDatabase.h"

namespace LFL {
const int TranslationUnit::Token::Punctuation = CXToken_Punctuation;  // = 0
const int TranslationUnit::Token::Keyword     = CXToken_Keyword;      // = 1
const int TranslationUnit::Token::Identifier  = CXToken_Identifier;   // = 2
const int TranslationUnit::Token::Literal     = CXToken_Literal;      // = 3
const int TranslationUnit::Token::Comment     = CXToken_Comment;      // = 4
const int TranslationUnit::Cursor::StringLiteral = CXCursor_StringLiteral;

bool TranslationUnit::Cursor::IsDeclaration    (int c) { return clang_isDeclaration    (CXCursorKind(c)); }
bool TranslationUnit::Cursor::IsReference      (int c) { return clang_isReference      (CXCursorKind(c)); }
bool TranslationUnit::Cursor::IsExpression     (int c) { return clang_isExpression     (CXCursorKind(c)); }
bool TranslationUnit::Cursor::IsStatement      (int c) { return clang_isStatement      (CXCursorKind(c)); }
bool TranslationUnit::Cursor::IsAttribute      (int c) { return clang_isAttribute      (CXCursorKind(c)); }
bool TranslationUnit::Cursor::IsInvalid        (int c) { return clang_isInvalid        (CXCursorKind(c)); }
bool TranslationUnit::Cursor::IsTranslationUnit(int c) { return clang_isTranslationUnit(CXCursorKind(c)); }
bool TranslationUnit::Cursor::IsPreprocessing  (int c) { return clang_isPreprocessing  (CXCursorKind(c)); }
bool TranslationUnit::Cursor::IsUnexposed      (int c) { return clang_isUnexposed      (CXCursorKind(c)); }

bool TranslationUnit::Cursor::IsBool(int c) { return CXCursorKind(c) == CXCursor_CXXBoolLiteralExpr; }
bool TranslationUnit::Cursor::IsCharacter(int c) { return CXCursorKind(c) == CXCursor_CharacterLiteral; }
bool TranslationUnit::Cursor::IsInteger(int c) { return CXCursorKind(c) == CXCursor_IntegerLiteral; }
bool TranslationUnit::Cursor::IsFloat(int c) {
  switch (CXCursorKind(c)) {
    case CXCursor_FloatingLiteral:
    case CXCursor_ImaginaryLiteral:
      return true;
    default:
      return false;
  }
}

bool TranslationUnit::Cursor::IsInclude(int c) { return CXCursorKind(c) == CXCursor_InclusionDirective; }
bool TranslationUnit::Cursor::IsMacroDefinition(int c) { return CXCursorKind(c) == CXCursor_MacroDefinition; }
bool TranslationUnit::Cursor::IsOperator(int c) {
  switch (CXCursorKind(c)) {
    case CXCursor_CXXTypeidExpr:
    case CXCursor_UnaryOperator:
    case CXCursor_BinaryOperator:
    case CXCursor_CompoundAssignOperator:
    case CXCursor_ConditionalOperator:
      return true;
    default:
      return false;
  }
}

bool TranslationUnit::Cursor::IsTypedef(int c) { return CXCursorKind(c) == CXCursor_TypedefDecl; }
bool TranslationUnit::Cursor::IsCStatement(int c) {
  switch (CXCursorKind(c)) {
    case CXCursor_GotoStmt:
    case CXCursor_BreakStmt:
    case CXCursor_ReturnStmt:
    case CXCursor_ContinueStmt:
    case CXCursor_AsmStmt:
      return true;
    default:
      return false;
  }
}

bool TranslationUnit::Cursor::IsCLabel(int c) {
  switch (CXCursorKind(c)) {
    case CXCursor_CaseStmt:
    case CXCursor_DefaultStmt:
      return true;
    default:
      return false;
  }
}

bool TranslationUnit::Cursor::IsCConditional(int c) {
  switch (CXCursorKind(c)) {
    case CXCursor_IfStmt:
    case CXCursor_SwitchStmt:
      return true;
    default:
      return false;
  }
}

bool TranslationUnit::Cursor::IsCStructure(int c) {
  switch (CXCursorKind(c)) {
    case CXCursor_StructDecl:
    case CXCursor_UnionDecl:
    case CXCursor_EnumDecl:
      return true;
    default:
      return false;
  }
}

bool TranslationUnit::Cursor::IsCCast(int c) { return CXCursorKind(c) == CXCursor_CStyleCastExpr; }
bool TranslationUnit::Cursor::IsCRepeat(int c) {
  switch (CXCursorKind(c)) {
    case CXCursor_WhileStmt:
    case CXCursor_ForStmt:
    case CXCursor_DoStmt:
      return true;
    default:
      return false;
  }
}

bool TranslationUnit::Cursor::IsCPPStatement(int c) {
  switch (CXCursorKind(c)) {
    case CXCursor_CXXNewExpr:
    case CXCursor_CXXDeleteExpr:
    case CXCursor_CXXThisExpr:
    case CXCursor_UsingDirective:
    case CXCursor_UsingDeclaration:
      return true;
    default:
      return false;
  }
}

bool TranslationUnit::Cursor::IsCPPAccess(int c) { return CXCursorKind(c) == CXCursor_CXXAccessSpecifier; }
bool TranslationUnit::Cursor::IsCPPExceptions(int c) {
  switch (CXCursorKind(c)) {
    case CXCursor_CXXTryStmt:
    case CXCursor_CXXCatchStmt:
    case CXCursor_CXXThrowExpr:
      return true;
    default:
      return false;
  }
}

bool TranslationUnit::Cursor::IsCPPStructure(int c) {
  switch (CXCursorKind(c)) {
    case CXCursor_ClassDecl:
    case CXCursor_Namespace:
    case CXCursor_NamespaceAlias:
      return true;
    default:
      return false;
  }
}

bool TranslationUnit::Cursor::IsCPPCast(int c) {
  switch (CXCursorKind(c)) {
    case CXCursor_CXXStaticCastExpr:
    case CXCursor_CXXDynamicCastExpr:
    case CXCursor_CXXReinterpretCastExpr:
    case CXCursor_CXXConstCastExpr:
      return true;
    default:
      return false;
  }
}

typedef function<CXChildVisitResult(CXCursor, CXCursor)> ClangCursorVisitor;
typedef function<void(CXFile, CXSourceLocation*, unsigned)> ClangInclusionVisitor;

unsigned VisitClangCursorChildren(CXCursor p, ClangCursorVisitor f) {
  auto visitor_closure = [](CXCursor c, CXCursor p, CXClientData v) -> CXChildVisitResult {
    return (*reinterpret_cast<const ClangCursorVisitor*>(v))(c, p);
  };
  return clang_visitChildren(p, visitor_closure, &f);
}

void VisitClangTranslationUnitInclusions(CXTranslationUnit tu, ClangInclusionVisitor f) {
  auto visitor_closure = [](CXFile f, CXSourceLocation *inc, unsigned inclen, CXClientData v) -> void {
    return (*reinterpret_cast<const ClangInclusionVisitor*>(v))(f, inc, inclen);
  };
  return clang_getInclusions(tu, visitor_closure, &f);
}

string GetClangString(const CXString &s) {
  string v = BlankNull(clang_getCString(s));
  clang_disposeString(s);
  return v;
}

vector<CXUnsavedFile> GetClangUnsavedFiles(const TranslationUnit::OpenedFiles &opened) {
  vector<CXUnsavedFile> ret;
  for (const auto &i : opened) {
    Editor *e = &i.second->view;
    if (!e->CacheModifiedText()) continue;
    ret.push_back({ e->file->Filename(), e->cached_text.buf.c_str(), e->cached_text.buf.size() });
  }
  return ret;
}

TranslationUnit::TranslationUnit(const string &f, const string &cc, const string &wd) :
  index(clang_createIndex(0, 0)), filename(f), compile_command(cc), working_directory(wd) {}

TranslationUnit::~TranslationUnit() {
  clang_disposeTranslationUnit(tu);
  clang_disposeIndex(index);
}

bool TranslationUnit::SaveTo(const string &f) {
  int err = clang_saveTranslationUnit(tu, f.c_str(), clang_defaultSaveOptions(tu));
  if (err != CXSaveError_None) return ERRORv(false, "clang_saveTranslationUnit ", err);
  return true;
}

bool TranslationUnit::Load(const string &fn) {
  CXErrorCode ret = clang_createTranslationUnit2(index, fn.c_str(), &tu);
  if (!tu) return ERRORv(false, "TranslationUnit ", fn, " load failed ", StringPrintf("%d",ret));
  return true;
}

bool TranslationUnit::Parse(const OpenedFiles &opened) { 
  vector<CXUnsavedFile> unsaved = GetClangUnsavedFiles(opened);
  vector<string> argv;
  vector<const char*> av = { "-xc++", "-std=c++11" };
  Split(compile_command, isspace, &argv);
  for (int i=1; i<int(argv.size())-4; i++) av.push_back(argv[i].data());
  INFO("TranslationUnit args ", Join(av, " "));
  chdir(working_directory.c_str());

  unsigned options = CXTranslationUnit_DetailedPreprocessingRecord | // CXTranslationUnit_KeepGoing |
    CXTranslationUnit_ForSerialization | // clang_defaultEditingTranslationUnitOptions();
    CXTranslationUnit_PrecompiledPreamble | CXTranslationUnit_CacheCompletionResults |
    CXTranslationUnit_CreatePreambleOnFirstParse;
  CXErrorCode ret = clang_parseTranslationUnit2(index, filename.c_str(), av.data(), av.size(),
                                                &unsaved[0], unsaved.size(), options, &tu);
  if (!tu) return ERRORv(false, "TranslationUnit ", filename, " create failed ", StringPrintf("%d",ret));
  // clang_reparseTranslationUnit(tu, 0, 0, clang_defaultReparseOptions(tu));

  parse_failed = false;
  int diagnostics = clang_getNumDiagnostics(tu);
  for(int i = 0; i != diagnostics; ++i) {
    CXDiagnostic diag = clang_getDiagnostic(tu, i);
    CXDiagnosticSeverity severity = clang_getDiagnosticSeverity(diag);
    if (severity == CXDiagnostic_Error || severity == CXDiagnostic_Error) parse_failed = true;
    INFO("TranslationUnit ", filename, " ",
         GetClangString(clang_formatDiagnostic(diag, clang_defaultDiagnosticDisplayOptions())));
  }

  CXFile cf = clang_getFile(tu, filename.c_str());
  CXSourceRangeList *skipped = clang_getSkippedRanges(tu, cf);
  for (CXSourceRange *r = skipped->ranges, *e = r + skipped->count; r != e; ++r) {
    unsigned by=0, ey=0;
    CXSourceLocation rb = clang_getRangeStart(*r), re = clang_getRangeEnd(*r);
    clang_getSpellingLocation(rb, NULL, &by, NULL, NULL);
    clang_getSpellingLocation(re, NULL, &ey, NULL, NULL);
    skipped_lines.emplace_back(by, ey);
  }
  clang_disposeSourceRangeList(skipped);
  return true;
}

bool TranslationUnit::Reparse(const OpenedFiles &opened) { 
  vector<CXUnsavedFile> unsaved = GetClangUnsavedFiles(opened);
  clang_reparseTranslationUnit(tu, unsaved.size(), &unsaved[0], clang_defaultReparseOptions(tu));
  return true;
}

void *TranslationUnit::CompleteCode(const OpenedFiles &opened, int line, int column) {
  unsigned options = clang_defaultCodeCompleteOptions();
  vector<CXUnsavedFile> unsaved = GetClangUnsavedFiles(opened);
  auto ret = clang_codeCompleteAt(tu, filename.c_str(), line, column, &unsaved[0], unsaved.size(), options);
  INFO("CompleteCode results ", ret->NumResults);
  clang_disposeCodeCompleteResults(ret);
  return nullptr;
}

FileNameAndOffset TranslationUnit::FindDefinition(const string &fn, int offset) {
  CXFile cf = clang_getFile(tu, fn.c_str());
  if (!cf) return FileNameAndOffset();
  CXCursor cursor = clang_getCursor(tu, clang_getLocationForOffset(tu, cf, offset));
  CXCursor cursor_canonical = clang_getCanonicalCursor(cursor), null = clang_getNullCursor();
  CXCursor parent = clang_getCursorSemanticParent(cursor), child = null;
  if (clang_equalCursors(parent, null)) return FileNameAndOffset();
  VisitClangCursorChildren(parent, [&](CXCursor c, CXCursor p){
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
  CXCursor null_cursor = clang_getNullCursor();
  CXCursorKind last_cursor_kind;
  for (CXToken *t = tokens, *e = t + num_tokens; t != e; ++t) {
    sr = clang_getTokenExtent(tu->tu, *t);
    CXSourceLocation tb=clang_getRangeStart(sr), te=clang_getRangeEnd(sr);
    clang_getSpellingLocation(tb, NULL, &by, &bx, NULL);
    clang_getSpellingLocation(te, NULL, &ey, &ex, NULL);
    if (1)                  CHECK_LE(last_token.y, (int)by);
    if (by == last_token.y) CHECK_LT(last_token.x, (int)bx);

    CXTokenKind token_kind = clang_getTokenKind(*t);
    // clang_annotateTokens(tu->tu, t, 1, &token_cursor);
    CXCursor token_cursor = clang_getCursor(tu->tu, tb);
    CXCursorKind cursor_kind = clang_getCursorKind(token_cursor);
    CXType type = clang_getCursorType(token_cursor);
    cb(this, GetClangString(clang_getTokenSpelling(tu->tu, *t)), token_kind, cursor_kind, type.kind, by, bx);
    last_token = point(bx, by);
  }
}

IDEProject::~IDEProject() { if (compilation_db) clang_CompilationDatabase_dispose(compilation_db); }
IDEProject::IDEProject(const string &d) : build_dir(d), source_dir(build_dir.c_str(), DirNameLen(build_dir)) {
  CXCompilationDatabase_Error error;
  if (!(compilation_db = clang_CompilationDatabase_fromDirectory(build_dir.c_str(), &error)) || error)
    ERROR("clang_CompilationDatabase_fromDirectory(", build_dir, "): ", error);
}

bool IDEProject::GetCompileCommand(const string &fn, string *out, string *dir) {
  if (!compilation_db) return false;
  CXCompileCommands cmds = clang_CompilationDatabase_getCompileCommands(compilation_db, fn.c_str());
  if (!cmds) return false;
  size_t num_cmds = clang_CompileCommands_getSize(cmds);
  if (!num_cmds) return false;
  CXCompileCommand cmd = clang_CompileCommands_getCommand(cmds, 0);
  size_t num_args = clang_CompileCommand_getNumArgs(cmd);
  *dir = GetClangString(clang_CompileCommand_getDirectory(cmd));
  *out = "";
  for (size_t i = 0; i != num_args; ++i) 
    StrAppend(out, out->empty() ? "" : " ", GetClangString(clang_CompileCommand_getArg(cmd, i)));
  clang_CompileCommands_dispose(cmds);
  return true;
}

void ClangCPlusPlusHighlighter::UpdateAnnotation(Editor *e, TranslationUnit *tu) {
  if (!tu) return;
  e->annotation.clear();
  using Token  = TranslationUnit::Token;
  using Cursor = TranslationUnit::Cursor;
  static unordered_set<string> inc_w{ "include", "import" }, ctype_w{ "int", "long", "short", "char",
    "void", "signed", "unsigned", "float", "double", "size_t", "ssize_t", "off_t", "wchar_t", "ptrdiff_t",
    "sig_atomic_t", "fpos_t", "clock_t", "time_t", "va_list", "jmp_buf", "FILE", "DIR", "div_t", "ldiv_t",
    "mbstate_t", "wctrans_t", "wint_t", "wctype_t", "bool", "complex", "int8_t", "int16_t", "int32_t",
    "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int_least8_t", "int_least16_t",
    "int_least32_t", "int_least64_t", "uint_least8_t", "uint_least16_t", "uint_least32_t", "uint_least64_t",
    "int_fast8_t", "int_fast16_t", "int_fast32_t", "int_fast64_t", "uint_fast8_t", "uint_fast16_t",
    "uint_fast32_t", "uint_fast64_t", "intptr_t", "uintptr_t", "intmax_t", "uintmax_t",
    "__label__", "__complex__", "__volatile__" };

  int fl_ind = 1, a = e->default_attr, last_a = a, last_line = -1, last_cursor_kind = 0;
  auto fl = e->file_line.Begin();
  Editor::LineOffset *file_line_data = 0;
  TranslationUnit::TokenVisitor(tu, TranslationUnit::TokenVisitor::TokenCB([&]
    (TranslationUnit::TokenVisitor *v, const string &text, int tk, int ck, int kk, int line, int column) {
      if (e->syntax) {
        string match;
        int input_cursor_kind = ck;
        if (Cursor::IsInvalid(ck) && line == last_line) ck = last_cursor_kind;
        last_line = line;
        last_cursor_kind = ck;
        bool is_id = tk == Token::Identifier, is_punct = tk == Token::Punctuation;

        if      (Cursor::IsBool           (ck))     match = "Boolean";
        else if (Cursor::IsCharacter      (ck))     match = "Character";
        else if (Cursor::IsInteger        (ck))     match = "Number";
        else if (Cursor::IsFloat          (ck))     match = "Float";
        else if (Token::Literal ==        (tk))     match = "String";
        else if (Token::Comment ==        (tk))     match = "Comment";
        else if (Cursor::IsInclude        (ck))
          match = ((is_punct && text != "#") || (is_id && !Contains(inc_w, text))) ? "String" : "Include";
        else if (Cursor::IsMacroDefinition(ck))     match = "Define";
        else if (Cursor::IsPreprocessing  (ck))     match = "PreProc";
        else if (Cursor::IsOperator       (ck))     match = "Operator";
        else if (is_punct)                          match = "Delimiter";
        else if (tk == Token::Keyword) {
          if      (Cursor::IsCLabel       (ck))     match = "Label";
          else if (Cursor::IsCStatement   (ck))     match = "Statement";
          else if (Cursor::IsCPPStatement (ck))     match = "Statement";
          else if (Cursor::IsCConditional (ck))     match = "Conditional";
          else if (Cursor::IsCRepeat      (ck))     match = "Repeat";
          else if (Cursor::IsCStructure   (ck))     match = "Structure";
          else if (Cursor::IsCPPStructure (ck))     match = "Structure";
          else if (Cursor::IsTypedef      (ck))     match = "Typedef";
          else if (Contains(ctype_w, text))         match = "Type";
          else                                      match = "Keyword";
        } else if (tk == is_id) {
          match = "Normal";
        }

        a = e->syntax->GetSyntaxStyle(match, e->default_attr);
        if (0) printf("%d %d %s %d %d (in %d) %d %s\n", line, column, text.c_str(), tk, ck,
                      input_cursor_kind, kk, match.c_str());
      }

      if (line != v->last_token.y) {
        if (v->last_token.y+1 < line && last_a != e->default_attr) PushBack(e->annotation, {0, last_a});
        if (fl_ind < line)
          for (++fl_ind, ++fl; fl.ind && fl_ind < line; ++fl_ind, ++fl)
            fl.val->annotation = PieceIndex(e->annotation.size()-1, 1);
        if (!fl.ind) return;
        (file_line_data = fl.val)->annotation.offset = e->annotation.size();
      }
      e->annotation.emplace_back(column-1, a);
      file_line_data->annotation.len++;
      last_a = a;
    })).Visit();
}

}; // namespace LFL
