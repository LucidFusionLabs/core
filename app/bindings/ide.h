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

#ifndef LFL_CORE_APP_BINDINGS_IDE_H__
#define LFL_CORE_APP_BINDINGS_IDE_H__
namespace LFL {

struct TranslationUnit {
  struct Token { static const int Punctuation, Keyword, Identifier, Literal, Comment; };
  struct Cursor {
    static const int StringLiteral;
    static bool IsDeclaration(int);
    static bool IsReference(int);
    static bool IsExpression(int);
    static bool IsStatement(int);
    static bool IsAttribute(int);
    static bool IsInvalid(int);
    static bool IsTranslationUnit(int);
    static bool IsPreprocessing(int);
    static bool IsUnexposed(int);

    static bool IsBool(int);
    static bool IsCharacter(int);
    static bool IsInteger(int);
    static bool IsFloat(int);
    static bool IsInclude(int);
    static bool IsOperator(int);
    static bool IsTypedef(int);
    static bool IsMacroDefinition(int);
    static bool IsCStatement(int);
    static bool IsCLabel(int);
    static bool IsCConditional(int);
    static bool IsCStructure(int);
    static bool IsCCast(int);
    static bool IsCRepeat(int);
    static bool IsCPPStatement(int);
    static bool IsCPPAccess(int);
    static bool IsCPPExceptions(int);
    static bool IsCPPStructure(int);
    static bool IsCPPCast(int);
  };

  struct TokenVisitor {
    typedef function<void(TokenVisitor*, const string&, int, int, int, int, int)> TokenCB;
    TranslationUnit *tu;
    point last_token;
    TokenCB cb;

    TokenVisitor(TranslationUnit *t, const TokenCB &c) : tu(t), cb(c) {}
    void Visit();
  };

  CXIndex index=0;
  CXTranslationUnit tu=0;
  string filename, compile_command, working_directory;
  vector<pair<int,int>> skipped_lines;

  virtual ~TranslationUnit();
  TranslationUnit(const string &f, const string &cc, const string &wd);
  FileNameAndOffset FindDefinition(const string &f, int offset);
};

struct IDEProject {
  string build_dir, source_dir;
  CXCompilationDatabase compilation_db=0;
  struct BuildRule { string dir, cmd; };
  unordered_map<string, BuildRule> build_rules;
  virtual ~IDEProject();
  IDEProject(const string&);
  bool GetCompileCommand(const string &fn, string *out, string *dir);
};

struct RegexCPlusPlusHighlighter { static void UpdateAnnotation(Editor*); };
struct ClangCPlusPlusHighlighter { static void UpdateAnnotation(Editor*); };

struct CMakeDaemon {
  enum { Null=0, Init=1, HaveTargets=2 };
  struct Proto {
    static const string header, footer;
    static string MakeBuildsystem();
    static string MakeHandshake(const string &v);
    static string MakeTargetInfo(const string &n, const string &c);
    static string MakeFileInfo(const string &n, const string &p, const string &c);
  };
  struct Target {
    string decl_file;
    int decl_line;
  };
  struct TargetInfo {
    string output;
    vector<string> compile_definitions, compile_options, include_directories, link_libraries, sources, generated_sources;
  };
  typedef function<void(const TargetInfo&)> TargetInfoCB;

  ProcessPipe process;
  int state = Null;
  vector<string> configs;
  unordered_map<string, Target> targets;
  deque<pair<string, TargetInfoCB>> target_info_cb;
  Callback init_targets_cb;
  CMakeDaemon() {}

  void Start(const string &bin, const string &builddir);
  void HandleClose(Connection *c);
  void HandleRead(Connection *c);
  bool GetTargetInfo(const string &target, TargetInfoCB&&);
};

}; // namespace LFL
#endif // LFL_CORE_APP_BINDINGS_IDE_H__
