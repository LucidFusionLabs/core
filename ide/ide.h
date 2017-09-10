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

struct CodeCompletions {
  virtual ~CodeCompletions() {}
  virtual size_t size() const = 0;
  virtual string GetText(size_t ind) = 0;
};

struct CodeCompletionsVector : public CodeCompletions {
  vector<string> data;
  size_t size() const { return data.size(); }
  string GetText(size_t ind) { return data[ind]; }
};

struct TranslationUnit {
  typedef vector<pair<string, shared_ptr<BufferFile>>> OpenedFiles;
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

  struct CodeCompletions : public LFL::CodeCompletions {
    void *impl=0;
    virtual ~CodeCompletions();
    CodeCompletions(void *I=0) : impl(I) {}
    size_t size() const;
    string GetText(size_t ind);
  };

  CXIndex index=0;
  CXTranslationUnit tu=0;
  bool parse_failed=0;
  string filename, compile_command, working_directory;
  vector<pair<int,int>> skipped_lines;

  virtual ~TranslationUnit();
  TranslationUnit(const string &f, const string &cc, const string &wd);
  bool SaveTo(const string &f);
  bool Load(const string &f);
  bool Parse(const OpenedFiles &unsaved = OpenedFiles());
  bool Reparse(const OpenedFiles &unsaved = OpenedFiles());
  unique_ptr<LFL::CodeCompletions> CompleteCode(const OpenedFiles&, int, int);
  pair<FileOffset, FileOffset> GetCursorExtent(const string &f, int, int);
  FileNameAndOffset FindDefinition(const string &f, int, int);
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

struct ClangCPlusPlusHighlighter {
  static void UpdateAnnotation(TranslationUnit*, Editor::SyntaxColors*, int,
                               vector<DrawableAnnotation> *out);
};

struct CMakeDaemon {
  enum { Null=0, SentHandshake=1, Init=2, HaveTargets=3 };
  struct Proto {
    static const string header, footer;
    static string MakeBuildsystem();
    static string MakeHandshake(const string &v);
    static string MakeTargetInfo(const string &n, const string &c);
    static string MakeFileInfo(const string &n, const string &p, const string &c);
    static string MakeCodeComplete(const string &f, int y, int x, const string &content);
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

  ThreadDispatcher *dispatch;
  ProcessPipe process;
  int state = Null;
  vector<string> configs;
  unordered_map<string, Target> targets;
  deque<pair<string, TargetInfoCB>> target_info_cb;
  Callback init_targets_cb;
  Semaphore *code_completions_done=0;
  unique_ptr<CodeCompletions> *code_completions_out=0;
  CMakeDaemon(ThreadDispatcher *d) : dispatch(d) {}

  bool Ready() const { return state >= HaveTargets; }
  void Start(ApplicationInfo*, SocketServices*, const string &bin, const string &builddir);
  void HandleClose(Connection *c);
  void HandleRead(Connection *c);
  bool GetTargetInfo(const string &target, TargetInfoCB&&);
  unique_ptr<LFL::CodeCompletions> CompleteCode(const string &fn, int y, int x, const string &content);
};

struct CodeCompletionsView : public PropertyView {
  unique_ptr<LFL::CodeCompletions> completions;
  mutable Node node;
  using PropertyView::PropertyView;

  Node* GetNode(Id id) { const auto *self = this; return const_cast<Node*>(self->GetNode(id)); }
  const Node* GetNode(Id id) const {
    node.text = completions ? completions->GetText(id-1) : string();
    return &node;
  }
  void VisitExpandedChildren(Id id, const Node::Visitor &cb, int depth) {
    if (!id) for (int i = 0, l = completions ? completions->size() : 0; i != l; ++i) cb(i+1, 0, 0);
  }
};

struct CodeCompletionsViewDialog : public TextViewDialogT<CodeCompletionsView> {
  using TextViewDialogT::TextViewDialogT;
};

}; // namespace LFL
#endif // LFL_CORE_APP_BINDINGS_IDE_H__
