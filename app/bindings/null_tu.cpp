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

#ifdef LFL_FLATBUFFERS
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#endif

#ifdef LFL_JSONCPP
#include "json/json.h"
#endif

namespace LFL {
const int TranslationUnit::Token::Punctuation = 0;
const int TranslationUnit::Token::Keyword     = 0;
const int TranslationUnit::Token::Identifier  = 0;
const int TranslationUnit::Token::Literal     = 0;
const int TranslationUnit::Token::Comment     = 0;

bool TranslationUnit::Cursor::IsDeclaration    (int) { return 0; }
bool TranslationUnit::Cursor::IsReference      (int) { return 0; }
bool TranslationUnit::Cursor::IsExpression     (int) { return 0; }
bool TranslationUnit::Cursor::IsStatement      (int) { return 0; }
bool TranslationUnit::Cursor::IsAttribute      (int) { return 0; }
bool TranslationUnit::Cursor::IsInvalid        (int) { return 0; }
bool TranslationUnit::Cursor::IsTranslationUnit(int) { return 0; }
bool TranslationUnit::Cursor::IsPreprocessing  (int) { return 0; }
bool TranslationUnit::Cursor::IsUnexposed      (int) { return 0; }

bool TranslationUnit::Cursor::IsBool           (int) { return 0; }
bool TranslationUnit::Cursor::IsCharacter      (int) { return 0; }
bool TranslationUnit::Cursor::IsInteger        (int) { return 0; }
bool TranslationUnit::Cursor::IsFloat          (int) { return 0; }
bool TranslationUnit::Cursor::IsInclude        (int) { return 0; }
bool TranslationUnit::Cursor::IsOperator       (int) { return 0; }
bool TranslationUnit::Cursor::IsTypedef        (int) { return 0; }
bool TranslationUnit::Cursor::IsMacroDefinition(int) { return 0; }
bool TranslationUnit::Cursor::IsCStatement     (int) { return 0; }
bool TranslationUnit::Cursor::IsCLabel         (int) { return 0; }
bool TranslationUnit::Cursor::IsCConditional   (int) { return 0; }
bool TranslationUnit::Cursor::IsCStructure     (int) { return 0; }
bool TranslationUnit::Cursor::IsCCast          (int) { return 0; }
bool TranslationUnit::Cursor::IsCRepeat        (int) { return 0; }
bool TranslationUnit::Cursor::IsCPPStatement   (int) { return 0; }
bool TranslationUnit::Cursor::IsCPPAccess      (int) { return 0; }
bool TranslationUnit::Cursor::IsCPPExceptions  (int) { return 0; }
bool TranslationUnit::Cursor::IsCPPStructure   (int) { return 0; }
bool TranslationUnit::Cursor::IsCPPCast        (int) { return 0; }

TranslationUnit::TranslationUnit(const string &f, const string &cc, const string &wd) {}
TranslationUnit::~TranslationUnit() {}
bool TranslationUnit::SaveTo(const string &f) { return false; }
bool TranslationUnit::Load(const string &f) { return false; }
bool TranslationUnit::Parse(const OpenedFiles &opened) { return false; }
void *TranslationUnit::CompleteCode(const OpenedFiles&, int, int) { return nullptr; }
FileNameAndOffset TranslationUnit::FindDefinition(const string&, int) { return FileNameAndOffset(); }
void TranslationUnit::TokenVisitor::Visit() {}

IDEProject::~IDEProject() {}
IDEProject::IDEProject(const string &d) {
  unique_ptr<File> f(make_unique<LocalFile>(StrCat(d, "/compile_commands.json"), "r"));
  if (!f->Opened()) return;

#if defined(LFL_JSONCPP)
  Json::Value root;
  Json::Reader reader;
  CHECK(reader.parse(f->Contents(), root, false));
  for (int i=0, l=root.size(); i<l; ++i) {
    const Json::Value &f = root[i];
    build_rules[f["file"].asString()] = { f["directory"].asString(), f["command"].asString() };
  }
#elif defined(LFL_FLATBUFFERS)
  flatbuffers::Parser parser;
  CHECK(parser.Parse("table BuildRule {\n"
                     "  directory: string;\n"
                     "  command:   string;\n"
                     "  file:      string;\n"
                     "}\n"
                     "table BuildRules {\n"
                     "  rule: [BuildRule];\n"
                     "}\n"
                     "root_type BuildRules;\n"));
  CHECK(parser.Parse(StrCat("{\nrule: \n", f->Contents(), "\n}\n").c_str()));

  auto buildrule = parser.structs_.Lookup("BuildRule");
  auto buildrule_dir = buildrule->fields.Lookup("directory");
  auto buildrule_cmd = buildrule->fields.Lookup("command");
  auto buildrule_file = buildrule->fields.Lookup("file");
  auto buildrules = flatbuffers::GetRoot<flatbuffers::Table>(parser.builder_.GetBufferPointer());
  auto buildrules_rule = parser.root_struct_def_->fields.Lookup("rule");
  auto buildrules_rules = reinterpret_cast<const flatbuffers::Vector<flatbuffers::Offset<void>>*>
    (buildrules->GetPointer<const void *>(buildrules_rule->value.offset));

  string dir, cmd, file;
  for (int i = 0, l = buildrules_rules->size(); i < l; ++i) {
    auto br_i = reinterpret_cast<const flatbuffers::Table*>((*buildrules_rules)[i]);
    if (auto s = reinterpret_cast<const flatbuffers::String*>(br_i->GetPointer<const void *>(buildrule_dir->value.offset)))  dir  = s->str();
    if (auto s = reinterpret_cast<const flatbuffers::String*>(br_i->GetPointer<const void *>(buildrule_cmd->value.offset)))  cmd  = s->str();
    if (auto s = reinterpret_cast<const flatbuffers::String*>(br_i->GetPointer<const void *>(buildrule_file->value.offset))) file = s->str();
    if (file.size()) build_rules[file] = { dir, cmd };
  }
#endif
}

bool IDEProject::GetCompileCommand(const string &fn, string *out, string *dir) {
  auto rule = build_rules.find(fn);
  if (rule == build_rules.end()) return false;
  *out = rule->second.cmd;
  *dir = rule->second.dir;
  return true;
}

}; // namespace LFL
