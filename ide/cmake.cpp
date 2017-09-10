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

#include "core/app/gl/view.h"
#include "core/app/gl/editor.h"
#include "core/app/ipc.h"
#include "core/ide/ide.h"
#include "json/json.h"

namespace LFL {
static void CopyJsonStringVector(const Json::Value &v, vector<string> *out) {
  for (int i=0, l=v.size(); i != l; ++i) out->push_back(v[i].asString());
}

const string CMakeDaemon::Proto::header = "\n[== CMake MetaMagic ==[\n";
const string CMakeDaemon::Proto::footer = "\n]== CMake MetaMagic ==]\n";

string CMakeDaemon::Proto::MakeBuildsystem() { return StrCat(header, "{\"type\":\"buildsystem\"}", footer); }
string CMakeDaemon::Proto::MakeHandshake(const string &v) {
  return StrCat(header, "{\"type\":\"handshake\",\"protocolVersion\":\"", v, "\"}", footer);
}

string CMakeDaemon::Proto::MakeTargetInfo(const string &n, const string &c) {
  return StrCat(header, "{\"type\":\"target_info\",\"target_name\":\"", n, "\",\"config\":\"", c, "\"}", footer);
}

string CMakeDaemon::Proto::MakeFileInfo(const string &n, const string &p, const string &c) {
  return StrCat(header, "{\"type\":\"file_info\",\"target_name\":\"", n,
                "\",\"file_path\":\"", p, "\",\"config\":\"", c, "\"}", footer);
}

string CMakeDaemon::Proto::MakeCodeComplete(const string &f, int y, int x, const string &content) {
  return StrCat(header, "{\"type\":\"code_complete\",\"file_path\":\"", f,
                "\",\"file_line\":", y, ",\"file_column\":", x, ",\"file_content\":\"",
                JSONEscape(content), "\"}", footer);
}

void CMakeDaemon::Start(ApplicationInfo *appinfo, SocketServices *net,
                        const string &bin, const string &builddir) {
  if (process.in) return;
  vector<const char*> argv{ bin.c_str(), "-E", "daemon", builddir.c_str(), nullptr };
  CHECK(!process.Open(argv.data(), appinfo->startdir.c_str()));
  dispatch->RunInNetworkThread([=](){ net->unix_client->AddConnectedSocket
                          (fileno(process.in), new Connection::CallbackHandler
                           (bind(&CMakeDaemon::HandleRead, this, _1),
                            bind(&CMakeDaemon::HandleClose, this, _1))); });
}

void CMakeDaemon::HandleClose(Connection *c) {
  INFO("CMakeDaemon died");
  process.Close();
}

void CMakeDaemon::HandleRead(Connection *c) {
  while (c->rb.size() >= Proto::header.size()) {
    if (!PrefixMatch(c->rb.buf.data(), Proto::header)) { c->ReadFlush(c->rb.size()); return; }
    const char *start = c->rb.buf.data() + Proto::header.size();
    const char *end = strstr(start, Proto::footer.c_str());
    if (!end) break;
    Json::Value json;
    Json::Reader reader;
    string json_text(start, end-start);
    CHECK(reader.parse(json_text, json, false));
    do { // HandleMessage
      // if (state >= HaveTargets) printf("CMakeDaemon read '%s'\n", json_text.c_str());
      if (state < SentHandshake && (state = SentHandshake)) {
        CHECK(FWriteSuccess(process.out, Proto::MakeHandshake("3.5")));
      } else if (state < Init && json["binary_dir"].asString().size() && (state = Init)) {
        CHECK(FWriteSuccess(process.out, Proto::MakeBuildsystem()));
      } else if (state < HaveTargets && json.isMember("buildsystem") && (state = HaveTargets)) {
        configs.clear();
        targets.clear();
        const Json::Value &json_configs = json["buildsystem"]["configs"];
        const Json::Value &json_targets = json["buildsystem"]["targets"];
        for (int i=0, l=json_configs.size(); i != l; ++i) configs.push_back(json_configs[i].asString());
        for (int i=0, l=json_targets.size(); i != l; ++i) {
          const Json::Value &v = json_targets[i], &bt0 = v["backtrace"][0];
          targets[v["name"].asString()] = { bt0["path"].asString(), bt0["line"].asInt() };
        }
        if (init_targets_cb) init_targets_cb();
        INFO("CMakeDaemon configs: ", Join(configs, " "));
      } else if (state >= HaveTargets && json.isMember("target_info")) {
        if (!target_info_cb.size()) { ERROR("empty target_info_cb"); break; }
        const Json::Value &json_target_info = json["target_info"];
        string json_name = json_target_info["target_name"].asString();
        string front_name = target_info_cb.front().first;
        if (front_name != json_name) { ERROR("target_info_cb ", front_name, " != ", json_name); break; }
        TargetInfo info;
        info.output = json_target_info["build_location"].asString();
        CopyJsonStringVector(json_target_info["compile_definitions"], &info.compile_definitions);
        CopyJsonStringVector(json_target_info["compile_options"],     &info.compile_options);
        CopyJsonStringVector(json_target_info["include_directories"], &info.include_directories);
        CopyJsonStringVector(json_target_info["link_libraries"],      &info.link_libraries);
        CopyJsonStringVector(json_target_info["object_sources"],      &info.sources);
        CopyJsonStringVector(json_target_info["header_sources"],      &info.sources);
        target_info_cb.front().second(info);
        target_info_cb.pop_front();
      } else if (state >= HaveTargets && json.isMember("completion")) {
        CHECK_NE(nullptr, code_completions_done);
        CHECK_NE(nullptr, code_completions_out);
        const Json::Value &completion_commands = json["completion"]["commands"];
        if (int l = completion_commands.size()) {
          unique_ptr<CodeCompletionsVector> ret = make_unique<CodeCompletionsVector>();
          for (int i=0; i != l; ++i) ret->data.emplace_back(completion_commands[i].asString());
          *code_completions_out = move(ret);
        }
        code_completions_done->Signal();
      }
    } while(0);
    c->ReadFlush(end + Proto::footer.size() - c->rb.begin());
  }
}

bool CMakeDaemon::GetTargetInfo(const string &target, TargetInfoCB &&cb) {
  if (!Contains(targets, target)) return false;
  dispatch->RunInNetworkThread([=](){
    CHECK(FWriteSuccess(process.out, Proto::MakeTargetInfo(target, ""))); 
    //CHECK(FWriteSuccess(process.out, Proto::MakeFileInfo("lterm", "/Users/p/lfl/core/app/app.h", ""))); 
    target_info_cb.emplace_back(target, cb);
  });
  return true;
}

unique_ptr<CodeCompletions> CMakeDaemon::CompleteCode(const string &fn, int y, int x, const string &content) {
  Semaphore done;
  unique_ptr<CodeCompletions> ret;
  dispatch->RunInNetworkThread([&](){
    CHECK_EQ(nullptr, code_completions_done);
    CHECK_EQ(nullptr, code_completions_out);
    CHECK(FWriteSuccess(process.out, Proto::MakeCodeComplete(fn, y, x, content))); 
    code_completions_done = &done;
    code_completions_out = &ret;
  });
  done.Wait();
  code_completions_done = nullptr;
  code_completions_out = nullptr;
  return ret;
}

}; // namespace LFL
