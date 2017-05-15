/*
 * $Id: master.cpp 1306 2014-09-04 07:13:16Z justin $
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

#include "core/app/network.h"
#include "core/app/ipc.h"

namespace LFL {
#ifdef _WIN32
DEFINE_bool(install,    false,                          "Win32 Register Server");
DEFINE_bool(uninstall,  false,                          "Win32 Unregister Server");
#endif                  

DEFINE_int  (port,       27994,                          "Port");
DEFINE_bool (run_server, false,                          "Run server");

struct ServerList : public HTTPServer::Resource {
  struct Server { Time last_updated; };
  map<string, Server> servers;
  string serialized;
  Time timeout = Seconds(5*60+5);

  HTTPServer::Response Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
    if (method == HTTPServer::Method::POST) {
      string cn = c->Name();
      string url = cn.substr(0, cn.find(':')) + string(":") + postdata;
      Server server = { Now() };
      servers[url] = server;
      return HTTPServer::Response("text/html; charset=UTF-8", "");
    }
    else {
      serialized.clear();
      for (map<string, Server>::iterator i = servers.begin(); i != servers.end(); /**/) {
        if (i->second.last_updated + timeout < Now()) { servers.erase(i++); continue; }
        serialized += (*i).first + "\r\n";
        i++;
      }
      return HTTPServer::Response("text/html; charset=UTF-8", serialized.c_str());
    }
  }
};

int MasterServer(int argc, const char* const* argv) {
  HTTPServer httpd(FLAGS_port, false);
  if (app->net->Enable(&httpd)) return -1;
  httpd.AddURL("/favicon.ico", new HTTPServer::FileResource("./assets/icon.ico", "image/x-icon"));
  httpd.AddURL("/spaceball", new ServerList());

  INFO("LFL master server initialized");
  return app->Main();
}

}; // namespace LFL
using namespace LFL;

extern "C" void MyAppCreate(int argc, const char* const* argv) {
  FLAGS_enable_camera = FLAGS_enable_audio = FLAGS_enable_video = FLAGS_enable_input = 0;
  app = new Application(argc, argv);
  app->focused = Window::Create();
}

extern "C" int MyAppMain() {
  static const char *service_name = "LFL Master Server";
#ifdef _WIN32
  if (app->argc>1) FLAGS_open_console = 1;
#endif
  if (app->Create(__FILE__)) return ERRORv(-1, "lfapp init failed: ", strerror(errno));
  if (app->Init()) return ERRORv(-1, "lfapp open failed: ", strerror(errno));

  bool exit=0;
#ifdef _WIN32
  if (install) { NTService::Install(service_name, argv[0]); exit=1; }
  if (uninstall) { NTService::Uninstall(service_name); exit=1; }
#endif
  if (FLAGS_run_server) { return MasterServer(app->argc, app->argv); }
  if (exit) return -1;

  return NTService::WrapMain(service_name, MasterServer, app->argc, app->argv);
}
