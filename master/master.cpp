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

#include "lfapp/lfapp.h"
#include "lfapp/network.h"

using namespace LFL;

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
    static const int Timeout = (5*60+5)*1000;

    HTTPServer::Response Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
        if (method == HTTPServer::Method::POST) {
            string cn = c->name();
            string url = cn.substr(0, cn.find(':')) + string(":") + postdata;
            Server server = { Now() };
            servers[url] = server;
            return HTTPServer::Response("text/html; charset=UTF-8", "");
        }
        else {
            serialized.clear();
            for (map<string, Server>::iterator i = servers.begin(); i != servers.end(); /**/) {
                if (i->second.last_updated + Timeout < Now()) { servers.erase(i++); continue; }
                serialized += (*i).first + "\r\n";
                i++;
            }
            return HTTPServer::Response("text/html; charset=UTF-8", serialized.c_str());
        }
    }
};

int master_server(int argc, const char **argv) {
    HTTPServer httpd(FLAGS_port, false);
    if (app->network.Enable(&httpd)) return -1;
    httpd.AddURL("/favicon.ico", new HTTPServer::FileResource("./assets/icon.ico", "image/x-icon"));
    httpd.AddURL("/spaceball", new ServerList());

    INFO("LFL master server initialized");
    return app->Main();
}

extern "C" {
int main(int argc, const char **argv) {
    app->logfilename = StrCat(dldir(), "masterserv.txt");
    static const char *service_name = "LFL Master Server";

    FLAGS_lfapp_camera = FLAGS_lfapp_audio = FLAGS_lfapp_video = FLAGS_lfapp_input = 0;

#ifdef _WIN32
    if (argc>1) open_console = 1;
#endif

    if (app->Create(argc, argv, __FILE__)) { ERROR("lfapp init failed: ", strerror(errno)); return app->Free(); }
    if (app->Init()) { ERROR("lfapp open failed: ", strerror(errno)); return app->Free(); }

    bool exit=0;
#ifdef _WIN32
    if (install) { NTService::Install(service_name, argv[0]); exit=1; }
    if (uninstall) { NTService::Uninstall(service_name); exit=1; }
#endif
    if (FLAGS_run_server) { return master_server(argc, argv); }
    if (exit) return app->Free();

    return NTService::MainWrapper(service_name, master_server, argc, argv);
}
}
