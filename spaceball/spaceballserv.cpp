/*
 * $Id: spaceballserv.cpp 1314 2014-10-16 04:43:45Z justin $
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
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/gui.h"
#include "lfapp/game.h"

using namespace LFL;

#include "spaceballserv.h"

#ifdef _WIN32
DEFINE_bool(install,    false,                                 "Win32 Register Server");
DEFINE_bool(uninstall,  false,                                 "Win32 Unregister Server");
DEFINE_bool (run_server,false,                                 "Run server");
#else
DEFINE_bool (run_server,true,                                  "Run server");
#endif                                                         
                                                               
DEFINE_int  (port,      27640,                                 "Port");
DEFINE_int  (framerate, 10,                                    "Server framerate");
DEFINE_string(name,     "My Spaceball Server",                 "Server name");
DEFINE_string(master,   "lucidfusionlabs.com:27994/spaceball", "Master server list");
DEFINE_string(rconpw,   "",                                    "Admin password");

vector<Asset> assets;
SpaceballServer *server;

struct SpaceballStatusServer : public HTTPServer::Resource {
    char response[128];
    HTTPServer::Response Request(Connection *, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
        char time[32]; httptime(time, sizeof(time));
        snprintf(response, sizeof(response), "<html><h1>Spaceball Server %s</h1></html>\r\n", time);
        return HTTPServer::Response("text/html; charset=UTF-8", response);
    }
};

int Frame(LFL::Window *W, unsigned clicks, unsigned mic_samples, bool cam_sample, int flag) { return server->Frame(); }

int SpaceballServer(int argc, const char **argv) {
    FLAGS_target_fps = FLAGS_framerate;
    app->frame_cb = Frame;

    // assets.push_back(Asset(name,      texture, scale, trans, rotate, geometry  hull, cubemap, texgen));
    assets.push_back(Asset("ship",       "",      0,     0,     0,      0,        0,    0,       0     ));
    assets.push_back(Asset("shipred",    "",      0,     0,     0,      0,        0,    0,       0     ));
    assets.push_back(Asset("shipblue",   "",      0,     0,     0,      0,        0,    0,       0     ));
    assets.push_back(Asset("ball",       "",      0,     0,     0,      0,        0,    0,       0     ));
    Asset::Load(&assets);

    HTTPServer httpd(FLAGS_port, false);
    if (app->network.Enable(&httpd)) return -1;
    httpd.AddURL("/favicon.ico", new HTTPServer::FileResource("./assets/icon.ico", "image/x-icon"));
    httpd.AddURL("/", new SpaceballStatusServer());

    server = new LFL::SpaceballServer(FLAGS_name, FLAGS_port, FLAGS_framerate, &assets);
    if (!FLAGS_rconpw.empty()) server->rcon_auth_passwd = FLAGS_rconpw;
    if (!FLAGS_master.empty()) server->master_sink_url = FLAGS_master;
    if (app->network.Enable(server->udp_transport)) return -1;

    INFO("Spaceball 6006 server initialized");
    return app->Main();
}

extern "C" int main(int argc, const char **argv) {
    static const char *service_name = "Spaceball 6006 Server";
    app->logfilename = StrCat(LFAppDownloadDir(), "spaceballserv.txt");
    FLAGS_lfapp_camera = FLAGS_lfapp_audio = FLAGS_lfapp_video = FLAGS_lfapp_input = 0;

#ifdef _WIN32
    if (argc>1) open_console = 1;
#endif

    if (app->Create(argc, argv, __FILE__)) { ERROR("lfapp init failed: ", strerror(errno)); return app->Free(); }
    if (app->Init())                       { ERROR("lfapp open failed: ", strerror(errno)); return app->Free(); }

    bool exit=0;
#ifdef _WIN32
    if (install) { NTService::Install(service_name, argv[0]); exit=1; }
    if (uninstall) { NTService::Uninstall(service_name); exit=1; }
#endif
    if (FLAGS_run_server) { return ::SpaceballServer(argc, argv); }
    if (exit) return app->Free();

    return NTService::WrapMain(service_name, ::SpaceballServer, argc, argv);
}
