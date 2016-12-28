/*
 * $Id: video.cpp 1336 2014-12-08 09:29:59Z justin $
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

#import <Cocoa/Cocoa.h>
#include "core/app/app.h"
#include "core/app/framework/apple_common.h"
#include "core/app/framework/osx_common.h"

namespace LFL {
SystemAdvertisingView::SystemAdvertisingView() {}
void SystemAdvertisingView::Show() {}
void SystemAdvertisingView::Hide() {}

void Application::OpenSystemBrowser(const string &url_text) {
  CFURLRef url = CFURLCreateWithBytes(0, MakeUnsigned(url_text.c_str()), url_text.size(), kCFStringEncodingASCII, 0);
  if (url) { LSOpenCFURLRef(url, 0); CFRelease(url); }
}

string Application::GetSystemDeviceName() {
  string ret(1024, 0);
  if (gethostname(&ret[0], ret.size())) return "";
  ret.resize(strlen(ret.c_str()));
  return ret;
}

Connection *Application::ConnectTCP(const string &hostport, int default_port, Connection::CB *connected_cb, bool background_services) {
#if 0
  INFO("Application::ConnectTCP ", hostport, " (default_port = ", default_port, ") background_services = ", background_services);
  if (background_services) return new NSURLSessionStreamConnection(hostport, default_port, connected_cb ? move(*connected_cb) : Connection::CB());
  else return app->net->tcp_client->Connect(hostport, default_port, connected_cb);
#else
  INFO("Application::ConnectTCP ", hostport, " (default_port = ", default_port, ") background_services = false"); 
  return app->net->tcp_client->Connect(hostport, default_port, connected_cb);
#endif
}

}; // namespace LFL
