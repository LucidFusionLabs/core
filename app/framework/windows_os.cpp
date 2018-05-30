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

#include <shellapi.h>
#include <commdlg.h>

namespace LFL {
string Application::PrintCallStack() { return ""; }
string Application::GetSetting(const string &key) { return string(); }
void Application::SaveSettings(const StringPairVec&) {}
void Application::LoadDefaultSettings(const StringPairVec &v) {}

string ApplicationInfo::GetVersion() { return string(); }
string ApplicationInfo::GetSystemDeviceName() {
  string ret(128, 0);
  DWORD len = ret.size();
  if (!GetComputerName(&ret[0], &len)) return "";
  ret.resize(len);
  return ret;
}

string ApplicationInfo::GetSystemDeviceId() {
  return "";
}
string Localization::GetLocalizedString(const char *key) { return string(); }
String16 Localization::GetLocalizedString16(const char *key) { return String16(); }
string Localization::GetLocalizedInteger(int number) { return string(); }
String16 Localization::GetLocalizedInteger16(int number) { return String16(); }

void SystemBrowser::OpenSystemBrowser(const string &url_text) {
  ShellExecute(NULL, "open", url_text.c_str(), NULL, NULL, SW_SHOWNORMAL);
}

Connection *Networking::ConnectTCP(const string &hostport, int default_port, Connection::CB *connected_cb, bool background_services) {
  INFO("Application::ConnectTCP ", hostport, " (default_port = ", default_port, ") background_services = false"); 
  return net->tcp_client->Connect(hostport, default_port, connected_cb);
}

}; // namespace LFL
