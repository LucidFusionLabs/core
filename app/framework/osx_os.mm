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

#import <Cocoa/Cocoa.h>
#include "core/app/app.h"
#include "core/app/framework/apple_common.h"

namespace LFL {
void SystemBrowser::OpenSystemBrowser(const string &url_text) {
  CFURLRef url = CFURLCreateWithBytes(0, MakeUnsigned(url_text.c_str()), url_text.size(), kCFStringEncodingASCII, 0);
  if (url) { LSOpenCFURLRef(url, 0); CFRelease(url); }
}

string ApplicationInfo::GetSystemDeviceName() {
  string ret(1024, 0);
  if (gethostname(&ret[0], ret.size())) return "";
  ret.resize(strlen(ret.c_str()));
  vector<string> word;
  Split(ret, isdot, &word);
  return word.size() ? word[0] : ret;
}

string ApplicationInfo::GetSystemDeviceId() {
  mach_port_t master_port;
  kern_return_t kernResult = IOMasterPort(MACH_PORT_NULL, &master_port);
  if (kernResult != KERN_SUCCESS) return ERRORv("", "IOMasterPort");

  CFMutableDictionaryRef matching_dict = IOBSDNameMatching(master_port, 0, "en0");
  if (!matching_dict) return ERRORv("", "IOBSDNameMatching");

  io_iterator_t iterator;
  kernResult = IOServiceGetMatchingServices(master_port, matching_dict, &iterator);
  if (kernResult != KERN_SUCCESS) return ERRORv("", "IOServiceGetMatchingServices");

  CFDataRef guid_cf_data = nil;
  io_object_t service, parent_service;
  while((service = IOIteratorNext(iterator)) != 0) {
    kernResult = IORegistryEntryGetParentEntry(service, kIOServicePlane, &parent_service);
    if (kernResult == KERN_SUCCESS) {
      if (guid_cf_data) CFRelease(guid_cf_data);
      guid_cf_data = (CFDataRef) IORegistryEntryCreateCFProperty(parent_service, CFSTR("IOMACAddress"), NULL, 0);
      IOObjectRelease(parent_service);
    }
    IOObjectRelease(service);
    if (guid_cf_data) break;
  }
  IOObjectRelease(iterator);

  string ret;
  if (guid_cf_data) ret.assign(MakeSigned(CFDataGetBytePtr(guid_cf_data)), CFDataGetLength(guid_cf_data));
  return ret;
}

Connection *Networking::ConnectTCP(const string &hostport, int default_port, Connection::CB *connected_cb, bool background_services) {
#if 0
  INFO("Networking::ConnectTCP ", hostport, " (default_port = ", default_port, ") background_services = ", background_services);
  if (background_services) {
    static SocketService svc(net.get(), "SystemNetworking");
    return new NSURLSessionStreamConnection(&svc, hostport, default_port, connected_cb ? move(*connected_cb) : Connection::CB());
  } else return net->tcp_client->Connect(hostport, default_port, connected_cb);
#else
  INFO("Networking::ConnectTCP ", hostport, " (default_port = ", default_port, ") background_services = false"); 
  return net->tcp_client->Connect(hostport, default_port, connected_cb);
#endif
}

}; // namespace LFL
