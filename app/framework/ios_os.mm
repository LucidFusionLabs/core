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

#import <UIKit/UIKit.h>
#import <GLKit/GLKit.h>

#include "core/app/app.h"
#include "core/app/framework/apple_common.h"
#include "core/app/framework/ios_common.h"

@interface IOSKeychain : NSObject
  + (void)save:(NSString *)service data:(id)data;
  + (id)load:(NSString *)service;
@end

@implementation IOSKeychain
  + (NSMutableDictionary *)getKeychainQuery:(NSString *)service {
    return [NSMutableDictionary dictionaryWithObjectsAndKeys:(id)kSecClassGenericPassword, (id)kSecClass, service,
           (id)kSecAttrService, service, (id)kSecAttrAccount, (id)kSecAttrAccessibleAfterFirstUnlock, (id)kSecAttrAccessible, nil];
  }

  + (void)save:(NSString *)service data:(id)data {
    NSMutableDictionary *keychainQuery = [self getKeychainQuery:service];
    SecItemDelete((CFDictionaryRef)keychainQuery);
    [keychainQuery setObject:[NSKeyedArchiver archivedDataWithRootObject:data] forKey:(id)kSecValueData];
    SecItemAdd((CFDictionaryRef)keychainQuery, NULL);
  }

  + (id)load:(NSString *)service {
    id ret = nil;
    NSMutableDictionary *keychainQuery = [self getKeychainQuery:service];
    [keychainQuery setObject:(id)kCFBooleanTrue forKey:(id)kSecReturnData];
    [keychainQuery setObject:(id)kSecMatchLimitOne forKey:(id)kSecMatchLimit];
    CFDataRef keyData = NULL;
    if (SecItemCopyMatching((CFDictionaryRef)keychainQuery, (CFTypeRef *)&keyData) == noErr) {
      @try { ret = [NSKeyedUnarchiver unarchiveObjectWithData:(NSData *)keyData]; }
      @catch (NSException *e) { NSLog(@"Unarchive of %@ failed: %@", service, e); }
      @finally {}
    }
    if (keyData) CFRelease(keyData);
    return ret;
  }
@end

namespace LFL {
void Application::OpenSystemBrowser(const string &url_text) {
  NSString *url_string = [[NSString alloc] initWithUTF8String: url_text.c_str()];
  NSURL *url = [NSURL URLWithString: url_string];
  [[UIApplication sharedApplication] openURL:url];
  [url_string release];
}

bool Application::OpenSystemAppPreferences() {
  NSURL *url = [NSURL URLWithString:UIApplicationOpenSettingsURLString];
  [[UIApplication sharedApplication] openURL:url];
  return true;
}

void Application::SaveKeychain(const string &keyname, const string &val_in) {
  NSString *k = [[NSString stringWithFormat:@"%s://%s", name.c_str(), keyname.c_str()] retain];
  NSMutableString *val = [[NSMutableString stringWithUTF8String: val_in.c_str()] retain];
  UIAlertController *alertController = [UIAlertController alertControllerWithTitle: [NSString stringWithFormat: @"%s Keychain", name.c_str()]
    message:[NSString stringWithFormat:@"Save password for %s?", keyname.c_str()] preferredStyle:UIAlertControllerStyleAlert];
  UIAlertAction *actionNo  = [UIAlertAction actionWithTitle:@"No"  style:UIAlertActionStyleDefault handler: nil];
  UIAlertAction *actionYes = [UIAlertAction actionWithTitle:@"Yes" style:UIAlertActionStyleDefault
    handler:^(UIAlertAction *){
      [IOSKeychain save:k data: val];
      [val replaceCharactersInRange:NSMakeRange(0, [val length]) withString:[NSString stringWithFormat:@"%*s", int([val length]), ""]];
      [val release];
      [k release];
    }];
  [alertController addAction:actionYes];
  [alertController addAction:actionNo];
  [[LFUIApplication sharedAppDelegate].controller presentViewController:alertController animated:YES completion:nil];
}

bool Application::LoadKeychain(const string &keyname, string *val_out) {
  NSString *k = [NSString stringWithFormat:@"%s://%s", name.c_str(), keyname.c_str()];
  NSString *val = [IOSKeychain load: k];
  if (val) val_out->assign(GetNSString(val));
  else     val_out->clear();
  return   val_out->size();
}

string Application::GetPackageName() { return GetNSString([[NSBundle mainBundle] bundleIdentifier]); }
string Application::GetSystemDeviceName() { return GetNSString([[UIDevice currentDevice] name]); }
string Application::GetSystemDeviceId() {
  string ret(16, 0);
  [[[UIDevice currentDevice] identifierForVendor] getUUIDBytes: MakeUnsigned(&ret[0])];
  return ret;
}

Connection *Application::ConnectTCP(const string &hostport, int default_port, Connection::CB *connected_cb, bool background_services) {
  static bool ios9_or_later = kCFCoreFoundationVersionNumber >= kCFCoreFoundationVersionNumber_iOS_9_0;
  INFO("Application::ConnectTCP ", hostport, " (default_port = ", default_port, ") background_services = ", background_services, " ios_9=", ios9_or_later);
  if (background_services && ios9_or_later) return new NSURLSessionStreamConnection(hostport, default_port, connected_cb ? move(*connected_cb) : Connection::CB());
  else return app->net->tcp_client->Connect(hostport, default_port, connected_cb);
}

}; // namespace LFL
