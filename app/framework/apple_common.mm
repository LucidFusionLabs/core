#import <GLKit/GLKit.h>

#include "core/app/app.h"
#include "core/app/framework/apple_common.h"

@implementation ObjcCallback { LFL::Callback cb; }
  - (id)initWithCB:(LFL::Callback)v { self = [super init]; cb = move(v); return self; }
  - (void)run { if (cb) cb(); }
@end

@implementation ObjcStringCallback { LFL::StringCB cb; }
  - (id)initWithStringCB:(LFL::StringCB)v { self = [super init]; cb = move(v); return self; }
  - (void)run:(const LFL::string&)a { if (cb) cb(a); }
@end

@implementation ObjcFileHandleCallback
  {
    std::function<bool()> cb;
    id<ObjcWindow> win;
  }

  - (id)initWithCB:(std::function<bool()>)v forWindow:(id<ObjcWindow>)w fileDescriptor:(int)fd {
    self = [super init];
    cb = move(v);
    win = w;
    _fh = [[NSFileHandle alloc] initWithFileDescriptor:fd];
    [[NSNotificationCenter defaultCenter] addObserver:self
      selector:@selector(fileDataAvailable:) name:NSFileHandleDataAvailableNotification object:_fh];
    [_fh waitForDataInBackgroundAndNotify];
    return self;
  }

  - (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self
      name:NSFileHandleDataAvailableNotification object:_fh];
    [super dealloc];
  }

  - (void)fileDataAvailable: (NSNotification *)notification {
    [win objcWindowSelect];
    if (cb()) [win objcWindowFrame];
    [_fh waitForDataInBackgroundAndNotify];
  }
@end

namespace LFL {
void NSLogString(const string &text) { NSLog(@"%@", MakeNSString(text)); }

string GetNSDocumentDirectory() {
  NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
  return GetNSString([paths objectAtIndex:0]);
}

NSArray *MakeNSStringArray(const vector<string> &v) {
  vector<NSString*> vs;
  for (auto &vi : v) vs.push_back(MakeNSString(vi));
  return [NSArray arrayWithObjects:&vs[0] count:vs.size()];
}

string Application::GetVersion() {
  NSDictionary *info = [[NSBundle mainBundle] infoDictionary];
  NSString *version = [info objectForKey:@"CFBundleVersion"];
  return version ? GetNSString(version) : "";
}

String16 Application::GetLocalizedString16(const char *key) { return String16(); }
string Application::GetLocalizedString(const char *key) {
  NSString *localized = 
    [[NSBundle mainBundle] localizedStringForKey: [NSString stringWithUTF8String: key] value:nil table:nil];
  if (!localized) return StrCat("<missing localized: ", key, ">");
  else            return [localized UTF8String];
}

String16 Application::GetLocalizedInteger16(int number) { return String16(); }
string Application::GetLocalizedInteger(int number) {
  static NSNumberFormatter *formatter=0;
  static dispatch_once_t onceToken;
  dispatch_once(&onceToken, ^{
    formatter = [[NSNumberFormatter alloc] init];
    [formatter setNumberStyle:NSNumberFormatterDecimalStyle];
    [formatter setMaximumFractionDigits:0];
    [formatter setMinimumIntegerDigits:1];
    [formatter setLocale:[NSLocale autoupdatingCurrentLocale]];
  });
  return [[formatter stringFromNumber: [NSNumber numberWithLong:number]] UTF8String];
}

void Application::LoadDefaultSettings(const StringPairVec &v) {
  NSMutableDictionary *defaults = [[NSMutableDictionary alloc] init];
  for (auto &i : v) [defaults setValue:MakeNSString(i.second) forKey:MakeNSString(i.first)];
  [[NSUserDefaults standardUserDefaults] registerDefaults:defaults];
  [defaults release];
}

string Application::GetSetting(const string &key) {
  return GetNSString([[NSUserDefaults standardUserDefaults] stringForKey:MakeNSString(key)]);
}

void Application::SaveSettings(const StringPairVec &v) {
  NSUserDefaults* defaults = [NSUserDefaults standardUserDefaults];
  for (auto &i : v) [defaults setObject:MakeNSString(i.second) forKey:MakeNSString(i.first)];
  [defaults synchronize];
}

}; // namespace LFL
