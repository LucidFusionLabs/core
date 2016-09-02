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

}; // namespace LFL
