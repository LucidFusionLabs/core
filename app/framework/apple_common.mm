#import <GLKit/GLKit.h>

#include "core/app/app.h"
#include "core/app/framework/apple_common.h"

namespace LFL {
void NSLogString(const string &text) { NSLog(@"%@", MakeNSString(text)); }
string GetNSDocumentDirectory() {
  NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
  return GetNSString([paths objectAtIndex:0]);
}
}; // namespace LFL
