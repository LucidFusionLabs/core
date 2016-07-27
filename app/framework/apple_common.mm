#import <GLKit/GLKit.h>

#include "core/app/app.h"
#include "core/app/framework/apple_common.h"

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
