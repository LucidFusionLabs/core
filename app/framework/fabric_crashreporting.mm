#ifdef LFL_IOS
#import <UIKit/UIKit.h>
#endif

#import <Fabric.h>
#import <Crashlytics.h>

#include "apple_common.h"

namespace LFL {
void InitCrashReporting(const string &id, const string &name, const string &email) {
  [Fabric with:@[[Crashlytics class]]];
  // [CrashlyticsKit setUserIdentifier:@"12345"];
  if (name .size()) [CrashlyticsKit setUserName:  MakeNSString(name)];
  if (email.size()) [CrashlyticsKit setUserEmail: MakeNSString(name)];
}

void TestCrashReporting() {
  [[Crashlytics sharedInstance] crash];
}

} // namespace LFL
