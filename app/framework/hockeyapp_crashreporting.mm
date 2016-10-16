#import <HockeySDK.h>
#include "apple_common.h"

namespace LFL {
void InitCrashReporting(const string &id, const string &name, const string &email) {
  [[BITHockeyManager sharedHockeyManager] configureWithIdentifier: MakeNSString(id)];
  [[BITHockeyManager sharedHockeyManager] startManager];
}

} // namespace LFL
