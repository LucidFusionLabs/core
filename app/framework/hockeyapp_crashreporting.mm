#import <HockeySDK.h>
#include "apple_common.h"

namespace LFL {
void InitCrashReporting(const string &id) {
  [[BITHockeyManager sharedHockeyManager] configureWithIdentifier: MakeNSString(id)];
  [[BITHockeyManager sharedHockeyManager] startManager];
}

} // namespace LFL
