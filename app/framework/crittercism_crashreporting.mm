#import "Crittercism.h"
#include "apple_common.h"

namespace LFL {
void InitCrashReporting(const string &id, const string &name, const string &email) {
  [Crittercism enableWithAppID: MakeNSString(id)];
}

} // namespace LFL
