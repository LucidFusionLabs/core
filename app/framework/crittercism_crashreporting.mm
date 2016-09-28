#import "Crittercism.h"
#include "apple_common.h"
namespace LFL {
void InitCrashReporting(const string &id) { [Crittercism enableWithAppID: MakeNSString(id)]; }
} // namespace LFL
