#import <HockeySDK.h>
#include "apple_common.h"

@interface LFHockeyApp : NSObject<BITHockeyManagerDelegate> {}
  @property (nonatomic, retain) NSString *user, *email;
@end

@implementation LFHockeyApp
  - (NSString *)userNameForHockeyManager:(BITHockeyManager *)hockeyManager componentManager:(BITHockeyBaseManager *)componentManager { return self.user; }
  - (NSString *)userEmailForHockeyManager:(BITHockeyManager *)hockeyManager componentManager:(BITHockeyBaseManager *)componentManager { return self.email; }
  + (id)singleton {
    static LFHockeyApp *inst = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{ inst = [[self alloc] init]; });
    return inst;
  }
@end

namespace LFL {
void InitCrashReporting(const string &id, const string &name, const string &email) {
  LFHockeyApp *ha = [LFHockeyApp singleton];
  ha.user = MakeNSString(name);
  ha.email = MakeNSString(email);
  [[BITHockeyManager sharedHockeyManager] configureWithIdentifier: MakeNSString(id)];
  [BITHockeyManager sharedHockeyManager].delegate = ha;
  [[BITHockeyManager sharedHockeyManager] startManager];
}

} // namespace LFL
