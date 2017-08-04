/*
 * $Id$
 * Copyright (C) 2009 Lucid Fusion Labs

 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#import <UIKit/UIKit.h>
#import <GLKit/GLKit.h>
#import <GoogleMobileAds/GoogleMobileAds.h>
#import <AdSupport/ASIdentifierManager.h>

#include "core/app/app.h"
#include "core/app/crypto.h"
#include "core/app/framework/apple_common.h"
#include "core/app/framework/ios_common.h"

@interface IOSAdMob : NSObject<GADBannerViewDelegate>
  @property (nonatomic, retain) GADBannerView *banner;
  @property (nonatomic, retain) NSMutableArray *testDevices;
  @property (nonatomic)         BOOL loaded, shown;
@end

@implementation IOSAdMob
  - (id)initWithAdUnitID:(NSString*)adid andTestDevices:(const LFL::StringVec&)test_devices {
    self = [super init];
    _banner = [[GADBannerView alloc] initWithAdSize:kGADAdSizeBanner];
    _banner.adUnitID = adid;
    _banner.delegate = self;
    [_banner setAutoresizingMask:UIViewAutoresizingFlexibleTopMargin|UIViewAutoresizingFlexibleWidth];

    _testDevices = [[NSMutableArray alloc] init];
    [_testDevices addObject: kGADSimulatorID];
    for (auto &id : test_devices) [_testDevices addObject: LFL::MakeNSString(id)];

    GADRequest *request = [GADRequest request];
    request.testDevices = _testDevices;
    [_banner loadRequest:request];
    return self;
  }

  - (void)updateBannerFrame {
    int bh = _banner.frame.size.height;
    CGRect frame = _banner.rootViewController.view.frame;
    [_banner setFrame: CGRectMake(frame.origin.x, frame.origin.y+frame.size.height-bh, frame.size.width, bh)];
  }

  - (void)show: (BOOL)show_or_hide withTable:(UIViewController*)table {
    [_banner removeFromSuperview];
    if ((_shown = show_or_hide)) {
      _banner.rootViewController = table;
      [self updateBannerFrame];
      if (!_loaded && (_loaded = YES)) [self adjustParentFrame: -_banner.frame.size.height];
      [_banner.rootViewController.view addSubview:_banner];
    } else {
      if (_loaded && !(_loaded = NO)) [self adjustParentFrame: _banner.frame.size.height];
    }
  }

  - (void)adViewDidReceiveAd:(GADBannerView *)view {
    [self updateBannerFrame];
    if (_shown && !_loaded && (_loaded = YES)) [self adjustParentFrame: -_banner.frame.size.height];
  }

  - (void)adjustParentFrame: (int)bh {
    if (auto table = LFL::objc_dynamic_cast<IOSTable>(_banner.rootViewController)) {
       CGRect frame = table.tableView.frame;
       frame.size.height += bh;
       table.tableView.frame = frame;
       if (table.toolbar) {
          frame = table.toolbar.frame;
          frame.origin.y += bh;
          table.toolbar.frame = frame;
       }
    }
  }
@end

namespace LFL {
struct iOSAdvertisingView : public AdvertisingViewInterface {
  IOSAdMob *admob;
  virtual ~iOSAdvertisingView() { [admob release]; }
  iOSAdvertisingView(int type, int placement, const string &adid, const StringVec &test_devices) :
    admob([[IOSAdMob alloc] initWithAdUnitID: MakeNSString(adid) andTestDevices:test_devices]) {
    // INFO("Starting iOS AdMob with DeviceId = ", DeviceId());
  }

  void Show(bool show_or_hide) {}
  void Show(TableViewInterface *t, bool show_or_hide) {
    [admob show: show_or_hide withTable: dynamic_cast<iOSTableView*>(t)->table];
  }

  static string DeviceId() {
    return HexEscape
      (Crypto::MD5(GetNSString([[ASIdentifierManager sharedManager] advertisingIdentifier].UUIDString)), "");
  }
};

void SystemToolkit::DisableAdvertisingCrashReporting() {
  [GADMobileAds disableSDKCrashReporting];
}

unique_ptr<AdvertisingViewInterface> SystemToolkit::CreateAdvertisingView(int type, int placement, const string &adid, const StringVec &test_devices) {
  return make_unique<iOSAdvertisingView>(type, placement, adid, test_devices);
}

}; // namespace LFL
