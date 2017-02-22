/*
 * $Id: video.cpp 1336 2014-12-08 09:29:59Z justin $
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
  @property (nonatomic)         BOOL started, displayed;
@end

@implementation IOSAdMob
  - (id)initWithAdUnitID:(NSString*)adid {
    self = [super init];
    _banner = [[GADBannerView alloc] initWithAdSize:kGADAdSizeBanner];
    _banner.adUnitID = adid;
    _banner.delegate = self;
    [_banner setAutoresizingMask:UIViewAutoresizingFlexibleTopMargin|UIViewAutoresizingFlexibleWidth];
    _testDevices = [[NSMutableArray alloc] init];
    [_testDevices addObject: kGADSimulatorID];
    return self;
  }

  - (void)updateBannerFrame {
    int bh = _banner.frame.size.height;
    CGRect frame = _banner.rootViewController.view.frame;
    [_banner setFrame: CGRectMake(frame.origin.x, frame.origin.y+frame.size.height-bh, frame.size.width, bh)];
  }

  - (void)show: (BOOL)show_or_hide withTable:(UIViewController*)table {
    if (show_or_hide) {
      _displayed = NO;
      _banner.rootViewController = table;
      [self updateBannerFrame];
      if (!_started && (_started = YES)) {
        GADRequest *request = [GADRequest request];
        request.testDevices = _testDevices;
        [_banner loadRequest:request];
      }
      [_banner.rootViewController.view addSubview:_banner];
    }
  }

  - (void)adViewDidReceiveAd:(GADBannerView *)view {
    [self updateBannerFrame];
    if (!_displayed && (_displayed = YES)) {
      if (auto table = LFL::objc_dynamic_cast<IOSTable>(_banner.rootViewController)) {
         int bh = _banner.frame.size.height;
         CGRect frame = table.tableView.frame;
         frame.size.height -= bh;
         table.tableView.frame = frame;
         if (table.toolbar) {
            frame = table.toolbar.frame;
            frame.origin.y -= bh;
            table.toolbar.frame = frame;
         }
      }
    }
  }
@end

namespace LFL {
struct iOSAdvertisingView : public SystemAdvertisingView {
  IOSAdMob *admob;
  virtual ~iOSAdvertisingView() { [admob release]; }
  iOSAdvertisingView(int type, int placement, const string &adid, const StringVec &test_devices) :
    admob([[IOSAdMob alloc] initWithAdUnitID: MakeNSString(adid)]) {
    // INFO("Starting iOS AdMob with DeviceId = ", DeviceId());
    for (auto id : test_devices) [admob.testDevices addObject: MakeNSString(id)];
  }

  void Show(bool show_or_hide) {}
  void Show(SystemTableView *t, bool show_or_hide) {
    [admob show: show_or_hide withTable: dynamic_cast<iOSTableView*>(t)->table];
  }

  static string DeviceId() {
    return HexEscape
      (Crypto::MD5(GetNSString([[ASIdentifierManager sharedManager] advertisingIdentifier].UUIDString)), "");
  }
};

unique_ptr<SystemAdvertisingView> SystemAdvertisingView::Create(int type, int placement, const string &adid, const StringVec &test_devices) {
  return make_unique<iOSAdvertisingView>(type, placement, adid, test_devices);
}

}; // namespace LFL
