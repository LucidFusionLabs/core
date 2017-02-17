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

#import <GoogleMobileAds/GoogleMobileAds.h>

#include "core/app/app.h"
#include "core/app/framework/apple_common.h"

@interface IOSAdMob : NSObject
@end

@implementation IOSAdMob
  float bannerHeight;
  UIView *bannerContainerView;
  GADBannerView *bannerView;

  - (id)initWithFrame:(CGRect)r adUnitID:(NSString*)adid {
    self = [super init];
    if ([UIDevice currentDevice].userInterfaceIdiom == UIUserInterfaceIdiomPhone) bannerHeight = 50;
    else bannerHeight = 90;
    bannerContainerView = [[UIView alloc] initWithFrame:CGRectMake(0, r.size.height, r.size.width, bannerHeight)];

    bannerView = [[GADBannerView alloc] initWithAdSize:kGADAdSizeBanner];
    bannerView.adUnitID = adid;
    bannerView.rootViewController = (id)self;
    bannerView.delegate = (id<GADBannerViewDelegate>)self;
    bannerView.frame = CGRectMake(0, 0, r.size.width, bannerHeight);

    GADRequest *request = [GADRequest request];
    request.testDevices = [NSArray arrayWithObjects:kGADSimulatorID, nil];
    [bannerView loadRequest:request];
    return self;
  }

  + (void)adViewDidReceiveAd:(GADBannerView *)view {
  #if 0
    [UIView animateWithDuration:0.5 animations:^{
      containerView.frame = CGRectMake(0, 0, senderView.frame.size.width, senderView.frame.size.height - bannerHeight);
      bannerContainerView.frame = CGRectMake(0, senderView.frame.size.height - bannerHeight, senderView.frame.size.width, bannerHeight);
      [bannerContainerView addSubview:bannerView];
    }];
    #endif
  }
@end

namespace LFL {
struct iOSAdvertisingView : public SystemAdvertisingView {
  IOSAdMob *admob;
  virtual ~iOSAdvertisingView() { [admob release]; }
  iOSAdvertisingView(int type, int placement, const string &adid) :
    admob([[IOSAdMob alloc] initWithFrame: [[UIScreen mainScreen] bounds] adUnitID: MakeNSString(adid)]) {}

  void Show(bool show_or_hide) {
  }
};

unique_ptr<SystemAdvertisingView> SystemAdvertisingView::Create(int type, int placement, const string &adid) { return make_unique<iOSAdvertisingView>(type, placement, adid); }
}; // namespace LFL
