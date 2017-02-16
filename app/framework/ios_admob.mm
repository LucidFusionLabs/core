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

@interface IOSAdMob : NSObject
@end

@implementation IOSAdMob
  static GADBannerView *bannerView;
  static UIView *senderView;
  static UIView *containerView;
  static UIView *bannerContainerView;
  static float bannerHeight;

  + (void)createBanner:(UIViewController *)sender {
    if ([UIDevice currentDevice].userInterfaceIdiom == UIUserInterfaceIdiomPhone) bannerHeight = 50;
    else bannerHeight = 90;

    GADRequest *request = [GADRequest request];
    // request.testDevices = [NSArray arrayWithObjects:GAD_SIMULATOR_ID, nil];
    bannerView = [[GADBannerView alloc] initWithAdSize:kGADAdSizeBanner];
    bannerView.adUnitID = @"ca-app-pub-xxxxxxxxxxxxxxxx/xxxxxxxxxx";
    bannerView.rootViewController = (id)self;
    bannerView.delegate = (id<GADBannerViewDelegate>)self;
    senderView = sender.view;
    bannerView.frame = CGRectMake(0, 0, senderView.frame.size.width, bannerHeight);
    [bannerView loadRequest:request];
    containerView = [[UIView alloc] initWithFrame:senderView.frame];
    bannerContainerView = [[UIView alloc] initWithFrame:CGRectMake(0, senderView.frame.size.height, senderView.frame.size.width, bannerHeight)];

    for (id object in sender.view.subviews) {
        [object removeFromSuperview];
        [containerView addSubview:object];
    }

    [senderView addSubview:containerView];
    [senderView addSubview:bannerContainerView];
  }

  + (void)adViewDidReceiveAd:(GADBannerView *)view {
    [UIView animateWithDuration:0.5 animations:^{
      containerView.frame = CGRectMake(0, 0, senderView.frame.size.width, senderView.frame.size.height - bannerHeight);
      bannerContainerView.frame = CGRectMake(0, senderView.frame.size.height - bannerHeight, senderView.frame.size.width, bannerHeight);
      [bannerContainerView addSubview:bannerView];
    }];
  }
@end

namespace LFL {
struct iOSAdvertisingView : public SystemAdvertisingView {
  iOSAdvertisingView(int type, int placement) {}
  void Show(bool show_or_hide) {
  }
};

unique_ptr<SystemAdvertisingView> SystemAdvertisingView::Create(int type, int placement) { return make_unique<iOSAdvertisingView>(type, placement); }
}; // namespace LFL
