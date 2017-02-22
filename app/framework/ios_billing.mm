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

#import <StoreKit/StoreKit.h>

#include "core/app/app.h"
#include "core/app/framework/apple_common.h"

namespace LFL {
struct iOSProduct : public SystemProduct {
  SKProduct *product;
  ~iOSProduct() { [product release]; }
  iOSProduct(const string &i, SKProduct *p) : SystemProduct(i), product(p) { [product retain]; }
  string Name() { return GetNSString(product.localizedTitle); }
  string Description() { return GetNSString(product.localizedDescription); }
  string Price() {
    NSNumberFormatter *numberFormatter = [[NSNumberFormatter alloc] init];
    [numberFormatter setFormatterBehavior:NSNumberFormatterBehavior10_4];
    [numberFormatter setNumberStyle:NSNumberFormatterCurrencyStyle];
    [numberFormatter setLocale:product.priceLocale];
    string ret = GetNSString([numberFormatter stringFromNumber:product.price]);
    [numberFormatter release];
    return ret;
  }
};
}; // namespace LFL

@interface IOSProductRequest : NSObject<SKProductsRequestDelegate>
  @property (nonatomic, retain) SKProductsRequest *productsRequest;
@end

@implementation IOSProductRequest
  {
    LFL::Callback done_cb;
    LFL::SystemPurchases::ProductCB product_cb;
  }

  - (id)initWithProducts:(NSSet*)products doneCB:(LFL::Callback)dcb productCB:(LFL::SystemPurchases::ProductCB)pcb {
    self = [super init];
    done_cb = LFL::move(dcb);
    product_cb = LFL::move(pcb);
    for (NSString *p in products) INFO("request product '", LFL::GetNSString(p), "'");
    _productsRequest = [[SKProductsRequest alloc] initWithProductIdentifiers:products];

    _productsRequest.delegate = self;
    [_productsRequest start];
    return self;
  }

  - (void)productsRequest:(SKProductsRequest *)request didReceiveResponse:(SKProductsResponse *)response {
    INFO("product request response");
    for (SKProduct *product in response.products) {
      INFO("omg calling product");
      product_cb(LFL::make_unique<LFL::iOSProduct>(LFL::GetNSString(product.productIdentifier), product));
    }
    if (done_cb) done_cb();
    [self release];
  }
@end

@interface IOSReceiptRequest : NSObject<SKRequestDelegate>
  @property (nonatomic, retain) SKReceiptRefreshRequest *receiptRequest;
@end

@implementation IOSReceiptRequest
  {
    LFL::Callback done_cb;
  }

  - (id)initWithDoneCB:(LFL::Callback)dcb {
    self = [super init];
    done_cb = LFL::move(dcb);
    _receiptRequest = [[SKReceiptRefreshRequest alloc] init];
    _receiptRequest.delegate = self;
    [_receiptRequest start];
    return self;
  }

  - (void)requestDidFinish:(SKRequest *)request {
    if (done_cb) done_cb();
    [self release];
  }

  - (void)request:(SKRequest *)request didFailWithError:(NSError *)error {
    if (done_cb) done_cb();
    [self release];
  }
@end

@interface IOSPurchaseManager : NSObject<SKPaymentTransactionObserver>
@end

@implementation IOSPurchaseManager
  {
    LFL::unordered_map<LFL::string, LFL::IntCB> outstanding_purchase;
  }

  - (id) init {
    self = [super init];
    [[SKPaymentQueue defaultQueue] addTransactionObserver:self];
    return self;
  }

  - (void)paymentQueue:(SKPaymentQueue *)queue updatedTransactions:(NSArray *)transactions {
    for (SKPaymentTransaction *transaction in transactions) {
      switch (transaction.transactionState) {
        case SKPaymentTransactionStatePurchased:
          [self finishTransaction:transaction wasSuccessful:YES];
          break;
        case SKPaymentTransactionStateFailed:
          if (transaction.error.code == SKErrorPaymentCancelled)
            [[SKPaymentQueue defaultQueue] finishTransaction:transaction];
          else 
            [self finishTransaction:transaction wasSuccessful:NO];
          break;
        case SKPaymentTransactionStateRestored:
          [self finishTransaction:transaction wasSuccessful:YES];
          break;
        default:
          break;
      }
    }
  }

  - (bool)makePurchase:(LFL::iOSProduct*)product withCB:(LFL::IntCB)cb {
    if (LFL::Contains(outstanding_purchase, product->id)) return false;
    outstanding_purchase[product->id] = LFL::move(cb);
    SKPayment *payment = [SKPayment paymentWithProduct: product->product];
    [[SKPaymentQueue defaultQueue] addPayment:payment];
    return true;
  }

  - (void)finishTransaction:(SKPaymentTransaction *)transaction wasSuccessful:(BOOL)wasSuccessful {
    LFL::string product_id = LFL::GetNSString(transaction.payment.productIdentifier);
    [[SKPaymentQueue defaultQueue] finishTransaction:transaction];
    auto it = outstanding_purchase.find(product_id);
    if (it != outstanding_purchase.end()) {
      it->second(wasSuccessful);
      outstanding_purchase.erase(it);
    }
  }
@end

namespace LFL {
struct iOSPurchases : public SystemPurchases {
  IOSPurchaseManager *purchaser;
  unordered_set<string> purchases;
  ~iOSPurchases() { [purchaser release]; }
  iOSPurchases() : purchaser([[IOSPurchaseManager alloc] init]) { LoadPurchases(); }

  bool CanPurchase() { return [SKPaymentQueue canMakePayments]; }
  bool HavePurchase(const string &product_id) { return Contains(purchases, product_id); }

  void PreparePurchase(const StringVec &products, Callback done_cb, ProductCB product_cb) {
    [[IOSProductRequest alloc] initWithProducts:MakeNSStringSet(products) doneCB:move(done_cb) productCB:move(product_cb)];
  }

  bool MakePurchase(SystemProduct *product, IntCB result_cb) {
    return [purchaser makePurchase:dynamic_cast<iOSProduct*>(product) withCB:result_cb];
  }

  void RestorePurchases(Callback done_cb) {
    [[IOSReceiptRequest alloc] initWithDoneCB:move(done_cb)];
  }

  void LoadPurchases() {
    purchases.clear();
    NSURL *url = [NSBundle mainBundle].appStoreReceiptURL;
  }
};

unique_ptr<SystemPurchases> SystemPurchases::Create() { return make_unique<iOSPurchases>(); }
}; // namespace LFL
