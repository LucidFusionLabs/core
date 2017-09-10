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

#import <StoreKit/StoreKit.h>

#include "core/app/app.h"
#include "core/app/crypto.h"
#include "core/app/framework/apple_common.h"

#include <openssl/pkcs7.h>
#include <openssl/x509_vfy.h>

namespace LFL {
struct AppleProduct : public ProductInterface {
  SKProduct *product;
  ~AppleProduct() { [product release]; }
  AppleProduct(const string &i, SKProduct *p) : ProductInterface(i), product(p) { [product retain]; }
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
    LFL::PurchasesInterface::ProductCB product_cb;
  }

  - (id)initWithProducts:(NSSet*)products doneCB:(LFL::Callback)dcb productCB:(LFL::PurchasesInterface::ProductCB)pcb {
    self = [super init];
    done_cb = LFL::move(dcb);
    product_cb = LFL::move(pcb);
    _productsRequest = [[SKProductsRequest alloc] initWithProductIdentifiers:products];

    _productsRequest.delegate = self;
    [_productsRequest start];
    return self;
  }

  - (void)productsRequest:(SKProductsRequest *)request didReceiveResponse:(SKProductsResponse *)response {
    for (SKProduct *product in response.products)
      product_cb(LFL::make_unique<LFL::AppleProduct>(LFL::GetNSString(product.productIdentifier), product));

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

  - (bool)makePurchase:(LFL::AppleProduct*)product withCB:(LFL::IntCB)cb {
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
struct ApplePurchases : public PurchasesInterface {
  struct ReceiptAttribute { enum { BundleIdentifier = 2, AppVersion = 3, OpaqueValue = 4, Hash = 5,
    InAppPurchaseReceipt = 17, OriginalAppVersion = 19, ExpirationDate = 21 }; };
  struct InAppPurchaseReceiptAttribute { enum { ProductIdentifier = 1702 }; };

  ApplicationInfo *appinfo;
  IOSPurchaseManager *purchaser;
  unordered_set<string> purchases;
  ~ApplePurchases() { [purchaser release]; }
  ApplePurchases(ApplicationInfo *a) : appinfo(a), purchaser([[IOSPurchaseManager alloc] init]) { LoadPurchases(); }

  bool CanPurchase() { return [SKPaymentQueue canMakePayments]; }
  bool HavePurchase(const string &product_id) { return Contains(purchases, product_id); }

  void PreparePurchase(const StringVec &products, Callback done_cb, ProductCB product_cb) {
    [[IOSProductRequest alloc] initWithProducts:MakeNSStringSet(products) doneCB:move(done_cb) productCB:move(product_cb)];
  }

  bool MakePurchase(ProductInterface *product, IntCB result_cb) {
    return [purchaser makePurchase:dynamic_cast<AppleProduct*>(product) withCB:result_cb];
  }

  void RestorePurchases(Callback done_cb) {
    [[IOSReceiptRequest alloc] initWithDoneCB:move(done_cb)];
  }

  void LoadPurchases() {
    INFO("AppleBilling::LoadPurchases");
    purchases.clear();
    unordered_set<string> next_purchases;
    NSURL *receiptURL = [NSBundle mainBundle].appStoreReceiptURL;
    NSData *receiptData = [NSData dataWithContentsOfURL:receiptURL];
    BIO *receiptBIO = BIO_new(BIO_s_mem());
    BIO_write(receiptBIO, [receiptData bytes], (int) [receiptData length]);
    PKCS7 *receiptPKCS7 = d2i_PKCS7_bio(receiptBIO, NULL);
    if (!receiptPKCS7) return ERROR("parse PKCS #7 failed");
    if (!PKCS7_type_is_signed(receiptPKCS7)) return ERROR("PKCS #7 not signed");
    if (!PKCS7_type_is_data(receiptPKCS7->d.sign->contents)) return ERROR("PKCS #7 missing payload");

    NSURL *appleRootURL = [[NSBundle mainBundle] URLForResource:@"AppleIncRootCertificate" withExtension:@"cer"];
    NSData *appleRootData = [NSData dataWithContentsOfURL:appleRootURL];
    BIO *appleRootBIO = BIO_new(BIO_s_mem());
    BIO_write(appleRootBIO, (const void *) [appleRootData bytes], (int) [appleRootData length]);
    X509 *appleRootX509 = d2i_X509_bio(appleRootBIO, NULL);
    X509_STORE *store = X509_STORE_new();
    X509_STORE_add_cert(store, appleRootX509);

    OpenSSL_add_all_digests();
    int result = PKCS7_verify(receiptPKCS7, NULL, store, NULL, NULL, 0);
    if (result != 1) return ERROR("receipt signature check failed");

    long length = 0;
    int type = 0, xclass = 0;
    ASN1_OCTET_STRING *octets = receiptPKCS7->d.sign->contents->d.data;
    const unsigned char *ptr = octets->data, *end = ptr + octets->length;
    string bundle_id, bundle_id_data, bundle_version, hash_data, opaque_data, product_id;
    if (!GetASN1ObjectType(V_ASN1_SET, &ptr, &length, &type, &xclass, end - ptr)) return ERROR("unexpected ASN1 type");
    for (/**/; ptr < end; ptr += length) {

      if (!GetASN1ObjectType(V_ASN1_SEQUENCE, &ptr, &length, &type, &xclass, end - ptr)) return ERROR("unexpected ASN1 type");
      const unsigned char *attr_end = ptr + length;
      long attr_type    = GetASN1Integer(&ptr, &length, &type, &xclass, end - ptr);
      long attr_version = GetASN1Integer(&ptr, &length, &type, &xclass, end - ptr);

      if (!GetASN1ObjectType(V_ASN1_OCTET_STRING, &ptr, &length, &type, &xclass, end - ptr)) return ERROR("unexpected ASN1 type");
      switch (attr_type) {
        case ReceiptAttribute::AppVersion:       bundle_version = GetASN1String(ptr, attr_end - ptr); break;
        case ReceiptAttribute::BundleIdentifier: bundle_id      = GetASN1String(ptr, attr_end - ptr);
                                                 bundle_id_data.assign(MakeSigned(ptr), length); break;
        case ReceiptAttribute::OpaqueValue:      opaque_data   .assign(MakeSigned(ptr), length); break;
        case ReceiptAttribute::Hash:             hash_data     .assign(MakeSigned(ptr), length); break;
        case ReceiptAttribute::InAppPurchaseReceipt:
          if ((product_id = GetInAppPurchase(ptr, attr_end)).size()) next_purchases.insert(product_id);
          break;
      }
    }

    if (bundle_id != appinfo->GetPackageName()) return ERROR("wrong receipt bundle id");

    Crypto::Digest sha1 = Crypto::DigestOpen(Crypto::DigestAlgos::SHA1());
    Crypto::DigestUpdate(sha1, appinfo->GetSystemDeviceId());
    Crypto::DigestUpdate(sha1, opaque_data);
    Crypto::DigestUpdate(sha1, bundle_id_data);
    string sha1hash = Crypto::DigestFinish(sha1);
    if (sha1hash != hash_data) return ERROR("invalid hash");

    swap(purchases, next_purchases);
    INFO("AppleBilling: Receipt Verified: ", purchases.size(), " In-App Purchases");
  }

  static bool GetASN1ObjectType(int expected_type, const unsigned char **ptr, long *outlen, int *type, int *xclass, int len) {
    ASN1_get_object(ptr, outlen, type, xclass, len);
    return expected_type == *type;
  }

  static long GetASN1Integer(const unsigned char **ptr, long *outlen, int *type, int *xclass, int len) {
    ASN1_get_object(ptr, outlen, type, xclass, len);
    if (*type != V_ASN1_INTEGER) return 0;
    ASN1_INTEGER *integer = c2i_ASN1_INTEGER(NULL, ptr, *outlen);
    long ret = ASN1_INTEGER_get(integer);
    ASN1_INTEGER_free(integer);
    return ret;
  }

  static string GetASN1String(const unsigned char *str_ptr, int len) {
    long str_length = 0;
    int str_type = 0, str_xclass = 0;
    ASN1_get_object(&str_ptr, &str_length, &str_type, &str_xclass, len);
    return str_type == V_ASN1_UTF8STRING ? string(MakeSigned(str_ptr), str_length) : string();
  }

  static string GetInAppPurchase(const unsigned char *ptr, const unsigned char *end) {
    long length = 0;
    int type = 0, xclass = 0;
    if (!GetASN1ObjectType(V_ASN1_SET, &ptr, &length, &type, &xclass, end - ptr)) return ERRORv("", "unexpected ASN1 type");
    for (/**/; ptr < end; ptr += length) {

      if (!GetASN1ObjectType(V_ASN1_SEQUENCE, &ptr, &length, &type, &xclass, end - ptr)) return ERRORv("", "unexpected ASN1 type");
      const unsigned char *attr_end = ptr + length;
      long attr_type    = GetASN1Integer(&ptr, &length, &type, &xclass, end - ptr);
      long attr_version = GetASN1Integer(&ptr, &length, &type, &xclass, end - ptr);

      if (!GetASN1ObjectType(V_ASN1_OCTET_STRING, &ptr, &length, &type, &xclass, end - ptr)) return ERRORv("", "unexpected ASN1 type");
      switch (attr_type) {
        case InAppPurchaseReceiptAttribute::ProductIdentifier: return GetASN1String(ptr, attr_end - ptr);
      }
    }
    return "";
  }
};

unique_ptr<PurchasesInterface> SystemToolkit::CreatePurchases(ApplicationInfo *a, string) { return make_unique<ApplePurchases>(a); }
}; // namespace LFL
