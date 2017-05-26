/*
 * $Id: android_billing.cpp 1336 2014-12-08 09:29:59Z justin $
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

#include "core/app/app.h"

namespace LFL {
static JNI *jni = Singleton<JNI>::Get();

struct AndroidProduct : public ProductInterface {
  ~AndroidProduct() {}
  AndroidProduct(const string &i) : ProductInterface(i) {}
  string Name() { return ""; }
  string Description() { return ""; }
  string Price() {
    return "";
  }
};

struct AndroidPurchases : public PurchasesInterface {
  GlobalJNIObject impl;
  AndroidPurchases() : impl(NewPurchaseManagerObject()) {}

  static jobject NewPurchaseManagerObject() {
    if (!jni->purchases_class) jni->purchases_class = CheckNotNull
      (jclass(jni->env->NewGlobalRef(jni->env->FindClass("com/lucidfusionlabs/billing/PurchaseManager"))));
    static jmethodID mid = CheckNotNull
      (jni->env->GetStaticMethodID(jni->purchases_class, "createStaticInstance",
                                   "(Lcom/lucidfusionlabs/core/LifecycleActivity;)Lcom/lucidfusionlabs/billing/PurchaseManager;"));
    return jni->env->CallStaticObjectMethod(jni->purchases_class, mid, jni->activity);
  }

  bool CanPurchase() { return 0; }
  bool HavePurchase(const string &product_id) { return 0; }
  void PreparePurchase(const StringVec &products, Callback done_cb, ProductCB product_cb) {}
  bool MakePurchase(ProductInterface *product, IntCB result_cb) { return false; }
  void RestorePurchases(Callback done_cb) {}
  void LoadPurchases() {}
};

unique_ptr<PurchasesInterface> SystemToolkit::CreatePurchases() { return make_unique<AndroidPurchases>(); }
}; // namespace LFL
