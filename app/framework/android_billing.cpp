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

#include "core/app/app.h"

namespace LFL {
static JNI *jni = Singleton<JNI>::Set();

static jobject ToNativeProductCB(JNIEnv *env, PurchasesInterface::ProductCB c) {
  static jclass nativeproductcb_class = CheckNotNull(jclass(env->NewGlobalRef(env->FindClass("com/lucidfusionlabs/billing/NativeProductCB"))));
  static jmethodID mid = CheckNotNull(env->GetMethodID(nativeproductcb_class, "<init>", "(J)V"));
  jlong cb = uintptr_t(new PurchasesInterface::ProductCB(move(c)));
  return env->NewObject(nativeproductcb_class, mid, cb);
}

struct AndroidProduct : public ProductInterface {
  string name, desc, price;
  ~AndroidProduct() {}
  AndroidProduct(const string &i, const string &n, const string &d, const string &p) :
    ProductInterface(i), name(n), desc(d), price(p) {}
  string Name()        { return name; }
  string Description() { return desc; }
  string Price()       { return price; }
};

struct AndroidPurchases : public PurchasesInterface {
  GlobalJNIObject impl;
  AndroidPurchases(string pubkey) : impl(NewPurchaseManagerObject(move(pubkey))) {}

  static jobject NewPurchaseManagerObject(string pubkey) {
    if (!jni->purchases_class) jni->purchases_class = CheckNotNull
      (jclass(jni->env->NewGlobalRef(jni->env->FindClass("com/lucidfusionlabs/billing/PurchaseManager"))));
    static jmethodID mid = CheckNotNull
      (jni->env->GetStaticMethodID(jni->purchases_class, "createStaticInstance",
                                   "(Lcom/lucidfusionlabs/core/LifecycleActivity;Ljava/lang/String;)Lcom/lucidfusionlabs/billing/PurchaseManager;"));
    LocalJNIString pk(jni->env, JNI::ToJString(jni->env, pubkey));
    return jni->env->CallStaticObjectMethod(jni->purchases_class, mid, jni->activity, pk.v);
  }

  bool CanPurchase() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->purchases_class, "canPurchase", "()Z"));
    return jni->env->CallBooleanMethod(impl.v, mid);
  }

  bool HavePurchase(const string &product_id) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->purchases_class, "havePurchase", "(Ljava/lang/String;)Z"));
    LocalJNIString p(jni->env, JNI::ToJString(jni->env, product_id));
    return jni->env->CallBooleanMethod(impl.v, mid, p.v);
  }

  void LoadPurchases() {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->purchases_class, "loadPurchasesFromInternalStorage", "()Z"));
    jni->env->CallBooleanMethod(impl.v, mid);
  }

  void RestorePurchases(Callback done_cb) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->purchases_class, "restorePurchases", "()Z"));
    if (jni->env->CallBooleanMethod(impl.v, mid) && done_cb) done_cb();
  }

  void PreparePurchase(const StringVec &products, Callback done_cb, PurchasesInterface::ProductCB product_cb) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->purchases_class, "queryPurchase", "(Ljava/util/ArrayList;Lcom/lucidfusionlabs/core/NativeCallback;Lcom/lucidfusionlabs/billing/NativeProductCB;)Z"));
    LocalJNIObject p(jni->env, JNI::ToJStringArrayList(jni->env, products));
    LocalJNIObject dcb(jni->env, done_cb ? JNI::ToNativeCallback(jni->env, move(done_cb)) : nullptr);
    LocalJNIObject pcb(jni->env, product_cb ? ToNativeProductCB(jni->env, move(product_cb)) : nullptr);
    jni->env->CallBooleanMethod(impl.v, mid, p.v, dcb.v, pcb.v);
  }

  bool MakePurchase(ProductInterface *product, IntCB result_cb) {
    static jmethodID mid = CheckNotNull
      (jni->env->GetMethodID(jni->purchases_class, "makePurchase", "(Ljava/lang/String;Lcom/lucidfusionlabs/core/NativeIntCB;)Z"));
    if (!product) return false;

    LocalJNIObject rcb(jni->env, result_cb ? JNI::ToNativeIntCB(jni->env, move(result_cb)) : nullptr);
    LocalJNIString pid(jni->env, JNI::ToJString(jni->env, product->id));
    return jni->env->CallBooleanMethod(impl.v, mid, pid.v, rcb.v);
  }
};

extern "C" void Java_com_lucidfusionlabs_billing_NativeProductCB_RunProductCBInMainThread(JNIEnv *e, jclass c, jlong cb, jstring id, jstring name, jstring desc, jstring price) {
  auto product = new AndroidProduct(JNI::GetJString(e, id), JNI::GetJString(e, name),
                                    JNI::GetJString(e, desc), JNI::GetJString(e, price));
  jni->app->RunCallbackInMainThread([=](){
    (*static_cast<PurchasesInterface::ProductCB*>(Void(cb)))(unique_ptr<ProductInterface>(product));
  });
}

extern "C" void Java_com_lucidfusionlabs_billing_NativeProductCB_FreeProductCB(JNIEnv *e, jclass c, jlong cb) {
  delete static_cast<PurchasesInterface::ProductCB*>(Void(cb));
}

unique_ptr<PurchasesInterface> SystemToolkit::CreatePurchases(ApplicationInfo*, string pubkey) { return make_unique<AndroidPurchases>(move(pubkey)); }
}; // namespace LFL
