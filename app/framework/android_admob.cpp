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

#include "core/app/framework/android_toolkit.h"

namespace LFL {
static JNI *jni = Singleton<JNI>::Get();

struct AndroidAdvertisingView : public AdvertisingViewInterface {
  GlobalJNIObject impl;
  AndroidAdvertisingView(int t, int p, const string &did, const StringVec &test_devices)
    : impl(NewAdvertisingObject(t, p, did, test_devices)) {}

  static jobject NewAdvertisingObject(int t, int p, const string &did, const StringVec &td) {
    if (!jni->advertising_class) jni->advertising_class = CheckNotNull
      (jclass(jni->env->NewGlobalRef(jni->env->FindClass("com/lucidfusionlabs/ads/Advertising"))));
    static jmethodID mid = CheckNotNull
      (jni->env->GetStaticMethodID(jni->advertising_class, "createStaticInstance",
                                   "(Lcom/lucidfusionlabs/core/LifecycleActivity;IILjava/lang/String;Ljava/util/List;)Lcom/lucidfusionlabs/ads/Advertising;"));
    LocalJNIString di(jni->env, JNI::ToJString(jni->env, did));
    LocalJNIObject l(jni->env, JNI::ToJStringArrayList(jni->env, td));
    return jni->env->CallStaticObjectMethod(jni->advertising_class, mid, jni->activity, jint(t), jint(p), di.v, l.v);
  }

  void Show(bool show_or_hide) {
    ERROR(not_implemented);
  }

  void Show(TableViewInterface *t, bool show_or_hide) {
    static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->tablescreen_class, "setAdvertising", "(Landroid/support/v7/app/AppCompatActivity;Lcom/lucidfusionlabs/core/ViewOwner;)V"));
    jni->env->CallVoidMethod(dynamic_cast<AndroidTableView*>(t)->impl.v, mid, jni->activity, show_or_hide ? impl.v : nullptr);
  }
};

void SystemToolkit::DisableAdvertisingCrashReporting() {}
unique_ptr<AdvertisingViewInterface> SystemToolkit::CreateAdvertisingView(int type, int placement, const string &adid, const StringVec &test_devices) {
  return make_unique<AndroidAdvertisingView>(type, placement, adid, test_devices);
}

}; // namespace LFL
