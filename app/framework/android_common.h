/*
 * $Id: android_common.h 770 2013-09-25 00:27:33Z justin $
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

#ifndef LFL_CORE_APP_FRAMEWORK_ANDROID_COMMON_H__
#define LFL_CORE_APP_FRAMEWORK_ANDROID_COMMON_H__

#include <jni.h>

namespace LFL {
struct AndroidEvent {
  enum { ACTION_DOWN=0, ACTION_UP=1, ACTION_MOVE=2, ACTION_CANCEL=3, ACTION_OUTSIDE=4, ACTION_POINTER_DOWN=5,
    ACTION_POINTER_UP=6 };
};

struct JNI {
  JNIEnv *env=0;
  Box activity_box;
  jobject activity=0, resources=0, view=0, gplus=0;
  jclass activity_class=0, resources_class=0, throwable_class=0, string_class=0, arraylist_class=0,
         pair_class=0, hashmap_class=0, view_class=0, frame_class=0, assetmgr_class=0, 
         inputstream_class=0, channels_class=0, readbytechan_class=0, r_string_class=0,
         toolbar_class=0, advertising_class=0, gplus_class=0, purchases_class=0, 
         modelitem_class=0, modelitemchange_class=0, pickeritem_class=0, alertscreen_class=0,
         menuscreen_class=0, tablescreen_class=0, textscreen_class=0, screennavigator_class=0,
         nativecallback_class=0, nativestringcb_class=0, nativeintintcb_class=0,
         nativepickeritemcb_class=0, int_class=0, long_class=0;
  jmethodID arraylist_construct=0, arraylist_size=0, arraylist_get=0, arraylist_add=0,
            hashmap_construct=0, hashmap_size=0, hashmap_get=0, hashmap_put=0, pair_construct=0,
            modelitem_construct=0, modelitemchange_construct=0, pickeritem_construct=0,
            int_intval=0, long_longval=0;
  jfieldID activity_resources=0, activity_view=0, activity_gplus=0, pair_first=0, pair_second=0;
  string package_name;

  void Init(jobject a, bool first);
  void Free();
  int CheckForException();
  void LogException(jthrowable &exception);
  string GetJString(jstring x) { return GetEnvJString(env, x); }
  jstring ToJString(const string &x) { return env->NewStringUTF(x.c_str()); }
  jstring ToJStringRaw(const string &x);
  jobjectArray ToJStringArray(const StringVec &items);
  pair<jobjectArray, jobjectArray> ToJStringArrays(const StringPairVec &items);
  jobject ToJStringArrayList(const StringVec &items);
  jobject ToJStringPairArrayList(const StringPairVec &items);
  jobject ToModelItemArrayList(AlertItemVec items);
  jobject ToModelItemArrayList(MenuItemVec items);
  jobject ToModelItemArrayList(TableItemVec items);
  jobject ToModelItemChangeList(const TableSection::ChangeList&);
  jobject ToModelItem(AlertItem);
  jobject ToModelItem(MenuItem);
  jobject ToModelItem(TableItem);
  jobject ToModelItemChange(const TableSection::Change&);
  jobject ToPickerItem(PickerItem*);
  jobject ToNativeCallback(Callback c);
  jobject ToNativeStringCB(StringCB c);
  jobject ToNativeIntIntCB(IntIntCB c);
  jobject ToNativePickerItemCB(const PickerItem::CB &c);

  BufferFile *OpenAsset(const string &fn);
  static string GetEnvJString(JNIEnv*, jstring);
};

template <class X> struct LocalJNIType {
  X v;
  JNIEnv *env;
  LocalJNIType(JNIEnv *E, X V) : v(V), env(E) {}
  virtual ~LocalJNIType() { if (v) env->DeleteLocalRef(v); }
};

template <class X> struct GlobalJNIType {
  X v;
  GlobalJNIType(X V) { auto e = Singleton<JNI>::Get()->env; v = e->NewGlobalRef(V); e->DeleteLocalRef(V); }
  virtual ~GlobalJNIType() { if (v) Singleton<JNI>::Get()->env->DeleteGlobalRef(v); }
};

typedef LocalJNIType<jobject> LocalJNIObject;
typedef LocalJNIType<jstring> LocalJNIString;
typedef GlobalJNIType<jobject> GlobalJNIObject;
typedef GlobalJNIType<jstring> GlobalJNIString;

struct GPlus {
  GPlusServer *server=0;
  void SignIn();
  void SignOut();
  int  GetSignedIn();
  int  Invite();
  int  Accept();
  int  QuickGame();
};

}; // namespace LFL
#endif // LFL_CORE_APP_FRAMEWORK_ANDROID_COMMON_H__
