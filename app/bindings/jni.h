/*
 * $Id: lfapp.h 770 2013-09-25 00:27:33Z justin $
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

#ifndef LFL_CORE_APP_BINDINGS_JNI_H__
#define LFL_CORE_APP_BINDINGS_JNI_H__

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
  jclass activity_class=0, resources_class=0, view_class=0, gplus_class=0, throwable_class=0, frame_class=0, assetmgr_class=0;
  jclass string_class=0, inputstream_class=0, channels_class=0, readbytechan_class=0, r_string_class=0;
  jfieldID resources_id=0, view_id=0, gplus_id=0;
  string package_name;

  void Init(jobject a, bool first);
  void Free();
  int CheckForException();
  void LogException(jthrowable &exception);
  string GetJString(jstring);
  jstring ToJString(const string  &x) { return env->NewStringUTF(x.c_str()); }
  pair<jobjectArray, jobjectArray> ToJObjectArray(const StringPairVec& items);
  tuple<jobjectArray, jobjectArray, jobjectArray> ToJObjectArray(const MenuItemVec &items);
  BufferFile *OpenAsset(const string &fn);
};

struct GPlus {
  GPlusServer *server=0;
  void SignIn();
  void SignOut();
  int  GetSignedIn();
  int  Invite();
  int  Accept();
  int  QuickGame();
};

int GetAlertWidgetID(SystemAlertWidget*);
int GetToolbarWidgetID(SystemToolbarWidget*);
int GetMenuWidgetID(SystemMenuWidget*);
int GetTableWidgetID(SystemTableWidget*);
int GetNavigationWidgetID(SystemNavigationWidget*);

}; // namespace LFL
#endif // LFL_CORE_APP_BINDINGS_JNI_H__
