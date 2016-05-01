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
  jobject activity=0, view=0, gplus=0;
  jclass activity_class=0, view_class=0, gplus_class=0, throwable_class=0, frame_class=0;
  jfieldID view_id=0, gplus_id=0;

  void Init(jobject a, bool first);
  void Free();

  int CheckForException();
  void LogException(jthrowable &exception);
  std::string GetJNIString(jstring);
};
}; // namespace LFL

#ifdef __cplusplus
extern "C" {
#endif
int   AndroidAssetRead(const char *filename, char **malloc_out, int *size_out);
int   AndroidDeviceName(char *out, int len);
void  AndroidGPlusSignin();
void  AndroidGPlusSignout();
int   AndroidGPlusSignedin();
int   AndroidGPlusInvite();
int   AndroidGPlusAccept();
int   AndroidGPlusQuickGame();
void  AndroidGPlusService(void *s);
#ifdef __cplusplus
};
#endif

#endif // LFL_CORE_APP_BINDINGS_JNI_H__
