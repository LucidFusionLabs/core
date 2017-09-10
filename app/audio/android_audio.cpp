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

namespace LFL {
int Audio::GetMaxVolume() {
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "maxVolume", "()I"));
  return jni->env->CallIntMethod(jni->activity, mid);
}

int Audio::GetVolume() {
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "getVolume", "()I"));
  return jni->env->CallIntMethod(jni->activity, mid);
}

void Audio::SetVolume(int v) {
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID mid = CheckNotNull(jni->env->GetMethodID(jni->activity_class, "setVolume", "(I)V"));
  jint jv = v;
  return jni->env->CallVoidMethod(jni->activity, mid, jv);
}

void Audio::PlaySoundEffect(SoundAsset *sa, const v3&, const v3&) {
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID jni_activity_method_play_music =
    CheckNotNull(jni->env->GetMethodID(jni->activity_class, "playMusic", "(Landroid/media/MediaPlayer;)V"));
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_play_music, jobject(sa->handle));
}

void Audio::PlayBackgroundMusic(SoundAsset *sa) {
  JNI *jni = Singleton<LFL::JNI>::Get();
  static jmethodID jni_activity_method_play_background_music =
    CheckNotNull(jni->env->GetMethodID(jni->activity_class, "playBackgroundMusic", "(Landroid/media/MediaPlayer;)V"));
  jni->env->CallVoidMethod(jni->activity, jni_activity_method_play_background_music, jobject(sa->handle));
}

unique_ptr<Module> CreateAudioModule(Audio *a) { return make_unique<Module>(); }
}; // namespace LFL
