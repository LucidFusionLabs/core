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

#ifndef __LFL_LFJNI_LFJNI_H__
#define __LFL_LFJNI_LFJNI_H__

#ifdef __cplusplus
extern "C" {
#endif

int   AndroidVideoInit(int gles_version);
int   AndroidVideoSwap();
int   AndroidInput(unsigned clicks, unsigned *events);
int   AndroidToggleKeyboard();
int   AndroidFileRead(const char *filename, char **malloc_out, int *size_out);
int   AndroidInternalRead(const char *fn, char **malloc_out, int *size_out);
void *AndroidInternalOpenWriter(const char *fn);
int   AndroidInternalWrite(void *ifw, const char *b, int l);
void  AndroidInternalCloseWriter(void *ifw);
void *AndroidLoadMusicAsset(const char *filename);
void  AndroidPlayMusic(void *handle);
void  AndroidPlayBackgroundMusic(void *handle);
void  AndroidSetVolume(int v);
int   AndroidGetVolume();
int   AndroidGetMaxVolume();
int   AndroidDeviceName(char *out, int len);
int   AndroidIPV4Address();
int   AndroidIPV4BroadcastAddress();
void  AndroidOpenBrowser(const char *url);
void  AndroidShowAds();
void  AndroidHideAds();
void  AndroidGPlusSignin();
void  AndroidGPlusSignout();
int   AndroidGPlusSignedin();
int   AndroidGPlusInvite();
int   AndroidGPlusAccept();
int   AndroidGPlusQuickGame();
void  AndroidGPlusService(void *s);
int   AndroidGPlusSendUnreliable(const char *participant_name, const char *buf, int len);
int   AndroidGPlusSendReliable(const char *participant_name, const char *buf, int len);

#ifdef __cplusplus
};
#endif

#endif
