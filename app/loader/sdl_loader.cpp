/*
 * $Id: camera.cpp 1330 2014-11-06 03:04:15Z justin $
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

#include "SDL.h"
#include "SDL_image.h"

namespace LFL {
const int SoundAsset::FromBufPad = 0;

struct SDLAssetLoader : public AssetLoaderInterface {
  SDLAssetLoader() { INFO("SDLAssetLoader"); }

  virtual void *LoadAudioFile(const string &fn) { return OpenFileHandle(fn); }
  virtual void UnloadAudioFile(void *h) { CloseHandle(h); }
  virtual void *LoadAudioBuf(const char *b, int l, const char*) { return OpenBufferHandle(b, l); }
  virtual void UnloadAudioBuf(void *h) { CloseHandle(h); }

  virtual void *LoadVideoFile(const string &fn) { return OpenFileHandle(fn); }
  virtual void UnloadVideoFile(void *h) { CloseHandle(h); }
  virtual void *LoadVideoBuf(const char *b, int l, const char*) { return OpenBufferHandle(b, l); }
  virtual void UnloadVideoBuf(void *h) { CloseHandle(h); }

  virtual void *LoadMovieFile(const string &fn) { return OpenFileHandle(fn); }
  virtual void UnloadMovieFile(void *h) { CloseHandle(h); }
  virtual void *LoadMovieBuf(const char *b, int l, const char*) { return OpenBufferHandle(b, l); }
  virtual void UnloadMovieBuf(void *h) { CloseHandle(h); }

  void LoadVideo(void *h, Texture *out, int load_flag=VideoAssetLoader::Flag::Default) {
    SDL_Surface *image = IMG_Load_RW(reinterpret_cast<SDL_RWops*>(h), false);
  }

  void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) {}
  void LoadMovie(void *handle, MovieAsset *ma) {}
  int PlayMovie(MovieAsset *ma, int seek) { return 0; }
  int RefillAudio(SoundAsset *a, int reset) { return 0; }

  static void *OpenBufferHandle(const char *b, int l) { return SDL_RWFromConstMem(b, l); }
  static void *OpenFileHandle(const string &fn) { return SDL_RWFromFile(fn.c_str(), "rb"); }
  static void CloseHandle(void *h) { SDL_RWclose(static_cast<SDL_RWops*>(h)); }
};

unique_ptr<AssetLoaderInterface> CreateAssetLoader() {
  ONCE({ IMG_Init(IMG_INIT_JPG|IMG_INIT_PNG); });
  return make_unique<SDLAssetLoader>();
}

}; // namespace LFL
