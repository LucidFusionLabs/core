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

namespace LFL {
struct SDLAudioModule : public Module {
  Audio *audio;
  SDLAudioModule(Audio *A) : audio(A) {}

  int Init() {
    INFO("SDLAudioModule::Init");
    if (FLAGS_print_audio_devices) {
      INFO("audio_input_device=", FLAGS_audio_input_device, " audio_output_device=", FLAGS_audio_output_device);

      int count = SDL_GetNumAudioDevices(true);
      for (int i=0; i < count; i++) INFO("audio_input_device[", i, "] = ", SDL_GetAudioDeviceName(i, true));

      count = SDL_GetNumAudioDevices(false);
      for (int i=0; i < count; i++) INFO("audio_output_device[", i, "] = ", SDL_GetAudioDeviceName(i, false));
    }

    SDL_AudioSpec desired, obtained;
    desired.freq = FLAGS_sample_rate;
    desired.format = AUDIO_S16;
    desired.channels = FLAGS_chans_out > 0 ? FLAGS_chans_out : 2;
    desired.samples = 1024;
    desired.callback = &SDLAudioModule::IOCB;
    desired.userdata = this;

    if (SDL_OpenAudio(&desired, &obtained) < 0) return ERRORv(-1, "SDL_OpenAudio:", SDL_GetError());
    if (obtained.freq != FLAGS_sample_rate) return ERRORv(-1, "SDL_OpenAudio: sample_rate ", obtained.freq, " != ", FLAGS_sample_rate);

    FLAGS_chans_out = obtained.channels;
    SDL_PauseAudio(0);
    return 0;
  }

  int Free() {
    SDL_PauseAudio(0);
    return 0;
  }

  void IO(Uint8 *stream, int len) {
    short *out = (short*)stream;
    int frames = len/sizeof(short)/FLAGS_chans_out;

    ScopedMutex ML(audio->outlock);
    for (int i=0; i<frames*FLAGS_chans_out; i++)
      out[i] = (audio->Out.size() ? PopFront(audio->Out) : 0) * 32768.0;
  }

  static void IOCB(void *udata, Uint8 *stream, int len) { ((SDLAudioModule*)udata)->IO(stream, len); }
};

int Application::GetMaxVolume() { return 0; }
int Application::GetVolume() { return 0; }
void Application::SetVolume(int v) {}
 
Module *CreateAudioModule(Audio *a) { return new SDLAudioModule(a); }

}; // namespace LFL
