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

#include <OpenAL/al.h>
#include <OpenAL/alc.h>

namespace LFL {
struct OpenALAudioModule : public Module {
  Audio *audio;
  ALCdevice *dev=0;
  ALCcontext *ctx=0;
  ALuint source=0;
  ALenum in_format=0, out_format=0;
  deque<ALuint> free_buffers, used_buffers;
  OpenALAudioModule(Audio *A) : audio(A) {}

  int Init() {
    INFO("OpenALAudioModule::Init");
    if (FLAGS_chans_in == -1) FLAGS_chans_in = 2;
    if (FLAGS_chans_out == -1) FLAGS_chans_out = 2;
    if (!(dev = alcOpenDevice(NULL))) return ERRORv(-1, "alcOpenDevice");
    if (!(ctx = alcCreateContext(dev, NULL))) return ERRORv(-1, "alcCreateContext");
    alcMakeContextCurrent(ctx);
    vector<ALuint> buffers(3, 0);
    alGenBuffers(buffers.size(), &buffers[0]);
    if (alGetError() != AL_NO_ERROR) return ERRORv(-1, "alGenBuffers");
    alGenSources(1, &source);
    if (alGetError() != AL_NO_ERROR) return ERRORv(-1, "alGenSources");
    in_format  = FLAGS_chans_in  == 2 ? AL_FORMAT_STEREO16 : AL_FORMAT_MONO16;
    out_format = FLAGS_chans_out == 2 ? AL_FORMAT_STEREO16 : AL_FORMAT_MONO16;
    for (auto &b : buffers) free_buffers.push_back(b);
    return 0;
  }

  int Free() {
    alcMakeContextCurrent(NULL);
    alcDestroyContext(ctx);
    alcCloseDevice(dev);
    return 0;
  }

  int Frame(unsigned clicks) {
    ALint val = 0;
    alGetSourcei(source, AL_BUFFERS_PROCESSED, &val);
    for (ALint i = 0; i < val; ++i) {
      ALuint b = 0;
      alSourceUnqueueBuffers(source, 1, &b);
      free_buffers.push_back(b);
    }

    if (audio->Out.size() && free_buffers.size()) {
      auto b = PopFront(free_buffers);
      // alBufferData(b, out_format, buf, ret, FLAGS_sample_rate);
    }


    return 0;
  }
};
 
unique_ptr<Module> CreateAudioModule(Audio *a) { return make_unique<OpenALAudioModule>(a); }

}; // namespace LFL
