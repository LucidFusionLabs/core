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

#ifdef LFL_APPLE
#include <OpenAL/al.h>
#include <OpenAL/alc.h>
#else
#include <AL/al.h>
#include <AL/alc.h>
#endif

namespace LFL {
struct OpenALAudioModule : public Module {
  Audio *audio;
  ALCdevice *dev=0;
  ALCcontext *ctx=0;
  ALuint source=0, in_maxsamples=0, out_maxsamples=0;
  ALenum in_format=0, out_format=0;
  deque<ALuint> free_buffers, used_buffers;
  vector<int16_t> samples;
  OpenALAudioModule(Audio *A) : audio(A) {}

  int Init() {
    INFO("OpenALAudioModule::Init");
    if (FLAGS_chans_in == -1) FLAGS_chans_in = 2;
    if (FLAGS_chans_out == -1) FLAGS_chans_out = 2;
    if (!(dev = alcOpenDevice(NULL))) return ERRORv(-1, "alcOpenDevice");
    if (!(ctx = alcCreateContext(dev, NULL))) return ERRORv(-1, "alcCreateContext");
    alcMakeContextCurrent(ctx);
    vector<ALuint> buffers(FLAGS_sample_secs, 0);
    alGenBuffers(buffers.size(), &buffers[0]);
    if (alGetError() != AL_NO_ERROR) return ERRORv(-1, "alGenBuffers");
    alGenSources(1, &source);
    if (alGetError() != AL_NO_ERROR) return ERRORv(-1, "alGenSources");
    in_format  = FLAGS_chans_in  == 2 ? AL_FORMAT_STEREO16 : AL_FORMAT_MONO16;
    out_format = FLAGS_chans_out == 2 ? AL_FORMAT_STEREO16 : AL_FORMAT_MONO16;
    in_maxsamples = FLAGS_chans_in * FLAGS_sample_rate;
    out_maxsamples = FLAGS_chans_out * FLAGS_sample_rate;
    samples.resize(max(FLAGS_chans_in, FLAGS_chans_out) * FLAGS_sample_rate);
    for (auto &b : buffers) free_buffers.push_back(b);
    return 0;
  }

  int Free() {
    for (auto &b : free_buffers) alDeleteBuffers(1, &b);
    for (auto &b : used_buffers) alDeleteBuffers(1, &b);
    alDeleteSources(1, &source);
    alcMakeContextCurrent(NULL);
    alcDestroyContext(ctx);
    alcCloseDevice(dev);
    return 0;
  }

  int Frame(unsigned clicks) {
    bool added = 0;
    ALint val = 0;
    alGetSourcei(source, AL_BUFFERS_PROCESSED, &val);
    for (ALint i = 0; i < val; ++i) {
      ALuint b = 0;
      alSourceUnqueueBuffers(source, 1, &b);
      free_buffers.push_back(b);
    }
    while (audio->Out.size() && free_buffers.size()) {
      auto b = PopFront(free_buffers);
      ALuint num_samples = min(ALuint(audio->Out.size()), out_maxsamples);
      for (int16_t *i=&samples[0], *e=i+num_samples; i != e; ++i) *i = int16_t(PopFront(audio->Out) * 32768.0);
      alBufferData(b, out_format, samples.data(), num_samples*sizeof(int16_t), FLAGS_sample_rate);
      alSourceQueueBuffers(source, 1, &b);
      used_buffers.push_back(b);
      added = true;
    }
    if (added) {
      alGetSourcei(source, AL_SOURCE_STATE, &val);
      if (val != AL_PLAYING) alSourcePlay(source);
    }
    return 0;
  }
};

void Application::PlayBackgroundMusic(SoundAsset *music) {
  audio->QueueMix(music);
  audio->loop = music;
}

void Application::PlaySoundEffect(SoundAsset *sa) {
  audio->QueueMix(sa, MixFlag::Reset | MixFlag::Mix | (audio->loop ? MixFlag::DontQueue : 0), -1, -1);
}
 
unique_ptr<Module> CreateAudioModule(Audio *a) { return make_unique<OpenALAudioModule>(a); }

}; // namespace LFL
