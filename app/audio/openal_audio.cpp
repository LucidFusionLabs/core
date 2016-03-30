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
  struct Effect { ALuint source, buffer; };    
  Audio *audio;
  ALCdevice *dev=0;
  ALCcontext *ctx=0;
  ALuint bg_source=0;
  deque<ALuint> bg_free_buffers, bg_used_buffers;
  deque<Effect> fx_free, fx_used;
  vector<int16_t> samples;
  OpenALAudioModule(Audio *A) : audio(A) {}

  int Init() {
    INFO("OpenALAudioModule::Init");
    if (FLAGS_chans_in == -1) FLAGS_chans_in = 2;
    if (FLAGS_chans_out == -1) FLAGS_chans_out = 2;
    if (!(dev = alcOpenDevice(NULL))) return ERRORv(-1, "alcOpenDevice");
    if (!(ctx = alcCreateContext(dev, NULL))) return ERRORv(-1, "alcCreateContext");
    alcMakeContextCurrent(ctx);

    alGenSources(1, &bg_source);
    if (alGetError() != AL_NO_ERROR) return ERRORv(-1, "alGenSources");
    alSourcei(bg_source, AL_SOURCE_RELATIVE, AL_TRUE);

    vector<ALuint> sources, buffers(FLAGS_sample_secs, 0);
    alGenBuffers(buffers.size(), &buffers[0]);
    if (alGetError() != AL_NO_ERROR) return ERRORv(-1, "alGenBuffers");
    for (auto &b : buffers) bg_free_buffers.push_back(b);

    sources.resize(3);
    buffers.resize(sources.size());
    alGenSources(sources.size(), &sources[0]);
    if (alGetError() != AL_NO_ERROR) return ERRORv(-1, "alGenSources");
    alGenBuffers(buffers.size(), &buffers[0]);
    if (alGetError() != AL_NO_ERROR) return ERRORv(-1, "alGenBuffers");
    for (int i=0, e=sources.size(); i != e; ++i) {
      alSourcei(sources[i], AL_SOURCE_RELATIVE, AL_FALSE);
      fx_free.push_back({ sources[i], buffers[i] });
    }
    return 0;
  }

  int Free() {
    for (auto &f : fx_used) { alDeleteBuffers(1, &f.buffer); alDeleteSources(1, &f.source); }
    for (auto &f : fx_free) { alDeleteBuffers(1, &f.buffer); alDeleteSources(1, &f.source); }
    for (auto &b : bg_free_buffers) alDeleteBuffers(1, &b);
    for (auto &b : bg_used_buffers) alDeleteBuffers(1, &b);
    alDeleteSources(1, &bg_source);
    alcMakeContextCurrent(NULL);
    alcDestroyContext(ctx);
    alcCloseDevice(dev);
    return 0;
  }

  int Frame(unsigned clicks) {
    ALuint b = 0;
    ALint val = 0;
    bool bg_added = 0;
    if (screen && screen->cam) {
      alListenerfv(AL_POSITION, screen->cam->pos);
      alListenerfv(AL_VELOCITY, screen->cam->vel);
      const v3 &ort = screen->cam->ort, &up = screen->cam->up;
      ALfloat ortup[6] = { ort.x, ort.y, ort.z, up.x, up.y, up.z };
      alListenerfv(AL_ORIENTATION, ortup);
    }

    alGetSourcei(bg_source, AL_BUFFERS_PROCESSED, &val);
    for (ALint i = 0; i < val; ++i) {
      alSourceUnqueueBuffers(bg_source, 1, &b);
      bg_free_buffers.push_back(b);
    }
    while (audio->Out.size() && bg_free_buffers.size()) {
      b = PopFront(bg_free_buffers);
      ALuint num_samples = min(ALuint(audio->Out.size()), ALuint(FLAGS_chans_out * FLAGS_sample_rate));
      EnsureSize(samples, num_samples);
      for (int16_t *i=&samples[0], *e=i+num_samples; i != e; ++i) *i = int16_t(PopFront(audio->Out) * 32768.0);
      alBufferData(b, FLAGS_chans_out == 2 ? AL_FORMAT_STEREO16 : AL_FORMAT_MONO16,
                   samples.data(), num_samples*sizeof(int16_t), FLAGS_sample_rate);
      alSourceQueueBuffers(bg_source, 1, &b);
      bg_used_buffers.push_back(b);
      bg_added = true;
    }
    if (bg_added) {
      alGetSourcei(bg_source, AL_SOURCE_STATE, &val);
      if (val != AL_PLAYING) alSourcePlay(bg_source);
    }
    for (auto i=fx_used.begin(); i != fx_used.end(); /**/) {
      alGetSourcei(i->source, AL_BUFFERS_PROCESSED, &val);
      if (!val) { ++i; continue; }
      CHECK_EQ(1, val);
      alSourceUnqueueBuffers(i->source, 1, &b);
      CHECK_EQ(b, i->buffer);
      fx_free.push_back(*i);
      i = fx_used.erase(i);
    }
    return 0;
  }

  void PlaySoundEffect(SoundAsset *sa, const v3 &pos, const v3 &vel) {
    if (fx_free.empty()) return;
    int num_samples = sa->wav->ring.size;
    EnsureSize(samples, num_samples);
    RingSampler::Handle H(sa->wav.get());
    for (int i=0; i != num_samples; ++i) samples[i] = int16_t(H.Read(i) * 32768.0);

    Effect fx = PopFront(fx_free);
    alBufferData(fx.buffer, sa->channels  == 2 ? AL_FORMAT_STEREO16 : AL_FORMAT_MONO16,
                 samples.data(), num_samples*sizeof(int16_t), sa->sample_rate);
    alSourcefv(fx.source, AL_POSITION, pos);
    alSourcefv(fx.source, AL_VELOCITY, vel);
    alSourcef(fx.source, AL_MAX_DISTANCE, sa->gain);
    if (sa->max_distance) alSourcef(fx.source, AL_MAX_DISTANCE, sa->max_distance);
    if (sa->reference_distance) alSourcef(fx.source, AL_REFERENCE_DISTANCE, sa->reference_distance);
    alSourceQueueBuffers(fx.source, 1, &fx.buffer);
    alSourcePlay(fx.source);
    fx_used.push_back(fx);
  }
};

void Application::PlayBackgroundMusic(SoundAsset *music) {
  audio->QueueMix(music);
  audio->loop = music;
  audio->Out.resize(max(audio->Out.size(), SoundAsset::Size(music)));
}

void Application::PlaySoundEffect(SoundAsset *sa, const v3 &pos, const v3 &vel) {
  // audio->QueueMix(sa, MixFlag::Reset | MixFlag::Mix | (audio->loop ? MixFlag::DontQueue : 0), -1, -1);
  dynamic_cast<OpenALAudioModule*>(audio->impl.get())->PlaySoundEffect(sa, pos, vel);
}
 
unique_ptr<Module> CreateAudioModule(Audio *a) { return make_unique<OpenALAudioModule>(a); }

}; // namespace LFL
