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

#include "core/app/app.h"
#include <portaudio.h>

namespace LFL {
struct PortaudioAudioModule : public Module {
  Audio *audio;
  PaStream *impl=0;
  PaTime input_latency, output_latency;
  PortaudioAudioModule(Audio *A) : audio(A) {}

  int Init() {
    PaError err = Pa_Initialize();
    if (err != paNoError) return ERRORv(-1, "init_portaudio: ", Pa_GetErrorText(err));

    int numDevices = Pa_GetDeviceCount();
    if (numDevices <= 0) return ERRORv(-1, "open_portaudio: devices=", numDevices);
    for (int i=0; false && i<numDevices; i++) {
      const PaDeviceInfo *di = Pa_GetDeviceInfo(i);
      ERROR(di->name, ",", di->hostApi, ", mi=", di->maxInputChannels, ", mo=", di->maxOutputChannels, ", ", di->defaultSampleRate, " ", di->defaultHighInputLatency);
    }

    if (FLAGS_audio_input_device  == -1) FLAGS_audio_input_device  = Pa_GetDefaultInputDevice();
    if (FLAGS_audio_output_device == -1) FLAGS_audio_output_device = Pa_GetDefaultOutputDevice();

    if (FLAGS_print_audio_devices) {
      INFO("audio_input_device=", FLAGS_audio_input_device, " audio_output_device=", FLAGS_audio_output_device);
      PaDeviceIndex num = Pa_GetDeviceCount();
      for (int i=0; i<num; i++) {
        const PaDeviceInfo *d = Pa_GetDeviceInfo(i);
        INFO("audio_device[", i, "] = ", d->name, " : ", d->maxInputChannels, "/", d->maxOutputChannels, " I/O chans @ ", d->defaultSampleRate);
      }
    }

    const PaDeviceInfo *ii=Pa_GetDeviceInfo(FLAGS_audio_input_device), *oi=Pa_GetDeviceInfo(FLAGS_audio_output_device);
    if (FLAGS_sample_rate == -1) FLAGS_sample_rate = ii->defaultSampleRate;
    if (FLAGS_chans_in == -1) FLAGS_chans_in = min(ii->maxInputChannels, 2);
    if (FLAGS_chans_out == -1) FLAGS_chans_out = min(oi->maxOutputChannels, 2);

    input_latency = ii->defaultHighInputLatency;
    output_latency = oi->defaultHighInputLatency;
    INFO("PortaudioAudioModule::Init sample_rate=", FLAGS_sample_rate, ", chans_in = ", FLAGS_chans_in, ", chans_out = ", FLAGS_chans_out);
    return 0;
  }

  int Start() {
    PaStreamParameters in  = { FLAGS_audio_input_device,  FLAGS_chans_in,  paFloat32, input_latency, 0 };
    PaStreamParameters out = { FLAGS_audio_output_device, FLAGS_chans_out, paFloat32, output_latency, 0 };

    PaError err = Pa_OpenStream(&impl, &in, FLAGS_chans_out ? &out : 0, FLAGS_sample_rate,
                                paFramesPerBufferUnspecified, 0, &PortaudioAudioModule::IOCB, this);
    if (err != paNoError) return ERRORv(-1, "open_portaudio: ", Pa_GetErrorText(err));

    err = Pa_StartStream(impl);
    if (err != paNoError) return ERRORv(-1, "open_portaudio2: ", Pa_GetErrorText(err));
    return 0;
  }

  int Free() {
    PaError err = Pa_StopStream(impl);
    if (err != paNoError) return ERRORv(0, "free_portaudio: stop ", Pa_GetErrorText(err));

    err = Pa_Terminate();
    if (err != paNoError) return ERRORv(0, "free_portaudio: term ", Pa_GetErrorText(err));
    return 0;
  }

  int IO(const void *input, void *output, unsigned long samplesPerFrame,
         const PaStreamCallbackTimeInfo *timeInfo, PaStreamCallbackFlags flags) {
    const float *in = static_cast<const float*>(input);
    float *out = static_cast<float*>(output);
    microseconds step(1000000/FLAGS_sample_rate), stamp =
      AudioResamplerInterface::MonotonouslyIncreasingTimestamp(audio->IL->ReadTimestamp(-1), ToMicroseconds(Now()), &step, samplesPerFrame);
    RingBuf::WriteAheadHandle IL(audio->IL.get()), IR(audio->IR.get());
    for (unsigned i=0; i<samplesPerFrame; i++) {
      microseconds timestamp = stamp + i * step;
      IL.Write(in[i*FLAGS_chans_in+0], RingBuf::Stamp, timestamp);
      if (FLAGS_chans_in == 1) continue;
      IR.Write(in[i*FLAGS_chans_in+1], RingBuf::Stamp, timestamp);
    }
    {
      ScopedMutex ML(audio->inlock);
      IL.Commit();
      IR.Commit();
      app->audio->samples_read += samplesPerFrame;
    }
    {
      ScopedMutex ML(audio->outlock);
      for (unsigned i=0; i<samplesPerFrame*FLAGS_chans_out; i++) 
        out[i] = audio->Out.size() ? PopFront(audio->Out) : 0;
    }
    return 0;
  }

  static int IOCB(const void *input, void *output, unsigned long samplesPerFrame,
                  const PaStreamCallbackTimeInfo *timeInfo, PaStreamCallbackFlags flags, void *opaque) {
    return static_cast<PortaudioAudioModule*>(opaque)->IO(input, output, samplesPerFrame, timeInfo, flags);
  }
};
 
Module *CreateAudioModule(Audio *a) { return new PortaudioAudioModule(a); }

}; // namespace LFL