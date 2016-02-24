/*
 * $Id: audio.cpp 1330 2014-11-06 03:04:15Z justin $
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

extern "C" {
#ifdef LFL_FFMPEG
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/avfft.h>
#include <libswresample/swresample.h>
#define AVCODEC_MAX_AUDIO_FRAME_SIZE 192000
#endif
};

#include "lfapp/lfapp.h"

#ifdef LFL_PORTAUDIO
#include <portaudio.h>
#endif
#ifdef _WIN32
#define inline _inline
#endif
#ifdef LFL_SDLAUDIO
#include "SDL.h"
#endif
#ifdef LFL_AUDIOQUEUE
#include <AudioToolbox/AudioToolbox.h>
#endif
#ifdef LFL_AUDIOUNIT
#include <AudioUnit/AudioUnit.h>
#include <AudioToolbox/AudioServices.h>
#endif
#ifdef LFL_OPENSL
#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>
#endif

namespace LFL {
DEFINE_int(sample_rate, 16000, "Audio sample rate");
DEFINE_int(sample_secs, 3, "Seconds of RingBuf audio");
DEFINE_int(chans_in, -1, "Audio input channels");
DEFINE_int(chans_out, -1, "Audio output channels");
DEFINE_int(audio_input_device, -1, "Audio input device index");
DEFINE_int(audio_output_device, -1, "Audio output device index");
DEFINE_bool(print_audio_devices, false, "Print audio device list");

int Sample::Size(int fmt) {
  switch (fmt) {
    case Sample::U8:     case Sample::U8P:     return 1;
    case Sample::S16:    case Sample::S16P:    return 2;
    case Sample::S32:    case Sample::S32P:    return 4;
    case Sample::FLOAT:  case Sample::FLOATP:  return 4;
    case Sample::DOUBLE: case Sample::DOUBLEP: return 8;
    default: return ERRORv(0, "unknown sample fmt: ", fmt);
  }
}

#ifdef LFL_FFMPEG
int Sample::FromFFMpegId(int fmt) {
  switch(fmt) {
    case AV_SAMPLE_FMT_U8:   return Sample::U8;
    case AV_SAMPLE_FMT_U8P:  return Sample::U8P;
    case AV_SAMPLE_FMT_S16:  return Sample::S16;
    case AV_SAMPLE_FMT_S16P: return Sample::S16P;
    case AV_SAMPLE_FMT_S32:  return Sample::S32;
    case AV_SAMPLE_FMT_S32P: return Sample::S32P;
    case AV_SAMPLE_FMT_FLT:  return Sample::FLOAT;
    case AV_SAMPLE_FMT_FLTP: return Sample::FLOATP;
    case AV_SAMPLE_FMT_DBL:  return Sample::DOUBLE;
    case AV_SAMPLE_FMT_DBLP: return Sample::DOUBLEP;
    default: return ERRORv(0, "unknown sample fmt: ", fmt);
  }
}

int Sample::ToFFMpegId(int fmt) {
  switch(fmt) {
    case Sample::U8:      return AV_SAMPLE_FMT_U8;
    case Sample::U8P:     return AV_SAMPLE_FMT_U8P;
    case Sample::S16:     return AV_SAMPLE_FMT_S16;
    case Sample::S16P:    return AV_SAMPLE_FMT_S16P;
    case Sample::S32:     return AV_SAMPLE_FMT_S32;
    case Sample::S32P:    return AV_SAMPLE_FMT_S32P;
    case Sample::FLOAT:   return AV_SAMPLE_FMT_FLT;
    case Sample::FLOATP:  return AV_SAMPLE_FMT_FLTP;
    case Sample::DOUBLE:  return AV_SAMPLE_FMT_DBL;
    case Sample::DOUBLEP: return AV_SAMPLE_FMT_DBLP;
    default: return ERRORv(0, "unknown sample fmt: ", fmt);
  }
}
#endif

#ifdef LFL_PORTAUDIO
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
    const float *in = FromVoid<const float*>(input);
    float *out = FromVoid<float*>(output);
    microseconds step(1000000/FLAGS_sample_rate), stamp =
      AudioResampler::MonotonouslyIncreasingTimestamp(audio->IL->ReadTimestamp(-1), ToMicroseconds(Now()), &step, samplesPerFrame);
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
    return FromVoid<PortaudioAudioModule*>(opaque)->IO(input, output, samplesPerFrame, timeInfo, flags);
  }
};
#endif /* LFL_PORTAUDIO */

#ifdef LFL_SDLAUDIO
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
#endif /* LFL_SDLAUDIO */

#ifdef LFL_AUDIOQUEUE
struct AudioQueueAudioModule : public Module {
  static const int AQbuffers=3, AQframesPerCB=512;
  AudioQueueRef input_audio_queue=0, output_audio_queue=0;
  Audio *audio;
  AudioQueueAudioModule(Audio *A) : audio(A) {}

  int Init() {
    INFO("AudioQueueAudioModule::Init");
    if (input_audio_queue || output_audio_queue) return -1;
    if (FLAGS_chans_in == -1) FLAGS_chans_in = 2;
    if (FLAGS_chans_out == -1) FLAGS_chans_out = 2;

    AudioStreamBasicDescription desc_in;
    memset(&desc_in, 0, sizeof(desc_in));
    desc_in.mSampleRate = FLAGS_sample_rate;
    desc_in.mFormatID = kAudioFormatLinearPCM;
    desc_in.mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked;
    desc_in.mChannelsPerFrame = FLAGS_chans_in;
    desc_in.mBytesPerFrame = desc_in.mChannelsPerFrame * sizeof(short);
    desc_in.mFramesPerPacket = 1;
    desc_in.mBytesPerPacket = desc_in.mFramesPerPacket * desc_in.mBytesPerFrame;
    desc_in.mBitsPerChannel = 8 * sizeof(short);

    int err = AudioQueueNewInput(&desc_in, AudioQueueAudioModule::IOInCB, this, 0, kCFRunLoopCommonModes, 0, &input_audio_queue);
    if (err == kAudioFormatUnsupportedDataFormatError) return ERROR(-1, "AudioQueueNewInput: format unsupported sr=", desc_in.mSampleRate);
    if (err || !input_audio_queue) return ERRORv(-1, "AudioQueueNewInput: error ", err);

    AudioStreamBasicDescription desc_out = desc_in;
    desc_out.mChannelsPerFrame = FLAGS_chans_out;
    desc_out.mBytesPerFrame = desc_out.mChannelsPerFrame * sizeof(short);
    desc_out.mBytesPerPacket = desc_out.mFramesPerPacket * desc_out.mBytesPerFrame;

    err = AudioQueueNewOutput(&desc_out, &AudioQueueAudioModule::IOOutCB, this, 0, kCFRunLoopCommonModes, 0, &output_audio_queue);
    if (err == kAudioFormatUnsupportedDataFormatError) return ERRORv(-1, "AudioQueueNewOutput: format unsupported sr=", desc_out.mSampleRate);
    if (err || !output_audio_queue) return ERRORv(-1, "AudioQueueNewOutput: error ", err);

    for (int i=0; i<AQbuffers; i++) {
      AudioQueueBufferRef buf_in, buf_out;
      if ((err = AudioQueueAllocateBuffer(input_audio_queue, AQframesPerCB*desc_in.mBytesPerFrame, &buf_in))) return ERRORv(-1, "AudioQueueAllocateBuffer: error ", err); return -1; }
      if ((err = AudioQueueEnqueueBuffer(input_audio_queue, buf_in, 0, 0))) return ERRORv(-1, "AudioQueueEnqueueBuffer: error ", err); return -1; }

      if ((err = AudioQueueAllocateBuffer(output_audio_queue, AQframesPerCB*desc_out.mBytesPerFrame, &buf_out))) return ERRORv(-1, "AudioQueueAllocateBuffer: error ", err); return -1; }
      buf_out->mAudioDataByteSize = AQframesPerCB * FLAGS_chans_out * sizeof(short);
      memset(buf_out->mAudioData, 0, buf_out->mAudioDataByteSize);
      if ((err = AudioQueueEnqueueBuffer(output_audio_queue, buf_out, 0, 0))) return ERRORv(-1, "AudioQueueEnqueueBuffer: error ", err); return -1; }
    }
    return 0;
  }

  int Start() {
    if ((err = AudioQueueStart(input_audio_queue, 0))) return ERRORv(-1, "AudioQueueStart: error ", err);
    if ((err = AudioQueueStart(output_audio_queue, 0))) return ERRORv(-1, "AudioQueueStart: error ", err);
    return 0;
  }

  int Free() {
    if (input_audio_queue) {
      AudioQueueStop(input_audio_queue, true);
      AudioQueueDispose(input_audio_queue, true);
    }
    if (output_audio_queue) {
      AudioQueueStop(output_audio_queue, true);
      AudioQueueDispose(output_audio_queue, true);
    }
    return 0;
  }

  void IOIn(AudioQueueRef inAQ, AudioQueueBufferRef inB, const AudioTimeStamp *inStartTime, UInt32 count, const AudioStreamPacketDescription *) {
    short *in = (short *)inB->mAudioData;
    double step = Seconds(1)*1000/FLAGS_sample_rate;
    Time stamp = AudioResampler::monotonouslyIncreasingTimestamp(audio->IL->ReadTimestamp(-1), Now()*1000, &step, count);
    RingBuf::WriteAheadHandle IL(audio->IL.get()), IR(audio->IR.get());
    for (int i=0; i<count; i++) {
      Time timestamp = stamp + i*step;
      IL.Write(in[i*FLAGS_chans_in+0]/32768.0, RingBuf::Stamp, timestamp);

      if (FLAGS_chans_in == 1) continue;
      IR.Write(in[i*FLAGS_chans_in+1]/32768.0, RingBuf::Stamp, timestamp);
    }
    { 
      ScopedMutex ML(audio->inlock);
      IL.Commit();
      IR.Commit();
      app->samples_read += count;
    }
    AudioQueueEnqueueBuffer(inAQ, inB, 0, 0);
  }

  void IOOut(AudioQueueRef outAQ, AudioQueueBufferRef outB) {
    short *out = (short *)outB->mAudioData;
    outB->mAudioDataByteSize = AQframesPerCB * FLAGS_chans_out * sizeof(short);

    ScopedMutex ML(audio->outlock);
    for (int i=0, l=AQframesPerCB*FLAGS_chans_out; i<l; i++)
      out[i] = (audio->Out.size() ? PopFront(audio->Out) : 0) * 32768.0;

    AudioQueueEnqueueBuffer(outAQ, outB, 0, 0);
  }

  static void IOInCB(void *opaque, AudioQueueRef inAQ, AudioQueueBufferRef inB,
                     const AudioTimeStamp *inStartTime, UInt32 count,
                     const AudioStreamPacketDescription *pd) {
    return ((AudioQueueAudioModule*)opaque)->IOIn(inAQ, inB, inStartTime, count, pd);
  }

  static void IOOutCB(void *opaque, AudioQueueRef outAQ, AudioQueueBufferRef outB) {
    return ((AudioQueueAudioModule*)opaque)->IOOut(outAQ, outB);
  }
};
#endif /* LFL_AUDIOQUEUE */

#ifdef LFL_AUDIOUNIT
struct AudioUnitAudioModule : public Module {
  static const int au_output_element=0, au_input_element=1;
  AudioComponentInstance inst_audiounit = 0;
  Audio *audio;
  AudioUnitAudioModule(Audio *A) : audio(A) {}

  int Init() {
    INFO("AudioUnitAudioModule::Init");
    if (FLAGS_chans_in == -1) FLAGS_chans_in = 2;
    if (FLAGS_chans_out == -1) FLAGS_chans_out = 2;

    AudioComponentDescription compdesc;
    compdesc.componentType = kAudioUnitType_Output;
    compdesc.componentSubType = kAudioUnitSubType_RemoteIO;
    compdesc.componentManufacturer = kAudioUnitManufacturer_Apple;
    compdesc.componentFlags = 0;
    compdesc.componentFlagsMask = 0;

    AudioComponent comp = AudioComponentFindNext(0, &compdesc);
    AudioComponentInstanceNew(comp, &inst_audiounit);

    UInt32 enableIO = 1; int err;
    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input,  au_input_element,  &enableIO, sizeof(enableIO)))) return ERRORv(-1, "AudioUnitSetProperty: ", err);
    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, au_output_element, &enableIO, sizeof(enableIO)))) return ERRORv(-1, "AudioUnitSetProperty: ", err);

    AudioStreamBasicDescription desc;
    memset(&desc, 0, sizeof(desc));
    desc.mSampleRate = FLAGS_sample_rate;
    desc.mFormatID = kAudioFormatLinearPCM;
    desc.mFormatFlags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked;
    desc.mChannelsPerFrame = FLAGS_chans_in;
    desc.mBytesPerFrame = desc.mChannelsPerFrame * sizeof(short);
    desc.mFramesPerPacket = 1;
    desc.mBytesPerPacket = desc.mFramesPerPacket * desc.mBytesPerFrame;
    desc.mBitsPerChannel = 8 * sizeof(short);

    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,  au_output_element, &desc, sizeof(AudioStreamBasicDescription)))) return ERRORv(-1, "AudioUnitSetProperty: ", err);
    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, au_input_element,  &desc, sizeof(AudioStreamBasicDescription)))) return ERRORv(-1, "AudioUnitSetProperty: ", err);

    AURenderCallbackStruct cbin, cbout;
    cbin.inputProc = &AudioUnitAudioModule::IOInCB;
    cbin.inputProcRefCon = inst_audiounit;
    cbout.inputProc = &AudioUnitAudioModule::IOOutCB;
    cbout.inputProcRefCon = inst_audiounit;

    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, au_input_element,  &cbin,  sizeof( cbin)))) return ERRORv(-1, "AudioUnitSetProperty: ", err);
    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioUnitProperty_SetRenderCallback,      kAudioUnitScope_Global,  au_output_element, &cbout, sizeof(cbout)))) return ERRORv(-1, "AudioUnitSetProperty: ", err);

    // UInt32 shouldAllocateBuffer = 0;
    // if ((err = AudioUnitSetProperty(inst_audiounit, kAudioUnitProperty_ShouldAllocateBuffer, kAudioUnitScope_Output, au_input_element,   &shouldAllocateBuffer, sizeof(shouldAllocateBuffer)))) return ERRORv(-1, "AudioUnitSetProperty: ", err);

    char errBuf[sizeof(int)+1]; errBuf[sizeof(errBuf)-1] = 0;
    if (err = AudioSessionInitialize(NULL, NULL, NULL, NULL)) { memcpy(errBuf, &err, sizeof(int)); INFO("AudioSessionInitialize - ", errBuf); }

    UInt32 sessionCategory = kAudioSessionCategory_PlayAndRecord;    
    if (err = AudioSessionSetProperty (kAudioSessionProperty_AudioCategory, sizeof (sessionCategory), &sessionCategory)) {  memcpy(errBuf, &err, sizeof(int)); INFO("AudioSessionSetProperty - kAudioSessionProperty_AudioCategory - ", errBuf); }

    UInt32 newRoute = kAudioSessionOverrideAudioRoute_Speaker;
    if (err = AudioSessionSetProperty(kAudioSessionProperty_OverrideAudioRoute, sizeof(newRoute), &newRoute)) { memcpy(errBuf, &err, sizeof(int)); INFO("AudioSessionSetProperty - kAudioSessionProperty_OverrideAudioRoute - Set speaker on ", errBuf); }

    if ((err = AudioUnitInitialize(inst_audiounit))) return ERRORv(-1, "AudioUnitInitialize: ", err);
    return 0;
  }

  int Start() {
    int err;
    if ((err = AudioOutputUnitStart(inst_audiounit))) return ERRORv(-1, "AudioOutputUnitState: ", err);
    return 0;
  }

  int Free() {
    AudioUnitUninitialize(inst_audiounit);
    return 0;
  }

  OSStatus IOIn(void *inRefCon, AudioUnitRenderActionFlags *ioActionFlags, const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames, AudioBufferList *ioData) {
    static const int insize=2*16384;
    static short in[insize];

    AudioBufferList list;
    list.mNumberBuffers = 1;
    list.mBuffers[0].mData = in;
    list.mBuffers[0].mNumberChannels = FLAGS_chans_in;
    list.mBuffers[0].mDataByteSize = sizeof(short) * inNumberFrames * FLAGS_chans_in;
    if (list.mBuffers[0].mDataByteSize > insize) FATAL("overflow ", list.mBuffers[0].mDataByteSize, " > ", insize);
    ioData = &list;
    AudioUnitRender(inst_audiounit, ioActionFlags, inTimeStamp, au_input_element, inNumberFrames, ioData);

    double step = Seconds(1)*1000/FLAGS_sample_rate;
    Time stamp = AudioResampler::monotonouslyIncreasingTimestamp(audio->IL->ReadTimestamp(-1), Now()*1000, &step, inNumberFrames);
    RingBuf::WriteAheadHandle IL(audio->IL.get()), IR(audio->IR.get());
    for (int i=0; i<inNumberFrames; i++) {
      Time timestamp = stamp + i*step;
      IL.Write(in[i*FLAGS_chans_in+0]/32768.0, RingBuf::Stamp, timestamp);

      if (FLAGS_chans_in == 1) continue;
      IR.Write(in[i*FLAGS_chans_in+1]/32768.0, RingBuf::Stamp, timestamp);
    }
    { 
      ScopedMutex ML(audio->inlock);
      IL.Commit();
      IR.Commit();
      app->samples_read += inNumberFrames;
    }
    return 0;
  }

  OSStatus IOOut(void *inRefCon, AudioUnitRenderActionFlags *ioActionFlags, const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames, AudioBufferList *ioData) {
    if (!ioData || ioData->mNumberBuffers != 1) FATAL("unexpected parameters ", ioData, " ", ioData?ioData->mNumberBuffers:-1);
    ioData->mBuffers[0].mDataByteSize = inNumberFrames * FLAGS_chans_out * sizeof(short);
    short *out = (short *)ioData->mBuffers[0].mData;
    {
      ScopedMutex ML(audio->outlock);
      for (int i=0, l=inNumberFrames*FLAGS_chans_out; i<l; i++)
        out[i] = (audio->Out.size() ? PopFront(audio->Out) : 0) * 32768.0;
    }
    return 0;
  }

  static OSStatus IOInCB (void *inRefCon, AudioUnitRenderActionFlags *ioActionFlags, const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames, AudioBufferList *ioData) {
    return ((AudioUnitAudioModule*)app->audio.impl)->IOIn(inRefCon, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, ioData);
  }

  static OSStatus IOOutCB(void *inRefCon, AudioUnitRenderActionFlags *ioActionFlags, const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames, AudioBufferList *ioData) {
    return ((AudioUnitAudioModule*)app->audio.impl)->IOOut(inRefCon, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, ioData);
  }
};
#endif /* LFL_AUDIOUNIT */

#ifdef LFL_OPENSL
struct OpenSLAudioModule : public Module {
  Audio *audio;
  OpenSLAudioModule(Audio *A) : audio(A) {}
  int Init() {
    INFO("OpenSLAudioModule::Init");
    if (FLAGS_chans_in == -1) FLAGS_chans_in = 2;
    if (FLAGS_chans_out == -1) FLAGS_chans_out = 2;
    return 0;
  }
};
#endif /* LFL_OPENSL */

int Audio::Init() {
  INFO("Audio::Init()");
#if defined(LFL_PORTAUDIO)
  impl = make_unique<PortaudioAudioModule>(this);
#elif defined(LFL_SDLAUDIO)
  impl = make_unique<SDLAudioModule>(this);
#elif defined(LFL_AUDIOQUEUE)
  impl = make_unique<AudioQueueAudioModule>(this);
#elif defined(LFL_AUDIOUNIT)
  impl = make_unique<AudioUnitAudioModule>(this);
#elif defined(LFL_OPENSL)
  impl = make_unique<OpenSLAudioModule>(this);
#elif defined(LFL_IPHONE)
  INFO("audio_engine: iPhone");
#elif defined(LFL_ANDROID)
  INFO("audio_engine: Android");
#else
  FLAGS_lfapp_audio = false;
#endif
  if (impl && impl->Init()) return -1;

  IL = make_unique<RingBuf>(FLAGS_sample_rate*FLAGS_sample_secs);
  IR = make_unique<RingBuf>(FLAGS_sample_rate*FLAGS_sample_secs);
  RL = RingBuf::Handle(IL.get());
  RR = RingBuf::Handle(IR.get());

  if (impl && impl->Start()) return -1;
  if (FLAGS_chans_out < 0) FLAGS_chans_out = 0;
  return 0;
}

int Audio::Start() {
  return 0;
}

int Audio::Frame(unsigned clicks) {
  {
    ScopedMutex ML(inlock);

    /* frame align stream handles */
    RL.next = IL->ring.back;
    RR.next = IR->ring.back;

    mic_samples = samples_read - samples_read_last;
    samples_read_last = samples_read;
  }

  const int refillWhen = FLAGS_sample_rate*FLAGS_chans_out/2;

  if (app->asset_loader->movie_playing) {
    app->asset_loader->movie_playing->Play(0);
  } else if ((playing || loop) && Out.size() < refillWhen) {
    // QueueMix(playing ? playing : loop, !playing ? MixFlag::Reset : 0, -1, -1);
    app->RunInMainThread(bind(&Audio::QueueMix, this,
                              playing ? playing : loop, !playing ? MixFlag::Reset : 0, -1, -1));
  }
  return 0;
}

int Audio::Free() {
  if (impl) impl->Free();
  return 0;
}

void Audio::QueueMixBuf(const RingBuf::Handle *B, int channels, int flag) {
  ScopedMutex ML(outlock);
  outlast = B->Len() * FLAGS_chans_out;
  bool mix = flag & MixFlag::Mix, dont_queue = flag & MixFlag::DontQueue;

  for (int i=0; i<B->Len(); i++) {
    if (channels == 1) {
      for (int j=0; j<FLAGS_chans_out; j++) {
        int ind = i*FLAGS_chans_out + j;
        bool mixable = mix && ind < Out.size();
        if (dont_queue && !mixable) return;
        if (mixable) Out[ind] = Clamp(Out[ind] + B->Read(i), -32767, 32768);
        else Out.push_back(B->Read(i));
      }
    }
    else if (channels == FLAGS_chans_out) {
      bool mixable = mix && i < Out.size();
      if (dont_queue && !mixable) return;
      if (mixable) Out[i] = Clamp(Out[i] + B->Read(i), -32767, 32768);
      else Out.push_back(B->Read(i));
    }
    else FATAL("QueueMix unsupported channel combo ", channels, ", ", FLAGS_chans_out);
  }
}

void Audio::QueueMix(SoundAsset *sa, int flag, int begin, int len) {
  if (!sa->wav) return ERROR("QueueMix: asset(", sa->name, ") missing wave data", sa->name);
  if (sa->refill) {
    int samples = sa->refill(sa, flag & MixFlag::Reset);
    playing = (samples == SoundAsset::Size(sa)) ? sa : 0;
  }
  RingBuf::Handle B(sa->wav.get(), begin, len);
  QueueMixBuf(&B, sa->channels, flag);
}

int Audio::Snapshot(SoundAsset *out) {
  ScopedMutex ML(inlock);
  RingBuf::Handle(out->wav.get()).CopyFrom(&RL);
  return 0;
}

double Audio::VisualDelay() {
#ifdef __APPLE__
  return .1;
#else
  return .4;
#endif
}

int Sinthesize(Audio *s, int hz1, int hz2, int hz3) {
  RingBuf tonebuf(FLAGS_sample_rate, FLAGS_sample_rate);
  RingBuf::Handle tone(&tonebuf);
  double f = 2 * M_PI / tone.Rate();
  for (int i=0; i<tone.Len(); i++) {
    double v=sin(f*hz1*i);
    if (hz2) v+=sin(f*hz2*i);
    if (hz3) v+=sin(f*hz3*i);
    tone.Write(v);
  }
  s->QueueMixBuf(&tone);
  return 0;
}

double LowPassFilter(int n, int i, int maxfreq) {
  int ind = i-(n-1)/2;
  double R=2.0*maxfreq/FLAGS_sample_rate;
  return R*Sinc(ind*R);
}

double HighPassFilter(int n, int i, int minfreq) {
  int ind = i-(n-1)/2;
  double impulse = ind ? 0 : 1;
  return impulse - LowPassFilter(n,i,minfreq);
}

float PseudoEnergy(const RingBuf::Handle *in, int window, int offset) {
  float e=0;
  for (int i = 0; i < window; i++) e += fabs(in->Read(offset + i) * 32768.0);
  return e;
}

int ZeroCrossings(const RingBuf::Handle *in, int window, int offset) {
  int zcr=0;
  float last = in->Read(offset);
  for (int i = 1; i < window; i++) {
    float cur = in->Read(offset + i);
    if ((cur < 0 && last >= 0) || (cur >= 0 && last < 0)) zcr++;
    last = cur;
  }
  return zcr;
}

RingBuf *Decimate(const RingBuf::Handle *inh, int factor) {
  const int SPS=inh->Rate(), SPB=inh->Len();
  unique_ptr<RingBuf> outbuf = make_unique<RingBuf>(SPS/factor, SPB/factor);
  RingBuf::Handle out(outbuf.get());

  double sum = 0;
  for (int i = 0, count = 0; i < SPB; i++) {
    sum += inh->Read(-SPB + i);
    if (++count >= factor) {
      out.Write(sum / count);
      count = 0; 
      sum = 0;
    }
  }
  return outbuf.release();
}

int CrossCorrelateTDOA(const RingBuf::Handle *a, const RingBuf::Handle *b, int window, int offset, int samps) {
  float max=0; int maxind=0;
  float res[100];

  /* a : b */
  for (int i=0; i<window; i++) {
    float sum=0;
    for (int j=0; j<samps; j++) sum += a->Read(offset+j) * b->Read(offset+i+j);
    if (sum > max) { max=sum; maxind=i; }
    res[i] = max;
  }

  /* b : a */
  for (int i=1; i<window; i++) {
    float sum=0;
    for (int j=0; j<samps; j++) sum += b->Read(offset+j) * a->Read(offset+i+j);
    if (sum > max) { max=sum; maxind=-i; }
  }

  return maxind;
}

/* AudioResampler */

void AudioResampler::Clear() { out=0; input_processed=output_available=output_rate=output_chans=0; }

#ifdef LFL_FFMPEG
void AudioResampler::Close() {
  if (swr) { swr_free(&swr); swr=0; }
  Clear();
}

int AudioResampler::Open(RingBuf *rb, int in_channels,  int in_sample_rate,  int in_sample_type,
                         int out_channels, int out_sample_rate, int out_sample_type) {
  Close();
  out = rb;
  input_chans = in_channels;
  output_chans = out_channels;
  output_rate = out_sample_rate;
  int input_layout  =  in_channels > 1 ? AV_CH_LAYOUT_STEREO : AV_CH_LAYOUT_MONO;
  int output_layout = out_channels > 1 ? AV_CH_LAYOUT_STEREO : AV_CH_LAYOUT_MONO;
  swr = swr_alloc_set_opts(swr, output_layout, AVSampleFormat(Sample::ToFFMpegId(out_sample_type)), output_rate,
                           input_layout,  AVSampleFormat(Sample::ToFFMpegId(in_sample_type)), in_sample_rate, 0, 0);
  if (swr_init(swr) < 0) ERROR("swr_init");
  return 0;
}

int AudioResampler::Update(int samples, const short *in) {
  if (!out || !in) return -1;
  const short *input[SWR_CH_MAX] = { in, 0 }; 
  Allocator *tlsalloc = ThreadLocalStorage::GetAllocator();
  short *rsout = FromVoid<short*>(tlsalloc->Malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE + FF_INPUT_BUFFER_PADDING_SIZE));
  return Update(samples, input, rsout, microseconds(-1), AVCODEC_MAX_AUDIO_FRAME_SIZE/output_chans/2);
}

int AudioResampler::Update(int samples, RingBuf::Handle *L, RingBuf::Handle *R) {
  int channels = (L!=0) + (R!=0);
  if (!out || !channels) return -1;
  Allocator *tlsalloc = ThreadLocalStorage::GetAllocator();
  short *rsin = FromVoid<short*>(tlsalloc->Malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE + FF_INPUT_BUFFER_PADDING_SIZE));
  short *rsout = FromVoid<short*>(tlsalloc->Malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE + FF_INPUT_BUFFER_PADDING_SIZE));
  memset(rsin+samples*channels, 0, FF_INPUT_BUFFER_PADDING_SIZE);

  RingBuf::Handle *chan[2] = { L, R }; 
  for (int i=0; i<samples; i++)
    for (int j=0; j<channels; j++)
      rsin[i*channels + j] = chan[j]->Read(i) * 32768.0;

  const short *input[SWR_CH_MAX] = { rsin, 0 }; 
  return Update(samples, input, rsout, chan[0]->ReadTimestamp(0), AVCODEC_MAX_AUDIO_FRAME_SIZE/output_chans/2);
}

int AudioResampler::Update(int samples, const short **in, short *rsout, microseconds timestamp, int max_samples_out) {
  CHECK(swr);
  uint8_t *aout[SWR_CH_MAX] = { reinterpret_cast<uint8_t*>(rsout), 0 };
  int resampled = swr_convert(swr, aout, max_samples_out, reinterpret_cast<const uint8_t**>(in), samples);
  if (resampled < 0) return ERRORv(-1, "av_resample return ", resampled);
  if (!resampled) return 0;

  microseconds step(1000000/output_rate);
  int output = resampled * output_chans;
  microseconds stamp = MonotonouslyIncreasingTimestamp(out->ReadTimestamp(-1), timestamp, &step, output);
  for (int i=0; i<output; i++) {
    if (timestamp != microseconds(-1)) out->stamp[out->ring.back] = stamp + i/output_chans*step;
    *FromVoid<float*>(out->Write()) = rsout[i] / 32768.0;
  }

  input_processed += samples;
  output_available += output;
  return 0;
}
#else
void AudioResampler::Close() {}
int AudioResampler::Open(RingBuf *rb, int in_channels,  int in_sample_rate,  int in_sample_type,
                         int out_channels, int out_sample_rate, int out_sample_type)             { FATAL("not implemented"); }
int AudioResampler::Update(int samples, const short **in, short *rsout, microseconds ts, int max_samples_out) { FATAL("not implemented"); }
int AudioResampler::Update(int samples, const short *in)                                                      { FATAL("not implemented"); }
int AudioResampler::Update(int samples, RingBuf::Handle *L, RingBuf::Handle *R)                               { FATAL("not implemented"); }
#endif /* LFL_FFMPEG */

microseconds AudioResampler::MonotonouslyIncreasingTimestamp(microseconds laststamp, microseconds stamp, microseconds *step, int steps) {
  if (laststamp > stamp) {
    microseconds end = stamp + (steps-1) * (*step);
    *step = max(microseconds(1), (end - laststamp) / steps);
    stamp = laststamp + *step;
  }
  return stamp;
}

/* StatefulFilter */

void StatefulFilter::Open(int FilterLenB, const double *FilterB, int FilterLenA, const double *FilterA) {
  next = samples = 0;

  filterB    = FilterB;    filterA    = FilterA;
  filterLenB = FilterLenB; filterLenA = FilterLenA;

  size = max(filterLenA, filterLenB);
  memset(state, 0, sizeof(state));
}

double StatefulFilter::Filter(double sample) {
  for (int i=0; i<filterLenB; i++) {
    int si = (next+i) % size;
    state[si] += sample * filterB[i];
  }
  double ret = state[next];

  for (int i=1; i<filterLenA; i++) {
    int si = (next+i) % size;
    state[si] -= ret * filterA[i];
  }
  state[next] = 0;

  samples++;
  next = (next+1) % size;
  return ret;
}

int StatefulFilter::Filter(const RingBuf::Handle *in, RingBuf::Handle *out, int start, int length) {
  int count = 0;
  if (!length) length = in->Len();
  for (int i=start; i<length; i++) {
    double yi = Filter(in->Read(i));
    if (out) out->Write(yi);
    count++;
  }
  return count;
}

/* stateless filters */

double Filter(const RingBuf::Handle *in, int offset1,
              const RingBuf::Handle *out, int offset2,
              int filterlenB, const double *filterB,
              int filterlenA, const double *filterA,
              double initialcondition, bool nohistory) {    
  double yi = initialcondition;
  for (int k=0; k<filterlenB; k++) {
    double xk = nohistory && offset1-k < 0 ? 0 : in->Read(offset1-k);
    yi += xk * filterB[k];
  }
  bool IIR = out && filterA && filterlenA;
  if (!IIR) return yi;

  for (int k=0; k<filterlenA; k++) {
    double yk = nohistory && offset2-k < 0 ? 0 : out->Read(offset2-k);
    yi -= yk * filterA[k];
  }
  return yi;
}

double Filter(const RingBuf::Handle *in, int offset, int filterlen, const double *filter, bool nohistory) { return LFL::Filter(in, offset, 0, 0, filterlen, filter, 0, 0, 0, nohistory); }

int Filter(const RingBuf::Handle *in, RingBuf::Handle *out, int filterlenB, const double *filterB, int filterlenA, const double *filterA, int start, double *ic, double iclen, bool nohistory) {
  for (int i=start; i<in->Len(); i++) {
    double yi = LFL::Filter(in, i, out, i, filterlenB, filterB, filterlenA, filterA, (iclen-- > 0) ? *ic++ : 0, nohistory);
    out->Write(yi);
  }
  return 0;
}

int Filter(const RingBuf::Handle *in, RingBuf::Handle *out, int filterlen, const double *filter, bool nohistory) { return LFL::Filter(in, out, filterlen, filter, 0, 0, 0, 0, 0, nohistory); }

/* streaming overlap-add fft/ifft */

int FFT(const RingBuf::Handle *in, int i, int window, int hop, int fftlen, float *out,
        const vector<double> &pef, bool doHamming, int scale) {
  int padding=fftlen-window, split=window/2, bits=WhichLog2(fftlen), ind;
  if (bits < 0 || window % 2 != 0) return -1;

  for (int j=0; j<fftlen; j++) {			
    if      (j < split)           { ind = i*hop + split + j; }
    else if (j < split + padding) { out[j] = 0; continue; }
    else                          { ind = i*hop + j - split - padding; }

    double v = pef.size() ? Filter(in, ind, pef.size(), &pef[0]) : in->Read(ind);

    if (doHamming) v *= Hamming(window, ind - i*hop);
    out[j] = v * scale;
  }

#ifdef LFL_FFMPEG
  RDFTContext *fftctx = av_rdft_init(bits, DFT_R2C);
  av_rdft_calc(fftctx, out);
  av_rdft_end(fftctx);
#endif
  return 0;
}

int IFFT(const Complex *in, int i, int window, int hop, int fftlen, RingBuf::Handle *out) {
  Complex zero={0,0};
  vector<float> fftbuf(fftlen*2);
  int split=fftlen/2, bits=WhichLog2(fftlen);
  if (bits < 0) return -1;

  for (int j=0; j<split; j++) {
    fftbuf[j*2]   = in[j].r;
    fftbuf[j*2+1] = in[j].i;
  }
  for (int j=0; j<split; j++) {
    Complex val = !j ? zero : in[split-j].Conjugate();
    fftbuf[(split+j)*2]   = val.r;
    fftbuf[(split+j)*2+1] = val.i;
  }

#ifdef LFL_FFMPEG
  RDFTContext *fftctx = av_rdft_init(bits, IDFT_C2R);
  av_rdft_calc(fftctx, &fftbuf[0]);
  av_rdft_end(fftctx);
#endif

  for (int j=0; j<split; j++) *out->Index(i*hop +         j) += fftbuf[split+j];
  for (int j=0; j<split; j++) *out->Index(i*hop + split + j) += fftbuf[j];

  double scalef = (1.0/fftlen) * (double(window)/fftlen);
  for (int j=0; j<hop; j++) *out->Index(i*hop + j) *= scalef;
  return 0;
}

/* fft filter */

void FFTFilterCompile(int n, double *filter) {
  RingBuf filtbuf(FLAGS_sample_rate, n);
  RingBuf::Handle filtcoef(&filtbuf);
  for (int i=0; i<n; i++) filtcoef.Write(filter[i]);

  vector<float> fftbuf(n);
  FFT(&filtcoef, 0, n, 0, n, &fftbuf[0]);
  for (int i=0; i<n; i++) filter[i] = fftbuf[i];
}

int FFTFilter(const RingBuf::Handle *in, RingBuf::Handle *out, int window, int hop, const double *filter) {
  int frames = (in->Len() - (window - hop)) / hop;
  vector<float> fftbuf(window);
  vector<Complex> ifftb(window/2);

  for (int j=0; j<in->Len(); j++) *out->Index(j) = 0;

  for (int i=0; i<frames; i++) {
    if (FFT(in, i, window, hop, window, &fftbuf[0])) return -1;

    for (int j=0; j<window/2; j++) {
      Complex x={fft_r(&fftbuf[0], window, j), fft_i(&fftbuf[0], window, j)};
      Complex h={fft_r(filter,     window, j), fft_i(filter,     window, j)};
      ifftb[j] = Complex::Mult(x, h); /* cyclic convolution */
    }

    if (IFFT(&ifftb[0], i, window, hop, window, out)) return -1;
  }
  return 0;
}

Matrix *Spectogram(const RingBuf::Handle *in, Matrix *out, int window, int hop, int fftlen,
                   const vector<double> &preemph, int pd, int scale) {
  bool complex = (pd==PowerDomain::complex);
  int frames = (in->Len() - (window - hop)) / hop;
  vector<float> fftbuf(fftlen);

  Matrix *m;
  if (out) { m=out; if (m->M != frames || m->N != fftlen/2 || (m->flag&Matrix::Flag::Complex && !complex)) return 0; }
  else m = new Matrix(frames, fftlen/2, 0.0, complex);

  for (int i=0; i<frames; i++) {
    double *row = m->row(i);
    if (FFT(in, i, window, hop, fftlen, &fftbuf[0], preemph, true, scale)) { if (!out) delete m; return 0; }

    for (int j=0; j<fftlen/2; j++) {
      if (pd == PowerDomain::complex) {
        row[j*2 + 0] = fft_r(&fftbuf[0], fftlen, j);
        row[j*2 + 1] = fft_i(&fftbuf[0], fftlen, j);
        continue;
      }

      if      (pd == PowerDomain::abs)  row[j] = sqrt(fft_abs2(&fftbuf[0], fftlen, j));
      else if (pd == PowerDomain::abs2) row[j] = fft_abs2(&fftbuf[0], fftlen, j);      
      else if (pd == PowerDomain::dB)   row[j] = AmplitudeRatioDecibels(sqrt(fft_abs2(&fftbuf[0], fftlen, j)), 1);
    }
  }

  return m;
}

RingBuf *ISpectogram(const Matrix *in, int window, int hop, int fftlen, int samplerate) {
  unique_ptr<RingBuf> outbuf = make_unique<RingBuf>(samplerate, fftlen + hop * (in->M-1));
  RingBuf::Handle out(outbuf.get());

  for (int i=0; i<in->M; i++) {
    if (IFFT(in->crow(i), i, window, hop, fftlen, &out)) return nullptr;
  }
  return outbuf.release();
}

Matrix *F0Stream(const RingBuf::Handle *in, Matrix *out, int window, int hop, int method) {
  int frames = (in->Len() - (window - hop)) / hop;

  Matrix *m;
  if (out) { m=out; if (m->M != frames || m->N != 1) return 0; }
  else m = new Matrix(frames, 1);

  for (int i=0; i<frames; i++) {
    double *row = m->row(i); 
    row[0] = FundamentalFrequency(in, window, i*hop, method);
  }
  return m;
}

float FundamentalFrequency(const RingBuf::Handle *in, int window, int offset, int method) {
  int bits = WhichLog2(window);
  if (bits<0) return -1;

  if (method == F0EstmMethod::fftbucket) { /* max(fft.bucket) */
    vector<float> samp(window);
    FFT(in, 1, window, offset, window, &samp[0]);

    int maxind=0; double max=0;
    for (int i=1; i<window/2; i++) {
      float abs = sqrt(fft_abs2(&samp[0], window, i));
      if (abs > max) { max=abs; maxind=i; }
    }
    return in->Rate() * maxind / (window/2.0);
  }
  else if (method == F0EstmMethod::xcorr) { /* auto correlation */

    /* zero pad */
    RingBuf zpi(in->Rate(), window*2);
    RingBuf::Handle zpin(&zpi);
    for (int i=0; i<window; i++) *zpin.Index(i) = in->Read(i+offset);

    /* fft */
    vector<float> buf(window*2);
    FFT(&zpin, 0, window*2, 0, window*2, &buf[0]);

    /* to spectrum */
    fft_r(&buf[0], window*2, 0) = pow(fft_r(&buf[0], window*2, 0), 2);
    fft_i(&buf[0], window*2, 0) = pow(fft_i(&buf[0], window*2, 0), 2);
    for (int i=1; i<window; i++) {
      fft_r(&buf[0], window*2, i) = fft_abs2(&buf[0], window*2, i);
      fft_i(&buf[0], window*2, i) = 0;
    }

    /* ifft */
#ifdef LFL_FFMPEG
    RDFTContext *fftctx = av_rdft_init(bits+1, IDFT_C2R);
    av_rdft_calc(fftctx, &buf[0]);
    av_rdft_end(fftctx);
#endif

    /* auto correlation */
    int minlag = min(in->Rate() / 500, window*2); /* 500 hz */
    int maxlag = min(in->Rate() / 50 + 1, window*2); /* 50 hz */
    vector<float> xcorr(maxlag);

    /* divide by window for ifft and zxc for auto-correlation coefficients */
    float zxc = buf[0] / window;
    for (int i=0; i<maxlag; i++)
      xcorr[i] = buf[i] / window / zxc;

    /* max(xcorr[minlag,maxlag]) */
    int maxind=-1; double max=-INFINITY;
    for (int i=minlag; i<maxlag; i++)
      if (xcorr[i] > max) { max=xcorr[i]; maxind=i; }

    /* xcorr index to hz */
    return float(in->Rate()) / (maxind+1);
  }
  else if (method == F0EstmMethod::cepstral) { /* cepstral analysis */
    vector<float> buf(window);
    FFT(in, 1, window, offset, window, &buf[0]);

    /* to log spectrum */
    RingBuf spe(in->Rate(), window);
    RingBuf::Handle spec(&spe);
    *spec.Index(0) = log(fabs(fft_r(&buf[0], window, 0)));
    for (int i=0; i<window/2; i++) *spec.Index(i) = log(sqrt(fft_abs2(&buf[0], window, i)));
    *spec.Index(window/2) = log(fabs(fft_i(&buf[0], window, 0)));
    for (int i=1; i<window/2; i++) *spec.Index(window/2+i) = *spec.Index(window/2-i);

    /* fft */
    FFT(&spec, 0, window, 0, window, &buf[0]); 

    /* cepstrum */
    int minlag = min(in->Rate() / 1000, window/2); /* 1000 hz */
    int maxlag = min(in->Rate() / 50, window/2); /* 50 hz */

    vector<float> cepstrum(window/2);
    for (int i=0; i<window/2; i++) cepstrum[i] = sqrt(fft_abs2(&buf[0], window, i));

    /* max(cepstrum[minlag,maxlag]) */
    int maxind=-1; double max=-INFINITY;
    for (int i=minlag; i<maxlag; i++) {
      if (cepstrum[i] > max) { max=cepstrum[i]; maxind=i; }
    }

    /* cepstrum index to hz */
    return float(in->Rate()) / (maxind-1);
  }
  else if (method == F0EstmMethod::harmonicity) { /* harmonicity */
    vector<float> buf(window);
    FFT(in, 1, window, offset, window, &buf[0]);
  }
  return -1;
}

}; // namespace LFL
