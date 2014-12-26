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
#ifdef LFL_IPHONE
extern void iphone_play_music(void *handle);
extern void iphone_play_background_music(void *handle);
#endif

const int feat_lpccoefs  = 12;
const int feat_barkbands = 21;

#ifdef __APPLE__
const double feat_progressbar_c = .1;
#else
const double feat_progressbar_c = .4;
#endif

const double feat_rastaB[] = { .2, .1, 0, -.1, -.2 };
const double feat_rastaA[] = { 1, -.94 };

DEFINE_int(audio_input_device, -1, "Audio input device index");
DEFINE_int(audio_output_device, -1, "Audio output device index");
DEFINE_bool(print_audio_devices, false, "Print audio device list");
DEFINE_string(feat_type, "MFCC", "Feature type");
DEFINE_bool(feat_preemphasis, true, "Use pre-emphasis filter");
DEFINE_bool(feat_dither, false, "Dither input");
DEFINE_double(feat_minfreq, 0, "Minimum feature frequency");
DEFINE_double(feat_maxfreq, FLAGS_sample_rate/2, "Maximum feature frequency");
DEFINE_double(feat_preemphasis_filter, -0.97, "Preemphasis filter = { 1, X }");
DEFINE_int(feat_window, 512, "Feature window length");
DEFINE_int(feat_hop, 256, "Feature window advance");
DEFINE_int(feat_melbands, 40, "Mel bands used to create MFCC features");
DEFINE_int(feat_cepcoefs, 20, "Number of cepstrum coefficients");

int Sample::Size(int fmt) {
    if      (fmt == Sample::U8     || fmt == Sample::U8P)     return 1;
    else if (fmt == Sample::S16    || fmt == Sample::S16P)    return 2;
    else if (fmt == Sample::S32    || fmt == Sample::S32P)    return 4;
    else if (fmt == Sample::FLOAT  || fmt == Sample::FLOATP)  return 4;
    else if (fmt == Sample::DOUBLE || fmt == Sample::DOUBLEP) return 8;
    else { ERROR("unknown sample fmt: ", fmt); return 0; }
}

#ifdef LFL_FFMPEG
int Sample::FromFFMpegId(int fmt) {
    if      (fmt == AV_SAMPLE_FMT_U8)   return Sample::U8;
    else if (fmt == AV_SAMPLE_FMT_U8P)  return Sample::U8P;
    else if (fmt == AV_SAMPLE_FMT_S16)  return Sample::S16;
    else if (fmt == AV_SAMPLE_FMT_S16P) return Sample::S16P;
    else if (fmt == AV_SAMPLE_FMT_S32)  return Sample::S32;
    else if (fmt == AV_SAMPLE_FMT_S32P) return Sample::S32P;
    else if (fmt == AV_SAMPLE_FMT_FLT)  return Sample::FLOAT;
    else if (fmt == AV_SAMPLE_FMT_FLTP) return Sample::FLOATP;
    else if (fmt == AV_SAMPLE_FMT_DBL)  return Sample::DOUBLE;
    else if (fmt == AV_SAMPLE_FMT_DBLP) return Sample::DOUBLEP;
    else { ERROR("unknown sample fmt: ", fmt); return 0; }
}

int Sample::ToFFMpegId(int fmt) {
    if      (fmt == Sample::U8)      return AV_SAMPLE_FMT_U8;
    else if (fmt == Sample::U8P)     return AV_SAMPLE_FMT_U8P;
    else if (fmt == Sample::S16)     return AV_SAMPLE_FMT_S16;
    else if (fmt == Sample::S16P)    return AV_SAMPLE_FMT_S16P;
    else if (fmt == Sample::S32)     return AV_SAMPLE_FMT_S32;
    else if (fmt == Sample::S32P)    return AV_SAMPLE_FMT_S32P;
    else if (fmt == Sample::FLOAT)   return AV_SAMPLE_FMT_FLT;
    else if (fmt == Sample::FLOATP)  return AV_SAMPLE_FMT_FLTP;
    else if (fmt == Sample::DOUBLE)  return AV_SAMPLE_FMT_DBL;
    else if (fmt == Sample::DOUBLEP) return AV_SAMPLE_FMT_DBLP;
    else { ERROR("unknown sample fmt: ", fmt); return 0; }
}
#endif

void SystemAudio::PlaySoundEffect(SoundAsset *sa) {
#if defined(LFL_ANDROID)
    android_play_music(sa->handle);
#elif defined(LFL_IPHONE)
    iphone_play_music(sa->handle);
#else
    app->audio.QueueMix(sa, MixFlag::Reset | MixFlag::Mix | (app->audio.loop ? MixFlag::DontQueue : 0), -1, -1);
#endif
}

void SystemAudio::PlayBackgroundMusic(SoundAsset *music) {
#if defined(LFL_ANDROID)
    android_play_background_music(music->handle);
#elif defined(LFL_IPHONE)
    iphone_play_background_music(music->handle);
#else
    app->audio.QueueMix(music);
    app->audio.loop = music;
#endif
}

void SystemAudio::SetVolume(int v) { 
#if defined(LFL_ANDROID)
    android_set_volume(v);
#endif
}

int SystemAudio::GetVolume() { 
#if defined(LFL_ANDROID)
    return android_get_volume();
#else
    return 0;
#endif
}

int SystemAudio::GetMaxVolume() { 
#if defined(LFL_ANDROID)
    return android_get_max_volume();
#else
    return 10;
#endif
}

#ifdef LFL_PORTAUDIO
int io_portaudio(const void *inputBuffer, void *outputBuffer, unsigned long samplesPerFrame,
                 const PaStreamCallbackTimeInfo *timeInfo, PaStreamCallbackFlags statusFlags, void *opaque) {
    Audio *s = (Audio*)opaque;
    float *in = (float*)inputBuffer;
    float *out = (float *)outputBuffer;

    double step = Seconds(1)*1000/FLAGS_sample_rate;
    Time stamp = AudioResampler::monotonouslyIncreasingTimestamp(s->IL->ReadTimestamp(-1), Now()*1000, &step, samplesPerFrame);

    RingBuf::WriteAheadHandle IL(s->IL), IR(s->IR);
    for (unsigned i=0; i<samplesPerFrame; i++) {
        Time timestamp = stamp + i*step;
        IL.Write(in[i*FLAGS_chans_in+0], RingBuf::Stamp, timestamp);
        if (FLAGS_chans_in == 1) continue;
        IR.Write(in[i*FLAGS_chans_in+1], RingBuf::Stamp, timestamp);
    }

    {
        ScopedMutex ML(s->inlock);
        IL.Commit();
        IR.Commit();
        app->samples_read += samplesPerFrame;
    }

    {
        ScopedMutex ML(s->outlock);
        for (unsigned i=0; i<samplesPerFrame*FLAGS_chans_out; i++) 
            out[i] = s->Out.size() ? PopFront(s->Out) : 0;
    }
    return 0;
}

int open_portaudio(Audio *s) {
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

    PaStreamParameters in  = { FLAGS_audio_input_device,  FLAGS_chans_in,  paFloat32, ii->defaultHighInputLatency, 0 };
    PaStreamParameters out = { FLAGS_audio_output_device, FLAGS_chans_out, paFloat32, oi->defaultHighInputLatency, 0 };

    PaError err = Pa_OpenStream(&s->impl, &in, FLAGS_chans_out ? &out : 0, FLAGS_sample_rate, paFramesPerBufferUnspecified, 0, io_portaudio, s);
    if (err != paNoError) { ERROR("open_portaudio: ", Pa_GetErrorText(err)); return -1; }

    err = Pa_StartStream(s->impl);
    if (err != paNoError) { ERROR("open_portaudio2: ", Pa_GetErrorText(err)); return -1; }

    return 0;
}

int init_portaudio(Audio *s) {
    PaError err = Pa_Initialize();
    if (err != paNoError) { ERROR("init_portaudio: ", Pa_GetErrorText(err)); return -1; }

    int numDevices = Pa_GetDeviceCount();
    if(numDevices<=0) { ERROR("open_portaudio: devices=", numDevices); return -1; }

    for(int i=0; false && i<numDevices; i++) {
        const PaDeviceInfo *di = Pa_GetDeviceInfo(i);
        ERROR(di->name, ",", di->hostApi, ", mi=", di->maxInputChannels, ", mo=", di->maxOutputChannels, ", ", di->defaultSampleRate, " ", di->defaultHighInputLatency);
    }

    return 0;
}

int free_portaudio(Audio *s) {
    PaError err = Pa_StopStream(s->impl);
    if (err != paNoError) { ERROR("free_portaudio: stop ", Pa_GetErrorText(err)); return 0; }

    err = Pa_Terminate();
    if (err != paNoError) { ERROR("free_portaudio: term ", Pa_GetErrorText(err)); return 0; }
    return 0;
}

#endif /* LFL_PORTAUDIO */

#ifdef LFL_SDLAUDIO
void io_sdlaudio(void *udata, Uint8 *stream, int len) {
    Audio *s = (Audio*)udata;
    short *out = (short*)stream;
    int frames = len/sizeof(short)/FLAGS_chans_out;
    
    ScopedMutex ML(s->outlock);
    for (int i=0; i<frames*FLAGS_chans_out; i++)
        out[i] = (s->Out.size() ? s->Out.pop_front() : 0) * 32768.0;
}

int init_sdlaudio(Audio *s) {
    if (print_audio_devices) {
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
    desired.callback = io_sdlaudio;
    desired.userdata = s;

    if (SDL_OpenAudio(&desired, &obtained) < 0) { ERROR("SDL_OpenAudio:", SDL_GetError()); return -1; }
    if (obtained.freq != FLAGS_sample_rate) { ERROR("SDL_OpenAudio: sample_rate ", obtained.freq, " != ", FLAGS_sample_rate); return -1; }

    FLAGS_chans_out = obtained.channels;
    SDL_PauseAudio(0);
    return 0;
}

int free_sdlaudio(Audio *s) {
    SDL_PauseAudio(0);
    return 0;
}
#endif /* LFL_SDLAUDIO */

#ifdef LFL_AUDIOQUEUE
AudioQueueRef inputAudioQueue=0, outputAudioQueue=0;
static const int AQbuffers=3, AQframesPerCB=512;

void io_audioqueue_in(void *opaque, AudioQueueRef inAQ, AudioQueueBufferRef inB, const AudioTimeStamp *inStartTime, UInt32 count, const AudioStreamPacketDescription *) {
    short *in = (short *)inB->mAudioData;
    Audio *s = (Audio*)opaque;

    double step = Seconds(1)*1000/FLAGS_sample_rate;
    Time stamp = AudioResampler::monotonouslyIncreasingTimestamp(s->IL->readtimestamp(-1), Now()*1000, &step, count);

    RingBuf::WriteAheadHandle IL(s->IL), IR(s->IR);
    for (int i=0; i<count; i++) {
        Time timestamp = stamp + i*step;
        IL.write(in[i*FLAGS_chans_in+0]/32768.0, RingBuf::Stamp, timestamp);

        if (FLAGS_chans_in == 1) continue;
        IR.write(in[i*FLAGS_chans_in+1]/32768.0, RingBuf::Stamp, timestamp);
    }
    
    { 
        ScopedMutex ML(s->inlock);
        IL.commit();
        IR.commit();
        samples_read += count;
    }

    AudioQueueEnqueueBuffer(inAQ, inB, 0, 0);
}

void io_audioqueue_out(void *opaque, AudioQueueRef outAQ, AudioQueueBufferRef outB) {
    Audio *s = (Audio*)opaque;
    short *out = (short *)outB->mAudioData;
    outB->mAudioDataByteSize = AQframesPerCB * FLAGS_chans_out * sizeof(short);

    ScopedMutex ML(s->outlock);
    for (int i=0, l=AQframesPerCB*FLAGS_chans_out; i<l; i++)
        out[i] = (s->Out.size() ? s->Out.pop_front() : 0) * 32768.0;

    AudioQueueEnqueueBuffer(outAQ, outB, 0, 0);
}

int init_audioqueue(Audio *s) {
    if (inputAudioQueue || outputAudioQueue) return -1;
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

    int err = AudioQueueNewInput(&desc_in, io_audioqueue_in, s, 0, kCFRunLoopCommonModes, 0, &inputAudioQueue);
    if (err == kAudioFormatUnsupportedDataFormatError) { ERROR("AudioQueueNewInput: format unsupported sr=", desc_in.mSampleRate); return -1; }
    if (err || !inputAudioQueue) { ERROR("AudioQueueNewInput: error ", err); return -1; }

    AudioStreamBasicDescription desc_out = desc_in;
    desc_out.mChannelsPerFrame = FLAGS_chans_out;
    desc_out.mBytesPerFrame = desc_out.mChannelsPerFrame * sizeof(short);
    desc_out.mBytesPerPacket = desc_out.mFramesPerPacket * desc_out.mBytesPerFrame;
   
    err = AudioQueueNewOutput(&desc_out, io_audioqueue_out, s, 0, kCFRunLoopCommonModes, 0, &outputAudioQueue);
    if (err == kAudioFormatUnsupportedDataFormatError) { ERROR("AudioQueueNewOutput: format unsupported sr=", desc_out.mSampleRate); return -1; }
    if (err || !outputAudioQueue) { ERROR("AudioQueueNewOutput: error ", err); return -1; }

    for (int i=0; i<AQbuffers; i++) {
        AudioQueueBufferRef buf_in, buf_out;
        if ((err = AudioQueueAllocateBuffer(inputAudioQueue, AQframesPerCB*desc_in.mBytesPerFrame, &buf_in))) { ERROR("AudioQueueAllocateBuffer: error ", err); return -1; }
        if ((err = AudioQueueEnqueueBuffer(inputAudioQueue, buf_in, 0, 0))) { ERROR("AudioQueueEnqueueBuffer: error ", err); return -1; }

        if ((err = AudioQueueAllocateBuffer(outputAudioQueue, AQframesPerCB*desc_out.mBytesPerFrame, &buf_out))) { ERROR("AudioQueueAllocateBuffer: error ", err); return -1; }
        buf_out->mAudioDataByteSize = AQframesPerCB * FLAGS_chans_out * sizeof(short);
        memset(buf_out->mAudioData, 0, buf_out->mAudioDataByteSize);
        if ((err = AudioQueueEnqueueBuffer(outputAudioQueue, buf_out, 0, 0))) { ERROR("AudioQueueEnqueueBuffer: error ", err); return -1; }
    }

    if ((err = AudioQueueStart(inputAudioQueue, 0))) { ERROR("AudioQueueStart: error ", err); return -1; }
    if ((err = AudioQueueStart(outputAudioQueue, 0))) { ERROR("AudioQueueStart: error ", err); return -1; }
    return 0;
}

int free_audioqueue(Audio *s) {
    if (inputAudioQueue) {
        AudioQueueStop(inputAudioQueue, true);
        AudioQueueDispose(inputAudioQueue, true);
    }
    if (outputAudioQueue) {
        AudioQueueStop(outputAudioQueue, true);
        AudioQueueDispose(outputAudioQueue, true);
    }
    return 0;
}
#endif /* LFL_AUDIOQUEUE */

#ifdef LFL_AUDIOUNIT
AudioComponentInstance inst_audiounit = 0;
static const int au_output_element=0, au_input_element=1;

OSStatus io_audiounit_in(void *inRefCon, AudioUnitRenderActionFlags *ioActionFlags, const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames, AudioBufferList *ioData) {
    static const int insize=2*16384;
    static short in[insize];
    Audio *s = &app->audio;

    AudioBufferList list;
    list.mNumberBuffers = 1;
    list.mBuffers[0].mData = in;
    list.mBuffers[0].mNumberChannels = FLAGS_chans_in;
    list.mBuffers[0].mDataByteSize = sizeof(short) * inNumberFrames * FLAGS_chans_in;
    if (list.mBuffers[0].mDataByteSize > insize) FATAL("overflow ", list.mBuffers[0].mDataByteSize, " > ", insize);
    ioData = &list;

    AudioUnitRender(inst_audiounit, ioActionFlags, inTimeStamp, au_input_element, inNumberFrames, ioData);

    double step = Seconds(1)*1000/FLAGS_sample_rate;
    Time stamp = AudioResampler::monotonouslyIncreasingTimestamp(s->IL->readtimestamp(-1), Now()*1000, &step, inNumberFrames);

    RingBuf::WriteAheadHandle IL(s->IL), IR(s->IR);
    for (int i=0; i<inNumberFrames; i++) {
        Time timestamp = stamp + i*step;
        IL.write(in[i*FLAGS_chans_in+0]/32768.0, RingBuf::Stamp, timestamp);

        if (FLAGS_chans_in == 1) continue;
        IR.write(in[i*FLAGS_chans_in+1]/32768.0, RingBuf::Stamp, timestamp);
    }
    
    { 
        ScopedMutex ML(s->inlock);
        IL.commit();
        IR.commit();
        samples_read += inNumberFrames;
    }
    return 0;
}

OSStatus io_audiounit_out(void *inRefCon, AudioUnitRenderActionFlags *ioActionFlags, const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames, AudioBufferList *ioData) {
    if (!ioData || ioData->mNumberBuffers != 1) FATAL("unexpected parameters ", ioData, " ", ioData?ioData->mNumberBuffers:-1);
    ioData->mBuffers[0].mDataByteSize = inNumberFrames * FLAGS_chans_out * sizeof(short);

    short *out = (short *)ioData->mBuffers[0].mData;
    Audio *s = &app->audio;

    {
        ScopedMutex ML(s->outlock);
        for (int i=0, l=inNumberFrames*FLAGS_chans_out; i<l; i++)
            out[i] = (s->Out.size() ? s->Out.pop_front() : 0) * 32768.0;
    }
    return 0;
}

int init_audiounit(Audio *s) {
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
    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input,  au_input_element,  &enableIO, sizeof(enableIO)))) { ERROR("AudioUnitSetProperty: ", err); return -1; }
    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, au_output_element, &enableIO, sizeof(enableIO)))) { ERROR("AudioUnitSetProperty: ", err); return -1; }

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

    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input,  au_output_element, &desc, sizeof(AudioStreamBasicDescription)))) { ERROR("AudioUnitSetProperty: ", err); return -1; }
    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, au_input_element,  &desc, sizeof(AudioStreamBasicDescription)))) { ERROR("AudioUnitSetProperty: ", err); return -1; }

    AURenderCallbackStruct cbin, cbout;
    cbin.inputProc = io_audiounit_in;
    cbin.inputProcRefCon = inst_audiounit;
    cbout.inputProc = io_audiounit_out;
    cbout.inputProcRefCon = inst_audiounit;

    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, au_input_element,  &cbin,  sizeof( cbin)))) { ERROR("AudioUnitSetProperty: ", err); return -1; }
    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioUnitProperty_SetRenderCallback,      kAudioUnitScope_Global,  au_output_element, &cbout, sizeof(cbout)))) { ERROR("AudioUnitSetProperty: ", err); return -1; }

#if 0
    UInt32 shouldAllocateBuffer = 0;
    if ((err = AudioUnitSetProperty(inst_audiounit, kAudioUnitProperty_ShouldAllocateBuffer, kAudioUnitScope_Output, au_input_element,   &shouldAllocateBuffer, sizeof(shouldAllocateBuffer)))) { ERROR("AudioUnitSetProperty: ", err); return -1; };
#endif

	char errBuf[sizeof(int)+1]; errBuf[sizeof(errBuf)-1] = 0;
	if (err = AudioSessionInitialize(NULL, NULL, NULL, NULL)) { memcpy(errBuf, &err, sizeof(int)); INFO("AudioSessionInitialize - ", errBuf); }

	UInt32 sessionCategory = kAudioSessionCategory_PlayAndRecord;    
	if (err = AudioSessionSetProperty (kAudioSessionProperty_AudioCategory, sizeof (sessionCategory), &sessionCategory)) {  memcpy(errBuf, &err, sizeof(int)); INFO("AudioSessionSetProperty - kAudioSessionProperty_AudioCategory - ", errBuf); }

	UInt32 newRoute = kAudioSessionOverrideAudioRoute_Speaker;
	if (err = AudioSessionSetProperty(kAudioSessionProperty_OverrideAudioRoute, sizeof(newRoute), &newRoute)) { memcpy(errBuf, &err, sizeof(int)); INFO("AudioSessionSetProperty - kAudioSessionProperty_OverrideAudioRoute - Set speaker on ", errBuf); }

    if ((err = AudioUnitInitialize(inst_audiounit))) { ERROR("AudioUnitInitialize: ", err); return -1; }
    return 0;
}

int start_audiounit(Audio *s) {
    int err;
    if ((err = AudioOutputUnitStart(inst_audiounit))) { ERROR("AudioOutputUnitState: ", err); return -1; }
    return 0;
}

int free_audiounit(Audio *s) {
    AudioUnitUninitialize(inst_audiounit);
    return 0;
};
#endif /* LFL_AUDIOUNIT */

#ifdef LFL_OPENSL
int init_opensl(Audio *s) {
    if (FLAGS_chans_in == -1) FLAGS_chans_in = 2;
    if (FLAGS_chans_out == -1) FLAGS_chans_out = 2;
    return 0;
}
#endif /* LFL_OPENSL */

int Audio::Init() {
    RL = RingBuf::Handle(IL);
    RR = RingBuf::Handle(IR);
#if defined(LFL_PORTAUDIO)
    INFO("audio_engine: PortAudio");
    if (init_portaudio(this) < 0) return -1;
    if (open_portaudio(this) < 0) return -1;
#elif defined(LFL_SDLAUDIO)
    INFO("audio_engine: SDL Audio");
    if (init_sdlaudio(this) < 0) return -1;
#elif defined(LFL_AUDIOQUEUE)
    INFO("audio_engine: AudioQueue");
    if (init_audioqueue(this) < 0) return -1;
#elif defined(LFL_AUDIOUNIT)
    INFO("audio_engine: AudioUnit");
    if (init_audiounit(this) < 0) return -1;
#elif defined(LFL_OPENSL)
    INFO("audio_engine: OpenSL");
    if (init_opensl(this) < 0) return -1;
#elif defined(LFL_IPHONE)
    INFO("audio_engine: iPhone");
#elif defined(LFL_ANDROID)
    INFO("audio_engine: Android");
#else
    FLAGS_lfapp_audio = false;
#endif
    if (FLAGS_chans_out < 0) FLAGS_chans_out = 0;
    return 0;
}

int Audio::Free() {
#if defined(LFL_PORTAUDIO)
    free_portaudio(this);
#elif defined(LFL_SDLAUDIO)
    free_sdlaudio(this);
#elif defined(LFL_AUDIOQUEUE)
    free_audioqueue(this);
#elif defined(LFL_AUDIOUNIT)
    free_audiounit(this);
#endif
    return 0;
}

int Audio::Start() {
    if (!FLAGS_lfapp_audio) return 0;
#ifdef LFL_AUDIOUNIT
    if (start_audiounit(this) < 0) return -1;
#endif
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
                if (mixable) Out[ind] = clamp(Out[ind] + B->Read(i), -32767, 32768);
                else Out.push_back(B->Read(i));
            }
        }
        else if (channels == FLAGS_chans_out) {
            bool mixable = mix && i < Out.size();
            if (dont_queue && !mixable) return;
            if (mixable) Out[i] = clamp(Out[i] + B->Read(i), -32767, 32768);
            else Out.push_back(B->Read(i));
        }
        else FATAL("QueueMix unsupported channel combo ", channels, ", ", FLAGS_chans_out);
    }
}

void Audio::QueueMix(SoundAsset *sa, int flag, int begin, int len) {
    if (!sa->wav) { ERROR("QueueMix: asset(", sa->name, ") missing wave data", sa->name); return; }
    if (sa->refill) {
        int samples = sa->refill(sa, flag & MixFlag::Reset);
        playing = (samples == SoundAssetSize(sa)) ? sa : 0;
    }
    RingBuf::Handle B(sa->wav, begin, len);
    QueueMixBuf(&B, sa->channels, flag);
}

int Audio::Snapshot(SoundAsset *out) {
    ScopedMutex ML(inlock);
    RingBuf::Handle(out->wav).CopyFrom(&RL);
    return 0;
}

int sinthesize(Audio *s, int hz1, int hz2, int hz3) {
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

double lowpass_filter(int n, int i, int maxfreq) {
    int ind = i-(n-1)/2;
    double R=2.0*maxfreq/FLAGS_sample_rate;
    return R*sinc(ind*R);
}

double highpass_filter(int n, int i, int minfreq) {
    int ind = i-(n-1)/2;
    double impulse = ind ? 0 : 1;
    return impulse - lowpass_filter(n,i,minfreq);
}

Matrix *EqualLoudnessCurve(int outrows, double max) {
    double maxbark = hz2bark(max), stepbark = maxbark/(outrows-1);   
    Matrix *ret = new Matrix(outrows, 1);
    MatrixIter(ret) {
        double bandcfhz = bark2hz(i*stepbark), fsq = pow(bandcfhz, 2);
        ret->row(i)[j] = pow(fsq/(fsq + 1.6e5), 2) * ((fsq + 1.44e6) / (fsq + 9.61e6));
    }
    return ret;
}

double *lifterMatrixROSA(int n, double L, bool inverse=false) {
    double *dm = new double[n];
    for (int i=0; i<n; i++) {
        dm[i] = !i ? 1 : pow(i, L);
        if (inverse) dm[i] = 1 / dm[i];
    }
    return dm;
}

double *lifterMatrixHTK(int n, double L, bool inverse=false) {
    double *dm = new double[n];
    for (int i=0; i<n; i++) {
        dm[i] = !i ? 1 : (1+L/2*sin(i*M_PI/L));
        if (inverse) dm[i] = 1 / dm[i];
    }
    return dm;
}

float pseudoEnergy(const RingBuf::Handle *in, int window, int offset) {
    float e=0;
    for (int i=0; i<window; i++) e += fabs(in->Read(offset+i) * 32768.0);
    return e;
}

int zeroCrossings(const RingBuf::Handle *in, int window, int offset) {
    int zcr=0;
    float last = in->Read(offset);
    for (int i=1; i<window; i++) {
        float cur = in->Read(offset+i);
        if ((cur<0 && last>=0) || (cur>=0 && last<0)) zcr++;
        last = cur;
    }
    return zcr;
}

RingBuf *decimate(const RingBuf::Handle *inh, int factor) {
    const int SPS=inh->Rate(), SPB=inh->Len();
    RingBuf *outbuf = new RingBuf(SPS/factor, SPB/factor);
    RingBuf::Handle out(outbuf);

    double sum=0; int count=0;
    for (int i=0; i<SPB; i++)
    {
        count++;
        sum += inh->Read(-SPB + i);

        if (count >= factor) {
            out.Write(sum / count);
            sum = 0; count = 0; 
        }
    }
    return outbuf;
}

int cross_correlateTDOA(const RingBuf::Handle *a, const RingBuf::Handle *b, int window, int offset, int samps) {

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

void AudioResampler::clear() { out=0; input_processed=output_available=output_rate=output_chans=0; }

#ifdef LFL_FFMPEG
void AudioResampler::close() {
    if (swr) { swr_free(&swr); swr=0; }
    clear();
}

int AudioResampler::open(RingBuf *rb, int in_channels,  int in_sample_rate,  int in_sample_type,
                                     int out_channels, int out_sample_rate, int out_sample_type) {
    close();
    out = rb;
    input_chans = in_channels;
    output_chans = out_channels;
    output_rate = out_sample_rate;
    int input_layout  =  in_channels > 1 ? AV_CH_LAYOUT_STEREO : AV_CH_LAYOUT_MONO;
    int output_layout = out_channels > 1 ? AV_CH_LAYOUT_STEREO : AV_CH_LAYOUT_MONO;
    swr = swr_alloc_set_opts(swr, output_layout, (AVSampleFormat)Sample::ToFFMpegId(out_sample_type), output_rate,
                                  input_layout,  (AVSampleFormat)Sample::ToFFMpegId( in_sample_type), in_sample_rate, 0, 0);
    if (swr_init(swr) < 0) ERROR("swr_init");
    return 0;
}

int AudioResampler::update(int samples, const short *in) {
    if (!out || !in) return -1;
    const short *input[SWR_CH_MAX] = { in, 0 }; 
    Allocator *tlsalloc = ThreadLocalStorage::GetAllocator();
    short *rsout = (short*)tlsalloc->malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE + FF_INPUT_BUFFER_PADDING_SIZE);
    return update(samples, input, rsout, -1, AVCODEC_MAX_AUDIO_FRAME_SIZE/output_chans/2);
}

int AudioResampler::update(int samples, RingBuf::Handle *L, RingBuf::Handle *R) {
    int channels = (L!=0) + (R!=0);
    if (!out || !channels) return -1;
    Allocator *tlsalloc = ThreadLocalStorage::GetAllocator();
    short *rsin = (short*)tlsalloc->malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE + FF_INPUT_BUFFER_PADDING_SIZE);
    short *rsout = (short*)tlsalloc->malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE + FF_INPUT_BUFFER_PADDING_SIZE);
    memset(rsin+samples*channels, 0, FF_INPUT_BUFFER_PADDING_SIZE);

    RingBuf::Handle *chan[2] = { L, R }; 
    for (int i=0; i<samples; i++)
        for (int j=0; j<channels; j++)
            rsin[i*channels + j] = chan[j]->Read(i) * 32768.0;

    const short *input[SWR_CH_MAX] = { rsin, 0 }; 
    return update(samples, input, rsout, chan[0]->ReadTimestamp(0), AVCODEC_MAX_AUDIO_FRAME_SIZE/output_chans/2);
}

int AudioResampler::update(int samples, const short **in, short *rsout, Time timestamp, int max_samples_out) {
    CHECK(swr);
    uint8_t *aout[SWR_CH_MAX] = { (uint8_t*)rsout, 0 };
    int resampled = swr_convert(swr, aout, max_samples_out, (const uint8_t**)in, samples);
    if (resampled < 0) { ERROR("av_resample return ", resampled); return -1; }
    if (!resampled) return 0;

    double step = Seconds(1)*1000/output_rate;
    int output = resampled * output_chans;
    Time stamp = monotonouslyIncreasingTimestamp(out->ReadTimestamp(-1), timestamp, &step, output);
    for (int i=0; i<output; i++) {
        if (timestamp != -1) out->stamp[out->ring.back] = stamp + i/output_chans*step;
        *(float*)out->Write() = rsout[i] / 32768.0;
    }

    input_processed += samples;
    output_available += output;
    return 0;
}
#else
void AudioResampler::close() {}
int AudioResampler::open(RingBuf *rb, int in_channels,  int in_sample_rate,  int in_sample_type,
                                      int out_channels, int out_sample_rate, int out_sample_type)            { FATAL("not implemented"); }
int AudioResampler::update(int samples, const short **in, short *rsout, Time timestamp, int max_samples_out) { FATAL("not implemented"); }
int AudioResampler::update(int samples, const short *in)                                                     { FATAL("not implemented"); }
int AudioResampler::update(int samples, RingBuf::Handle *L, RingBuf::Handle *R)                              { FATAL("not implemented"); }
#endif /* LFL_FFMPEG */

long long AudioResampler::monotonouslyIncreasingTimestamp(Time laststamp, Time stamp, double *step, int steps) {
    if (laststamp > stamp) {
        Time end = (Time)(stamp + (steps-1) * (*step));
        *step = (double)(end - laststamp) / steps;
        if (*step < 1) *step = 1;
        stamp = (Time)(laststamp + *step);
    }
    return stamp;
}

/* stateful filter */

void Filter::open(int FilterLenB, const double *FilterB, int FilterLenA, const double *FilterA) {
    next = samples = 0;

    filterB    = FilterB;    filterA    = FilterA;
    filterLenB = FilterLenB; filterLenA = FilterLenA;
    
    size = max(filterLenA, filterLenB);
    memset(state, 0, sizeof(state));
}

double Filter::filter(double sample) {
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

int Filter::filter(const RingBuf::Handle *in, RingBuf::Handle *out, int start, int length) {
    int count = 0;
    if (!length) length = in->Len();
    for (int i=start; i<length; i++) {
        double yi = filter(in->Read(i));
        if (out) out->Write(yi);
        count++;
    }
    return count;
}

/* stateless filters */

double filter(const RingBuf::Handle *in, int offset1,
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

double filter(const RingBuf::Handle *in, int offset, int filterlen, const double *filter, bool nohistory) { return LFL::filter(in, offset, 0, 0, filterlen, filter, 0, 0, 0, nohistory); }

int filter(const RingBuf::Handle *in, RingBuf::Handle *out, int filterlenB, const double *filterB, int filterlenA, const double *filterA, int start, double *ic, double iclen, bool nohistory) {
    for (int i=start; i<in->Len(); i++) {
        double yi = LFL::filter(in, i, out, i, filterlenB, filterB, filterlenA, filterA, (iclen-- > 0) ? *ic++ : 0, nohistory);
        out->Write(yi);
    }
    return 0;
}

int filter(const RingBuf::Handle *in, RingBuf::Handle *out, int filterlen, const double *filter, bool nohistory) { return LFL::filter(in, out, filterlen, filter, 0, 0, 0, 0, 0, nohistory); }

/* streaming overlap-add fft/ifft */

int fft(const RingBuf::Handle *in, int i, int window, int hop, int fftlen, float *out, bool doPreEmphasis, bool doHamming, int scale) {
    vector<double> pef = doPreEmphasis ? PreEmphasisFilter() : vector<double>();
    int padding=fftlen-window, split=window/2, bits=which_log2(fftlen), ind;
    if (bits < 0 || window % 2 != 0) return -1;

    for (int j=0; j<fftlen; j++) {			
        if      (j < split)           { ind = i*hop + split + j; }
        else if (j < split + padding) { out[j] = 0; continue; }
        else                          { ind = i*hop + j - split - padding; }

        double v = pef.size() ? filter(in, ind, pef.size(), &pef[0]) : in->Read(ind);

        if (doHamming) v *= hamming(window, ind - i*hop);
        out[j] = v * scale;
    }

#ifdef LFL_FFMPEG
    RDFTContext *fftctx = av_rdft_init(bits, DFT_R2C);
    av_rdft_calc(fftctx, out);
    av_rdft_end(fftctx);
#endif
    return 0;
}

int ifft(const Complex *in, int i, int window, int hop, int fftlen, RingBuf::Handle *out) {
    float *fftbuf=(float*)alloca(sizeof(float)*fftlen*2);
    int split=fftlen/2, bits=which_log2(fftlen); Complex zero={0,0};
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
    av_rdft_calc(fftctx, (float*)fftbuf);
    av_rdft_end(fftctx);
#endif

    for (int j=0; j<split; j++) *out->Index(i*hop +         j) += fftbuf[split+j];
    for (int j=0; j<split; j++) *out->Index(i*hop + split + j) += fftbuf[j];

    double scalef = (1.0/fftlen) * ((double)window/fftlen);
    for (int j=0; j<hop; j++) *out->Index(i*hop + j) *= scalef;
    return 0;
}

/* fft filter */

void fft_filter_compile(int n, double *filter) {
    RingBuf filtbuf(FLAGS_sample_rate, n);
    RingBuf::Handle filtcoef(&filtbuf);
    for (int i=0; i<n; i++) filtcoef.Write(filter[i]);

    float *fftbuf = (float *)alloca(sizeof(float)*n);
    fft(&filtcoef, 0, n, 0, n, fftbuf);
    for (int i=0; i<n; i++) filter[i] = fftbuf[i];
}

int fft_filter(const RingBuf::Handle *in, RingBuf::Handle *out, int window, int hop, const double *filter) {
    int frames = (in->Len() - (window - hop)) / hop;
    float *fftbuf = (float*)alloca(sizeof(float)*window);
    Complex *ifftb = (Complex*)alloca(sizeof(Complex)*window/2);

    for (int j=0; j<in->Len(); j++) *out->Index(j) = 0;

    for (int i=0; i<frames; i++) {
        if (fft(in, i, window, hop, window, fftbuf)) return -1;

        for (int j=0; j<window/2; j++) {
            Complex x={fft_r(fftbuf, window, j), fft_i(fftbuf, window, j)};
            Complex h={fft_r(filter, window, j), fft_i(filter, window, j)};
            ifftb[j] = Complex::Mult(x, h); /* cyclic convolution */
        }

        if (ifft(ifftb, i, window, hop, window, out)) return -1;
    }
    return 0;
}

float fundamentalFrequency(const RingBuf::Handle *in, int window, int offset, int method) {
    int bits = which_log2(window);
    if (bits<0) return -1;

    if (method == F0EstmMethod::fftbucket) { /* max(fft.bucket) */
        float *samp = (float *)alloca(window*sizeof(float));
        fft(in, 1, window, offset, window, samp);

        int maxind=0; double max=0;
        for (int i=1; i<window/2; i++) {
            float abs = sqrt(fft_abs2(samp, window, i));
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
        float *buf = (float *)alloca(window*2*sizeof(float));
        fft(&zpin, 0, window*2, 0, window*2, buf, false);

        /* to spectrum */
        fft_r(buf, window*2, 0) = pow(fft_r(buf, window*2, 0), 2);
        fft_i(buf, window*2, 0) = pow(fft_i(buf, window*2, 0), 2);
        for (int i=1; i<window; i++) {
            fft_r(buf, window*2, i) = fft_abs2(buf, window*2, i);
            fft_i(buf, window*2, i) = 0;
        }

        /* ifft */
#ifdef LFL_FFMPEG
        RDFTContext *fftctx = av_rdft_init(bits+1, IDFT_C2R);
        av_rdft_calc(fftctx, buf);
        av_rdft_end(fftctx);
#endif

        /* auto correlation */
        int minlag = min(in->Rate() / 500, window*2); /* 500 hz */
        int maxlag = min(in->Rate() / 50 + 1, window*2); /* 50 hz */
        float *xcorr = (float*)alloca(maxlag*sizeof(float));

        /* divide by window for ifft and zxc for auto-correlation coefficients */
        float zxc = buf[0] / window;
        for (int i=0; i<maxlag; i++)
            xcorr[i] = buf[i] / window / zxc;

        /* max(xcorr[minlag,maxlag]) */
        int maxind=-1; double max=-INFINITY;
        for (int i=minlag; i<maxlag; i++)
            if (xcorr[i] > max) { max=xcorr[i]; maxind=i; }

        /* xcorr index to hz */
        return (float)in->Rate() / (maxind+1);
    }
    else if (method == F0EstmMethod::cepstral) { /* cepstral analysis */
        float *buf = (float *)alloca(window*sizeof(float));
        fft(in, 1, window, offset, window, buf);

        /* to log spectrum */
        RingBuf spe(in->Rate(), window);
        RingBuf::Handle spec(&spe);
        *spec.Index(0) = log(fabs(fft_r(buf, window, 0)));
        for (int i=0; i<window/2; i++) *spec.Index(i) = log(sqrt(fft_abs2(buf, window, i)));
        *spec.Index(window/2) = log(fabs(fft_i(buf, window, 0)));
        for (int i=1; i<window/2; i++) *spec.Index(window/2+i) = *spec.Index(window/2-i);

        /* fft */
        fft(&spec, 0, window, 0, window, buf, false); 

        /* cepstrum */
        int minlag = min(in->Rate() / 1000, window/2); /* 1000 hz */
        int maxlag = min(in->Rate() / 50, window/2); /* 50 hz */

        float *cepstrum = (float*)alloca(sizeof(float)*window/2);
        for (int i=0; i<window/2; i++) cepstrum[i] = sqrt(fft_abs2(buf, window, i));

        /* max(cepstrum[minlag,maxlag]) */
        int maxind=-1; double max=-INFINITY;
        for (int i=minlag; i<maxlag; i++) {
            if (cepstrum[i] > max) { max=cepstrum[i]; maxind=i; }
        }

        /* cepstrum index to hz */
        return (float)in->Rate() / (maxind-1);
    }
    else if (method == F0EstmMethod::harmonicity) { /* harmonicity */
        float *buf = (float *)alloca(window*sizeof(float));
        fft(in, 1, window, offset, window, buf);
    }
    return -1;
}

Matrix *fft2bark(int outrows, double minfreq, double maxfreq, int fftlen, int samplerate) {
    Matrix *m = new Matrix(outrows, fftlen/2);
    double minbark = hz2bark(minfreq), maxbark = hz2bark(maxfreq);
    double nyqbark = maxbark - minbark, stepbark = nyqbark/(outrows-1);

    double *binbark = (double *)alloca(fftlen/2 * sizeof(double));
    for (int i=0; i<fftlen/2; i++)
        binbark[i] = hz2bark((double)i/fftlen*samplerate);

    for (int i=0; i<outrows; i++) {
        double *row = m->row(i), midbark = minbark + i*stepbark;

        for (int j=0; j<fftlen/2; j++) {
            double lofreq = (binbark[j] - midbark) - 0.5;
            double hifreq = (binbark[j] - midbark) + 0.5;
            row[j] = pow(10, min(0.0, min(hifreq, -2.5*lofreq)));
        }
    }
    return m;
}

Matrix *fft2mel(int outrows, double minfreq, double maxfreq, int fftlen, int samplerate) {
    Matrix *m = new Matrix(outrows, fftlen/2);
    double minmel = hz2mel(minfreq), maxmel = hz2mel(maxfreq);

    double *mel = (double *)alloca((outrows+2) * sizeof(double));
    for (int i=0; i<outrows+2; i++)
        mel[i] = mel2hz(minmel + ((double)i/(double)(outrows+1)) * (maxmel-minmel)); 

    for (int i=0; i<outrows; i++) {
        double *row = m->row(i);

        for (int j=0; j<fftlen/2; j++) {
            double binfreq = (double)j/fftlen*samplerate;
            double loslope = ( binfreq - mel[i+0]) / (mel[i+1] - mel[i+0]);
            double hislope = (-binfreq + mel[i+2]) / (mel[i+2] - mel[i+1]);

            row[j] = max(0.0, min(loslope, hislope));
        }
    }
    return m;
}

Matrix *mel2fft(int outrows, double minfreq, double maxfreq, int fftlen, int samplerate) {
    Matrix *m = fft2mel(outrows, minfreq, maxfreq, fftlen, samplerate);
    Matrix *w = Matrix::Mult(m, m, mTrnpA);

    float diagmean=0; int diagcount=0;
    for (int i=0; i<w->M && i<w->N; i++) { diagmean += w->row(i)[i]; diagcount++; }
    diagmean /= (diagcount+1);

    m = Matrix::Transpose(m);
    for (int i=0; i<m->M; i++) {
        double sum=0;
        for (int j=0; j<m->N; j++) sum += w->row(i)[j];

        if (sum < diagmean) sum = diagmean;
        for (int j=0; j<m->N; j++) m->row(i)[j] /= sum;
    }
    return m;
};

Matrix *spectogram(const RingBuf::Handle *in, Matrix *out, int window, int hop, int fftlen, bool preemph, int pd, int scale) {
    bool complex = (pd==PowerDomain::complex);
    int frames = (in->Len() - (window - hop)) / hop;
    float *fftbuf = (float*)alloca(sizeof(float)*fftlen);

    Matrix *m;
    if (out) { m=out; if (m->M != frames || m->N != fftlen/2 || (m->flag&Matrix::Flag::Complex && !complex)) return 0; }
    else m = new Matrix(frames, fftlen/2, 0.0, complex);

    for (int i=0; i<frames; i++) {
        double *row = m->row(i);
        if (fft(in, i, window, hop, fftlen, fftbuf, preemph, true, scale)) { if (!out) delete m; return 0; }

        for (int j=0; j<fftlen/2; j++) {
            if (pd == PowerDomain::complex) {
                row[j*2 + 0] = fft_r(fftbuf, fftlen, j);
                row[j*2 + 1] = fft_i(fftbuf, fftlen, j);
                continue;
            }

            if      (pd == PowerDomain::abs)  row[j] = sqrt(fft_abs2(fftbuf, fftlen, j));
            else if (pd == PowerDomain::abs2) row[j] = fft_abs2(fftbuf, fftlen, j);      
            else if (pd == PowerDomain::dB)   row[j] = fft_abs2_to_dB(fft_abs2(fftbuf, fftlen, j));
        }
    }

    return m;
}

RingBuf *ispectogram(const Matrix *in, int window, int hop, int fftlen, int samplerate) {
    RingBuf *outbuf = new RingBuf(samplerate, fftlen + hop * (in->M-1));
    RingBuf::Handle out(outbuf);
    
    for (int i=0; i<in->M; i++) {
        if (ifft(in->crow(i), i, window, hop, fftlen, &out)) { delete outbuf; return 0; }
    }
    return outbuf;
}

Matrix *f0stream(const RingBuf::Handle *in, Matrix *out, int window, int hop, int method) {
    int frames = (in->Len() - (window - hop)) / hop;

    Matrix *m;
    if (out) { m=out; if (m->M != frames || m->N != 1) return 0; }
    else m = new Matrix(frames, 1);

    for (int i=0; i<frames; i++) {
        double *row = m->row(i); 
        row[0] = fundamentalFrequency(in, window, i*hop, method);
    }
    return m;
}

Matrix *PLP(const RingBuf::Handle *in, Matrix *out, vector<Filter> *rastaFilters, Allocator *alloc) {
    /* http://seed.ucsd.edu/mediawiki/images/5/5c/PLP.pdf */
    if (!alloc) alloc = Singleton<MallocAlloc>::Get();

    /* plp = melfcc(x, sr, 'lifterexp', -22, 'nbands', 21, 'maxfreq', sr/2, 'fbtype', 'bark', 'modelorder', 12, 'usecmp',1, 'wintime', 512/sr, 'hoptime', 256/sr, 'preemph', 0, 'dcttype', 1) */
    static const Matrix *barktrans = fft2bark(feat_barkbands, FLAGS_feat_minfreq, FLAGS_feat_maxfreq, FLAGS_feat_window, FLAGS_sample_rate)->Transpose(mDelA);

    /* fft */
    int frames = (in->Len() - (FLAGS_feat_window - FLAGS_feat_hop)) / FLAGS_feat_hop;
    Matrix fftbuf(frames, FLAGS_feat_window/2, 0.0, 0, alloc);
    if (!spectogram(in, &fftbuf, FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window, false, PowerDomain::abs2, 32768)) return 0;
    if (FLAGS_feat_dither) MatrixIter(&fftbuf) fftbuf.row(i)[j] += FLAGS_feat_window;

    /* to bark scale critcal bands */
    Matrix barkbuf(fftbuf.M, barktrans->N, 0.0, 0, alloc);
    Matrix::Mult(&fftbuf, barktrans, &barkbuf);

    if (1) { /* rasta */
        MatrixIter(&barkbuf) barkbuf.row(i)[j] = log(barkbuf.row(i)[j] ? barkbuf.row(i)[j] : 1e7);
        Matrix rastabuf(barkbuf.M, barkbuf.N, 0.0, 0, alloc);

        if (rastaFilters) {
            if (!rastaFilters->size()) MatrixColIter(&rastabuf) rastaFilters->push_back(Filter());
            if (rastaFilters->size() != rastabuf.N) FATAL("mismatch ", rastaFilters, " != ", rastabuf.N);
        } 

        MatrixColIter(&rastabuf) {
            ColMatPtrRingBuf bandTrajectory(&barkbuf, j), rastaOut(&rastabuf, j);
            RingBuf::HandleT<double> bt(&bandTrajectory), ro(&rastaOut);
            Filter RFbuf, *RF = rastaFilters ? &(*rastaFilters)[j] : &RFbuf;
            int processed = 0;

            if (!RF->samples) RF->open(sizeof(feat_rastaB)/sizeof(double), feat_rastaB, 0, feat_rastaA);
            if (RF->samples < 4) {
                processed = RF->filter(&bt, 0, 0, min(4-RF->samples, bt.Len()));
                for (int i=0; i<processed; i++) ro.Write(0.0);
            }
            if (RF->samples == 4) RF->filterLenA = sizeof(feat_rastaA)/sizeof(double);
            RF->filter(&bt, &ro, processed);
        }
        MatrixIter(&barkbuf) barkbuf.row(i)[j] = exp(rastabuf.row(i)[j]);
    }

    static const Matrix *equalLoudnessCurve = EqualLoudnessCurve(barkbuf.N, FLAGS_sample_rate/2);
    MatrixIter(&barkbuf) {
        /* preemphasize = weight critical bands by equal loudness curve */
        barkbuf.row(i)[j] *= equalLoudnessCurve->row(j)[0];

        /* cube root compress */
        barkbuf.row(i)[j] = pow(barkbuf.row(i)[j], 0.33);
    }
    MatrixRowIter(&barkbuf) {
        barkbuf.row(i)[0]           = barkbuf.row(i)[1];
        barkbuf.row(i)[barkbuf.N-1] = barkbuf.row(i)[barkbuf.N-2];
    }
    
    /* auto-correlation */
    static Matrix *IDFTtrans = IDFT((feat_barkbands-1)*2, feat_barkbands)->Transpose(mDelA);
    Matrix xcorr(barkbuf.M, IDFTtrans->N, 0.0, 0, alloc);
    Matrix::Mult(&barkbuf, IDFTtrans, &xcorr);
    MatrixIter(&xcorr) xcorr.row(i)[j] /= IDFTtrans->N;

    /* levinson durbin recursion to LPC */
    int order = feat_lpccoefs, ceps = order+1;
    Matrix LPC(xcorr.M, ceps, 0.0, 0, alloc);
    double *reflect = (double*)alloca(order*sizeof(double)), *lpc = (double*)alloca(order*sizeof(double));
    MatrixRowIter(&xcorr) {
        double err = levinsondurbin(order, xcorr.row(i), reflect, lpc);
        MatrixColIter(&LPC) LPC.row(i)[j] = (!j ? 1.0 : lpc[j-1]) / err;
    }

    /* prepare output */
    if (out) { if (out->M != LPC.M || out->N != LPC.N) return 0; }
    else out = new Matrix(LPC.M, LPC.N);

    /* first cepstral coefficient is log(error) from levinson durbin recursion */
    MatrixRowIter(out) out->row(i)[0] = -log(LPC.row(i)[0]);

    /* normalize LPC */
    MatrixRowIter(&LPC) {
        double denom = LPC.row(i)[0];
        MatrixColIter(&LPC) LPC.row(i)[j] /= denom;
    }

    /* LPC to cepstral coefficients */
    MatrixRowIter(out) for (int j=1; j<out->N; j++) {
        double sum = 0;
        for (int j2=1; j2<j; j2++) sum += (j-j2) * LPC.row(i)[j2] * out->row(i)[j-j2];
        out->row(i)[j] = -LPC.row(i)[j] - sum / j;
    }

    /* lifter */
    static double *lifter = lifterMatrixROSA(ceps, 0.6); // lifterMatrixHTK(ceps, 22);
    out->MultdiagR(lifter, ceps);

    return out;
}

RingBuf *InvPLP(const Matrix *in, int samplerate, Allocator *alloc) {
    /* [dr,aspec,spec] = invmelfcc() */
    if (!alloc) alloc = Singleton<MallocAlloc>::Get();
    Matrix plpcc(in->M, in->N, 0.0, 0, alloc);
    plpcc.AssignL(in);

    int ceps = feat_lpccoefs+1;
    static double *unlifter = lifterMatrixHTK(ceps, 22, true); // lifterMatrixROSA(ceps, 0.6, true);
    plpcc.MultdiagR(unlifter, ceps);
    Matrix::Print(&plpcc, "sec plp");

    return 0;
}

Matrix *MFCC(const RingBuf::Handle *in, Matrix *out, Allocator *alloc) {
    if (!alloc) alloc = Singleton<MallocAlloc>::Get();

    /* [mm,aspc,pspc] = melfcc(y/32768, sr, 'maxfreq', sr/2, 'numcep', 20, 'nbands', 40, 'fbtype', 'htkmel', 'dcttype', 3, 'usecmp', 0, 'wintime', 512/44100, 'hoptime', 256/44100, 'dither', 0, 'lifterexp', 0) */
    static const Matrix *meltrans = fft2mel(FLAGS_feat_melbands, FLAGS_feat_minfreq, FLAGS_feat_maxfreq, FLAGS_feat_window, FLAGS_sample_rate)->Transpose(mDelA);
    static const Matrix *ceptrans = DCT2(FLAGS_feat_cepcoefs, FLAGS_feat_melbands)->Transpose(mDelA);
    
    /* fft */
    int frames = (in->Len() - (FLAGS_feat_window - FLAGS_feat_hop)) / FLAGS_feat_hop;
    Matrix fftbuf(frames, FLAGS_feat_window/2, 0.0, 0, alloc);
    if (!spectogram(in, &fftbuf, FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window, FLAGS_feat_preemphasis, PowerDomain::abs2)) return 0;
    
    /* to mel scale */
    Matrix melbuf(fftbuf.M, meltrans->N, 0.0, 0, alloc);
    Matrix::Mult(&fftbuf, meltrans, &melbuf);

    /* log */
    MatrixIter(&melbuf) melbuf.row(i)[j] = log(melbuf.row(i)[j]);

    /* prepare output */
    if (out) { if (out->M != melbuf.M || out->N != ceptrans->N) return 0; }
    else out = new Matrix(melbuf.M, ceptrans->N);

    /* to cepstral coefs */
    return Matrix::Mult(&melbuf, ceptrans, out);
}

RingBuf *InvMFCC(const Matrix *in, int samplerate, const Matrix *f0) {
    /* [dr,aspec,spec] = invmelfcc() */
    Matrix *transform = DCT2(in->N, FLAGS_feat_melbands);
    for (int j=0; j<transform->N; j++) transform->row(0)[j] /= 2;

    Matrix *m = Matrix::Mult(in, transform, mDelB);
    MatrixIter(m) m->row(i)[j] = exp(m->row(i)[j]);

    transform = mel2fft(FLAGS_feat_melbands, FLAGS_feat_minfreq, FLAGS_feat_maxfreq, FLAGS_feat_window, samplerate);
    m = Matrix::Mult(m, transform, mTrnpB|mDelB);

    if (0 && f0) {
        if (f0->M != in->M || m->M != in->M) return 0;
        RingBuf *outbuf = new RingBuf(samplerate, FLAGS_feat_window + FLAGS_feat_hop * (m->M-1));
        RingBuf::Handle out(outbuf);

        for (int i=0; i<m->M; i++) {
            float F0 = f0->row(i)[0];
            float wavs[] = { F0, F0*2, F0*4, F0*8, F0*16, F0*32 };

            for (int j=0; j<sizeofarray(wavs); j++) {
                double f = (2 * M_PI * wavs[j]) / out.Rate();
                int ind = (int)(wavs[j] * m->N / out.Rate());
                double a = m->row(i)[ind];

                for (int k=0; k<FLAGS_feat_window; k++) 
                    *out.Index(i*FLAGS_feat_hop + k) += a*sin(f*i);
            }
        }

        return outbuf;
    }
    else {
        RingBuf *outbuf = new RingBuf(samplerate, FLAGS_feat_window + FLAGS_feat_hop * (m->M-1));
        RingBuf::Handle out(outbuf);
        { 
            double f = (2 * M_PI * 440) / out.Rate();
            for (int i=0; i<out.Len(); i++) out.Write(sin(f*i)/16 + rand(-1,1)/2);
        }
        Matrix *spec = spectogram(&out, 0, FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window, 0, PowerDomain::complex);
        delete outbuf;

        if (m->M != spec->M || m->N != spec->N) { delete m; delete spec; return 0; }	
        MatrixIter(spec) {
            Complex v = {sqrt(m->row(i)[j]), 0}; 
            spec->crow(i)[j].Mult(v);
        }
        delete m;

        outbuf = ispectogram(spec, FLAGS_feat_window, FLAGS_feat_hop, FLAGS_feat_window/2, samplerate);
        delete spec;
        return outbuf;
    }
}

}; // namespace LFL
