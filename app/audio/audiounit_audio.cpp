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

#include <AudioUnit/AudioUnit.h>
#include <AudioToolbox/AudioServices.h>

namespace LFL {
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
    if ((err = AudioSessionInitialize(NULL, NULL, NULL, NULL))) { memcpy(errBuf, &err, sizeof(int)); INFO("AudioSessionInitialize - ", errBuf); }

    UInt32 sessionCategory = kAudioSessionCategory_PlayAndRecord;    
    if ((err = AudioSessionSetProperty (kAudioSessionProperty_AudioCategory, sizeof (sessionCategory), &sessionCategory))) {  memcpy(errBuf, &err, sizeof(int)); INFO("AudioSessionSetProperty - kAudioSessionProperty_AudioCategory - ", errBuf); }

    UInt32 newRoute = kAudioSessionOverrideAudioRoute_Speaker;
    if ((err = AudioSessionSetProperty(kAudioSessionProperty_OverrideAudioRoute, sizeof(newRoute), &newRoute))) { memcpy(errBuf, &err, sizeof(int)); INFO("AudioSessionSetProperty - kAudioSessionProperty_OverrideAudioRoute - Set speaker on ", errBuf); }

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

    microseconds step(1000000/FLAGS_sample_rate), stamp = AudioResamplerInterface::MonotonouslyIncreasingTimestamp(audio->IL->ReadTimestamp(-1), ToMicroseconds(Now()), &step, inNumberFrames);
    RingSampler::WriteAheadHandle IL(audio->IL.get()), IR(audio->IR.get());
    for (int i=0; i<inNumberFrames; i++) {
      microseconds timestamp = stamp + i*step;
      IL.Write(in[i*FLAGS_chans_in+0]/32768.0, RingSampler::Stamp, timestamp);

      if (FLAGS_chans_in == 1) continue;
      IR.Write(in[i*FLAGS_chans_in+1]/32768.0, RingSampler::Stamp, timestamp);
    }
    { 
      ScopedMutex ML(audio->inlock);
      IL.Commit();
      IR.Commit();
      app->audio->samples_read += inNumberFrames;
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
    return ((AudioUnitAudioModule*)app->audio.get())->IOIn(inRefCon, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, ioData);
  }

  static OSStatus IOOutCB(void *inRefCon, AudioUnitRenderActionFlags *ioActionFlags, const AudioTimeStamp *inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames, AudioBufferList *ioData) {
    return ((AudioUnitAudioModule*)app->audio.get())->IOOut(inRefCon, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, ioData);
  }
};

int Application::GetMaxVolume() { return 0; }
int Application::GetVolume() { return 0; }
void Application::SetVolume(int v) {}

unique_ptr<Module> CreateAudioModule(Audio *a) { return make_unique<AudioUnitAudioModule>(a); }

}; // namespace LFL
