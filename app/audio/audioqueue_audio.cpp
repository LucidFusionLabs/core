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

#include <AudioToolbox/AudioToolbox.h>

namespace LFL {
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
    RingSampler::WriteAheadHandle IL(audio->IL.get()), IR(audio->IR.get());
    for (int i=0; i<count; i++) {
      Time timestamp = stamp + i*step;
      IL.Write(in[i*FLAGS_chans_in+0]/32768.0, RingSampler::Stamp, timestamp);

      if (FLAGS_chans_in == 1) continue;
      IR.Write(in[i*FLAGS_chans_in+1]/32768.0, RingSampler::Stamp, timestamp);
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

int Application::GetMaxVolume() { return 0; }
int Application::GetVolume() { return 0; }
void Application::SetVolume(int v) {}

Module *CreateAudioModule(Audio *a) { return new AudioQueueAudioModule(a); }

}; // namespace LFL
