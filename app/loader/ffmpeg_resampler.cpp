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

#include "lfapp/lfapp.h"

extern "C" {
#ifdef LFL_FFMPEG
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
#define AVCODEC_MAX_AUDIO_FRAME_SIZE 192000
#endif
};

#include "lfapp/bindings/ffmpeg.h"

namespace LFL {
struct FFMpegAudioResampler {
  SwrContext *swr=0;
  ~FFMpegAudioResampler() { if (swr) swr_free(&swr); }
  bool Opened() { return swr; }

  int Open(RingBuf *rb, int in_channels,  int in_sample_rate,  int in_sample_type,
           int out_channels, int out_sample_rate, int out_sample_type) {
    Close();
    out = rb;
    input_chans = in_channels;
    output_chans = out_channels;
    output_rate = out_sample_rate;
    int input_layout  =  in_channels > 1 ? AV_CH_LAYOUT_STEREO : AV_CH_LAYOUT_MONO;
    int output_layout = out_channels > 1 ? AV_CH_LAYOUT_STEREO : AV_CH_LAYOUT_MONO;
    swr = swr_alloc_set_opts(swr, output_layout, AVSampleFormat(SampleToFFMpegId(out_sample_type)), output_rate,
                             input_layout,  AVSampleFormat(SampleToFFMpegId(in_sample_type)), in_sample_rate, 0, 0);
    if (swr_init(swr) < 0) ERROR("swr_init");
    return 0;
  }

  int Update(int samples, const short *in) {
    if (!out || !in) return -1;
    const short *input[SWR_CH_MAX] = { in, 0 }; 
    Allocator *tlsalloc = ThreadLocalStorage::GetAllocator();
    short *rsout = static_cast<short*>(tlsalloc->Malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE + FF_INPUT_BUFFER_PADDING_SIZE));
    return Update(samples, input, rsout, microseconds(-1), AVCODEC_MAX_AUDIO_FRAME_SIZE/output_chans/2);
  }

  int Update(int samples, RingBuf::Handle *L, RingBuf::Handle *R) {
    int channels = (L!=0) + (R!=0);
    if (!out || !channels) return -1;
    Allocator *tlsalloc = ThreadLocalStorage::GetAllocator();
    short *rsin = static_cast<short*>(tlsalloc->Malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE + FF_INPUT_BUFFER_PADDING_SIZE));
    short *rsout = static_cast<short*>(tlsalloc->Malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE + FF_INPUT_BUFFER_PADDING_SIZE));
    memset(rsin+samples*channels, 0, FF_INPUT_BUFFER_PADDING_SIZE);

    RingBuf::Handle *chan[2] = { L, R }; 
    for (int i=0; i<samples; i++)
      for (int j=0; j<channels; j++)
        rsin[i*channels + j] = chan[j]->Read(i) * 32768.0;

    const short *input[SWR_CH_MAX] = { rsin, 0 }; 
    return Update(samples, input, rsout, chan[0]->ReadTimestamp(0), AVCODEC_MAX_AUDIO_FRAME_SIZE/output_chans/2);
  }

  int Update(int samples, const short *const *inbuf, short *rsout, microseconds timestamp, int max_samples_out) {
    CHECK(swr);
    auto in = const_cast<const u_int8_t**>(reinterpret_cast<const uint8_t*const*>(inbuf));
    uint8_t *aout[SWR_CH_MAX] = { reinterpret_cast<uint8_t*>(rsout), 0 };
    int resampled = swr_convert(swr, aout, max_samples_out, in, samples);
    if (resampled < 0) return ERRORv(-1, "av_resample return ", resampled);
    if (!resampled) return 0;

    microseconds step(1000000/output_rate);
    int output = resampled * output_chans;
    microseconds stamp = MonotonouslyIncreasingTimestamp(out->ReadTimestamp(-1), timestamp, &step, output);
    for (int i=0; i<output; i++) {
      if (timestamp != microseconds(-1)) out->stamp[out->ring.back] = stamp + i/output_chans*step;
      *static_cast<float*>(out->Write()) = rsout[i] / 32768.0;
    }

    input_processed += samples;
    output_available += output;
    return 0;
  }
};

struct FFMPEGVideoResampler : public VideoResamplerInterface {
  SwsContext *conv=0;
  bool simple_resampler_passthru=0;
  ~FFMPEGVideoResampler() { if (conv.value) sws_freeContext(conv); }

  bool Opened() const { return conv || simple_resampler_passthru; }

  void Open(int sw, int sh, int sf, int dw, int dh, int df) {
    s_fmt = sf; s_width = sw; s_height = sh;
    d_fmt = df; d_width = dw; d_height = dh;
    // INFO("resample ", BlankNull(Pixel::Name(s_fmt)), " -> ", BlankNull(Pixel::Name(d_fmt)), " : (", sw, ",", sh, ") -> (", dw, ",", dh, ")");

    if (SimpleVideoResampler::Supports(s_fmt) && SimpleVideoResampler::Supports(d_fmt) && sw == dw && sh == dh)
    { simple_resampler_passthru = 1; return; }

    conv = MakeTyped(sws_getContext(sw, sh, PixelFormat(Pixel::ToFFMpegId(sf)),
                                    dw, dh, PixelFormat(Pixel::ToFFMpegId(df)), SWS_BICUBIC, 0, 0, 0));
  }

  void Resample(const unsigned char *s, int sls, unsigned char *d, int dls, bool flip_x, bool flip_y) {
    if (simple_resampler_passthru) return SimpleVideoResampler::Resample(s, sls, d, dls, flip_x, flip_y);
    const uint8_t *source[4] = { MakeUnsigned(s), 0, 0, 0 };
    /**/  uint8_t *dest  [4] = { MakeUnsigned(d), 0, 0, 0 };
    int sourcels[4] = { sls, 0, 0, 0 }, destls[4] = { dls, 0, 0, 0 };
    if (flip_y) {
      source[0] += sls * (s_height - 1);
      sourcels[0] *= -1;
    }
    sws_scale(GetTyped<SwsContext*>(conv),
              flip_y ? source   : source,
              flip_y ? sourcels : sourcels, 0, s_height, dest, destls);
  }
};

AudioResamplerInterface *CreateAudioResampler() { return new FFMpegAudioResampler(); }
VideoResamplerInterface *CreateVideoResampler() { return new FFMpegVideoResampler(); }

}; // namespace LFL
