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

extern "C" {
//#define INT64_C (long long)
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#define AVCODEC_MAX_AUDIO_FRAME_SIZE 192000
};

#include "core/app/bindings/ffmpeg.h"

namespace LFL {
const int SoundAsset::FromBufPad = FF_INPUT_BUFFER_PADDING_SIZE;

struct FFBIOFile {
  static void *Alloc(File *f) {
    int bufsize = 16384;
    return avio_alloc_context(static_cast<unsigned char*>(malloc(bufsize)), bufsize, 0, f, Read, Write, Seek);
  }
  static void Free(void *in) {
    AVIOContext *s = static_cast<AVIOContext*>(in);
    delete static_cast<File*>(s->opaque);
    free(s->buffer);
    av_free(s);
  }
  static int     Read(void *f, uint8_t *buf, int buf_size) { return static_cast<File*>(f)->Read (buf, buf_size); }
  static int    Write(void *f, uint8_t *buf, int buf_size) { return static_cast<File*>(f)->Write(buf, buf_size); }
  static int64_t Seek(void *f, int64_t offset, int whence) { return static_cast<File*>(f)->Seek (offset, whence); }
};

struct FFBIOC {
  void const *buf;
  int len, offset;
  FFBIOC(void const *b, int l) : buf(b), len(l), offset(0) {}

  static void *Alloc(void const *buf, int len) {
    static const int bufsize = 32768;
    return avio_alloc_context(static_cast<unsigned char*>(malloc(bufsize)), bufsize, 0, new FFBIOC(buf, len), Read, Write, Seek);
  }
  static void Free(void *in) {
    AVIOContext *s = static_cast<AVIOContext*>(in);
    delete static_cast<FFBIOC*>(s->opaque);
    free(s->buffer);
    av_free(s);
  }
  static int Read(void *opaque, uint8_t *buf, int buf_size) {
    FFBIOC *s = static_cast<FFBIOC*>(opaque);
    int len = min(buf_size, s->len - s->offset);
    if (len <= 0) return len;

    memcpy(buf, static_cast<const char*>(s->buf) + s->offset, len);
    s->offset += len;
    return len;
  }
  static int Write(void *opaque, uint8_t *buf, int buf_size) { return -1; }
  static int64_t Seek(void *opaque, int64_t offset, int whence) {
    FFBIOC *s = static_cast<FFBIOC*>(opaque);
    if      (whence == SEEK_SET) s->offset = offset;
    else if (whence == SEEK_CUR) s->offset += offset;
    else if (whence == SEEK_END) s->offset = s->len + offset;
    else return -1;
    return 0;
  }
};

struct FFMpegAssetLoader : public AssetLoaderInterface {
  FFMpegAssetLoader() { INFO("FFMpegAssetLoader"); }

  virtual void *LoadFile(const string &filename) { return LoadFile(filename, 0); }
  AVFormatContext *LoadFile(const string &filename, AVIOContext **pbOut) {
#if !defined(LFL_ANDROID) && !defined(LFL_IPHONE)
    AVFormatContext *fctx = 0;
    if (avformat_open_input(&fctx, filename.c_str(), 0, 0)) return ERRORv(nullptr, "av_open_input_file: ", filename);
    return fctx;
#else
    unique_ptr<LocalFile> lf = make_unique<LocalFile>(filename, "r");
    if (!lf->opened()) return ERRORv(nullptr, "FFLoadFile: open ", filename);
    void *pb = FFBIOFile::Alloc(lf.release());
    AVFormatContext *ret = Load((AVIOContext*)pb, filename, 0, 0, pbOut);
    if (!ret) FFBIOFile::Free(pb);
    return ret;
#endif
  }

  virtual void UnloadFile(void *h) {
    AVFormatContext *handle = static_cast<AVFormatContext*>(h);
    for (int i = handle->nb_streams - 1; handle->streams && i >= 0; --i) {
      AVStream* stream = handle->streams[i];
      if (!stream || !stream->codec || !stream->codec->codec) continue;
      stream->discard = AVDISCARD_ALL;
      avcodec_close(stream->codec);
    }
#if defined(LFL_ANDROID) || defined(LFL_IPHONE)
    FFBIOFile::Free(handle->pb);
#endif
    avformat_close_input(&handle);
  }

  virtual void *LoadBuf(const char *buf, int len, const char *filename) { return LoadBuf(buf, len, filename, 0); }
  AVFormatContext *LoadBuf(void const *buf, int len, const char *filename, AVIOContext **pbOut) {
    const char *suffix = strchr(filename, '.');
    void *pb = FFBIOC::Alloc(buf, len);
    AVFormatContext *ret = Load(static_cast<AVIOContext*>(pb), suffix ? suffix+1 : filename,
                                const_cast<char*>(static_cast<const char*>(buf)), min(4096, len), pbOut);
    if (!ret) FFBIOC::Free(pb);
    return ret;
  }

  virtual void UnloadBuf(void *h) {
    AVFormatContext *handle = static_cast<AVFormatContext*>(h);
    FFBIOC::Free(handle->pb);
    avformat_close_input(&handle);
  }

  virtual void *LoadAudioFile(const string &filename) { return LoadFile(filename); }
  virtual void UnloadAudioFile(void *h) { return UnloadFile(h); }
  virtual void *LoadAudioBuf(const char *buf, int len, const char *mimetype) { return LoadBuf(buf, len, mimetype); }
  virtual void UnloadAudioBuf(void *h) { return UnloadBuf(h); }

  virtual void *LoadVideoFile(const string &filename) { return LoadFile(filename); }
  virtual void UnloadVideoFile(void *h) { return UnloadFile(h); }
  virtual void *LoadVideoBuf(const char *buf, int len, const char *mimetype) { return LoadBuf(buf, len, mimetype); }
  virtual void UnloadVideoBuf(void *h) { return UnloadBuf(h); }

  virtual void *LoadMovieFile(const string &filename) { return LoadFile(filename); }
  virtual void UnloadMovieFile(void *h) { return UnloadFile(h); }
  virtual void *LoadMovieBuf(const char *buf, int len, const char *mimetype) { return LoadBuf(buf, len, mimetype); }
  virtual void UnloadMovieBuf(void *h) { return UnloadBuf(h); }

  void LoadVideo(void *handle, Texture *out, int load_flag=VideoAssetLoader::Flag::Default) {
    AVFormatContext *fctx = static_cast<AVFormatContext*>(handle);
    int video_index = -1, got=0;
    for (int i=0; i<fctx->nb_streams; i++) {
      AVStream *st = fctx->streams[i];
      AVCodecContext *avctx = st->codec;
      if (avctx->codec_type == AVMEDIA_TYPE_VIDEO) video_index = i;
    }
    if (video_index < 0) return ERROR("no stream: ", fctx->nb_streams);

    AVCodecContext *avctx = fctx->streams[video_index]->codec;
    AVCodec *codec = avcodec_find_decoder(avctx->codec_id);
    if (!codec || avcodec_open2(avctx, codec, 0) < 0) return ERROR("avcodec_open2: ", handle);

    AVPacket packet;
    if (av_read_frame(fctx, &packet) < 0) { avcodec_close(avctx); return ERROR("av_read_frame: ", handle); }

    AVFrame *frame = av_frame_alloc(); int ret;
    if ((ret = avcodec_decode_video2(avctx, frame, &got, &packet)) <= 0 || !got) {
      char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
      ERROR("avcodec_decode_video2: ", codec->name, " ", ret, ": ", errstr);
    } else {
      int pf = PixelFromFFMpegId(avctx->pix_fmt);
      out->width  = avctx->width;
      out->height = avctx->height;
      out->LoadGL(*frame->data, point(out->width, out->height), pf, frame->linesize[0]);
      if (!(load_flag & Flag::Clear)) out->LoadBuffer(frame->data[0], point(out->width, out->height), pf, frame->linesize[0]);
      // av_frame_unref(frame);
    }

    av_free_packet(&packet);
    // av_frame_free(&frame);
  }

  void LoadAudio(void *handle, SoundAsset *a, int seconds, int flag) {
    AVFormatContext *fctx = static_cast<AVFormatContext*>(handle);
    LoadMovie(a, 0, fctx);

    int samples = RefillAudio(a, 1);
    if (samples == SoundAsset::Size(a) && !(flag & SoundAsset::FlagNoRefill)) a->refill = RefillAudioCB;

    if (!a->refill) {
      avcodec_close(fctx->streams[a->handle_arg1]->codec);
      a->resampler.reset();
    }
  }

  void LoadMovie(void *handle, MovieAsset *ma) {
    LoadMovie(&ma->audio, &ma->video, static_cast<AVFormatContext*>(handle));
    PlayMovie(&ma->audio, &ma->video, static_cast<AVFormatContext*>(handle), 0);
  }

  int PlayMovie(MovieAsset *ma, int seek) { return PlayMovie(&ma->audio, &ma->video, static_cast<AVFormatContext*>(ma->handle), seek); }
  int RefillAudio(SoundAsset *a, int reset) { return RefillAudioCB(a, reset); }

  static AVFormatContext *Load(AVIOContext *pb, const string &filename, char *probe_buf, int probe_buflen, AVIOContext **pbOut=0) {
    AVProbeData probe_data;
    memzero(probe_data);
    probe_data.filename = BaseName(filename);

    bool probe_buf_data = probe_buf && probe_buflen;
    if (probe_buf_data) {
      probe_data.buf = MakeUnsigned(probe_buf);
      probe_data.buf_size = probe_buflen;
    }

    AVInputFormat *fmt = av_probe_input_format(&probe_data, probe_buf_data);
    if (!fmt) return ERRORv(nullptr, "no AVInputFormat for ", probe_data.filename);

    AVFormatContext *fctx = avformat_alloc_context(); int ret;
    fctx->flags |= AVFMT_FLAG_CUSTOM_IO;

    fctx->pb = pb;
    if ((ret = avformat_open_input(&fctx, probe_data.filename, fmt, 0))) {
      char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
      return ERRORv(nullptr, "av_open_input ", probe_data.filename, ": ", ret, " ", errstr);
    }

    if (pbOut) *pbOut = fctx->pb;
    return fctx;
  }

  static int RefillAudioCB(SoundAsset *a, int reset) {
    AVFormatContext *fctx = static_cast<AVFormatContext*>(a->handle);
    AVCodecContext *avctx = fctx->streams[a->handle_arg1]->codec;
    bool open_resampler = false;
    if (reset) {
      av_seek_frame(fctx, a->handle_arg1, 0, AVSEEK_FLAG_BYTE);
      a->wav->ring.size = SoundAsset::Size(a);
      a->wav->bytes = a->wav->ring.size * a->wav->width;
      open_resampler = true;
    }
    if (!a->wav) {
      a->wav = make_unique<RingBuf>(a->sample_rate, SoundAsset::Size(a));
      open_resampler = true;
    }
    if (open_resampler)
      a->resampler->Open(a->wav.get(), avctx->channels, avctx->sample_rate, SampleFromFFMpegId(avctx->sample_fmt),
                         a->channels, a->sample_rate, Sample::S16);
    a->wav->ring.back = 0;
    int wrote = PlayMovie(a, 0, fctx, 0);
    if (wrote < SoundAsset::Size(a)) {
      a->wav->ring.size = wrote;
      a->wav->bytes = a->wav->ring.size * a->wav->width;
    }
    return wrote;
  }

  static void LoadMovie(SoundAsset *sa, Asset *va, AVFormatContext *fctx) {
    if (avformat_find_stream_info(fctx, 0) < 0) return ERROR("av_find_stream_info");

    int audio_index = -1, video_index = -1, got=0;
    for (int i=0; i<fctx->nb_streams; i++) {
      AVStream *st = fctx->streams[i];
      AVCodecContext *avctx = st->codec;
      if (avctx->codec_type == AVMEDIA_TYPE_AUDIO) audio_index = i;
      if (avctx->codec_type == AVMEDIA_TYPE_VIDEO) video_index = i;
    }
    if (va) {
      if (video_index < 0) return ERROR("no v-stream: ", fctx->nb_streams);
      AVCodecContext *avctx = fctx->streams[video_index]->codec;
      AVCodec *codec = avcodec_find_decoder(avctx->codec_id);
      if (!codec || avcodec_open2(avctx, codec, 0) < 0) return ERROR("avcodec_open2: ", codec);
      va->tex.CreateBacked(avctx->width, avctx->height, Pixel::BGR32);
    }
    if (sa) {
      if (audio_index < 0) return ERROR("no a-stream: ", fctx->nb_streams);
      AVCodecContext *avctx = fctx->streams[audio_index]->codec;
      AVCodec *codec = avcodec_find_decoder(avctx->codec_id);
      if (!codec || avcodec_open2(avctx, codec, 0) < 0) return ERROR("avcodec_open2: ", codec);

      sa->handle = fctx;
      sa->handle_arg1 = audio_index;
      sa->channels = avctx->channels == 1 ? 1 : FLAGS_chans_out;
      sa->sample_rate = FLAGS_sample_rate;
      sa->seconds = FLAGS_soundasset_seconds;
      sa->wav = make_unique<RingBuf>(sa->sample_rate, SoundAsset::Size(sa));
      sa->resampler = unique_ptr<AudioResamplerInterface>(CreateAudioResampler());
      sa->resampler->Open(sa->wav.get(), avctx->channels, avctx->sample_rate, SampleFromFFMpegId(avctx->sample_fmt),
                          sa->channels, sa->sample_rate, Sample::S16);
    }
  }

  static int PlayMovie(SoundAsset *sa, Asset *va, AVFormatContext *fctx, int seek_unused) {
    int begin_resamples_available = sa->resampler->output_available, wrote=0, done=0;
    Allocator *tlsalloc = ThreadLocalStorage::GetAllocator();

    while (!done && wrote != SoundAsset::Size(sa)) {
      AVPacket packet; int ret, got=0;
      if ((ret = av_read_frame(fctx, &packet)) < 0) {
        char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
        ERROR("av_read_frame: ", errstr); break;
      }

      AVFrame *frame = av_frame_alloc();
      AVCodecContext *avctx = fctx->streams[packet.stream_index]->codec;
      if (sa && packet.stream_index == sa->handle_arg1) {
        if ((ret = avcodec_decode_audio4(avctx, frame, &got, &packet)) <= 0 || !got) {
          char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
          ERROR("avcodec_decode_audio4: ", errstr);
        } else {
          tlsalloc->Reset();
          short *rsout = static_cast<short*>(tlsalloc->Malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE + FF_INPUT_BUFFER_PADDING_SIZE));

          int sampsIn  = frame->nb_samples;
          int sampsOut = max(0, SoundAsset::Size(sa) - wrote) / sa->channels;

          sa->resampler->Update(sampsIn, reinterpret_cast<const short* const*>(frame->extended_data), rsout, Time(-1), sampsOut);
          wrote = sa->resampler->output_available - begin_resamples_available;
          av_frame_unref(frame);
        }
      } else if (va) {
        if ((ret = avcodec_decode_video2(avctx, frame, &got, &packet)) <= 0 || !got) {
          char errstr[128]; av_strerror(ret, errstr, sizeof(errstr));
          ERROR("avcodec_decode_video2 ", ret, ": ", errstr);
        } else {
          screen->gd->BindTexture(GraphicsDevice::Texture2D, va->tex.ID);
          va->tex.UpdateBuffer(*frame->data, point(avctx->width, avctx->height), PixelFromFFMpegId(avctx->pix_fmt), 
                               frame->linesize[0], Texture::Flag::Resample);
          va->tex.UpdateGL();
          av_frame_unref(frame);
          done = 1;
        }
      }
      av_free_packet(&packet);
      av_frame_free(&frame);
    }
    return wrote;
  }
};

AssetLoaderInterface *CreateFFMpegAssetLoader() { 
  ONCE({ INFO("FFMpegInit()");
         // av_log_set_level(AV_LOG_DEBUG);
         av_register_all(); });

  return new FFMpegAssetLoader();
}

}; // namespace LFL