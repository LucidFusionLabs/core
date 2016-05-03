/*
 * $Id: crypto.cpp 1335 2014-12-02 04:13:46Z justin $
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
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavcodec/avfft.h>
#include <libswscale/swscale.h>
#define AVCODEC_MAX_AUDIO_FRAME_SIZE 192000 
};
#include "core/app/bindings/ffmpeg.h"

namespace LFL {
struct StreamResourceClient : public Connection::Handler {
  Connection *conn;
  HTTPServer::StreamResource *resource;
  AVFormatContext *fctx;
  microseconds start;

  StreamResourceClient(Connection *c, HTTPServer::StreamResource *r) : conn(c), resource(r), start(0) {
    resource->subscribers[this] = conn;
    fctx = avformat_alloc_context();
    CopyAVFormatContextStreams(fctx, resource->fctx);
    fctx->max_delay = int(0.7*AV_TIME_BASE);
  }
  virtual ~StreamResourceClient() {
    resource->subscribers.erase(this);
    FreeAVFormatContext(fctx);
  }

  int Flushed(Connection *c) { return 1; }
  void Open() { if (avio_open_dyn_buf(&fctx->pb)) ERROR("avio_open_dyn_buf"); }

  void Write(AVPacket *pkt, microseconds timestamp) {        
    Open();
    if (start == microseconds(0)) start = timestamp;
    if (timestamp != microseconds(0)) {
      AVStream *st = fctx->streams[pkt->stream_index];
      AVRational r = {1, 1000000};
      unsigned t = (timestamp - start).count();
      pkt->pts = av_rescale_q(t, r, st->time_base);
    }
    int ret;
    if ((ret = av_interleaved_write_frame(fctx, pkt))) ERROR("av_interleaved_write_frame: ", ret);
    Flush();
  }

  void WriteHeader() {
    Open();
    if (avformat_write_header(fctx, 0)) ERROR("av_write_header");
    avio_flush(fctx->pb);
    Flush();
  }

  void Flush() {
    int len=0;
    char *buf=0;
    if (!(len = avio_close_dyn_buf(fctx->pb, reinterpret_cast<uint8_t**>(&buf)))) return;
    if (len < 0) return ERROR("avio_close_dyn_buf");
    if (conn->Write(buf, len) < 0) conn->SetError();
    av_free(buf);
  }

  static void FreeAVFormatContext(AVFormatContext *fctx) {
    for (int i=0; i<fctx->nb_streams; i++) av_freep(&fctx->streams[i]);
    av_free(fctx);
  }

  static void CopyAVFormatContextStreams(AVFormatContext *dst, AVFormatContext *src) {
    if (!dst->streams) {
      dst->nb_streams = src->nb_streams;
      dst->streams = static_cast<AVStream**>(av_mallocz(sizeof(AVStream*) * src->nb_streams));
    }

    for (int i=0; i<src->nb_streams; i++) {
      AVStream *s = static_cast<AVStream*>(av_mallocz(sizeof(AVStream)));
      *s = *src->streams[i];
      s->priv_data = 0;
      s->codec->frame_number = 0;
      dst->streams[i] = s;
    }

    dst->oformat = src->oformat;
    dst->nb_streams = src->nb_streams;
  }

  static AVFrame *AllocPicture(AVPixelFormat pix_fmt, int width, int height) {
    AVFrame *picture = av_frame_alloc();
    if (!picture) return 0;
    int size = avpicture_get_size(pix_fmt, width, height);
    uint8_t *picture_buf = static_cast<uint8_t*>(av_malloc(size));
    if (!picture_buf) { av_free(picture); return 0; }
    avpicture_fill(reinterpret_cast<AVPicture*>(picture), picture_buf, pix_fmt, width, height);
    return picture;
  } 
  static void FreePicture(AVFrame *picture) {
    av_free(picture->data[0]);
    av_free(picture);
  }

  static AVFrame *AllocSamples(int num_samples, int num_channels, short **samples_out) {
    AVFrame *samples = av_frame_alloc();
    if (!samples) return 0;
    samples->nb_samples = num_samples;
    int size = 2 * num_samples * num_channels;
    uint8_t *samples_buf = static_cast<uint8_t*>(av_malloc(size + FF_INPUT_BUFFER_PADDING_SIZE));
    if (!samples_buf) { av_free(samples); return 0; }
    avcodec_fill_audio_frame(samples, num_channels, AV_SAMPLE_FMT_S16, samples_buf, size, 1);
    memset(samples_buf+size, 0, FF_INPUT_BUFFER_PADDING_SIZE);
    if (samples_out) *samples_out = reinterpret_cast<short*>(samples_buf);
    return samples;
  }
  static void FreeSamples(AVFrame *picture) {
    av_free(picture->data[0]);
    av_free(picture);
  }
};

HTTPServer::StreamResource::~StreamResource() {
  if (resampler) delete resampler->out;
  delete resampler;
  if (audio && audio->codec) avcodec_close(audio->codec);
  if (video && video->codec) avcodec_close(video->codec);
  if (picture) StreamResourceClient::FreePicture(picture);
  if (samples) StreamResourceClient::FreeSamples(picture);
  StreamResourceClient::FreeAVFormatContext(fctx);
}

HTTPServer::StreamResource::StreamResource(const char *oft, int Abr, int Vbr) : abr(Abr), vbr(Vbr) {
  fctx = avformat_alloc_context();
  fctx->oformat = av_guess_format(oft, 0, 0);
  if (!fctx->oformat) { ERROR("guess_format '", oft, "' failed"); return; }
  INFO("StreamResource: format ", fctx->oformat->mime_type);
  OpenStreams(FLAGS_enable_audio, FLAGS_enable_camera);
}

HTTPServer::Response HTTPServer::StreamResource::Request(Connection *c, int method, const char *url, const char *args, const char *headers, const char *postdata, int postlen) {
  if (!open) return HTTPServer::Response::_400;
  Response response(fctx->oformat->mime_type, -1, new StreamResourceClient(c, this), false);
  if (c->Write(HTTP::MakeHeaders(response.code, response.content_length, response.type)) < 0)
  { c->SetError(); return response; }
  dynamic_cast<StreamResourceClient*>(response.refill)->WriteHeader();
  return response;
}

void HTTPServer::StreamResource::OpenStreams(bool A, bool V) {
  if (V) {
    CHECK(!video);
    video = avformat_new_stream(fctx, 0);
    video->id = fctx->nb_streams;
    AVCodecContext *vc = video->codec;

    vc->codec_type = AVMEDIA_TYPE_VIDEO;
    vc->codec_id = AV_CODEC_ID_H264;
    vc->codec_tag = av_codec_get_tag(fctx->oformat->codec_tag, vc->codec_id);

    vc->width = 576;
    vc->height = 342;
    vc->bit_rate = vbr;
    vc->time_base.num = 1;
    vc->time_base.den = FLAGS_camera_fps;
    vc->pix_fmt = AV_PIX_FMT_YUV420P;

    /* x264 defaults */
    vc->me_range = 16;
    vc->max_qdiff = 4;
    vc->qmin = 10;
    vc->qmax = 51;
    vc->qcompress = 0.6;

    if (fctx->oformat->flags & AVFMT_GLOBALHEADER) vc->flags |= CODEC_FLAG_GLOBAL_HEADER;

    AVCodec *codec = avcodec_find_encoder(vc->codec_id);
    if (avcodec_open2(vc, codec, 0) < 0) return ERROR("avcodec_open2");
    if (!vc->codec) return ERROR("no video codec");

    if (vc->pix_fmt != AV_PIX_FMT_YUV420P) return ERROR("pix_fmt ", vc->pix_fmt, " != ", AV_PIX_FMT_YUV420P);
    if (!(picture = StreamResourceClient::AllocPicture(vc->pix_fmt, vc->width, vc->height))) return ERROR("AllocPicture");
  }

  if (0 && A) {
    audio = avformat_new_stream(fctx, 0);
    audio->id = fctx->nb_streams;
    AVCodecContext *ac = audio->codec;

    ac->codec_type = AVMEDIA_TYPE_AUDIO;
    ac->codec_id = AV_CODEC_ID_MP3;
    ac->codec_tag = av_codec_get_tag(fctx->oformat->codec_tag, ac->codec_id);

    ac->channels = FLAGS_chans_in;
    ac->bit_rate = abr;
    ac->sample_rate = 22050;
    ac->sample_fmt = AV_SAMPLE_FMT_S16P;
    ac->channel_layout = AV_CH_LAYOUT_STEREO;

    if (fctx->oformat->flags & AVFMT_GLOBALHEADER) ac->flags |= CODEC_FLAG_GLOBAL_HEADER;

    AVCodec *codec = avcodec_find_encoder(ac->codec_id);
    if (avcodec_open2(ac, codec, 0) < 0) return ERROR("avcodec_open2");
    if (!ac->codec) return ERROR("no audio codec");

    if (!(frame = ac->frame_size)) return ERROR("empty frame size");
    channels = ac->channels;

    if (!(samples = StreamResourceClient::AllocSamples(frame, channels, &sample_data))) return ERROR("AllocPicture");
  }

  open = 1;
}

void HTTPServer::StreamResource::Update(int audio_samples, bool video_sample) {
  if (!open || !subscribers.size()) return;
  AVCodecContext *vc = video ? video->codec : 0;
  AVCodecContext *ac = audio ? audio->codec : 0;

  if (ac && audio_samples) {
    if (!resampler) {
      resampler = CreateAudioResampler();
      resampler->out = new RingSampler(ac->sample_rate, ac->sample_rate*channels);
      resampler->Open(resampler->out, FLAGS_chans_in, FLAGS_sample_rate, Sample::S16,
                      channels, ac->sample_rate, SampleFromFFMpegId(ac->channel_layout));
    };
    RingSampler::Handle L(app->audio->IL.get(), app->audio->IL->ring.back-audio_samples, audio_samples);
    RingSampler::Handle R(app->audio->IR.get(), app->audio->IR->ring.back-audio_samples, audio_samples);
    if (resampler->Update(audio_samples, &L, FLAGS_chans_in > 1 ? &R : 0)) open=0;
  }

  for (;;) {
    bool asa = ac && resampler->output_available >= resamples_processed + frame * channels;
    bool vsa = vc && video_sample;
    if (!asa && !vsa) break;
    if (vc && !vsa) break;

    if (!vsa) { SendAudio(); continue; }       
    if (!asa) { SendVideo(); video_sample=0; continue; }

    int audio_behind = resampler->output_available - resamples_processed;
    microseconds audio_timestamp = resampler->out->ReadTimestamp(0, resampler->out->ring.back - audio_behind);

    if (audio_timestamp < microseconds(app->camera->state.image_timestamp_us)) SendAudio();
    else { SendVideo(); video_sample=0; }
  }
}

void HTTPServer::StreamResource::SendAudio() {
  int behind = resampler->output_available - resamples_processed, got = 0;
  resamples_processed += frame * channels;

  AVCodecContext *ac = audio->codec;
  RingSampler::Handle H(resampler->out, resampler->out->ring.back - behind, frame * channels);

  /* linearize */
  for (int i=0; i<frame; i++) 
    for (int c=0; c<channels; c++)
      sample_data[i*channels + c] = H.Read(i*channels + c) * 32768.0;

  /* broadcast */
  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;

  avcodec_encode_audio2(ac, &pkt, samples, &got);
  if (got) Broadcast(&pkt, H.ReadTimestamp(0));

  av_free_packet(&pkt);
}

void HTTPServer::StreamResource::SendVideo() {
  AVCodecContext *vc = video->codec;

  /* convert video */
  if (!conv)
    conv = sws_getContext(FLAGS_camera_image_width, FLAGS_camera_image_height,
                          AVPixelFormat(PixelToFFMpegId(app->camera->state.image_format)),
                          vc->width, vc->height, vc->pix_fmt, SWS_BICUBIC, 0, 0, 0);

  int camera_linesize[4] = { app->camera->state.image_linesize, 0, 0, 0 }, got = 0;
  sws_scale(conv, reinterpret_cast<uint8_t**>(&app->camera->state.image), camera_linesize, 0,
            FLAGS_camera_image_height, picture->data, picture->linesize);

  /* broadcast */
  AVPacket pkt;
  av_init_packet(&pkt);
  pkt.data = NULL;
  pkt.size = 0;

  avcodec_encode_video2(vc, &pkt, picture, &got);
  if (got) Broadcast(&pkt, microseconds(app->camera->state.image_timestamp_us));

  av_free_packet(&pkt);
}

void HTTPServer::StreamResource::Broadcast(AVPacket *pkt, microseconds timestamp) {
  for (auto i = subscribers.begin(); i != subscribers.end(); i++) {
    StreamResourceClient *client = static_cast<StreamResourceClient*>(i->first);
    client->Write(pkt, timestamp);
  }
}

}; // namespace LFL
