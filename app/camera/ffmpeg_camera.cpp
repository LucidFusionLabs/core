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
#ifdef LFL_FFMPEG_CAMERA
#include <libavformat/avformat.h>
#include <libavdevice/avdevice.h>
#endif
};

namespace LFL {
struct FFmpegCameraModule : public Module {
  Thread thread;
  mutex lock;
  struct Stream { 
    struct FramePtr {
      AVPacket data;
      bool dirty;
      FramePtr(bool D=0) : dirty(D) {}
      ~FramePtr() { if (dirty) free(); }
      void free() { av_free_packet(&data); clear(); }
      void clear() { dirty=0; }
      static void swap(FramePtr *A, FramePtr *B) { FramePtr C = *A; *A = *B; *B = C; C.clear(); }
    };
    AVFormatContext *fctx=0;
    unique_ptr<RingBuf> frames;
    int next=0;
  } L, R;
  CameraState *camera;
  FFmpegCameraModule(CameraState *C) : thread(bind(&FFMpegCameraModule::Threadproc, this)), camera(C) {}

  int Free() {
    if (thread.started) thread.Wait();
    return 0;
  }

  int Init() {
#ifdef _WIN32
    static const char *ifmtname = "vfwcap";
    static const char *ifilename[2] = { "0", "1" };
#endif
#ifdef __linux__
    static const char *ifmtname = "video4linux2";
    static const char *ifilename[2] = { "/dev/video0", "/dev/video1" };
#endif
    avdevice_register_all();
    AVInputFormat *ifmt = av_find_input_format(ifmtname);

    AVDictionary *options = 0;
    av_dict_set(&options, "video_size", "640x480", 0);

    L.fctx = 0;
    if (avformat_open_input(&L.fctx, ifilename[0], ifmt, &options) < 0) {
      FLAGS_lfapp_camera = 0;
      return 0;
    }
    L.frames = make_unique<RingBuf>(FLAGS_camera_fps, FLAGS_camera_fps, sizeof(Stream::FramePtr));
    av_dict_free(&options);

    if (!thread.Start()) { FLAGS_lfapp_camera=0; return -1; }

    AVCodecContext *codec = L.fctx->streams[0]->codec;
    FLAGS_camera_image_width = codec->width;
    FLAGS_camera_image_height = codec->height;
    camera->image_format = Pixel::FromFFMpegId(codec->pix_fmt);
    camera->image_linesize = FLAGS_camera_image_width * Pixel::size(camera->image_format);
    return 0;
  }

  int Frame(unsigned) {
    bool new_frame = false;
    {
      ScopedMutex ML(lock);
      if (camera->frames_read > camera->last_frames_read) {
        if (L.fctx) L.next = L.frames->ring.back;
        if (R.fctx) R.next = R.frames->ring.back;
        new_frame = true;
      }
      camera->last_frames_read = camera->frames_read;
    }
    if (!new_frame) return 0;

    AVPacket *f = (AVPacket*)L.frames->Read(-1, L.next);
    camera->image = f->data;
    camera->image_timestamp = L.frames->ReadTimestamp(-1, L.next);
    return 1;
  }

  int Threadproc() {
    while (app->run) {
      /* grab */
      Stream::FramePtr Lframe(0);
      if (av_read_frame(L.fctx, &Lframe.data) < 0) return ERRORv(-1, "av_read_frame");
      else Lframe.dirty = 1;

      Stream::FramePtr::swap((Stream::FramePtr*)L.frames->Write(RingBuf::Peek | RingBuf::Stamp), &Lframe);

      /* commit */  
      {
        ScopedMutex ML(lock);
        L.frames->Write();
        camera->frames_read++;
      }
    }
    return 0;
  }
};

extern "C" void *LFAppCreateCameraModule(CameraState *state) { return new FFMpegCameraModule(state); }

}; // namespace LFL
