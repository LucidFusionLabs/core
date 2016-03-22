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

#include "dsvl.h"

namespace LFL {
struct DsvlCameraModule : public Module {
  Thread thread;
  mutex lock;
  struct Stream { 
    unique_ptr<DSVL_VideoSource> vs;
    unique_ptr<RingSampler> frames;
    int next=0;
  } L, R;
  CameraState *camera;
  DsvlCameraModule(CameraState *C) : thread(bind(&DsvlCameraModule::Threadproc, this)), camera(C) {}

  int Free() {
    if (thread.started) thread.Wait();
    return 0;
  }

  int Init() {
    char config[] = "<?xml version='1.0' encoding='UTF-8'?> <dsvl_input>"
      "<camera frame_rate='5.0' show_format_dialog='true'>"
      "<pixel_format><RGB24 flip_v='true'/></pixel_format>"
      "</camera></dsvl_input>\r\n";

    CoInitialize(0);
    L.vs = make_unique<DSVL_VideoSource>();

    LONG w, h;
    double fps;
    PIXELFORMAT pf;
    if (FAILED(L.vs->BuildGraphFromXMLString(config))) return -1;
    if (FAILED(L.vs->GetCurrentMediaFormat(&w, &h, &fps, &pf))) return -1;
    if (FAILED(L.vs->EnableMemoryBuffer())) return -1;
    if (FAILED(L.vs->Run())) return -1;

    FLAGS_camera_image_width = w;
    FLAGS_camera_image_height = h;
    FLAGS_camera_fps = fps;

    int depth;
    if (pf == PIXELFORMAT_RGB24) {
      depth = 3;
      camera->image_format = Pixel::RGB24;
      camera->image_linesize = FLAGS_camera_image_width * depth;
    }
    else return ERRORv(-1, "unknown pixel format: ", pf);

    L.frames = make_unique<RingSampler>(FLAGS_camera_fps, FLAGS_camera_fps, FLAGS_camera_image_width*FLAGS_camera_image_height*depth);

    if (!thread.Start()) { FLAGS_lfapp_camera=0; return -1; }

    INFO("opened camera ", FLAGS_camera_image_width, "x", FLAGS_camera_image_height, " @ ", FLAGS_camera_fps, "fps");
    return 0;
  }

  int Frame(unsigned) {
    bool new_frame = false;
    {
      ScopedMutex ML(lock);
      if (camera->frames_read > camera->last_frames_read) {

        /* frame align stream handles */
        if (L.vs) L.next = L.frames->ring.back;
        if (R.vs) R.next = R.frames->ring.back;

        new_frame = true;
      }
      camera->last_frames_read = camera->frames_read;
    }
    if (!new_frame) return 0;

    camera->image = (unsigned char*)L.frames->read(-1, L.next);
    camera->image_timestamp = L.frames->readtimestamp(-1, L.next);
    return 1;
  }

  int Threadproc() {
    while (app->run) {
      DWORD ret = L.vs->WaitForNextSample(1000/FLAGS_camera_fps);
      if (ret != WAIT_OBJECT_0) continue;

      char *b;
      MemoryBufferHandle h;
      L.vs->CheckoutMemoryBuffer(&h, (BYTE**)&b);
      memcpy(L.frames->write(RingSampler::Peek | RingSampler::Stamp), b, L.frames->width);
      L.vs->CheckinMemoryBuffer(h);

      { /* commit */ 
        ScopedMutex ML(lock);
        L.frames->write();
        camera->frames_read++;
      }
    }
    return 0;
  };
};

extern "C" void *LFAppCreateCameraModule(CameraState *state) { return new DsvlCameraModule(state); }

}; // namespace LFL
