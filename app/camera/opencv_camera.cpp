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

#include "opencv/highgui.h"
#include "lfapp/lfapp.h"
LFL_IMPORT extern int OPENCV_FPS;

namespace LFL {
struct OpenCvCameraModule : public Module {
  Thread thread;
  mutex lock;
  struct Stream { 
    CvCapture *capture=0;
    unique_ptr<RingSampler> frames;
    int width=0, height=0, next=0;
    void SetDimensions(int W, int H) { width=W; height=H; }
  } L, R;
  CameraState *camera;
  OpenCvCameraModule(CameraState *C) : thread(bind(&OpenCvCamera::Threadproc, this)), camera(C) {}

  int Free() {
    if (thread.started) thread.Wait();
    if (&L.capture) cvReleaseCapture(&L.capture);
    if (&R.capture) cvReleaseCapture(&R.capture);
    return 0;
  }

  int Init() {
    OPENCV_FPS = FLAGS_camera_fps;

    if (!(L.capture = cvCaptureFromCAM(0))) { FLAGS_lfapp_camera=0; return 0; }
    // if (!(R.capture = cvCaptureFromCAM(1))) { /**/ }

    if (!thread.Start()) { FLAGS_lfapp_camera=0; return -1; }
    return 0;
  }

  int Frame(unsigned) {
    bool new_frame = false;
    {
      ScopedMutex ML(lock);
      if (camera->frames_read > camera->last_frames_read) {

        /* frame align stream handles */
        if (L.capture) L.next = L.frames->ring.back;
        if (R.capture) R.next = R.frames->ring.back;

        new_frame = true;
      }
      camera->last_frames_read = camera->frames_read;
    }

    if (!new_frame) return 0;

    camera->image = (unsigned char*)L.frames->read(-1, L.next);
    camera->image_timestamp = L.frames->readtimestamp(-1, L.next);
    FLAGS_camera_image_width = L.width;
    FLAGS_camera_image_height = L.height;

    camera->image_format = Pixel::BGR24;
    camera->image_linesize = camera->image_width*3;

    return 1;
  }

  int Threadproc() {
    while (app->run) {
      /* grab */
      bool lg=0, rg=0;
      if (L.capture) lg = cvGrabFrame(L.capture);
      if (R.capture) rg = cvGrabFrame(R.capture);

      /* retrieve */
      IplImage *lf=0, *rf=0;
      if (lg) lf = cvRetrieveFrame(L.capture);
      if (rg) rf = cvRetrieveFrame(R.capture);

      /* 1-time cosntruct */
      if (lf && !L.frames) {
        L.dimensions(lf->width, lf->height);
        L.frames = make_unique<RingSampler>(FLAGS_camera_fps, FLAGS_camera_fps, lf->imageSize);
      }
      if (rf && !R.frames) {
        R.dimensions(rf->width, rf->height);
        R.frames = make_unique<RingSampler>(FLAGS_camera_fps, FLAGS_camera_fps, rf->imageSize);
      }

      /* write */
      if (lf) memcpy(L.frames->write(RingSampler::Peek | RingSampler::Stamp), lf->imageData, lf->imageSize);
      if (rf) memcpy(R.frames->write(RingSampler::Peek | RingSampler::Stamp), rf->imageData, rf->imageSize);

      /* commit */  
      if (lf || rf) {
        ScopedMutex ML(lock);
        if (lf) L.frames->write();
        if (rf) R.frames->write();
        camera->frames_read++;
      }
      else Msleep(1);
    }
    return 0;
  }
};

extern "C" void *LFAppCreateCameraModule(CameraState *state) { return new OpenCVCameraModule(state); }

}; // namespace LFL
