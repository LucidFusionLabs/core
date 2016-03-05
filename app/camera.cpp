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
#include "core/app/camera.h"

namespace LFL {
DEFINE_int(camera_fps, 20, "Camera capture frames per second");
DEFINE_int(camera_device, 0, "Camera device index");
DEFINE_bool(print_camera_devices, false, "Print camera device list");
DEFINE_int(camera_image_width, 0, "Camera capture image width");
DEFINE_int(camera_image_height, 0, "Camera capture image height");

int Camera::Init() {
  INFO("Camera::Init()");
  impl = CreateCameraModule(&state);

  int ret = 0;
  if (impl) ret = impl->Init();
  else FLAGS_lfapp_camera = 0;

  if (!FLAGS_lfapp_camera) INFO("no camera found");
  return ret;
}

int Camera::Frame(unsigned clicks) { 
  state.since_last_frame += clicks;
  if ((state.have_sample = (impl->Frame(clicks) == 1))) {
    fps.Add(state.since_last_frame);
    state.since_last_frame = 0;
  }
  return 0;
}

int Camera::Free() {
  int ret = impl ? impl->Free() : 0;
  impl.reset();
  return ret;
}

}; // namespace LFL
