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

#ifndef LFL_CORE_APP_CAMERA_H__
#define LFL_CORE_APP_CAMERA_H__
namespace LFL {

DECLARE_int(camera_fps);
DECLARE_int(camera_image_width);
DECLARE_int(camera_image_height);

struct Camera : public Module {
  RollingAvg<unsigned> fps;
  unique_ptr<Module> impl;
  CameraState state;
  Camera() : fps(64) { memzero(state); }

  int Init()               override;
  int Free()               override;
  int Frame(unsigned time) override;
};

}; // namespace LFL
#endif // LFL_CORE_APP_CAMERA_H__
