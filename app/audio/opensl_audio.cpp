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
#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>

namespace LFL {
struct OpenSLAudioModule : public Module {
  Audio *audio;
  OpenSLAudioModule(Audio *A) : audio(A) {}
  int Init() {
    INFO("OpenSLAudioModule::Init");
    if (FLAGS_chans_in == -1) FLAGS_chans_in = 2;
    if (FLAGS_chans_out == -1) FLAGS_chans_out = 2;
    return 0;
  }
};
 
Module *CreateAudioModule(Audio *a) { return new OpenSLAudioModule(a); }

}; // namespace LFL
