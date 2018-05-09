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

#ifndef LFL_CORE_APP_BINDINGS_OPENCV_H__
#define LFL_CORE_APP_BINDINGS_OPENCV_H__

namespace LFL {
inline void TextureToIplImage(const Texture &in, _IplImage *out) {
  memset(out, 0, sizeof(IplImage));
  out->nSize = sizeof(IplImage);
  out->nChannels = Pixel::Size(in.pf);
  out->depth = IPL_DEPTH_8U;
  out->origin = 1;
  out->width = in.width;
  out->height = in.height;
  out->widthStep = out->width * out->nChannels;
  out->imageSize = out->widthStep * out->height;
  out->imageData = MakeSigned(in.buf);
  out->imageDataOrigin = out->imageData;
}

}; // namespace LFL
#endif // LFL_CORE_APP_BINDINGS_OPENCV_H__
