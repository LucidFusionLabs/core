/*
 * $Id: video.h 1336 2014-12-08 09:29:59Z justin $
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

#ifndef LFL_LFAPP_BINDINGS_FFMPEG_H__
#define LFL_LFAPP_BINDINGS_FFMPEG_H__

extern "C" {
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libswresample/swresample.h>
};

namespace LFL {
inline int PixelFromFFMpegId(int fmt) {
  switch (fmt) {
    case AV_PIX_FMT_RGB32:    return Pixel::RGB32;
    case AV_PIX_FMT_BGR32:    return Pixel::BGR32;
    case AV_PIX_FMT_RGB24:    return Pixel::RGB24;
    case AV_PIX_FMT_BGR24:    return Pixel::BGR24;
    case AV_PIX_FMT_GRAY8:    return Pixel::GRAY8;
    case AV_PIX_FMT_YUV410P:  return Pixel::YUV410P;
    case AV_PIX_FMT_YUV420P:  return Pixel::YUV420P;
    case AV_PIX_FMT_YUYV422:  return Pixel::YUYV422;
    case AV_PIX_FMT_YUVJ420P: return Pixel::YUVJ420P;
    case AV_PIX_FMT_YUVJ422P: return Pixel::YUVJ422P;
    case AV_PIX_FMT_YUVJ444P: return Pixel::YUVJ444P;
    default: return ERRORv(0, "unknown pixel fmt: ", fmt);
  }
}

inline int PixelToFFMpegId(int fmt) {
  switch (fmt) {
    case Pixel::RGB32:    return AV_PIX_FMT_RGB32;
    case Pixel::BGR32:    return AV_PIX_FMT_BGR32;
    case Pixel::RGB24:    return AV_PIX_FMT_RGB24;
    case Pixel::BGR24:    return AV_PIX_FMT_BGR24;
    case Pixel::RGBA:     return AV_PIX_FMT_RGBA;
    case Pixel::BGRA:     return AV_PIX_FMT_BGRA;
    case Pixel::GRAY8:    return AV_PIX_FMT_GRAY8;
    case Pixel::YUV410P:  return AV_PIX_FMT_YUV410P;
    case Pixel::YUV420P:  return AV_PIX_FMT_YUV420P;
    case Pixel::YUYV422:  return AV_PIX_FMT_YUYV422;
    case Pixel::YUVJ420P: return AV_PIX_FMT_YUVJ420P;
    case Pixel::YUVJ422P: return AV_PIX_FMT_YUVJ422P;
    case Pixel::YUVJ444P: return AV_PIX_FMT_YUVJ444P;
    default: return ERRORv(0, "unknown pixel fmt: ", fmt);
  }
}

inline int SampleFromFFMpegId(int fmt) {
  switch(fmt) {
    case AV_SAMPLE_FMT_U8:   return Sample::U8;
    case AV_SAMPLE_FMT_U8P:  return Sample::U8P;
    case AV_SAMPLE_FMT_S16:  return Sample::S16;
    case AV_SAMPLE_FMT_S16P: return Sample::S16P;
    case AV_SAMPLE_FMT_S32:  return Sample::S32;
    case AV_SAMPLE_FMT_S32P: return Sample::S32P;
    case AV_SAMPLE_FMT_FLT:  return Sample::FLOAT;
    case AV_SAMPLE_FMT_FLTP: return Sample::FLOATP;
    case AV_SAMPLE_FMT_DBL:  return Sample::DOUBLE;
    case AV_SAMPLE_FMT_DBLP: return Sample::DOUBLEP;
    default: return ERRORv(0, "unknown sample fmt: ", fmt);
  }
}

inline int SampleToFFMpegId(int fmt) {
  switch(fmt) {
    case Sample::U8:      return AV_SAMPLE_FMT_U8;
    case Sample::U8P:     return AV_SAMPLE_FMT_U8P;
    case Sample::S16:     return AV_SAMPLE_FMT_S16;
    case Sample::S16P:    return AV_SAMPLE_FMT_S16P;
    case Sample::S32:     return AV_SAMPLE_FMT_S32;
    case Sample::S32P:    return AV_SAMPLE_FMT_S32P;
    case Sample::FLOAT:   return AV_SAMPLE_FMT_FLT;
    case Sample::FLOATP:  return AV_SAMPLE_FMT_FLTP;
    case Sample::DOUBLE:  return AV_SAMPLE_FMT_DBL;
    case Sample::DOUBLEP: return AV_SAMPLE_FMT_DBLP;
    default: return ERRORv(0, "unknown sample fmt: ", fmt);
  }
}

}; // namespace LFL
#endif // LFL_LFAPP_BINDINGS_FFMPEG_H__
