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

#include "gif_lib.h"

namespace LFL {
static int GIFInput(GifFileType *gif, GifByteType *out, int size) {
  return static_cast<BufferFile*>(gif->UserData)->Read(out, size);
}

int GIFReader::Read(File *lf,           Texture *out) { return Read(lf->Contents(), out); }
int GIFReader::Read(const string &data, Texture *out) {
  int error_code = 0;
  BufferFile bf(data);
  GifFileType *gif = DGifOpen(&bf, &GIFInput, &error_code);
  if (!gif) { INFO("gif open failed: ", error_code); return -1; }
  if (DGifSlurp(gif) != GIF_OK) { INFO("gif slurp failed"); DGifCloseFile(gif, &error_code); return -1; }
  out->Resize(gif->SWidth, gif->SHeight, Pixel::RGBA, Texture::Flag::CreateBuf);

  SavedImage *image = &gif->SavedImages[0];
  int gif_linesize = image->ImageDesc.Width, ls = out->LineSize(), ps = out->PixelSize(), transparent = -1;
  ColorMapObject *color_map = X_or_Y(gif->SColorMap, image->ImageDesc.ColorMap);
  for (int i=0; i<image->ExtensionBlockCount; i++) {
    ExtensionBlock *block = &image->ExtensionBlocks[i];
    if (block->Function == GRAPHICS_EXT_FUNC_CODE && block->ByteCount == 4 && (block->Bytes[0] & 1))
    { transparent = block->Bytes[3]; break; }
  }
  CHECK_LE(image->ImageDesc.Left + image->ImageDesc.Width,  out->width);
  CHECK_LE(image->ImageDesc.Top  + image->ImageDesc.Height, out->height);
  for (int y=image->ImageDesc.Top, yl=y+image->ImageDesc.Height; y<yl; y++) {
    unsigned char *gif_row = &image->RasterBits[y * gif_linesize], *pix_row = &(out->buf)[y * ls];
    for (int x=image->ImageDesc.Left, xl=x+image->ImageDesc.Width; x<xl; x++) {
      unsigned char gif_in = gif_row[x], *pix_out = &pix_row[x * ps];
      if (gif_in == transparent || gif_in >= color_map->ColorCount) { pix_out[3] = 0; continue; }
      *pix_out++ = color_map->Colors[gif_in].Red;
      *pix_out++ = color_map->Colors[gif_in].Green;
      *pix_out++ = color_map->Colors[gif_in].Blue;
      *pix_out++ = 255;
    }
  }
  DGifCloseFile(gif, &error_code);
  return 0;
}

}; // namespace LFL
