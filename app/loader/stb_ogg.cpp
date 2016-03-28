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

#undef CHECK
#include "core/imports/stb/stb_vorbis.c"

namespace LFL {
void *OGGReader::OpenBuffer(const char *buf, size_t len, int *sr_out, int *chans_out, int *total_out) { return 0; }
void *OGGReader::OpenFile(const string &fn, int *sr_out, int *chans_out, int *total_out) {
  stb_vorbis *stream = stb_vorbis_open_filename(&fn[0], NULL, NULL);
  stb_vorbis_info info = stb_vorbis_get_info(stream);
  if (sr_out)    *sr_out    = info.sample_rate;
  if (chans_out) *chans_out = info.channels;
  if (total_out) *total_out = stb_vorbis_stream_length_in_samples(stream);
  return stream;
}

void OGGReader::Close(void *h) {
  auto stream = static_cast<stb_vorbis*>(h);
  if (stream) stb_vorbis_close(stream);
}

int OGGReader::Read(void *h, int chans, int samples, RingSampler *out, bool reset) { 
  Allocator *tlsalloc = ThreadLocalStorage::GetAllocator();
  float *buf = static_cast<float*>(tlsalloc->Malloc(samples*chans*sizeof(float)));
  auto stream = static_cast<stb_vorbis*>(h);
  if (reset) stb_vorbis_seek_start(stream);
  size_t decoded = stb_vorbis_get_samples_float_interleaved(stream, chans, buf, samples * chans);
  for (const float *i = buf, *e = i+decoded*chans; i != e; ++i) out->Write(*i);
  return decoded*chans;
}

}; // namespace LFL
