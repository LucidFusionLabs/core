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

#include <google/protobuf/message.h>
#include "core/app/app.h"

namespace LFL {
void ProtoFile::Open(const char *fn) {
  if (file) delete file;
  file = fn ? new LocalFile(fn, "r+", true) : 0;
  read_offset = 0;
  write_offset = -1;
  done = (file ? file->Size() : 0) <= 0;
  nr.Init(file);
}

int ProtoFile::Add(const Proto *msg, int status) {
  done = 0;
  write_offset = file->Seek(0, File::Whence::END);

  ProtoHeader ph(status);
  int wrote = WriteProto(file, &ph, msg, true);
  nr.SetFileOffset(wrote > 0 ? write_offset + wrote : write_offset);
  return wrote > 0;
}

bool ProtoFile::Update(int offset, const ProtoHeader *ph, const Proto *msg) {
  if (offset < 0 || (write_offset = file->Seek(offset, File::Whence::SET)) != offset) return false;
  int wrote = WriteProto(file, ph, msg, true);
  nr.SetFileOffset(wrote > 0 ? offset + wrote : offset);
  return wrote > 0;
}

bool ProtoFile::Update(int offset, int status) {
  if (offset < 0 || (write_offset = file->Seek(offset, File::Whence::SET)) != offset) return false;
  ProtoHeader ph(status);
  int wrote = WriteProtoFlag(file, &ph, true);
  nr.SetFileOffset(wrote > 0 ? offset + wrote : offset);
  return wrote > 0;
}

bool ProtoFile::Get(Proto *out, int offset, int status) {
  int record_offset;
  write_offset = 0;
  file->Seek(offset, File::Whence::SET);
  bool ret = Next(out, &record_offset, status);
  if (!ret) return 0;
  return offset == record_offset;
}

bool ProtoFile::Next(Proto *out, int *offsetOut, int status) { ProtoHeader hdr; return Next(&hdr, out, offsetOut, status); }
bool ProtoFile::Next(ProtoHeader *hdr, Proto *out, int *offsetOut, int status) {
  if (done) return false;

  if (write_offset >= 0) {
    write_offset = -1;
    file->Seek(read_offset, File::Whence::SET);
  }

  for (;;) {
    const char *text; int offset;
    if (!(text = nr.NextProto(&offset, &read_offset, hdr))) { done=true; return false; }
    if (!out->ParseFromArray(text, hdr->len)) { done=1; app->run=0; return ERRORv(false, "parse failed, shutting down"); }
    if (status >= 0 && status != hdr->GetFlag()) continue;
    if (offsetOut) *offsetOut = offset;
    return true;
  }
}

int ProtoFile::WriteProto(File *f, const ProtoHeader *hdr, const Proto *msg, bool doflush) {
  std::string v = msg->SerializeAsString();
  CHECK_EQ(hdr->len, v.size());
  v.insert(0, (const char *)hdr, ProtoHeader::size);
  int ret = (f->Write(v.c_str(), v.size()) == v.size()) ? v.size() : -1;
  if (doflush) f->Flush();
  return ret;
}

int ProtoFile::WriteProto(File *f, ProtoHeader *hdr, const Proto *msg, bool doflush) {
  std::string v = msg->SerializeAsString();
  hdr->SetLength(v.size());
  v.insert(0, (const char *)hdr, ProtoHeader::size);
  int ret = (f->Write(v.c_str(), v.size()) == v.size()) ? v.size() : -1;
  if (doflush) f->Flush();
  return ret;
}

int ProtoFile::WriteProtoFlag(File *f, const ProtoHeader *hdr, bool doflush) {
  int ret = f->Write(&hdr->flag, sizeof(int)) == sizeof(int) ? sizeof(int) : -1;
  if (doflush) f->Flush();
  return ret;
}
}; // namespace LFL
