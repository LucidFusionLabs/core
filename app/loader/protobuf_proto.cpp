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

#include <google/protobuf/message.h>
#include "core/app/app.h"

namespace LFL {
int ProtoFile::Add(const Proto *msg, int status) {
  return ContainerFile::Add(msg->SerializeAsString(), status);
}

bool ProtoFile::Update(int offset, const ContainerFileHeader *ph, const Proto *msg) {
  return ContainerFile::Update(offset, ph, msg->SerializeAsString());
}

bool ProtoFile::Get(Proto *out, int offset, int status) {
  StringPiece buf;
  if (!ContainerFile::Get(&buf, offset, status)) return false;
  if (!out->ParseFromArray(buf.buf, buf.len)) { done=1; return ERRORv(false, "parse failed"); }
  return true;
}

bool ProtoFile::Next(Proto *out, int *offsetOut, int status) {
  ContainerFileHeader hdr;
  return Next(&hdr, out, offsetOut, status);
}

bool ProtoFile::Next(ContainerFileHeader *hdr, Proto *out, int *offsetOut, int status) {
  StringPiece buf;
  if (!ContainerFile::Next(hdr, &buf, offsetOut, status)) return false;
  if (!out->ParseFromArray(buf.buf, buf.len)) { done=1; return ERRORv(false, "parse failed"); }
  return true;
}

}; // namespace LFL
