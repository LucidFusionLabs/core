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

namespace LFL {
void ProtoFile::Open(const char *fn) {}
int ProtoFile::Add(const Proto *msg, int status) { return -1; }
bool ProtoFile::Update(int offset, const ProtoHeader *ph, const Proto *msg) { return false; }
bool ProtoFile::Update(int offset, int status) { return false; }
bool ProtoFile::Get(Proto *out, int offset, int status) { return false; }
bool ProtoFile::Next(Proto *out, int *offsetOut, int status) { return false; }
bool ProtoFile::Next(ProtoHeader *hdr, Proto *out, int *offsetOut, int status) { return false; }
int ProtoFile::WriteProto(File *f, const ProtoHeader *hdr, const Proto *msg, bool doflush) { return -1; }
int ProtoFile::WriteProto(File *f, ProtoHeader *hdr, const Proto *msg, bool doflush) { return -1; }
int ProtoFile::WriteProtoFlag(File *f, const ProtoHeader *hdr, bool doflush) { return -1; }

}; // namespace LFL
