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

namespace LFL {
int ProtoFile::Add(const Proto *msg, int status) { return -1; }
bool ProtoFile::Update(int offset, const ContainerFileHeader *ph, const Proto *msg) { return false; }
bool ProtoFile::Get(Proto *out, int offset, int status) { return false; }
bool ProtoFile::Next(Proto *out, int *offsetOut, int status) { return false; }
bool ProtoFile::Next(ContainerFileHeader *hdr, Proto *out, int *offsetOut, int status) { return false; }

}; // namespace LFL
