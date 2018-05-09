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

#include "libarchive/archive.h"
#include "libarchive/archive_entry.h"

namespace LFL {
ArchiveIter::~ArchiveIter() { if (impl) archive_read_finish((archive*)impl); }
ArchiveIter::ArchiveIter(const char *path) {
  if (!(impl = archive_read_new())) return;
  if (archive_read_support_format_zip          ((archive*)impl) != 0) INFO("no zip support");
  if (archive_read_support_format_tar          ((archive*)impl) != 0) INFO("no tar support");
  if (archive_read_support_format_ar           ((archive*)impl) != 0) INFO("no ar support");
  if (archive_read_support_compression_gzip    ((archive*)impl) != 0) INFO("no gzip support");
  if (archive_read_support_compression_none    ((archive*)impl) != 0) INFO("no none support");
  if (archive_read_support_compression_compress((archive*)impl) != 0) INFO("no compress support");
  if (archive_read_open_filename((archive*)impl, path, 65536) != 0) {
    archive_read_finish((archive*)impl);
    impl = nullptr;
  }
}

const char *ArchiveIter::Next() {
  if (!impl) return 0;
  int ret = archive_read_next_header((archive*)impl, (archive_entry**)&entry);
  if (ret) {
    if (const char *errstr = archive_error_string((archive*)impl)) ERROR("read_next: ", ret, " ", errstr);
    return 0;
  }
  return archive_entry_pathname((archive_entry*)entry);
}

bool ArchiveIter::LoadData() {
  buf.resize(Size());
  return buf.size() == archive_read_data((archive*)impl, &buf[0], buf.size());
}

void ArchiveIter::Skip() { archive_read_data_skip((archive*)impl); }
long long ArchiveIter::Size() { return archive_entry_size((archive_entry*)entry); }

}; // namespace LFL
