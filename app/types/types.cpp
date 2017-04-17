/*
 * $Id: lftypes.cpp 1334 2014-11-28 09:14:21Z justin $
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

#include "core/app/network.h"

#include <fcntl.h>
#include <sys/stat.h>
#ifndef WIN32
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/resource.h>
#endif

namespace LFL {
void Allocator::Reset() { FATAL("unimplemented reset"); }
Allocator *Allocator::Default() { return Singleton<MallocAllocator>::Get(); }

void *MallocAllocator::Malloc(int size) { return ::malloc(size); }
void *MallocAllocator::Realloc(void *p, int size) { 
  if (!p) return ::malloc(size);
#ifdef __APPLE__
  else return ::reallocf(p, size);
#else
  else return ::realloc(p, size);
#endif
}
void MallocAllocator::Free(void *p) { return ::free(p); }

MMapAllocator::~MMapAllocator() {
#ifdef WIN32
  UnmapViewOfFile(addr);
  CloseHandle(map);
  CloseHandle(file);
#else
  munmap(addr, size);
#endif
}

unique_ptr<MMapAllocator> MMapAllocator::Open(const char *path, bool logerror, bool readonly, long long size) {
#ifdef LFL_ANDROID
  return 0;
#endif
#ifdef WIN32
  HANDLE file = CreateFile(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
  if (file == INVALID_HANDLE_VALUE) { if (logerror) ERROR("CreateFile ", path, " failed ", GetLastError()); return 0; }

  DWORD hsize, lsize=GetFileSize(file, &hsize);

  HANDLE map = CreateFileMapping(file, 0, PAGE_READONLY, 0, 0, 0);
  if (!map) return ERRORv(nullptr, "CreateFileMapping ", path, " failed");

  void *addr = MapViewOfFile(map, readonly ? FILE_MAP_READ : FILE_MAP_COPY, 0, 0, 0);
  if (!addr) return ERRORv(nullptr, "MapViewOfFileEx ", path, " failed ", GetLastError());

  INFO("MMapAllocator::open(", path, ")");
  return make_unique<MMapAllocator>(file, map, addr, lsize);
#else
  int fd = ::open(path, O_RDONLY);
  if (fd < 0) { if (logerror) ERROR("open ", path, " failed: ", strerror(errno)); return nullptr; }

  if (!size) {
    struct stat s;
    if (fstat(fd, &s)) { ERROR("fstat failed: ", strerror(errno)); close(fd); return nullptr; }
    size = s.st_size;
  }

  char *buf = static_cast<char*>(mmap(0, size, PROT_READ | (readonly ? 0 : PROT_WRITE) , MAP_PRIVATE, fd, 0));
  if (buf == MAP_FAILED) { ERROR("mmap failed: ", strerror(errno)); close(fd); return nullptr; }

  close(fd);
  INFO("MMapAllocator::open(", path, ")");
  return make_unique<MMapAllocator>(buf, size);
#endif
}

void *BlockChainAllocator::Malloc(int n) { 
  n = NextMultipleOfPowerOfTwo(n, 16);
  CHECK_LT(n, block_size);
  if (cur_block_ind == -1 || blocks[cur_block_ind].len + n > block_size) {
    cur_block_ind++;
    if (cur_block_ind >= blocks.size()) blocks.emplace_back(block_size);
    CHECK_EQ(blocks[cur_block_ind].len, 0);
    CHECK_LT(n, block_size);
  }
  Block *b = &blocks[cur_block_ind];
  char *ret = &b->buf[b->len];
  b->len += n;
  return ret;
}

/* RingSampler */

void RingSampler::Resize(int SPS, int SPB, int Width) {
  if (SPS != samples_per_sec || SPB != ring.size || Width != width) { 
    ring.size = SPB;
    samples_per_sec = SPS;
    width = Width ? Width : sizeof(float);
    bytes = ring.size * width;
    if (buf) alloc->Free(buf);
    buf = static_cast<char*>(alloc->Malloc(bytes));
    memset(buf, 0, bytes);
    if (stamp) alloc->Free(stamp);
    stamp = static_cast<microseconds*>(alloc->Malloc(ring.size * sizeof(microseconds)));
    memset(stamp, 0, ring.size * sizeof(microseconds));
  }
  ring.back = 0;
}

void *RingSampler::Write(int writeFlag, microseconds timestamp) {
  void *ret = Void(buf + ring.back*width);
  if (writeFlag & Stamp) stamp[ring.back] = timestamp != microseconds(-1) ? timestamp : Now();
  if (!(writeFlag & Peek)) ring.back = ring.Index(1);
  return ret;
}

int RingSampler::Dist(int indexB, int indexE) const { return Since(Bucket(indexB), Bucket(indexE)); }

int RingSampler::Since(int index, int Next) const {
  Next = Next>=0 ? Next : ring.back;
  return (Next < index ? ring.size : 0) + Next - index;
}

void *RingSampler::Read(int index, int Next) const { 
  Next = Next>=0 ? Next : ring.back;
  int ind = Bucket(Next+index);
  return Void(buf + ind * width);
}

microseconds RingSampler::ReadTimestamp(int index, int Next) const { 
  Next = Next>=0 ? Next : ring.back;
  int ind = Bucket(Next+index);
  return stamp[ind];
}

void RingSampler::Handle::CopyFrom(const RingSampler::Handle *src) {
  next=0; int N=Len(), B=0;
  if (N > src->Len()) { B=N-src->Len(); N=src->Len(); }
  for (int i=0; i<N; i++) Write(src->Read(-N+i));
  for (int i=0; i<B; i++) Write(0.0);
}

TableItem::TableItem(string K, int T, string V, string RT, int TG, int LI, int RI, Callback CB, StringCB RC,
                     int F, bool H, PickerItem *P, string DDK, const Color &fg, const Color &bg, float MinV, float MaxV)
  : TableItem(move(K), T, move(V), move(RT), TG, LI, RI, move(CB), move(RC), F, H, P, DDK) {
  minval = MinV;
  maxval = MaxV;
  SetFGColor(fg);
  SetBGColor(bg);
}

void TableItem::SetFGColor(const Color &c) { fg_r=c.R(); fg_g=c.G(); fg_b=c.B(); fg_a=c.A(); }
void TableItem::SetBGColor(const Color &c) { bg_r=c.R(); bg_g=c.G(); bg_b=c.B(); bg_a=c.A(); }

vector<TableSection> TableSection::Convert(vector<TableItem> in) {
  vector<TableSection> ret;
  ret.emplace_back();
  for (auto &i : in) {
    if (i.type == LFL::TableItem::Separator) ret.emplace_back(move(i));
    else                                     ret.back().item.emplace_back(move(i));
  }
  return ret;
}

void TableSection::FindSectionOffset(const vector<TableSection> &data, int collapsed_row, int *section_out, int *row_out) {
  auto it = lower_bound(data.begin(), data.end(), TableSection(collapsed_row),
                        MemberLessThanCompare<TableSection, int, &TableSection::start_row>());
  if (it != data.end() && it->start_row == collapsed_row) { *section_out = it - data.begin(); return; }
  CHECK_NE(data.begin(), it);
  *section_out = (it != data.end() ? (it - data.begin()) : data.size()) - 1;
  *row_out = collapsed_row - data[*section_out].start_row - 1;
}

void TableSection::ApplyChangeList(const TableSection::ChangeList &changes, vector<TableSection> *out, function<void(const TableSection::Change&)> f) {
  for (auto &d : changes) {
    CHECK_LT(d.section, out->size());
    CHECK_LT(d.row, (*out)[d.section].item.size());
    auto &ci = (*out)[d.section].item[d.row];
    ApplyChange(&ci, d);
    if (f) f(d);
  }
}

void TableSection::ApplyChange(TableItem *out, const TableSection::Change &d) {
  if (1)            out->val        = d.val;
  if (1)            out->hidden     = d.hidden;
  if (d.left_icon)  out->left_icon  = d.left_icon  == -1 ? 0 : d.left_icon;
  if (d.right_icon) out->right_icon = d.right_icon == -1 ? 0 : d.right_icon;
  if (d.key.size()) out->key        = d.key;
  if (d.cb)         out->cb         = d.cb;
  if (d.type)       out->type       = d.type;
  if (d.flags)      out->flags      = d.flags;
}

}; // namespace LFL
