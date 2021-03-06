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

#include "core/app/network.h"

#include <fcntl.h>
#include <sys/stat.h>
#ifndef WIN32
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/resource.h>
#endif

namespace LFL {
const char *IterWordIter::Next() {
  if (!iter) return 0;
  const char *w = word.in.buf ? word.Next() : 0;
  while (!w) {
    first_count++;
    const char *line = iter->Next();
    if (!line) return 0;
    word = StringWordIter(line, iter->CurrentLength(), word.IsSpace);
    w = word.Next();
  }
  return w;
}    

void Allocator::Reset() { FATAL("unimplemented reset"); }
Allocator *Allocator::Default() { return Singleton<MallocAllocator>::Set(); }

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

string GraphViz::Footer() { return "}\r\n"; }
string GraphViz::DigraphHeader(const string &name) {
  return StrCat("digraph ", name, " {\r\n"
                "rankdir=LR;\r\n"
                "size=\"8,5\"\r\n"
                "node [style = solid];\r\n"
                "node [shape = circle];\r\n");
}

string GraphViz::NodeColor(const string &s) { return StrCat("node [color = ", s, "];\r\n"); }
string GraphViz::NodeShape(const string &s) { return StrCat("node [shape = ", s, "];\r\n"); }
string GraphViz::NodeStyle(const string &s) { return StrCat("node [style = ", s, "];\r\n"); }

void GraphViz::AppendNode(string *out, const string &n1, const string &label) {
  StrAppend(out, "\"", n1, "\"",
            (label.size() ? StrCat(" [ label = \"", label, "\" ] ") : ""),
            ";\r\n");
}

void GraphViz::AppendEdge(string *out, const string &n1, const string &n2, const string &label) {
  StrAppend(out, "\"", n1, "\" -> \"", n2, "\"",
            (label.size() ? StrCat(" [ label = \"", label, "\" ] ") : ""),
            ";\r\n");
}

TableItem::TableItem(string K, int T, string V, string RT, int TG, int LI, int RI, Callback CB, StringCB RC,
                     int F, bool H, PickerItem *P, string DDK, const Color &fg, const Color &bg, float MinV, float MaxV)
  : TableItem(move(K), T, move(V), move(RT), TG, LI, RI, move(CB), move(RC), F, H, P, DDK) {
  minval = MinV;
  maxval = MaxV;
  font.fg = fg;
  font.bg = bg;
}

void TableViewInterface::ApplyChangeSet(const string &v, const TableSectionInterface::ChangeSet &changes) {
  auto it = changes.find(v);
  if (it == changes.end()) return ERROR("Missing TableView ChangeSet ", v);
  ApplyChangeList(it->second);
}

void TableSectionInterface::ApplyChange(TableItem *out, const TableSectionInterface::Change &d) {
  if (1)            out->val        = d.val;
  if (1)            out->hidden     = d.hidden;
  if (1)            out->flags      = d.flags;
  if (d.left_icon)  out->left_icon  = d.left_icon  == -1 ? 0 : d.left_icon;
  if (d.right_icon) out->right_icon = d.right_icon == -1 ? 0 : d.right_icon;
  if (d.key.size()) out->key        = d.key;
  if (d.cb)         out->cb         = d.cb;
  if (d.type)       out->type       = d.type;
}

}; // namespace LFL
