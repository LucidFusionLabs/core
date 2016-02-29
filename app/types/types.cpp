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

#include "core/app/app.h"
#include "core/app/network.h"

#include <fcntl.h>
#include <sys/stat.h>
#ifdef WIN32
#include <Shlobj.h>
#else
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

/* RingBuf */

void RingBuf::Resize(int SPS, int SPB, int Width) {
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

void *RingBuf::Write(int writeFlag, microseconds timestamp) {
  void *ret = Void(buf + ring.back*width);
  if (writeFlag & Stamp) stamp[ring.back] = timestamp != microseconds(-1) ? timestamp : Now();
  if (!(writeFlag & Peek)) ring.back = ring.Index(1);
  return ret;
}

int RingBuf::Dist(int indexB, int indexE) const { return Since(Bucket(indexB), Bucket(indexE)); }

int RingBuf::Since(int index, int Next) const {
  Next = Next>=0 ? Next : ring.back;
  return (Next < index ? ring.size : 0) + Next - index;
}

void *RingBuf::Read(int index, int Next) const { 
  Next = Next>=0 ? Next : ring.back;
  int ind = Bucket(Next+index);
  return Void(buf + ind * width);
}

microseconds RingBuf::ReadTimestamp(int index, int Next) const { 
  Next = Next>=0 ? Next : ring.back;
  int ind = Bucket(Next+index);
  return stamp[ind];
}

void RingBuf::Handle::CopyFrom(const RingBuf::Handle *src) {
  next=0; int N=Len(), B=0;
  if (N > src->Len()) { B=N-src->Len(); N=src->Len(); }
  for (int i=0; i<N; i++) Write(src->Read(-N+i));
  for (int i=0; i<B; i++) Write(0.0);
}
}; // namespace LFL
