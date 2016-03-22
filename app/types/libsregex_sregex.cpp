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

extern "C" {
#include "sregex.h"
};

namespace LFL {
StreamRegex::~StreamRegex() {
  if (ppool) sre_destroy_pool(static_cast<sre_pool_t*>(ppool));
  if (cpool) sre_destroy_pool(static_cast<sre_pool_t*>(cpool));
}

StreamRegex::StreamRegex(const string &patternstr) : ppool(sre_create_pool(1024)), cpool(sre_create_pool(1024)) {
  sre_uint_t ncaps;
  sre_int_t err_offset = -1;
  sre_regex_t *re = sre_regex_parse(static_cast<sre_pool_t*>(cpool),
                                    MakeUnsigned(const_cast<char*>(patternstr.c_str())),
                                    &ncaps, 0, &err_offset);
  prog = sre_regex_compile(static_cast<sre_pool_t*>(ppool), re);
  sre_reset_pool(static_cast<sre_pool_t*>(cpool));
  res.resize(2*(ncaps+1));
  ctx = sre_vm_pike_create_ctx(static_cast<sre_pool_t*>(cpool), static_cast<sre_program_t*>(prog), &res[0], res.size()*sizeof(sre_int_t));
}

int StreamRegex::Match(const string &text, vector<Regex::Result> *out, bool eof) {
  int offset = last_end + since_last_end;
  sre_int_t rc = sre_vm_pike_exec(static_cast<sre_vm_pike_ctx_t*>(ctx),
                                  MakeUnsigned(const_cast<char*>(text.data())), text.size(), eof, NULL);
  if (rc >= 0) {
    since_last_end = 0;
    for (int i = 0, l = res.size(); i < l; i += 2) 
      out->emplace_back(res[i] - offset, (last_end = res[i+1]) - offset);
  } else since_last_end += text.size();
  return 1;
}

}; // namespace LFL
