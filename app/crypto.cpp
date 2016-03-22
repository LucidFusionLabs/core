/*
 * $Id: crypto.cpp 1335 2014-12-02 04:13:46Z justin $
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

#include "core/app/crypto.h"

namespace LFL {
string Crypto::MD5   (const string &in) { return ComputeDigest(DigestAlgos::MD5   (), in); }
string Crypto::SHA1  (const string &in) { return ComputeDigest(DigestAlgos::SHA1  (), in); }
string Crypto::SHA256(const string &in) { return ComputeDigest(DigestAlgos::SHA256(), in); }

string Crypto::ComputeDigest(DigestAlgo algo, const string &in) {
  Digest d = DigestOpen(algo);
  DigestUpdate(d, in);
  return DigestFinish(d);
}

string ECPointGetData(ECGroup g, ECPoint p, BigNumContext ctx) {
  string ret(ECPointDataSize(g, p, ctx), 0);
  ECPointGetData(g, p, &ret[0], ret.size(), ctx);
  return ret;
}

}; // namespace LFL
