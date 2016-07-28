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

#include "core/app/crypto.h"
#include "curve25519/nacl_includes/crypto_scalarmult_curve25519.h"
#include "ed25519/nacl_includes/crypto_sign.h"

namespace LFL {
void Crypto::X25519DiffieHellman::GeneratePair(std::mt19937 &rand_eng) {
  privkey = RandBytes(crypto_scalarmult_curve25519_BYTES, rand_eng);
  pubkey.resize(crypto_scalarmult_curve25519_BYTES);
  unsigned char g[crypto_scalarmult_curve25519_BYTES] = {9};
  crypto_scalarmult_curve25519(MakeUnsigned(&pubkey[0]), MakeUnsigned(privkey.data()), g);
}

bool Crypto::X25519DiffieHellman::ComputeSecret(BigNum *K) {
  string sharedkey(crypto_scalarmult_curve25519_BYTES, 0);
  if (remotepubkey.size() != crypto_scalarmult_curve25519_BYTES) return false;
  if (sharedkey == remotepubkey) return false;
  crypto_scalarmult_curve25519(MakeUnsigned(&sharedkey[0]), MakeUnsigned(privkey.data()),
                               MakeUnsigned(remotepubkey.data()));
  *K = BigNumSetData(*K, sharedkey);
  sharedkey.assign(sharedkey.size(), 0);
  return true;
}

int ED25519Verify(const StringPiece &digest, const StringPiece &sig, const StringPiece &ed25519_key) {
  if (sig.size() != 64) return false;
  if (ed25519_key.size() != 32) return false;
  string verify = StrCat(sig.str(), digest.str()), result(verify.size(), 0);
  unsigned long long result_size = result.size();
  return 0 == crypto_sign_edwards25519sha512batch_open(MakeUnsigned(&result[0]), &result_size,
                                                       MakeUnsigned(verify.data()), verify.size(),
                                                       MakeUnsigned(ed25519_key.data()));
}

}; // namespace LFL
