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

#include "core/app/app.h"
#include "core/app/crypto.h"

namespace LFL {
void FreeECPoint(ECPoint p) { FATAL(not_implemented); }
void FreeECPair(ECPair p) { FATAL(not_implemented); }
int ECPointDataSize(const ECGroup g, const ECPoint p, BigNumContext x) { FATAL(not_implemented); }
void ECPointGetData(const ECGroup g, const ECPoint p, char *out, int len, BigNumContext x) { FATAL(not_implemented) }
void ECPointSetData(const ECGroup g, ECPoint v, const StringPiece &data) { FATAL(not_implemented); }

RSAKey NewRSAPubKey() { return nullptr; }
DSAKey NewDSAPubKey() { return nullptr; }
DSASig NewDSASig()    { return nullptr; }
ECDSASig NewECDSASig() { return nullptr; }
BigNum GetRSAKeyE(RSAKey k) { return nullptr; }
BigNum GetRSAKeyN(RSAKey k) { return nullptr; }
BigNum GetDSAKeyP(DSAKey k) { return nullptr; }
BigNum GetDSAKeyQ(DSAKey k) { return nullptr; }
BigNum GetDSAKeyG(DSAKey k) { return nullptr; }
BigNum GetDSAKeyK(DSAKey k) { return nullptr; }
BigNum GetDSASigR(DSASig k) { return nullptr; }
BigNum GetDSASigS(DSASig k) { return nullptr; }
BigNum GetECDSASigR(ECDSASig k) { return nullptr; }
BigNum GetECDSASigS(ECDSASig k) { return nullptr; }
void RSAKeyFree(RSAKey k) { FATAL(not_implemented); }
void DSAKeyFree(DSAKey k) { FATAL(not_implemented); }
void DSASigFree(DSASig s) { FATAL(not_implemented); }
void ECDSASigFree(ECDSASig s) { FATAL(not_implemented); }

int RSAGeneratePair(RSAKey rsa_key, int bits) { return 0; }
int RSAGeneratePair(RSAKey key, int bits, BigNum e) { return 0; }
int DSAGeneratePair(DSAKey key, int bits) { return 0; }

string RSAOpenSSHPublicKey(RSAKey key, const string &comment) { return ""; }
string DSAOpenSSHPublicKey(DSAKey key, const string &comment) { return ""; }
string ECDSAOpenSSHPublicKey(ECPair key, const string &comment) { return ""; }
string RSAPEMPublicKey(RSAKey key) { return ""; }
string DSAPEMPublicKey(DSAKey key) { return ""; }
string ECDSAPEMPublicKey(ECPair key) { return ""; }
string RSAPEMPrivateKey(RSAKey key, string pw) { return ""; }
string DSAPEMPrivateKey(DSAKey key, string pw) { return ""; }
string ECDSAPEMPrivateKey(ECPair key, string pw) { return ""; }
  
void Crypto::PublicKeyInit() {}
ECDef Crypto::EllipticCurve::NISTP256() { return nullptr; }
ECDef Crypto::EllipticCurve::NISTP384() { return nullptr; }
ECDef Crypto::EllipticCurve::NISTP521() { return nullptr; }
ECPair Crypto::EllipticCurve::NewPair(ECDef id, bool generate) { return nullptr; }

int Ed25519GeneratePair(Ed25519Pair *key, std::mt19937 &rand_eng) { return 0; }
string Ed25519OpenSSHPublicKey(const Ed25519Pair &key, const string &comment) { return ""; }
string Ed25519PEMPrivateKey(const Ed25519Pair &key, const string &pw, const string &comment, int checkint) { return ""; }

}; // namespace LFL
