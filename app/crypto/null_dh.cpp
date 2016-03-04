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
BigNum        NewBigNum        ()                { FATAL("not implemented"); }
BigNumContext NewBigNumContext ()                { FATAL("not implemented"); }
void          FreeBigNumContext(BigNumContext c) { FATAL("not implemented"); }
void          FreeBigNum       (BigNum        n) { FATAL("not implemented"); }
void BigNumSetValue(BigNum v, int val)        { FATAL("not implemented"); }
void BigNumGetData(const BigNum v, char *out) { FATAL("not implemented"); }
int  BigNumDataSize       (const BigNum v)    { FATAL("not implemented"); }
int  BigNumSignificantBits(const BigNum v)    { FATAL("not implemented"); }
void BigNumModExp(BigNum v, const BigNum a, const BigNum e, const BigNum m, BigNumContext) { FATAL("not implemented"); }
BigNum BigNumSetData(BigNum v, const StringPiece &data)       { FATAL("not implemented"); }
BigNum BigNumRand   (BigNum v, int bits, int top, int bottom) { FATAL("not implemented"); }
void FreeECPoint(ECPoint p) { FATAL("not implemented"); }
void FreeECPair(ECPair p) { FATAL("not implemented"); }
int ECPointDataSize(const ECGroup g, const ECPoint p, BigNumContext x) { FATAL("not implemented"); }
void ECPointGetData(const ECGroup g, const ECPoint p, char *out, int len, BigNumContext x) { FATAL("not implemented") }
void ECPointSetData(const ECGroup g, ECPoint v, const StringPiece &data) { FATAL("not implemented"); }

}; // namespace LFL
