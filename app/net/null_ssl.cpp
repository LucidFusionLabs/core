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

namespace LFL {
SSLSocket::~SSLSocket() {}
string SSLSocket::ErrorString() const { return ""; }
ptrdiff_t SSLSocket::Write(const StringPiece &b) { return -1; }
ptrdiff_t SSLSocket::Read(char *buf, int readlen) { return -1; }
Socket SSLSocket::Connect(CTXPtr sslctx, const string &hostport) { return InvalidSocket; }
Socket SSLSocket::Connect(CTXPtr sslctx, IPV4::Addr addr, int port) { return InvalidSocket; }
Socket SSLSocket::Listen(int port, bool reuse) { return InvalidSocket; }
Socket SSLSocket::Accept(SSLSocket *out) { return InvalidSocket; }
SSLSocket::CTXPtr SSLSocket::Init() { return nullptr; }
void SSLSocket::Free() {}

}; // namespace LFL
