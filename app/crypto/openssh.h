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

namespace LFL {
struct OpenSSHKey : public Serializable {
  StringPiece magic, ciphername, kdfname, kdfoptions, privatekey;
  vector<StringPiece> publickey;
  OpenSSHKey() : Serializable(0) {}

  int HeaderSize() const { return 5*4 + 15; }
  int Size() const {
    int ret = HeaderSize() + publickey.size() * 4;
    for (auto &k : publickey) ret += k.size();
    return ret;
  }

  int In(const Serializable::Stream *i) {
    int publickeys = 0;
    magic = StringPiece(i->Get(15), 14);
    if (magic.str() != "openssh-key-v1") { i->error = true; return -1; }
    i->ReadString(&ciphername);
    i->ReadString(&kdfname);
    i->ReadString(&kdfoptions);
    i->Ntohl(&publickeys);
    publickey.resize(publickeys);
    for (auto &k : publickey) i->ReadString(&k);
    i->ReadString(&privatekey);
    return 0;
  }

  void Out(Serializable::Stream *o) const {}
};

struct OpenSSHPrivateKeyHeader : public Serializable {
  int checkint1=0, checkint2=0;
  OpenSSHPrivateKeyHeader(int n) : Serializable(0) {}

  int HeaderSize() const { return 2*4; }
  int Size() const { return HeaderSize(); }
  void Out(Serializable::Stream *o) const { o->Htonl(checkint1); o->Htonl(checkint2); }
  int In(const Serializable::Stream *i) {
    i->Ntohl(&checkint1);
    i->Ntohl(&checkint2);
    if (checkint1 != checkint2) { i->error = true; return -1; }
    return 0;
  }
};

struct OpenSSHBCryptKDFOptions : public Serializable {
  StringPiece salt;
  int rounds=0;
  OpenSSHBCryptKDFOptions() : Serializable(0) {}

  int HeaderSize() const { return 2*4; }
  int Size() const { return HeaderSize() + salt.size(); }

  int In(const Serializable::Stream *i) { i->ReadString(&salt); i->Ntohl(&rounds); return 0; }
  void Out(Serializable::Stream *o) const { o->BString(salt); o->Htonl(rounds); }
};

}; // namespace LFL
