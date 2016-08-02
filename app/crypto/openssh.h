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
  OpenSSHKey(const StringPiece &CN=StringPiece(), const StringPiece &KN=StringPiece(),
             const StringPiece &KO=StringPiece(), const StringPiece &PK=StringPiece()) :
    Serializable(0), magic("openssh-key-v1"), ciphername(CN), kdfname(KN), kdfoptions(KO), privatekey(PK) {}

  int HeaderSize() const { return 5*4 + 15; }
  int Size() const {
    int ret = HeaderSize() + ciphername.size() + kdfname.size() + kdfoptions.size() + privatekey.size();
    for (auto &k : publickey) ret += 4 + k.size();
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

  void Out(Serializable::Stream *o) const {
    o->NTString(magic);
    o->BString(ciphername);
    o->BString(kdfname);
    o->BString(kdfoptions);
    o->Htonl(int(publickey.size()));
    for (auto &k : publickey) o->BString(k);
    o->BString(privatekey);
  }
};

struct OpenSSHPrivateKeyHeader : public Serializable {
  int checkint1=0, checkint2=0;
  OpenSSHPrivateKeyHeader() : Serializable(0) {}

  int HeaderSize() const { return 2*4; }
  int Size() const { return HeaderSize(); }

  int In(const Serializable::Stream *i) {
    i->Ntohl(&checkint1);
    i->Ntohl(&checkint2);
    if (checkint1 != checkint2) { i->error = true; return -1; }
    return 0;
  }

  void Out(Serializable::Stream *o) const {
    o->Htonl(checkint1);
    o->Htonl(checkint2);
  }
};

struct OpenSSHEd25519PrivateKey : public Serializable {
  StringPiece keytype, pubkey, privkey, comment;
  OpenSSHEd25519PrivateKey(const StringPiece &kpub=StringPiece(), const StringPiece &kpri=StringPiece(), const StringPiece &C=StringPiece()) :
    Serializable(0), keytype("ssh-ed25519"), pubkey(kpub), privkey(kpri), comment(C) {}

  int HeaderSize() const { return 4*4; }
  int Size() const { return HeaderSize()+keytype.size()+pubkey.size()+privkey.size()+comment.size(); }

  int In(const Serializable::Stream *i) {
    i->ReadString(&keytype);
    if (keytype.str() != "ssh-ed25519") { i->error = true; return -1; }
    i->ReadString(&pubkey);
    if (pubkey.size() != 32) { i->error = true; return -1; }
    i->ReadString(&privkey);
    if (privkey.size() != 64) { i->error = true; return -1; }
    i->ReadString(&comment);
    return 0;
  }

  void Out(Serializable::Stream *o) const {
    o->BString(keytype);
    o->BString(pubkey);
    o->BString(privkey);
    o->BString(comment);
  }
};

struct OpenSSHBCryptKDFOptions : public Serializable {
  StringPiece salt;
  int rounds;
  OpenSSHBCryptKDFOptions(const StringPiece &S=StringPiece(), int R=0) :
    Serializable(0), salt(S), rounds(R) {}

  int HeaderSize() const { return 2*4; }
  int Size() const { return HeaderSize() + salt.size(); }

  int In(const Serializable::Stream *i) { i->ReadString(&salt); i->Ntohl(&rounds); return 0; }
  void Out(Serializable::Stream *o) const { o->BString(salt); o->Htonl(rounds); }
};

}; // namespace LFL
