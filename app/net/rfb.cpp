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
#include "core/app/network.h"
#include "core/app/net/rfb.h"

#define LFL_RFB_DEBUG
#ifdef LFL_RFB_DEBUG
#define RFBTrace(...) INFO(__VA_ARGS__)
#else
#define RFBTrace(...)
#endif

namespace LFL {
struct RFB {
  struct ProtocolVersion : public Serializable {
    int major, minor;
    ProtocolVersion(int A=0, int B=0) : Serializable(0), major(A), minor(B) {}
 
    int HeaderSize() const { return 12; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { o->String(StringPrintf("RFB %03d.%03d\n", major, minor)); }
    int In(const Serializable::Stream *i) {
      string in(i->Get(Size()), Size()); 
      if (!PrefixMatch(in, "RFB ") || in[7] != '.' || in[11] != '\n') { i->error = true; return -1; }
      major = atoi(in.data() + 4);
      minor = atoi(in.data() + 8);
      return i->Result();
    }
  };
  
  struct SecurityTypeList : public Serializable {
    enum { Invalid=0, None=1, VNC=2 };
    StringPiece security_type, error;
    SecurityTypeList() : Serializable(0) {}
 
    int HeaderSize() const { return 1; }
    int Size() const { return HeaderSize() + security_type.size(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) {
      unsigned char len=0;
      i->Read8(&len);
      if (!len) i->ReadString(&error);
      else security_type = StringPiece(i->Get(len), len);
      return i->Result();
    }
  };

  struct WhichSecurityType : public Serializable {
    unsigned char st;
    WhichSecurityType(unsigned char v=0) : Serializable(0), st(v) {}
    int HeaderSize() const { return 1; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { o->Write8(st); }
    int In(const Serializable::Stream *i) { i->Read8(&st); return i->Result(); }
  };

  struct VNCAuthenticationChallenge : public Serializable {
    StringPiece text;
    VNCAuthenticationChallenge(const StringPiece &T=StringPiece()) : Serializable(0), text(T) {}
    int HeaderSize() const { return 16; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { CHECK_EQ(Size(), text.size()); o->String(text); }
    int In(const Serializable::Stream *i) { text = StringPiece(i->Get(Size()), Size()); return i->Result(); }
  };

  struct SecurityResult : public Serializable {
    int error=0;
    SecurityResult() : Serializable(0) {}
    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize(); }
    int In(const Serializable::Stream *i) { i->Ntohl(&error); return i->Result(); }
    void Out(Serializable::Stream *o) const {}
  };

  struct ReasonString : public Serializable {
    StringPiece text;
    ReasonString() : Serializable(0) {}
    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize() + text.size(); }
    int In(const Serializable::Stream *i) { i->ReadString(&text); return i->Result(); }
    void Out(Serializable::Stream *o) const {}
  };
}; 
  
struct RFBClientConnection : public Connection::Handler {
  enum { HANDSHAKE=0, SECURITY_HANDSHAKE=1, VNC_AUTH=2, SECURITY_RESULT=3, AUTH_FAILED=4, INIT=5 };
  RFBClient::Params params;
  RFBClient::LoadPasswordCB load_password_cb;
  RFBClient::UpdateCB update_cb;
  Callback success_cb;
  int state=0;
  uint8_t security_type=0;
  RFBClientConnection(RFBClient::Params p, RFBClient::LoadPasswordCB PCB, RFBClient::UpdateCB UCB, Callback SCB) :
    params(move(p)), load_password_cb(move(PCB)), update_cb(move(UCB)), success_cb(move(SCB)) {}

  void Close(Connection *c) { update_cb(c, StringPiece()); }
  int Connected(Connection *c) { return state == HANDSHAKE ? 0 : -1; }

  int Read(Connection *c) {
    int total_processed = 0, processed = 1;
    for (/**/; processed; total_processed += processed) {
      Serializable::ConstStream msg_s(c->rb.buf.data() + total_processed, c->rb.size() - total_processed);
      processed = 0;

      switch (state) {
        case HANDSHAKE: {
          RFB::ProtocolVersion msg;
          if (c->rb.size() < total_processed + msg.Size()) break;
          if (msg.In(&msg_s)) return ERRORv(-1, c->Name(), ": read ProtocolVersion");
          RFBTrace(c->Name(), ": ProtocolVersion ", msg.major, ".", msg.minor);

          if (!Write(c, RFB::ProtocolVersion(3, 8))) return ERRORv(-1, c->Name(), ": write");
          processed = msg.Size();
          state++;
        } break;

        case SECURITY_HANDSHAKE: {
          RFB::SecurityTypeList msg;
          if (c->rb.size() < total_processed + msg.Size()) break;
          if (msg.In(&msg_s)) {
            if (c->rb.size() < total_processed + msg.Size()) break;
            return ERRORv(-1, c->Name(), ": read SecurityTypeList");
          }
          if (!msg.security_type.size()) return ERRORv(-1, c->Name(), ": empty SecurityTypeList: ", msg.error.str());
          RFBTrace(c->Name(), ": SecurityTypeList ", msg.security_type.size());

          security_type = 0;
          for (int i=0, l=msg.security_type.size(); i != l; ++i) {
            auto st = msg.security_type.buf[i];
            if      (st == RFB::SecurityTypeList::None) { security_type = st; break; }
            else if (st == RFB::SecurityTypeList::VNC)  { security_type = st; }
          }
          if      (security_type == RFB::SecurityTypeList::VNC)  state = VNC_AUTH;
          else if (security_type == RFB::SecurityTypeList::None) state = SECURITY_RESULT;
          else return ERRORv(-1, c->Name(), ": invalid SecurityType ", security_type);

          if (!Write(c, RFB::WhichSecurityType(security_type))) return ERRORv(-1, c->Name(), ": write");
          processed = msg.Size();
        } break;

        case VNC_AUTH: {
          RFB::VNCAuthenticationChallenge msg;
          if (c->rb.size() < total_processed + msg.Size()) break;
          if (msg.In(&msg_s)) return ERRORv(-1, c->Name(), ": read VNCAuthenticationChallenge");
          RFBTrace(c->Name(), ": VNCAuthenticationChallenge ", HexEscape(msg.text.str(), ""));

          string pw="", response(16, 0);
          if (load_password_cb) load_password_cb(&pw);
          pw.resize(8, 0);
          for (auto i = MakeUnsigned(&pw[0]), e = i + pw.size(); i != e; ++i) *i = BitString::Reverse(*i);

          Crypto::Cipher enc = Crypto::CipherInit();
          Crypto::CipherOpen(enc, Crypto::CipherAlgos::DES_ECB(), true, pw, string(8, 0));
          int len = Crypto::CipherUpdate(enc, msg.text, &response[0], response.size());
          len += Crypto::CipherFinal(enc, &response[0+len], response.size()-len);
          Crypto::CipherFree(enc);
          if (len != 16) return ERRORv(-1, c->Name(), ": invalid challenge response len=", len);

          if (!Write(c, RFB::VNCAuthenticationChallenge(response))) return ERRORv(-1, c->Name(), ": write");
          state = SECURITY_RESULT;
          processed = msg.Size();
        } break;

        case SECURITY_RESULT: {
          RFB::SecurityResult msg;
          if (c->rb.size() < total_processed + msg.Size()) break;
          if (msg.In(&msg_s)) return ERRORv(-1, c->Name(), ": read SecurityResult");
          RFBTrace(c->Name(), ": SecurityResult ", msg.error);
          state = msg.error ? AUTH_FAILED : INIT;
          processed = msg.Size();
        } break;

        case AUTH_FAILED: {
          RFB::ReasonString msg;
          if (c->rb.size() < total_processed + msg.Size()) break;
          if (msg.In(&msg_s)) {
            if (c->rb.size() < total_processed + msg.Size()) break;
            return ERRORv(-1, c->Name(), ": read SecurityTypeList");
          }
          INFO(c->Name(), ": vnc auth failed: ", msg.text.str());
        } break;

        case INIT: {
        } break;
      }
    }

    c->ReadFlush(total_processed);
    return 0;
  }

  bool Write(Connection *c, const Serializable &m) {
    string text = m.ToString();
    return c->WriteFlush(text) == text.size();
  }
};

Connection *RFBClient::Open(Params p, RFBClient::LoadPasswordCB pcb, RFBClient::UpdateCB ucb,
                            Callback *detach, Callback *success) { 
  Connection *c = app->net->tcp_client->Connect(p.hostport, 5900, detach);
  if (!c) return 0;
  c->handler = make_unique<RFBClientConnection>(move(p), move(pcb), move(ucb),
                                                move(success ? *success : Callback()));
  return c;
}

}; // namespace LFL
