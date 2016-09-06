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

#ifdef LFL_RFB_DEBUG
#define RFBTrace(...) INFO(__VA_ARGS__)
#else
#define RFBTrace(...)
#endif

namespace LFL {
struct RFB {
  struct ByteType : public Serializable {
    unsigned char v;
    ByteType(unsigned char V=0) : Serializable(0), v(V) {}
    int HeaderSize() const { return 1; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { o->Write8(v); }
    int In(const Serializable::Stream *i) { i->Read8(&v); return i->Result(); }
  };

  struct IntType : public Serializable {
    int v=0;
    IntType(int V=0) : Serializable(0) {}
    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize(); }
    int In(const Serializable::Stream *i) { i->Ntohl(&v); return i->Result(); }
    void Out(Serializable::Stream *o) const { o->Htonl(v); }
  };

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

  struct WhichSecurityType : ByteType {
    using ByteType::ByteType;
  };

  struct VNCAuthenticationChallenge : public Serializable {
    StringPiece text;
    VNCAuthenticationChallenge(const StringPiece &T=StringPiece()) : Serializable(0), text(T) {}
    int HeaderSize() const { return 16; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { CHECK_EQ(Size(), text.size()); o->String(text); }
    int In(const Serializable::Stream *i) { text = StringPiece(i->Get(Size()), Size()); return i->Result(); }
  };

  struct SecurityResult : public IntType {
    using IntType::IntType;
  };

  struct ReasonString : public Serializable {
    StringPiece text;
    ReasonString() : Serializable(0) {}
    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize() + text.size(); }
    int In(const Serializable::Stream *i) { i->ReadString(&text); return i->Result(); }
    void Out(Serializable::Stream *o) const { o->BString(text); }
  };

  struct ClientInit : ByteType {
    using ByteType::ByteType;
  };
  
  struct PixelFormat : public Serializable {
    uint8_t bits_per_pixel, depth, big_endian_flag, true_color_flag, red_shift, green_shift, blue_shift;
    uint16_t red_max, green_max, blue_max;
    PixelFormat() : Serializable(0) {}

    string DebugString() const {
      return StrCat("PF{ bpp=", int(bits_per_pixel), ", depth=", int(depth),
                    ", flags=", int(big_endian_flag), ",", int(true_color_flag), " }");
    }

    string RGBParamString() const {
      return StrCat("{ red_shift=", int(red_shift), ", red_max=", int(red_max), ", green_shift=", int(green_shift),
                    ", green_max = ", int(green_max), ", blue_shift=", int(blue_shift), ", blue_max=", int(blue_max), "}");
    }

    int HeaderSize() const { return 16; }
    int Size() const { return HeaderSize(); }
    int In(const Serializable::Stream *i) {
      i->Read8(&bits_per_pixel);   i->Read8(&depth);
      i->Read8(&big_endian_flag);  i->Read8(&true_color_flag);
      i->Ntohs(&red_max);          i->Ntohs(&green_max);        i->Ntohs(&blue_max);
      i->Read8(&red_shift);        i->Read8(&green_shift);      i->Read8(&blue_shift);
      i->Get(3);
      return i->Result();
    }

    void Out(Serializable::Stream *o) const {
      o->Write8(bits_per_pixel);   o->Write8(depth);
      o->Write8(big_endian_flag);  o->Write8(true_color_flag);
      o->Htons(red_max);           o->Htons(green_max);         o->Htons(blue_max);
      o->Write8(red_shift);        o->Write8(green_shift);      o->Write8(blue_shift);
      o->String(string(3, 0));
    }
  };

  struct ServerInit : public Serializable {
    uint16_t fb_w, fb_h;
    PixelFormat pixel_format;
    StringPiece name;
    ServerInit() : Serializable(0) {}

    int HeaderSize() const { return 8 + pixel_format.HeaderSize(); }
    int Size() const { return HeaderSize() + name.size(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) {
      i->Ntohs(&fb_w);
      i->Ntohs(&fb_h);
      pixel_format.In(i);
      i->ReadString(&name);
      return i->Result();
    }
  };

  struct FramebufferUpdateRequest : public Serializable {
    static const int ID = 3;
    uint8_t incremental;
    uint16_t x, y, w, h;
    FramebufferUpdateRequest(const Box &b=Box(), bool inc=1) :
      Serializable(ID), incremental(inc), x(b.x), y(b.y), w(b.w), h(b.h) {}

    int HeaderSize() const { return 10; }
    int Size() const { return HeaderSize(); }
    int In(const Serializable::Stream *i) { return i->Result(); }
    void Out(Serializable::Stream *o) const {
      o->Write8(uint8_t(ID));  o->Write8(incremental);
      o->Htons(x);             o->Htons(y);
      o->Htons(w);             o->Htons(h);
    }
  };

  struct KeyEvent : public Serializable {
    static const int ID = 4;
    uint32_t key;
    uint8_t down;
    KeyEvent(uint32_t k=0, uint8_t d=0) : Serializable(ID), key(k), down(d) {}

    int HeaderSize() const { return 8; }
    int Size() const { return HeaderSize(); }
    int In(const Serializable::Stream *i) { return i->Result(); }
    void Out(Serializable::Stream *o) const {
      o->Write8(uint8_t(ID));  o->Write8(down);
      o->Write16(uint16_t(0)); o->Htonl(key);
    }
  };

  struct PointerEvent : public Serializable {
    static const int ID = 5;
    uint16_t x, y;
    uint8_t buttons;
    PointerEvent(uint16_t X=0, uint16_t Y=0, uint8_t B=0) : Serializable(ID), x(X), y(Y), buttons(B) {}

    int HeaderSize() const { return 6; }
    int Size() const { return HeaderSize(); }
    int In(const Serializable::Stream *i) { return i->Result(); }
    void Out(Serializable::Stream *o) const {
      o->Write8(uint8_t(ID));  o->Write8(buttons);
      o->Htons(x);             o->Htons(y);
    }
  };

  struct ClientCutText : public Serializable {
    static const int ID = 6;
    StringPiece text;
    ClientCutText(const StringPiece &T=StringPiece()) : Serializable(ID), text(T) {}

    int HeaderSize() const { return 8; }
    int Size() const { return HeaderSize() + text.size(); }
    int In(const Serializable::Stream *i) { return i->Result(); }
    void Out(Serializable::Stream *o) const {
      o->Write8(uint8_t(ID));  o->Write16(uint16_t(0));
      o->Write8(uint8_t(0));   o->BString(text);
    }
  };

  struct PixelData : public Serializable {
    enum { Raw=0, CopyRect=1, RRE=2, Hextile=5, TRLE=15, ZRLE=16, DesktopSizePE=-223, CursorPE=-239 };
    const PixelFormat *pf;
    uint16_t x, y, w, h;
    int encoding;
    StringPiece data;
    PixelData(const PixelFormat *p) : Serializable(0), pf(p) {}

    string DebugString() const { return StrCat("{ x=", x, ", y=", y, ", w=", w, ", h=", h, ", encoding=", encoding, " }"); }
    int HeaderSize() const { return 12; }
    int Size() const { return HeaderSize() + data.size(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) {
      i->Ntohs(&x);  i->Ntohs(&y);
      i->Ntohs(&w);  i->Ntohs(&h);  i->Ntohl(&encoding);
      if (encoding == Raw) data.len = w * h * pf->bits_per_pixel/8;
      else return ERRORv(-1, "unknown encoding: ", encoding);
      data = StringPiece(i->Get(data.len), data.len);
      return i->Result();
    }
  };

  struct FramebufferUpdate : public Serializable {
    static const int ID = 0;
    const PixelFormat *pf;
    vector<PixelData> rect;
    FramebufferUpdate(const PixelFormat *p) : Serializable(ID), pf(p) {}

    int HeaderSize() const { return 3; }
    int Size() const { int v=HeaderSize(); for (auto &r : rect) v += r.Size(); return v; }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) {
      i->Get(1);
      uint16_t rects = 0;
      i->Ntohs(&rects);
      rect.clear();
      for (uint16_t j = 0; !i->error && j != rects; ++j) {
        rect.emplace_back(pf);
        PixelData &r = rect.back();
        if (i->Remaining() < r.HeaderSize()) { i->error=true; return -1; }
        r.In(i);
      }
      return i->Result();
    }
  };

  struct SetColorMapEntries {
    static const int ID = 1;
  };

  struct Bell {
    static const int ID = 2;
  };

  struct ServerCutText : public Serializable {
    static const int ID = 3;
    StringPiece text;
    ServerCutText(const StringPiece &T=StringPiece()) : Serializable(ID), text(T) {}

    int HeaderSize() const { return 7; }
    int Size() const { return HeaderSize() + text.size(); }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) {
      uint16_t padding1=0;
      uint8_t padding2=0;
      i->Read16(&padding1);
      i->Read8(&padding2);
      i->ReadString(&text);
      return i->Result();
    }
  };
};

struct RFBClientConnection : public Connection::Handler {
  enum { HANDSHAKE=0, SECURITY_HANDSHAKE=1, VNC_AUTH=2, SECURITY_RESULT=3, AUTH_FAILED=4, INIT=5, RUN=6 };
  RFBClient::Params params;
  RFBClient::LoadPasswordCB load_password_cb;
  RFBClient::UpdateCB update_cb;
  Callback success_cb;
  int state=0, fb_w=0, fb_h=0;
  uint8_t security_type=0;
  bool share_desktop=1;
  RFB::PixelFormat pixel_format;

  RFBClientConnection(RFBClient::Params p, RFBClient::LoadPasswordCB PCB, RFBClient::UpdateCB UCB, Callback SCB) :
    params(move(p)), load_password_cb(move(PCB)), update_cb(move(UCB)), success_cb(move(SCB)) {}

  void Close(Connection *c) { update_cb(c, Box(), 0, StringPiece()); }
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
          RFBTrace(c->Name(), ": SecurityResult ", msg.v);
          state = msg.v ? AUTH_FAILED : INIT;
          processed = msg.Size();
          if (state == INIT) { if (!Write(c, RFB::ClientInit(share_desktop))) return ERRORv(-1, c->Name(), ": write"); }
        } break;

        case AUTH_FAILED: {
          RFB::ReasonString msg;
          if (c->rb.size() < total_processed + msg.Size()) break;
          if (msg.In(&msg_s)) {
            if (c->rb.size() < total_processed + msg.Size()) break;
            return ERRORv(-1, c->Name(), ": read SecurityTypeList");
          }
          INFO(c->Name(), ": vnc auth failed: ", msg.text.str());
          processed = msg.Size();
        } break;

        case INIT: {
          RFB::ServerInit msg;
          if (c->rb.size() < total_processed + msg.Size()) break;
          if (msg.In(&msg_s)) {
            if (c->rb.size() < total_processed + msg.Size()) break;
            return ERRORv(-1, c->Name(), ": read SecurityTypeList");
          }
          RFBTrace(c->Name(), ": ServerInit ", msg.name.str(), " w=", msg.fb_w, ", h=", msg.fb_h, 
                   " ", msg.pixel_format.DebugString(), " ", msg.pixel_format.RGBParamString());

          if (!msg.pixel_format.true_color_flag) return ERRORv(-1, c->Name(), ": only true-color supported");
          if (!Write(c, RFB::FramebufferUpdateRequest(Box(msg.fb_w, msg.fb_h), false))) return ERRORv(-1, c->Name(), ": write");
          processed = msg.Size();
          state = RUN;
          fb_w = msg.fb_w;
          fb_h = msg.fb_h;
          pixel_format = msg.pixel_format;
          c->rb.Resize(max(c->rb.size(), 2 * fb_w * fb_h * pixel_format.bits_per_pixel/8));
          int pf = GetPixelFormat(RFB::PixelData::Raw, pixel_format.bits_per_pixel);
          if (!pf) return ERRORv(-1, c->Name(), ": unknown pixel format ", RFB::PixelData::Raw, " ", pixel_format.bits_per_pixel);
          update_cb(c, Box(fb_w, fb_h), pf, StringPiece());
        } break;
        
        case RUN: {
          unsigned char msg_type = 0;
          if (c->rb.size() < total_processed + 1) break;
          msg_s.Read8(&msg_type);

          if (msg_type == RFB::FramebufferUpdate::ID) {
            RFB::FramebufferUpdate msg(&pixel_format);
            if (c->rb.size() < total_processed + msg.Size()) break;
            if (msg.In(&msg_s)) {
              if (c->rb.size() < total_processed + msg.Size()) break;
              return ERRORv(-1, c->Name(), ": read FramebufferUpdate");
            }
            RFBTrace(c->Name(), ": FramebufferUpdate rects=", msg.rect.size());
            processed = msg.Size() + 1;
            for (auto &r : msg.rect) {
              if (!r.w && !r.h) continue;
              int pf = GetPixelFormat(r.encoding, pixel_format.bits_per_pixel);
              if (pf) update_cb(c, Box(r.x, fb_h - r.y - r.h, r.w, r.h), pf, r.data);
              else ERROR("unknown encoding ", r.encoding, " ", pixel_format.bits_per_pixel);
            }
            if (!Write(c, RFB::FramebufferUpdateRequest(Box(fb_w, fb_h), true))) return ERRORv(-1, c->Name(), ": write");

          } else if (msg_type == RFB::SetColorMapEntries::ID) {
            return ERRORv(-1, c->Name(), ": SetColorMapEntries not supported");

          } else if (msg_type == RFB::Bell::ID) {
            return ERRORv(-1, c->Name(), ": Bell not supported");

          } else if (msg_type == RFB::ServerCutText::ID) {
            return ERRORv(-1, c->Name(), ": ServerCutText not supported");

          } else return ERRORv(-1, c->Name(), ": unknown message ", int(msg_type));
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

  static int GetPixelFormat(int encoding, int bits_per_pixel) {
    int pf = 0;
    if (encoding == RFB::PixelData::Raw) {
      if      (bits_per_pixel == 32) pf = Pixel::BGR32;
      else if (bits_per_pixel == 24) pf = Pixel::BGR24;
    }
    return pf;
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

int RFBClient::WriteKeyEvent(Connection *c, uint32_t key, uint8_t down) {
  if (!dynamic_cast<RFBClientConnection*>(c->handler.get())->Write(c, RFB::KeyEvent(key, down)))
  { c->SetError(); return ERRORv(-1, c->Name(), ": write"); }
  return 0;
}

int RFBClient::WritePointerEvent(Connection *c, uint16_t x, uint16_t y, uint8_t buttons) {
  if (!dynamic_cast<RFBClientConnection*>(c->handler.get())->Write(c, RFB::PointerEvent(x, y, buttons)))
  { c->SetError(); return ERRORv(-1, c->Name(), ": write"); }
  return 0;
}

int RFBClient::WriteClientCutText(Connection *c, const StringPiece &text) {
  if (!dynamic_cast<RFBClientConnection*>(c->handler.get())->Write(c, RFB::ClientCutText(text)))
  { c->SetError(); return ERRORv(-1, c->Name(), ": write"); }
  return 0;
}

}; // namespace LFL
