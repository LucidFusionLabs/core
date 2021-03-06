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
    int ver_major, ver_minor;
    ProtocolVersion(int A=0, int B=0) : Serializable(0), ver_major(A), ver_minor(B) {}
 
    int HeaderSize() const { return 12; }
    int Size() const { return HeaderSize(); }
    void Out(Serializable::Stream *o) const { o->String(StringPrintf("RFB %03d.%03d\n", ver_major, ver_minor)); }
    int In(const Serializable::Stream *i) {
      string in(i->Get(Size()), Size()); 
      if (!PrefixMatch(in, "RFB ") || in[7] != '.' || in[11] != '\n') { i->error = true; return -1; }
      ver_major = atoi(in.data() + 4);
      ver_minor = atoi(in.data() + 8);
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

  struct SetPixelFormat : public Serializable {
    static const int ID = 0;
    PixelFormat pixel_format;
    SetPixelFormat() : Serializable(ID) {}

    int HeaderSize() const { return 4 + pixel_format.HeaderSize(); }
    int Size() const { return 4 + pixel_format.Size(); }
    int In(const Serializable::Stream *i) { return i->Result(); }
    void Out(Serializable::Stream *o) const
    { o->Write8(uint8_t(ID)); o->String(string(3, 0)); pixel_format.Out(o); }
  };

  struct SetEncodings : public Serializable {
    static const int ID = 2;
    vector<int> encodings;
    SetEncodings(vector<int> E=vector<int>()) : Serializable(ID), encodings(move(E)) {}

    int HeaderSize() const { return 4; }
    int Size() const { return HeaderSize() + encodings.size() * 4; }
    int In(const Serializable::Stream *i) { return i->Result(); }
    void Out(Serializable::Stream *o) const {
      o->Write8(uint8_t(ID));
      o->Write8(uint8_t(0));
      o->Htons(uint16_t(encodings.size()));
      for (auto &e : encodings) o->Htonl(e);
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
    KeyEvent(uint32_t k=0, uint8_t d=0) : Serializable(ID), down(d) {
      static const unordered_map<uint32_t, uint32_t> keymap{
      { Key::Backspace,  0xff08 }, { Key::Tab,        0xff09 }, { Key::Return,     0xff08 },
      { Key::Escape,     0xff1b }, { Key::Insert,     0xff63 }, { Key::Delete,     0xffff },
      { Key::Home,       0xff50 }, { Key::End,        0xff57 }, { Key::PageUp,     0xff55 },
      { Key::PageDown,   0xff56 }, { Key::Left,       0xff51 }, { Key::Up,         0xff52 },
      { Key::Right,      0xff53 }, { Key::Down,       0xff54 }, { Key::F1,         0xffbe },
      { Key::F2,         0xffbf }, { Key::F3,         0xffc0 }, { Key::F4,         0xffc1 },
      { Key::F5,         0xffc2 }, { Key::F6,         0xffc3 }, { Key::F7,         0xffc4 },
      { Key::F8,         0xffc5 }, { Key::F9,         0xffc6 }, { Key::F10,        0xffc7 },
      { Key::F11,        0xffc8 }, { Key::F12,        0xffc9 }, { Key::LeftShift,  0xffe1 },
      { Key::RightShift, 0xffe2 }, { Key::LeftCtrl,   0xffe3 }, { Key::RightCtrl,  0xffe4 },
      { Key::LeftCmd,    0xffe9 }, { Key::RightCmd,   0xffea } };
      auto it = keymap.find(k);
      key = it != keymap.end() ? it->second : k;
    }

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
    struct Subencoding { enum { Raw=0, Solid=1, PackedPalette=2, PackedPaletteEnd=16, PlainRLE=128, PaletteRLE=130, PaletteRLEEnd=255 }; };
    const PixelFormat *pf;
    ZLibReader *zreader;
    uint16_t x, y, w, h, copy_x=0, copy_y=0;
    int encoding;
    StringPiece data;
    PixelData(const PixelFormat *p, ZLibReader *r) : Serializable(0), pf(p), zreader(r) {}

    string DebugString() const { return StrCat("{ x=", x, ", y=", y, ", w=", w, ", h=", h, ", encoding=", encoding, " }"); }
    int HeaderSize() const { return 12; }
    int Size() const { return HeaderSize() + data.size(); }
    void Out(Serializable::Stream *o) const {}

    int In(const Serializable::Stream *i) {
      i->Ntohs(&x);  i->Ntohs(&y);
      i->Ntohs(&w);  i->Ntohs(&h);  i->Ntohl(&encoding);

      if (encoding == Raw) {
        int len = w * h * pf->bits_per_pixel/8;
        data = StringPiece(i->Get(len), len);
        return i->Result();

      } else if (encoding == CopyRect) {
        data = StringPiece(0, 4);
        i->Ntohs(&copy_x);
        i->Ntohs(&copy_y);
        return i->Result();

      } else if (encoding == ZRLE) {
        if (i->Remaining() < 4) { data = StringPiece(0, 4); i->error = true; return i->Result(); }
        int len = 0;
        i->Ntohl(&len);
        const char *buf = i->Get(len);
        data = StringPiece(buf ? buf - 4 : nullptr, len + 4);
        return i->Result();

      } else return ERRORv(-1, "unknown encoding: ", encoding);
    }
  };

  struct FramebufferUpdate : public Serializable {
    static const int ID = 0;
    const PixelFormat *pf;
    ZLibReader *zreader;
    vector<PixelData> rect;
    FramebufferUpdate(const PixelFormat *p, ZLibReader *r) : Serializable(ID), pf(p), zreader(r) {}

    int HeaderSize() const { return 3; }
    int Size() const { int v=HeaderSize(); for (auto &r : rect) v += r.Size(); return v; }
    void Out(Serializable::Stream *o) const {}
    int In(const Serializable::Stream *i) {
      i->Get(1);
      uint16_t rects = 0;
      i->Ntohs(&rects);
      rect.clear();
      for (uint16_t j = 0; !i->error && j != rects; ++j) {
        rect.emplace_back(pf, zreader);
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
  RFBClient::CopyCB copy_cb;
  Callback success_cb;
  int state=0, fb_w=0, fb_h=0;
  uint8_t security_type=0;
  bool share_desktop=1;
  RFB::PixelFormat pixel_format;
  ZLibReader zreader;
  string challenge, decoded;

  RFBClientConnection(RFBClient::Params p, RFBClient::LoadPasswordCB PCB, RFBClient::UpdateCB UCB,
                      RFBClient::CopyCB CCB, Callback SCB) : params(move(p)), load_password_cb(move(PCB)),
  update_cb(move(UCB)), copy_cb(move(CCB)), success_cb(move(SCB)), zreader(66536) {}

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
          if (16 != msg.text.size()) return ERRORv(-1, c->Name(), ": invalid challenge length");
          challenge = msg.text.str();
          processed = msg.Size();
          string pw;
          if (load_password_cb && !load_password_cb(&pw)) break;
          if (int ret = SendChallengeResponse(c, move(pw))) return ret;
        } break;

        case SECURITY_RESULT: {
          RFB::SecurityResult msg;
          if (c->rb.size() < total_processed + msg.Size()) break;
          if (msg.In(&msg_s)) return ERRORv(-1, c->Name(), ": read SecurityResult");
          RFBTrace(c->Name(), ": SecurityResult ", msg.v);
          state = msg.v ? AUTH_FAILED : INIT;
          processed = msg.Size();
          if (state == INIT) {
            if (success_cb) success_cb();
            if (!Write(c, RFB::ClientInit(share_desktop))) return ERRORv(-1, c->Name(), ": write");
          }
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

          vector<int> encodings{ RFB::PixelData::ZRLE, RFB::PixelData::Raw };
          if (copy_cb) encodings.insert(encodings.begin(), RFB::PixelData::CopyRect);
          if (!msg.pixel_format.true_color_flag) return ERRORv(-1, c->Name(), ": only true-color supported");
          if (!Write(c, RFB::SetEncodings(encodings))) return ERRORv(-1, c->Name(), ": write");
          if (!Write(c, RFB::FramebufferUpdateRequest(Box(msg.fb_w, msg.fb_h), false))) return ERRORv(-1, c->Name(), ": write");
          processed = msg.Size();
          state = RUN;
          fb_w = msg.fb_w;
          fb_h = msg.fb_h;
          pixel_format = msg.pixel_format;
          c->rb.Resize(max(c->rb.size(), 2 * fb_w * fb_h * pixel_format.bits_per_pixel/8));
          update_cb(c, Box(fb_w, fb_h), 0, StringPiece());
        } break;
        
        case RUN: {
          unsigned char msg_type = 0;
          if (c->rb.size() < total_processed + 1) break;
          msg_s.Read8(&msg_type);

          if (msg_type == RFB::FramebufferUpdate::ID) {
            RFB::FramebufferUpdate msg(&pixel_format, &zreader);
            if (c->rb.size() < total_processed + msg.Size()) break;
            if (msg.In(&msg_s)) {
              if (c->rb.size() < total_processed + msg.Size()) break;
              return ERRORv(-1, c->Name(), ": read FramebufferUpdate");
            }
            RFBTrace(c->Name(), ": FramebufferUpdate rects=", msg.rect.size());
            processed = msg.Size() + 1;
            for (auto &r : msg.rect) {
              if (!r.w && !r.h) continue;
              Box b(r.x, fb_h - r.y - r.h, r.w, r.h);
              if (r.encoding == RFB::PixelData::CopyRect) {
                copy_cb(c, b, point(r.copy_x, r.copy_y));
              } else if (r.encoding == RFB::PixelData::ZRLE) {
                if (DecodeZRLE(c, b, StringPiece(r.data.buf + 4, r.data.len - 4))) return ERRORv(-1, "decode zrle");
              } else if (r.encoding == RFB::PixelData::Raw) {
                int pf = 0;
                if      (pixel_format.bits_per_pixel == 32) pf = Pixel::BGR32;
                else if (pixel_format.bits_per_pixel == 24) pf = Pixel::BGR24;
                else return ERRORv(-1, "unknown encoding ", r.encoding, " ", pixel_format.bits_per_pixel);
                update_cb(c, b, pf, r.data);
              } else return ERRORv(-1, "unknown encoding ", r.encoding);
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

  bool SendChallengeResponse(Connection *c, string pw) {
    if (VNC_AUTH != state) return ERRORv(-1, "invalid state");
    pw.resize(8, 0);
    for (auto i = MakeUnsigned(&pw[0]), e = i + pw.size(); i != e; ++i) *i = BitString::Reverse(*i);

    string response(16, 0);
    Crypto::Cipher enc = Crypto::CipherInit();
    Crypto::CipherOpen(enc, Crypto::CipherAlgos::DES_ECB(), true, pw, string(8, 0));
    int len = Crypto::CipherUpdate(enc, challenge, &response[0], response.size());
    len += Crypto::CipherFinal(enc, &response[0+len], response.size()-len);
    Crypto::CipherFree(enc);
    if (len != 16) return ERRORv(-1, c->Name(), ": invalid challenge response len=", len);

    if (!Write(c, RFB::VNCAuthenticationChallenge(response))) return ERRORv(-1, c->Name(), ": write");
    state = SECURITY_RESULT;
    return 0;
  }

  int DecodeZRLE(Connection *c, const Box &b, const StringPiece &in) {
    zreader.out.clear();
    if (!zreader.Add(in)) return ERRORv(-1, "decompress failed");
    return DecodeTRLE(c, b, 64, 64, Serializable::ConstStream(zreader.out.data(), zreader.out.size()));
  }

  int DecodeTRLE(Connection *c, const Box &box, int tile_width, int tile_height, const Serializable::ConstStream &zi) {
    int cpixel_size = 3, ipf = cpixel_size == 3 ? Pixel::BGR24 : Pixel::BGR32, opf = Texture::preferred_pf;
    int tiles_width  = NextMultipleOfN(box.w, tile_width)  / tile_width;
    int tiles_height = NextMultipleOfN(box.h, tile_height) / tile_height;
    int last_tile_width  = box.w - (tiles_width  - 1) * tile_width;
    int last_tile_height = box.h - (tiles_height - 1) * tile_height;
    int ips = Pixel::Size(ipf), ops = Pixel::Size(opf);
#ifdef LFL_RFB_DEBUG
    decoded.assign(box.w * box.h * ops, 0);
#else 
    decoded.resize(box.w * box.h * ops);
#endif
    for (int tile_y = 0; tile_y != tiles_height; tile_y++) {
      for (int tile_x = 0; tile_x != tiles_width; tile_x++) {
        int tile_w = tile_x == tiles_width -1 ? last_tile_width  : tile_width;
        int tile_h = tile_y == tiles_height-1 ? last_tile_height : tile_height;
        unsigned char subencoding = 0;
        zi.Read8(&subencoding);

        if (subencoding == RFB::PixelData::Subencoding::Raw) {
          const char *pixel_data = zi.Get(tile_w * tile_h * ips);
          if (int ret = zi.Result()) return ERRORv(ret, "parse raw subencoding: ", ret);
          SimpleVideoResampler::Blit(MakeUnsigned(pixel_data), MakeUnsigned(&decoded[0]), tile_w, tile_h,
                                     ipf, tile_w * ips, 0, 0,
                                     opf, box.w * ops, tile_x * tile_width, tile_y * tile_height);

        } else if (subencoding == RFB::PixelData::Subencoding::Solid) {
          Color color = ReadRGB(zi);
          if (int ret = zi.Result()) return ERRORv(ret, "parse solid subencoding: ", ret);
          SimpleVideoResampler::Fill(MakeUnsigned(&decoded[0]), tile_w, tile_h,
                                     opf, box.w * ops, tile_x * tile_width, tile_y * tile_height, color);

        } else if (subencoding == RFB::PixelData::Subencoding::PlainRLE) {
          int tile_pixels = tile_w * tile_h, p_i = 0;
          while (p_i < tile_pixels && !zi.error) {
            point ts(p_i % tile_w, p_i / tile_w);
            Color color = ReadRGB(zi);
            int run_len = ReadRunLength(zi);
            p_i += run_len;
            if (FillTileRange(MakeUnsigned(&decoded[0]), opf, box.w * ops,
                              point(tile_x * tile_width, tile_y * tile_height),
                              ts, point((p_i-1) % tile_w, (p_i-1) / tile_w), tile_w, color)) return -1;
          }
          if (int ret = zi.Result()) return ERRORv(ret, "parse plain rle subencoding: ", ret);
          if (p_i != tile_pixels) return ERRORv(-1, "parse ", p_i, " instead of ", tile_pixels);

        } else if (subencoding >= RFB::PixelData::Subencoding::PackedPalette &&
                   subencoding <= RFB::PixelData::Subencoding::PackedPaletteEnd) {
          int palette_size = subencoding, palette_bytes = palette_size * cpixel_size, line_size = box.w * ops;
          int indices_per_byte = palette_size == 2 ? 8 : (palette_size < 5 ? 4 : 2), padded_bytes, ind_rowlen;
          if      (indices_per_byte == 8) padded_bytes = (ind_rowlen = (tile_w + 7) / 8) * tile_h;
          else if (indices_per_byte == 4) padded_bytes = (ind_rowlen = (tile_w + 3) / 4) * tile_h;
          else if (indices_per_byte == 2) padded_bytes = (ind_rowlen = (tile_w + 1) / 2) * tile_h;
          else return ERRORv(-1, "unknown indices per byte ", indices_per_byte);
          const unsigned char *palette  = MakeUnsigned(zi.Get(palette_bytes));
          const unsigned char *ind_data = MakeUnsigned(zi.Get(padded_bytes));
          if (int ret = zi.Result()) return ERRORv(ret, "parse packed palette subencoding: ", ret);
          for (int t_y = 0; t_y != tile_h; ++t_y) {
            for (int t_x = 0; t_x != tile_w; ++t_x) {
              int bit_offset = t_x % indices_per_byte;
              uint8_t byte = ind_data[t_y * ind_rowlen + t_x / indices_per_byte], ind=0;
              if      (indices_per_byte == 8) ind = (byte >> (7 - bit_offset))   & 1;
              else if (indices_per_byte == 4) ind = (byte >> (6 - bit_offset*2)) & 3;
              else if (indices_per_byte == 2) ind = (byte >> (4 - bit_offset*4)) & 0xf;
              else return ERRORv(-1, "unknown indices per byte ", indices_per_byte);
              SimpleVideoResampler::CopyPixel
                (ipf, opf, palette + int(ind) * cpixel_size, MakeUnsigned
                 (&decoded[0] + line_size*(tile_y*tile_height + t_y) + ops*(tile_x*tile_width + t_x)));
            }
          }

        } else if (subencoding >= RFB::PixelData::Subencoding::PaletteRLE &&
                   subencoding <= RFB::PixelData::Subencoding::PaletteRLEEnd) {
          int palette_size = subencoding - 128, palette_bytes = palette_size * cpixel_size;
          const unsigned char *palette = MakeUnsigned(zi.Get(palette_bytes));
          int tile_pixels = tile_w * tile_h, p_i = 0;
          while (p_i < tile_pixels && !zi.error) {
            point ts(p_i % tile_w, p_i / tile_w);
            unsigned char ind = 0;
            zi.Read8(&ind);
            auto c = palette + (int(ind) & 0x7f) * cpixel_size;
            Color color(*(c+2), *(c+1), *(c+0));
            int run_len = ind < 128 ? 1 : ReadRunLength(zi);
            p_i += run_len;
            if (FillTileRange(MakeUnsigned(&decoded[0]), opf, box.w * ops,
                              point(tile_x * tile_width, tile_y * tile_height),
                              ts, point((p_i-1) % tile_w, (p_i-1) / tile_w), tile_w, color)) return -1;
          }
          if (int ret = zi.Result()) return ERRORv(ret, "parse plain rle subencoding: ", ret);
          if (p_i != tile_pixels) return ERRORv(-1, "parse ", p_i, " instead of ", tile_pixels);

        } else return ERRORv(-1, "unknown subencoding: ", int(subencoding));
      }
    }
    if (int remaining = zi.Remaining()) return ERRORv(-1, remaining, " extra zrle bytes");
    update_cb(c, box, opf, decoded);
    return 0;
  }

  bool Write(Connection *c, const Serializable &m) {
    string text = m.ToString();
    return c->WriteFlush(text) == text.size();
  }

  static Color ReadRGB(const Serializable::ConstStream &i) { 
    unsigned char r = 0, g = 0, b = 0;
    i.Read8(&b); i.Read8(&g); i.Read8(&r);
    return Color(r, g, b);
  }

  static int ReadRunLength(const Serializable::ConstStream &i) {
    int run_len = 0;
    unsigned char byte_len = 0;
    do { i.Read8(&byte_len); run_len += byte_len; }
    while(byte_len == 255 && !i.error);
    return run_len + 1;
  }

  static int FillTileRange(unsigned char *dst, int pf, int ls,
                           point b, point ts, point te, int tile_w, const Color &color) {
    int d_y = te.y - ts.y;
    if (d_y == 0) {
      if (te.x < ts.x) return ERRORv(-1, "negative advance ", ts.x, " to ", te.x);
      SimpleVideoResampler::Fill(dst, te.x + 1 - ts.x, 1, pf, ls, b.x + ts.x, b.y + ts.y, color);
    } else if (d_y > 0) {
      SimpleVideoResampler::Fill(dst, tile_w - ts.x, 1, pf, ls, b.x + ts.x, b.y + ts.y, color);
      if (d_y > 1)
        SimpleVideoResampler::Fill(dst, tile_w, d_y - 1, pf, ls, b.x + 0, b.y + ts.y + 1, color);
      SimpleVideoResampler::Fill(dst, te.x + 1, 1, pf, ls, b.x + 0, b.y + te.y, color);
    } else return ERRORv(-1, "negative advance: ", ts.y, " to ", te.y);
    return 0;
  }
};

Connection *RFBClient::Open(Networking *net, Params p, RFBClient::LoadPasswordCB pcb, RFBClient::UpdateCB ucb,
                            RFBClient::CopyCB ccb, Connection::CB *detach, Callback *success) { 
  Connection *c = net->ConnectTCP(p.hostport, 5900, detach, p.background_services);
  if (!c) return 0;
  c->handler = make_unique<RFBClientConnection>(move(p), move(pcb), move(ucb), move(ccb),
                                                success ? move(*success) : Callback());
  return c;
}

int RFBClient::SendChallengeResponse(Connection *c, string pw) {
  return dynamic_cast<RFBClientConnection*>(c->handler.get())->SendChallengeResponse(c, move(pw));
}

int RFBClient::SendKeyEvent(Connection *c, uint32_t key, uint8_t down) {
  if (!dynamic_cast<RFBClientConnection*>(c->handler.get())->Write(c, RFB::KeyEvent(key, down)))
  { c->SetError(); return ERRORv(-1, c->Name(), ": write"); }
  return 0;
}

int RFBClient::SendPointerEvent(Connection *c, uint16_t x, uint16_t y, uint8_t buttons) {
  if (!dynamic_cast<RFBClientConnection*>(c->handler.get())->Write(c, RFB::PointerEvent(x, y, buttons)))
  { c->SetError(); return ERRORv(-1, c->Name(), ": write"); }
  return 0;
}

int RFBClient::SendClientCutText(Connection *c, const StringPiece &text) {
  if (!dynamic_cast<RFBClientConnection*>(c->handler.get())->Write(c, RFB::ClientCutText(text)))
  { c->SetError(); return ERRORv(-1, c->Name(), ": write"); }
  return 0;
}

}; // namespace LFL
