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

#ifndef __LFL_LFAPP_IPC_H__
#define __LFL_LFAPP_IPC_H__

namespace LFL {
struct NTService {
    static int Install  (const char *name, const char *path);
    static int Uninstall(const char *name);
    static int WrapMain (const char *name, MainCB main_cb, int argc, const char **argv);
};

struct ProcessPipe {
    int pid=0;
    FILE *in=0, *out=0;
    virtual ~ProcessPipe() { Close(); }
    int Open(const char **argv, const char *startdir=0);
    int OpenPTY(const char **argv, const char *startdir=0);
    int Close();
};

struct MultiProcessBuffer {
    string url;
    char *buf=0;
    int len=0, transfer_handle=-1;
#ifdef WIN32
    HANDLE impl = INVALID_HANDLE_VALUE, share_process = INVALID_HANDLE_VALUE;
#else
    int impl = -1; void *share_process = 0;
#endif
    MultiProcessBuffer(void *share_with) : share_process(share_with) {}
    MultiProcessBuffer(Connection *c, const InterProcessProtocol::ResourceHandle &h);
    virtual ~MultiProcessBuffer();
    virtual void Close();
    virtual bool Open();
    bool Create(int s) { len=s; return Open(); }
    bool Create(const Serializable &s) { bool ret; if ((ret = Create(Size(s)))) s.ToString(buf, len, 0); return ret; }
    bool Copy(const Serializable &s) { bool ret; if ((ret = len >= Size(s))) s.ToString(buf, len, 0); return ret; }
    static int Size(const Serializable &s) { return Serializable::Header::size + s.Size(); }
};

struct ProcessAPIClient {
    typedef function<void(const MultiProcessResource::Texture&)> LoadResourceCompleteCB;
    struct LoadResourceQuery {
        LoadResourceCompleteCB cb;
        MultiProcessBuffer response;
        LoadResourceQuery(void *share_with, const LoadResourceCompleteCB &c) : response(share_with), cb(c) {}
    };
    struct ConnectionHandler : public Connection::Handler {
        ProcessAPIClient *parent;
        ConnectionHandler(ProcessAPIClient *P) : parent(P) {}
        int Read(Connection *c);
        bool ReadTexture(const MultiProcessBuffer&, MultiProcessResource::Texture *out);
    };
    int pid=0;
    Connection *conn=0;
    unsigned short seq=0;
    void *server_process=0;
    unordered_map<unsigned short, LoadResourceQuery*> reqmap;

    void StartServer(const string &server_program);
    void LoadResource(const string &content, const string &fn, const LoadResourceCompleteCB &cb);
};

struct ProcessAPIServer {
    Connection *conn=0;
    unordered_map<unsigned short, Texture*> resmap;

    void Start(const string &socket_name);
    void HandleMessagesLoop();
    Texture *LoadTexture(const InterProcessProtocol::LoadResourceRequest&, const MultiProcessBuffer&);
};

}; // namespace LFL
#endif // __LFL_LFAPP_IPC_H__
