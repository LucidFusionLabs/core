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
    int Open(const char **argv);
    int OpenPTY(const char **argv);
    int Close();
};

struct MultiProcessBuffer {
    string url;
    char *buf=0;
    int len=0, impl=-1, transfer_socket=-1;
    MultiProcessBuffer() {}
    MultiProcessBuffer(Connection *c, const InterProcessProtocol::ResourceHandle &h);
    virtual ~MultiProcessBuffer();
    virtual void Close();
    virtual bool Open();
    bool Create(const Serializable &s) { 
        len = Serializable::Header::size + s.Size();
        if (Open()) { s.ToString(buf, len, 0); return true; }
        return false;
    }
};

struct ProcessAPIClient {
    typedef function<void(const MultiProcessResource::Texture&)> LoadResourceCompleteCB;
    struct ConnectionHandler : public Connection::Handler {
        ProcessAPIClient *parent;
        ConnectionHandler(ProcessAPIClient *P) : parent(P) {}
        int Read(Connection *c);
        bool ReadTexture(const InterProcessProtocol::LoadResourceResponse&, const MultiProcessBuffer&, MultiProcessResource::Texture *out);
    };
    int pid=0;
    Connection *conn=0;
    unsigned short seq=0;
    unordered_map<unsigned short, LoadResourceCompleteCB> reqmap;

    void StartServer(const string &server_program);
    void LoadResource(const string &content, const string &fn, const LoadResourceCompleteCB &cb);
};

struct ProcessAPIServer {
    Connection *conn=0;
    void Start(const string &socket_name);
    void HandleMessagesLoop();
    Texture *LoadTexture(const InterProcessProtocol::LoadResourceRequest&, const MultiProcessBuffer&, Texture *orig, Texture *scaled);
};

}; // namespace LFL
#endif // __LFL_LFAPP_IPC_H__
