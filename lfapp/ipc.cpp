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

#include "lfapp/lfapp.h"
#include "lfapp/ipc.h"

#include <fcntl.h>

#if defined(LFL_MOBILE)
#elif defined(WIN32)
#else
#include <signal.h>
#include <sys/types.h>
#include <sys/shm.h>
#endif

namespace LFL {
#ifndef WIN32
int NTService::Install  (const char *name, const char *path) { FATAL("%s", "not implemented"); }
int NTService::Uninstall(const char *name)                   { FATAL("%s", "not implemented"); }
int NTService::WrapMain (const char *name, MainCB main_cb, int argc, const char **argv) { return main_cb(argc, argv); }
#endif
#if defined(LFL_MOBILE)
int ProcessPipe::OpenPTY(const char **argv) { FATAL("%s", "not implemented"); }
int ProcessPipe::Open   (const char **argv) { FATAL("%s", "not implemented"); }
int ProcessPipe::Close()                    { FATAL("%s", "not implemented"); }
#elif defined(WIN32)
MainCB nt_service_main = 0;
const char *nt_service_name = 0;
SERVICE_STATUS_HANDLE nt_service_status_handle = 0;

BOOL UpdateSCMStatus(DWORD dwCurrentState, DWORD dwWin32ExitCode,
                     DWORD dwServiceSpecificExitCode, DWORD dwCheckPoint,
                     DWORD dwWaitHint) {
    SERVICE_STATUS serviceStatus;
    serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    serviceStatus.dwCurrentState = dwCurrentState;
    serviceStatus.dwServiceSpecificExitCode = dwServiceSpecificExitCode;
    serviceStatus.dwCheckPoint = dwCheckPoint;
    serviceStatus.dwWaitHint = dwWaitHint;

    if (dwCurrentState == SERVICE_START_PENDING) serviceStatus.dwControlsAccepted = 0;
    else serviceStatus.dwControlsAccepted = SERVICE_ACCEPT_STOP |SERVICE_ACCEPT_SHUTDOWN;

    if (dwServiceSpecificExitCode == 0) serviceStatus.dwWin32ExitCode = dwWin32ExitCode;
    else serviceStatus.dwWin32ExitCode = ERROR_SERVICE_SPECIFIC_ERROR;

    return SetServiceStatus(nt_service_status_handle, &serviceStatus);
}

void HandleNTServiceControl(DWORD controlCode) {
    if (controlCode == SERVICE_CONTROL_SHUTDOWN || controlCode == SERVICE_CONTROL_STOP) {
        UpdateSCMStatus(SERVICE_STOPPED, NO_ERROR, 0, 0, 0);
        app->run = 0;
    } else {
        UpdateSCMStatus(SERVICE_RUNNING, NO_ERROR, 0, 0, 0);
    }
}

int DispatchNTServiceMain(int argc, char **argv) {
    nt_service_status_handle = RegisterServiceCtrlHandler(nt_service_name, (LPHANDLER_FUNCTION)HandleNTServiceControl);
    if (!nt_service_status_handle) { ERROR("RegisterServiceCtrlHandler: ", GetLastError()); return -1; }

    if (!UpdateSCMStatus(SERVICE_RUNNING, NO_ERROR, 0, 0, 0)) {
        ERROR("UpdateSCMStatus: ", GetLastError()); return -1;
    }
    
    return nt_service_main(argc, (const char **)argv);
}

int NTService::Install(const char *name, const char *path) {
    SC_HANDLE schSCManager = OpenSCManager(0, 0, SC_MANAGER_CREATE_SERVICE);
    if (!schSCManager) { ERROR("OpenSCManager: ", GetLastError()); return -1; }

    SC_HANDLE schService = CreateService( 
        schSCManager,    	  /* SCManager database      */ 
        name,			      /* name of service         */ 
        name,                 /* service name to display */ 
        SERVICE_ALL_ACCESS,   /* desired access          */ 
        SERVICE_WIN32_SHARE_PROCESS|SERVICE_INTERACTIVE_PROCESS, 
        SERVICE_DEMAND_START, /* start type              */ 
        SERVICE_ERROR_NORMAL, /* error control type      */ 
        path,			      /* service's binary        */ 
        0,                    /* no load ordering group  */ 
        0,                    /* no tag identifier       */ 
        0,                    /* no dependencies         */ 
        0,                    /* LocalSystem account     */ 
        0);                   /* no password             */
    if (!schService) { ERROR("CreateService: ", GetLastError()); return -1; }

    INFO("service ", name, " installed - see Control Panel > Services");
    CloseServiceHandle(schSCManager);
    return 0;
}

int NTService::Uninstall(const char *name) {
    SC_HANDLE schSCManager = OpenSCManager(0, 0, SC_MANAGER_CREATE_SERVICE);
    if (!schSCManager) { ERROR("OpenSCManager: ", GetLastError()); return -1; }

    SC_HANDLE schService = OpenService(schSCManager, name, SERVICE_ALL_ACCESS);
    if (!schService) { ERROR("OpenService: ", GetLastError()); return -1; }

    if (!DeleteService(schService)) { ERROR("DeleteService: ", GetLastError()); return -1; }

    INFO("service ", name, " uninstalled");
    CloseServiceHandle(schService);
    CloseServiceHandle(schSCManager);
    return 0;
}

int NTService::WrapMain(const char *name, MainCB main_cb, int argc, const char **argv) {
    nt_service_name = name;
    nt_service_main = main_cb;

    SERVICE_TABLE_ENTRY serviceTable[] = {
        { (LPSTR)name, (LPSERVICE_MAIN_FUNCTION)DispatchNTServiceMain},
        { 0, 0 }
    };

    if (!StartServiceCtrlDispatcher(serviceTable)) {
        ERROR("StartServiceCtrlDispatcher ", GetLastError());
        return -1;
    }
    return 0;
}

int ProcessPipe::OpenPTY(const char **argv) { return Open(argv); }
int ProcessPipe::Open(const char **argv) {
    SECURITY_ATTRIBUTES sa;
    memset(&sa, 0, sizeof(sa));
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = 1;
    HANDLE pipeinR, pipeinW, pipeoutR, pipeoutW, h;
    if (!CreatePipe(&pipeinR, &pipeinW, &sa, 0)) return -1;
    if (!CreatePipe(&pipeoutR, &pipeoutW, &sa, 0)) { CloseHandle(pipeinR); CloseHandle(pipeinW); return -1; }

    STARTUPINFO si;
    memset(&si, 0, sizeof(si));
    si.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
    si.wShowWindow = SW_HIDE;
    si.hStdInput = pipeoutR;
    si.hStdOutput = pipeinW;
    si.hStdError = pipeinW;

    PROCESS_INFORMATION pi;
    if (!CreateProcess(0, (LPSTR)argv[0], 0, 0, 1, CREATE_NEW_PROCESS_GROUP, 0, 0, &si, &pi)) return -1;
    CloseHandle(pi.hThread);
    CloseHandle(pipeinW);
    CloseHandle(pipeoutR);

    in = fdopen(_open_osfhandle((long)pipeinR, O_TEXT), "r"); // leaks ?
    out = fdopen(_open_osfhandle((long)pipeoutW, O_TEXT), "w");
    return 0;
}
InterProcessResource::InterProcessResource(int size, const string &u) : len(size), url(u) {}
InterProcessResource::~InterProcessResource() {}
#else /* WIN32 */
int ProcessPipe::Open(const char **argv) {
    int pipein[2], pipeout[2], ret;
    if (pipe(pipein) < 0) return -1;
    if (pipe(pipeout) < 0) { close(pipein[0]); close(pipein[1]); return -1; }

    if ((ret = fork())) { 
        close(pipein[1]);
        close(pipeout[0]);
        if (ret < 0) { close(pipein[0]); close(pipeout[1]); return -1; }
        in = fdopen(pipein[0], "r");
        out = fdopen(pipeout[1], "w");
    } else {
        close(pipein[0]);
        close(pipeout[1]);
        close(0);
        close(1);
        close(2);
        dup2(pipein[1], 2);
        dup2(pipein[1], 1);
        dup2(pipeout[0], 0);
        execvp(argv[0], (char*const*)argv);
    }
    return 0;
}

extern "C" pid_t forkpty(int *, char *, struct termios *, struct winsize *);
int ProcessPipe::OpenPTY(const char **argv) {
    // struct termios term;
    // struct winsize win;
    char name[PATH_MAX];
    int fd = -1;
    if ((pid = forkpty(&fd, name, 0, 0))) {
        if (pid < 0) { close(fd); return -1; }
        fcntl(fd, F_SETFL, O_NONBLOCK);
        in = fdopen(fd, "r");
        out = fdopen(fd, "w");
    } else {
        execvp(argv[0], (char*const*)argv);
    }
    return 0;
}

int ProcessPipe::Close() {
    if (pid) { kill(pid, SIGHUP); pid = 0; }
    if (in)  { fclose(in);        in  = 0; }
    if (out) { fclose(out);       out = 0; }
    return 0;
}

static int ShmKeyFromInterProcessResourceURL(const string &u) {
    static string shm_url = "shm://";
    CHECK(PrefixMatch(u, shm_url));
    return atoi(u.c_str() + shm_url.size());
}

InterProcessResource::~InterProcessResource() {
    if (buf)     shmdt(buf);
    if (id >= 0) shmctl(id, IPC_RMID, NULL);
}

InterProcessResource::InterProcessResource(int size, const string &u) : len(size), url(u) {
    CHECK(len);
    int key = url.empty() ? rand() : ShmKeyFromInterProcessResourceURL(url);
    if ((id = shmget(key, size, url.empty() ? (IPC_CREAT | 0600) : 0400)) < 0)
        FATAL("id=", id, ", size=", size, ", url=", url, ": ", strerror(errno));

    CHECK_GE(id, 0);
    buf = reinterpret_cast<char*>(shmat(id, NULL, 0));
    CHECK(buf);
    CHECK_NE((char*)-1, buf);
    if (url.empty()) url = StrCat("shm://", key);
}
#endif /* WIN32 */

#ifdef LFL_IPC_DEBUG
#define IPCTrace(...) printf(__VA_ARGS__)
#else
#define IPCTrace(...)
#endif

#ifndef LFL_MOBILE
void ProcessAPIServer::Start(const string &client_program) {
    int fd[2];
    CHECK(SystemNetwork::OpenSocketPair(fd));
    INFO("ProcessAPIServer starting ", client_program);
#ifdef WIN32
	FATAL("not implemented")
#else
    if ((pid = fork())) {
        CHECK_GT(pid, 0);
        close(fd[0]);
        conn = new Connection(Singleton<UnixClient>::Get(), new Query(this));
        conn->state = Connection::Connected;
        conn->socket = fd[1];
        conn->svc->conn[conn->socket] = conn;
        app->network.active.Add(conn->socket, SocketSet::READABLE, &conn->self_reference);
    } else {
        close(fd[1]);
        // close(0); close(1); close(2);
        string arg0 = client_program, arg1 = StrCat("fd://", fd[0]);
        vector<char*> av = { &arg0[0], &arg1[0], 0 };
        CHECK(!execvp(av[0], &av[0]));
    }
#endif
}

void ProcessAPIServer::LoadResource(const string &content, const string &fn, const ProcessAPIServer::LoadResourceCompleteCB &cb) { 
    CHECK(conn);
    InterProcessProtocol::ContentResource resource(content, fn, "");
    InterProcessResource ipr(Serializable::Header::size + resource.Size());
    resource.ToString(ipr.buf, ipr.len);
    ipr.id = -1;

    string msg;
    reqmap[seq] = cb;
    InterProcessProtocol::LoadResourceRequest(resource.Type(), ipr.url, ipr.len).ToString(&msg, seq++);
    IPCTrace("ProcessAPIServer::LoadResource fn='%s' url='%s' msg_size=%zd\n", fn.c_str(), ipr.url.c_str(), msg.size());
    if (conn->state != Connection::Connected) { ERROR("no process api client"); cb(InterProcessProtocol::TextureResource()); }
    else CHECK_EQ(msg.size(), conn->WriteFlush(msg));
}

int ProcessAPIServer::Query::Read(Connection *c) {
    while (c->rl >= Serializable::Header::size) {
        IPCTrace("ProcessAPIServer::Query::Read begin parse %d bytes\n", c->rl);
        Serializable::ConstStream in(c->rb, c->rl);
        Serializable::Header hdr;
        hdr.In(&in);
        auto reply = parent->reqmap.find(hdr.seq);
        CHECK_NE(parent->reqmap.end(), reply);
        if (hdr.id == Serializable::GetType<InterProcessProtocol::LoadResourceResponse>()) {
            InterProcessProtocol::LoadResourceResponse req;
            if (req.Read(&in)) break;
            parent->reqmap.erase(hdr.seq);
            if (!req.ipr_len) { IPCTrace("TextureResource failed\n"); reply->second(InterProcessProtocol::TextureResource()); }
            else {
                InterProcessResource res(req.ipr_len, req.ipr_url);
                IPCTrace("ProcessAPIServer::Query::Read LoadResourceResponse url='%s' ", res.url.c_str());
                if (req.ipr_type == Serializable::GetType<InterProcessProtocol::TextureResource>()) {
                    Serializable::ConstStream res_in(res.buf, res.len);
                    Serializable::Header res_hdr;
                    res_hdr.In(&res_in);
                    CHECK_EQ(req.ipr_type, res_hdr.id);
                    InterProcessProtocol::TextureResource tex_res;
                    CHECK(!tex_res.Read(&res_in));
                    IPCTrace("TextureResource width=%d height=%d\n", tex_res.width, tex_res.height);
                    reply->second(tex_res);
                } else FATAL("unknown ipr type", req.ipr_type);
            }
        } else FATAL("unknown id ", hdr.id);
        IPCTrace("ProcessAPIServer::Query::Read flush %d bytes\n", in.offset);
        c->ReadFlush(in.offset);
    }
    return 0;
}

void ProcessAPIClient::Start(const string &socket_name) {
    static string fd_url = "fd://";
    if (PrefixMatch(socket_name, fd_url)) {
        conn = new Connection(Singleton<UnixClient>::Get(), Singleton<Query>::Get());
        conn->state = Connection::Connected;
        conn->socket = atoi(socket_name.c_str() + fd_url.size());
    } else return;
    INFO("ProcessAPIClient opened ", socket_name);
}

void ProcessAPIClient::HandleMessagesLoop() {
    int l;
    while (app->run) {
        if ((l = NBRead(conn->socket, conn->rb + conn->rl, Connection::BufSize - conn->rl, -1)) <= 0) break;
        conn->rl += l;
        while (conn->rl >= Serializable::Header::size) {
            IPCTrace("ProcessAPIClient:HandleMessagesLoop begin parse %d bytes\n", conn->rl);
            Serializable::ConstStream in(conn->rb, conn->rl);
            Serializable::Header hdr;
            hdr.In(&in);
            if (hdr.id == Serializable::GetType<InterProcessProtocol::LoadResourceRequest>()) {
                InterProcessProtocol::LoadResourceRequest req;
                if (req.Read(&in)) break;
                IPCTrace("ProcessAPIClient:HandleMessagesLoop LoadResourceRequest url='%s' ", req.ipr_url.c_str());
                InterProcessResource res(req.ipr_len, req.ipr_url);
                if (req.ipr_type == Serializable::GetType<InterProcessProtocol::ContentResource>()) {
                    Serializable::ConstStream res_in(res.buf, res.len);
                    Serializable::Header res_hdr;
                    res_hdr.In(&res_in);
                    CHECK_EQ(req.ipr_type, res_hdr.id);
                    InterProcessProtocol::ContentResource content_res;
                    CHECK(!content_res.Read(&res_in));
                    IPCTrace("ContentResource fn='%s' %p %d\n", content_res.name.buf, content_res.buf.buf, content_res.buf.len);

                    const int max_image_size = 1000000;
                    Texture orig_tex, scaled_tex, *tex = &orig_tex;
                    Asset::LoadTexture(content_res.buf.data(), content_res.name.data(), content_res.buf.size(), &orig_tex, 0);
                    if (orig_tex.BufferSize() >= max_image_size) {
                        tex = &scaled_tex;
                        float scale_factor = sqrt((float)max_image_size/orig_tex.BufferSize());
                        scaled_tex.Resize(orig_tex.width*scale_factor, orig_tex.height*scale_factor, Pixel::RGB24, Texture::Flag::CreateBuf);
                        VideoResampler resampler;
                        resampler.Open(orig_tex.width, orig_tex.height, orig_tex.pf, scaled_tex.width, scaled_tex.height, scaled_tex.pf);
                        resampler.Resample(orig_tex.buf, orig_tex.LineSize(), scaled_tex.buf, scaled_tex.LineSize());
                    }

                    string msg;
                    if (tex->buf) {
                        InterProcessProtocol::TextureResource tex_res(*tex);
                        InterProcessResource ipr(Serializable::Header::size + tex_res.Size());
                        tex_res.ToString(ipr.buf, ipr.len);
                        ipr.id = -1;
                        IPCTrace("ProcessAPIClient:HandleMessagesLoop LoadResourceResponse url='%s' width=%d height=%d ", ipr.url.c_str(), tex_res.width, tex_res.height);
                        InterProcessProtocol::LoadResourceResponse(tex_res.Type(), ipr.url, ipr.len).ToString(&msg, hdr.seq);
                    } else {
                        IPCTrace("ProcessAPIClient:HandleMessagesLoop LoadResourceResponse failed ");
                        InterProcessProtocol::LoadResourceResponse(0, "", 0).ToString(&msg, hdr.seq);
                    }

                    IPCTrace("TextureResource msg_size=%zd\n", msg.size());
                    CHECK_EQ(msg.size(), conn->WriteFlush(msg));

                } else FATAL("unknown ipr type", req.ipr_type);
            } else FATAL("unknown id ", hdr.id);
            IPCTrace("ProcessAPIClient:HandleMessagesLoop flush %d bytes\n", in.offset);
            conn->ReadFlush(in.offset);
        }
    }
}
#endif

}; // namespace LFL
