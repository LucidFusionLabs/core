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
#include <sys/mman.h>
#endif

namespace LFL {
#ifndef WIN32
int NTService::Install  (const char *name, const char *path) { FATAL("not implemented"); }
int NTService::Uninstall(const char *name)                   { FATAL("not implemented"); }
int NTService::WrapMain (const char *name, MainCB main_cb, int argc, const char **argv) { return main_cb(argc, argv); }
#endif
#if defined(LFL_MOBILE)
int ProcessPipe::OpenPTY(const char **argv, const char *startdir) { FATAL("not implemented"); }
int ProcessPipe::Open   (const char **argv, const char *startdir) { FATAL("not implemented"); }
int ProcessPipe::Close()                                          { FATAL("not implemented"); }
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

  SC_HANDLE schService = CreateService(schSCManager,         /* SCManager database      */ 
                                       name,			           /* name of service         */ 
                                       name,                 /* service name to display */ 
                                       SERVICE_ALL_ACCESS,   /* desired access          */ 
                                       SERVICE_WIN32_SHARE_PROCESS|SERVICE_INTERACTIVE_PROCESS, 
                                       SERVICE_DEMAND_START, /* start type              */ 
                                       SERVICE_ERROR_NORMAL, /* error control type      */ 
                                       path,			           /* service's binary        */ 
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

int ProcessPipe::Close() { return 0; }
int ProcessPipe::OpenPTY(const char **argv, const char *startdir) { return Open(argv); }
int ProcessPipe::Open(const char **argv, const char *startdir) {
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

MultiProcessBuffer::MultiProcessBuffer(Connection *c, const InterProcessProtocol::ResourceHandle &h) : url(h.url), len(h.len) {}
MultiProcessBuffer::~MultiProcessBuffer() {}
bool MultiProcessBuffer::Open() { return 0; }
void MultiProcessBuffer::Close() {}

#else /* WIN32 */

int ProcessPipe::Open(const char **argv, const char *startdir) {
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
    if (startdir) chdir(startdir);
    execvp(argv[0], (char*const*)argv);
  }
  return 0;
}

extern "C" pid_t forkpty(int *, char *, struct termios *, struct winsize *);
int ProcessPipe::OpenPTY(const char **argv, const char *startdir) {
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
    if (startdir) chdir(startdir);
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

#if 1
static string MultiProcessBufferURL = "fd://transferred";
MultiProcessBuffer::MultiProcessBuffer(Connection *c, const InterProcessProtocol::ResourceHandle &h) : url(h.url), len(h.len) {
  if (url == MultiProcessBufferURL) swap(impl, c->transferred_socket);
}
MultiProcessBuffer::~MultiProcessBuffer() {
  if (buf && buf != MAP_FAILED) munmap(buf, len);
  if (impl >= 0) close(impl);
}
void MultiProcessBuffer::Close() {}
bool MultiProcessBuffer::Open() {
  bool read_only = url.size();
  if (!len || (url.size() && impl < 0)) return ERRORv("mpb open url=", url, " len=", len, " fd=", impl);
  if (url.empty()) {
    CHECK_EQ(-1, impl);
#ifdef __APPLE__
    string dir = "/var/tmp/";
#else
    string dir = app->dldir;
#endif
    string path = StrCat(dir, app->name, "_mpb.XXXXXXXX");
    if ((impl = open(mktemp(&path[0]), O_RDWR|O_CREAT|O_EXCL, 0600)) < 0) return ERRORv(false, "open ", path);
    if (unlink(path.c_str())) return ERRORv(false, "unlink ", path);
    if (ftruncate(impl, len)) return ERRORv(false, "ftruncate ", path, " ", len);
    url = MultiProcessBufferURL;
    transfer_socket = impl;
  }
  if ((buf = (char*)mmap(0, len, PROT_READ | (read_only ? 0 : PROT_WRITE), MAP_SHARED, impl, 0)) == MAP_FAILED) return ERRORv(false, "mmap ", impl); 
  return true;
}
#else
static int ShmKeyFromMultiProcessBufferURL(const string &u) {
  static string shm_url = "shm://";
  CHECK(PrefixMatch(u, shm_url));
  return atoi(u.c_str() + shm_url.size());
}
MultiProcessBuffer::MultiProcessBuffer(Connection*, const InterProcessProtocol::ResourceHandle &h) : url(h.url), len(h.len) {}
MultiProcessBuffer::~MultiProcessBuffer() { if (buf) shmdt(buf); }
void MultiProcessBuffer::Close() { if (impl >= 0) shmctl(impl, IPC_RMID, NULL); }
bool MultiProcessBuffer::Open() {
  if (!len) return false;
  int key = url.empty() ? rand() : ShmKeyFromMultiProcessBufferURL(url);
  if ((impl = shmget(key, len, url.empty() ? (IPC_CREAT | 0600) : 0400)) < 0)
    return ERRORv(false, "MultiProcessBuffer Open id=", impl, ", size=", len, ", url=", url, ": ", strerror(errno));

  CHECK_GE(impl, 0);
  buf = reinterpret_cast<char*>(shmat(impl, NULL, 0));
  CHECK(buf);
  CHECK_NE((char*)-1, buf);
  if (url.empty()) url = StrCat("shm://", key);
  return true;
}
#endif

#endif /* WIN32 */

#ifdef LFL_IPC_DEBUG
#define IPCTrace(...) printf(__VA_ARGS__)
#else
#define IPCTrace(...)
#endif

#ifdef LFL_MOBILE
void ProcessAPIClient::StartServer(const string &server_program) {}
void ProcessAPIClient::LoadResource(const string &content, const string &fn, const ProcessAPIClient::LoadResourceCompleteCB &cb) {}
void ProcessAPIServer::Start(const string &socket_name) {}
void ProcessAPIServer::HandleMessagesLoop() {}
#else
static int ProcessAPIWrite(Connection *conn, const Serializable &req, int seq, int transfer_socket=-1) {
  if (conn->state != Connection::Connected) return 0;
  string msg;
  req.ToString(&msg, seq);
  if (transfer_socket >= 0) return conn->WriteFlush(msg.data(), msg.size(), transfer_socket) == msg.size() ? msg.size() : 0;
  else                      return conn->WriteFlush(msg)                                     == msg.size() ? msg.size() : 0;
}

void ProcessAPIClient::StartServer(const string &server_program) {
  Socket fd[2];
  CHECK(SystemNetwork::OpenSocketPair(fd));
  if (!LocalFile(server_program, "r").Opened()) return ERROR("ProcessAPIClient: \"", server_program, "\" doesnt exist");
  INFO("ProcessAPIClient starting server ", server_program);

#ifdef WIN32
  FATAL("not implemented")
#else
  if ((pid = fork())) {
    CHECK_GT(pid, 0);
    close(fd[0]);
    conn = new Connection(Singleton<UnixClient>::Get(), new ConnectionHandler(this));
    conn->state = Connection::Connected;
    conn->control_messages = true;
    conn->socket = fd[1];
    conn->svc->conn[conn->socket] = conn;
    app->network.active.Add(conn->socket, SocketSet::READABLE, &conn->self_reference);
  } else {
    close(fd[1]);
    // close(0); close(1); close(2);
    string arg0 = server_program, arg1 = StrCat("fd://", fd[0]);
    vector<char*> av = { &arg0[0], &arg1[0], 0 };
    CHECK(!execvp(av[0], &av[0]));
  }
#endif
}

void ProcessAPIClient::LoadResource(const string &content, const string &fn, const ProcessAPIClient::LoadResourceCompleteCB &cb) { 
  MultiProcessBuffer mpb;
  if (!conn || !mpb.Create(MultiProcessResource::File(content, fn, ""))) return cb(MultiProcessResource::Texture());

  reqmap[seq] = cb;
  int wrote = ProcessAPIWrite(conn, InterProcessProtocol::LoadResourceRequest(MultiProcessResource::File::Type, mpb.url, mpb.len),
                              seq++, mpb.transfer_socket);
  IPCTrace("ProcessAPIClient LoadResource fn='%s' url='%s' response=%zd\n", fn.c_str(), mpb.url.c_str(), wrote);
  if (!wrote) cb(MultiProcessResource::Texture());
}

int ProcessAPIClient::ConnectionHandler::Read(Connection *c) {
  while (c->rl >= Serializable::Header::size) {
    IPCTrace("ProcessAPIClient begin parse %d bytes\n", c->rl);
    Serializable::ConstStream in(c->rb, c->rl);
    Serializable::Header hdr;
    hdr.In(&in);

    auto reply = parent->reqmap.find(hdr.seq);
    CHECK_NE(parent->reqmap.end(), reply);

    if (hdr.id == InterProcessProtocol::LoadResourceResponse::Type) {
      InterProcessProtocol::LoadResourceResponse res;
      if (res.Read(&in)) break;
      IPCTrace("ProcessAPIClient LoadResourceResponse url='%s'\n", res.mpb.url.c_str());

      MultiProcessResource::Texture tex;
      MultiProcessBuffer res_mpb(c, res.mpb);
      if (res.mpb.type == MultiProcessResource::Texture::Type && res_mpb.Open() && ReadTexture(res, res_mpb, &tex)) reply->second(tex);
      else { ERROR("ProcessAPIClient res_mpb.Open: ", res.mpb.url); reply->second(MultiProcessResource::Texture()); }
      res_mpb.Close();

    } else FATAL("ProcessAPIClient unknown hdr id ", hdr.id);

    parent->reqmap.erase(hdr.seq);
    c->ReadFlush(in.offset);
    IPCTrace("ProcessAPIClient flushed %d bytes\n", in.offset);
  }
  return 0;
}

bool ProcessAPIClient::ConnectionHandler::ReadTexture(const InterProcessProtocol::LoadResourceResponse &res, const MultiProcessBuffer &res_mpb,
                                                      MultiProcessResource::Texture *tex) {
  if (!MultiProcessResource::Read(res_mpb, res.mpb.type, tex)) { ERROR("mpb read"); return false; }
  IPCTrace("ProcessAPIClient Texture width=%d height=%d\n", tex->width, tex->height);
  return true;
}

void ProcessAPIServer::Start(const string &socket_name) {
  static string fd_url = "fd://";
  if (PrefixMatch(socket_name, fd_url)) {
    conn = new Connection(Singleton<UnixClient>::Get(), static_cast<Connection::Handler*>(nullptr));
    conn->state = Connection::Connected;
    conn->socket = atoi(socket_name.c_str() + fd_url.size());
    conn->control_messages = true;
  } else return;
  INFO("ProcessAPIServer opened ", socket_name);
}

void ProcessAPIServer::HandleMessagesLoop() {
  while (app->run) {
    if (!NBReadable(conn->socket, -1)) continue;
    if (conn->Read() <= 0) { ERROR(conn->Name(), ": read "); break; }

    while (conn->rl >= Serializable::Header::size) {
      IPCTrace("ProcessAPIServer begin parse %d bytes\n", conn->rl);
      Serializable::ConstStream in(conn->rb, conn->rl);
      Serializable::Header hdr;
      hdr.In(&in);

      if (hdr.id == InterProcessProtocol::LoadResourceRequest::Type) {
        InterProcessProtocol::LoadResourceRequest req;
        if (req.Read(&in)) break;
        IPCTrace("ProcessAPIServer LoadResourceRequest url='%s'\n", req.mpb.url.c_str());

        Texture orig_tex, scaled_tex, *input_tex = 0;
        MultiProcessBuffer req_mpb(conn, req.mpb), res_mpb;
        if (req.mpb.type == MultiProcessResource::File::Type && req_mpb.Open() &&
            (input_tex = LoadTexture(req, req_mpb, &orig_tex, &scaled_tex)) &&
            res_mpb.Create(MultiProcessResource::Texture(*input_tex))) {
          IPCTrace("ProcessAPIServer LoadResourceResponse url='%s'\n", res_mpb.url.c_str());
          if (!ProcessAPIWrite(conn, InterProcessProtocol::LoadResourceResponse(MultiProcessResource::Texture::Type, res_mpb.url, res_mpb.len),
                               hdr.seq, res_mpb.transfer_socket)) ERROR("ProcessAPIServer write");
        } else {
          IPCTrace("ProcessAPIServer LoadResourceResponse failed\n");
          if (!ProcessAPIWrite(conn, InterProcessProtocol::LoadResourceResponse(0, "", 0), hdr.seq)) ERROR("ProcessAPIServer write");
        }
        req_mpb.Close();

      } else FATAL("unknown id ", hdr.id);

      IPCTrace("ProcessAPIServer flush %d bytes\n", in.offset);
      conn->ReadFlush(in.offset);
    }
  }
}

Texture *ProcessAPIServer::LoadTexture(const InterProcessProtocol::LoadResourceRequest &req, const MultiProcessBuffer &req_mpb,
                                       Texture *orig_tex, Texture *scaled_tex) {
  MultiProcessResource::File file;
  if (!MultiProcessResource::Read(req_mpb, req.mpb.type, &file)) { ERROR("mpb read"); return 0; }
  IPCTrace("ProcessAPIServer File fn='%s' %p %d\n", file.name.buf, file.buf.buf, file.buf.len);

  Asset::LoadTexture(file.buf.data(), file.name.data(), file.buf.size(), orig_tex, 0);
  Texture *tex = orig_tex;

  const int max_image_size = 1000000;
  if (orig_tex->BufferSize() >= max_image_size) {
    tex = scaled_tex;
    float scale_factor = sqrt((float)max_image_size/orig_tex->BufferSize());
    scaled_tex->Resize(orig_tex->width*scale_factor, orig_tex->height*scale_factor, Pixel::RGB24, Texture::Flag::CreateBuf);

    VideoResampler resampler;
    resampler.Open(orig_tex->width, orig_tex->height, orig_tex->pf, scaled_tex->width, scaled_tex->height, scaled_tex->pf);
    resampler.Resample(orig_tex->buf, orig_tex->LineSize(), scaled_tex->buf, scaled_tex->LineSize());
  }
  return tex->buf ? tex : 0;
}
#endif // LFL_MOBILE

}; // namespace LFL
