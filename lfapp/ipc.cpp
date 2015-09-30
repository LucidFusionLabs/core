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
#include "lfapp/flow.h"
#include "lfapp/gui.h"
#include "lfapp/dom.h"
#include "lfapp/css.h"
#include "lfapp/browser.h"

#include <fcntl.h>

#if defined(LFL_MOBILE)
#elif defined(WIN32)
#define LFL_TCP_IPC
#else
#include <signal.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/mman.h>
#define LFL_MMAPXFER_MPB
#define LFL_SOCKETPAIR_IPC
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

  SC_HANDLE schService = CreateService(schSCManager, name, name, SERVICE_ALL_ACCESS, SERVICE_WIN32_SHARE_PROCESS|SERVICE_INTERACTIVE_PROCESS, 
                                       SERVICE_DEMAND_START, SERVICE_ERROR_NORMAL, path, 0, 0, 0, 0, 0);
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

static string MultiProcessHandleURL = "handle://";
MultiProcessBuffer::MultiProcessBuffer(Connection *c, const ResourceHandle *h) : url(h->url()->str()), len(h->len()) {
  if (PrefixMatch(url, MultiProcessHandleURL)) impl = (void*)strtoul(url.c_str() + MultiProcessHandleURL.size(), 0, 16);
}
MultiProcessBuffer::~MultiProcessBuffer() {
  if (buf) UnmapViewOfFile(buf);
  if (impl != INVALID_HANDLE_VALUE) CloseHandle(impl);
}
void MultiProcessBuffer::Close() {}
bool MultiProcessBuffer::Open() {
  bool read_only = url.size();
  if (!len || (url.size() && !impl)) return ERRORv("mpb open url=", url, " len=", len);
  if (url.empty()) {
    if (!(impl = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, 0, len, NULL))) return ERRORv(false, "CreateFileMapping");
    CHECK(share_process);
    HANDLE rfh = INVALID_HANDLE_VALUE;
    if (FAILED(DuplicateHandle(GetCurrentProcess(), impl, (HANDLE)share_process, &rfh, 0, FALSE, DUPLICATE_SAME_ACCESS))) return ERRORv(false, "DuplicateHandle");
    url = StringPrintf("%s%p", MultiProcessHandleURL.c_str(), rfh);
  }
  if (!(buf = (char*)MapViewOfFile(impl, FILE_MAP_ALL_ACCESS, 0, 0, len))) return ERRORv(false, "MapViewOfFile ", impl);
  return true;
}

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

#if defined(LFL_MMAPXFER_MPB)
static string MultiProcessBufferURL = "fd://transferred";
MultiProcessBuffer::MultiProcessBuffer(Connection *c, const ResourceHandle *h) : url(h->url()->c_str()), len(h->len()) {
  if (url == MultiProcessBufferURL) swap(impl, c->transferred_socket);
}
MultiProcessBuffer::~MultiProcessBuffer() {
  if (buf && buf != MAP_FAILED) munmap(buf, len);
  if (impl >= 0) close(impl);
}
void MultiProcessBuffer::Close() {}
bool MultiProcessBuffer::Open() {
  bool read_only = 0 && url.size();
  if (!len || (url.size() && impl < 0)) return ERRORv("mpb open url=", url, " len=", len, " fd=", impl);
  if (url.empty()) {
    CHECK_EQ(-1, impl);
#ifdef __APPLE__
    string dir = "/var/tmp/";
#else
    string dir = app->dldir;
#endif
    string path = StrCat(dir, app->name, "_mpb.XXXXXXXX");
    if ((impl = mkstemp(&path[0])) < 0) return ERRORv(false, "mkstemp ", path);
    if (unlink(path.c_str())) return ERRORv(false, "unlink ", path);
    if (ftruncate(impl, len)) return ERRORv(false, "ftruncate ", path, " ", len);
    url = MultiProcessBufferURL;
    transfer_handle = impl;
  }
  if ((buf = (char*)mmap(0, len, PROT_READ | (read_only ? 0 : PROT_WRITE), MAP_SHARED, impl, 0)) == MAP_FAILED) return ERRORv(false, "mmap ", impl); 
  return true;
}
#elif defined(LFL_SHM_MPB)
static int ShmKeyFromMultiProcessBufferURL(const string &u) {
  static string shm_url = "shm://";
  CHECK(PrefixMatch(u, shm_url));
  return atoi(u.c_str() + shm_url.size());
}
MultiProcessBuffer::MultiProcessBuffer(Connection *c, const ResourceHandle *h) : url(h->url()->str()), len(h->len()) {}
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
#else
#error no_mpb_impl
#endif

#endif /* WIN32 */

#if defined(LFL_MOBILE) || !defined(LFL_FLATBUFFERS)
void ProcessAPIClient::StartServer(const string &server_program) {}
void ProcessAPIClient::LoadResource(const string &content, const string &fn, const ProcessAPIClient::LoadResourceCompleteCB &cb) {}
void ProcessAPIClient::Navigate(const string &url) {}
void ProcessAPIServer::Start(const string &socket_name) {}
void ProcessAPIServer::WGet(const string &url, const HTTPClient::ResponseCB &c) {}
#else

#ifdef WIN32
static bool EqualLuid(const LUID &l, const LUID &r) { return l.HighPart == r.HighPart && l.LowPart == r.LowPart; }
static HANDLE MakeUninheritableHandle(HANDLE in) {
  HANDLE uninheritable_handle = INVALID_HANDLE_VALUE;
  CHECK(DuplicateHandle(GetCurrentProcess(), in, GetCurrentProcess(), &uninheritable_handle, TOKEN_ALL_ACCESS, 0, 0));
  CloseHandle(in);
  return uninheritable_handle;
}
static HANDLE MakeImpersonationToken(HANDLE in) {
  HANDLE impersonation_token = INVALID_HANDLE_VALUE;
  CHECK(DuplicateToken(in, SecurityImpersonation, &impersonation_token));
  CloseHandle(in);
  return impersonation_token;
}
#endif

bool RPC::Write(Connection *conn, unsigned short id, unsigned short seq, const StringPiece &rpc_text, int transfer_handle) {
  if (conn->state != Connection::Connected) return 0;
  char hdrbuf[RPC::Header::size];
  Serializable::MutableStream o(hdrbuf, sizeof(hdrbuf));
  RPC::Header hdr = { 8+rpc_text.size(), id, seq };
  hdr.Out(&o);
  struct iovec iov[2] = { { hdrbuf, 8 }, { (void*)(rpc_text.data()), static_cast<size_t>(rpc_text.size()) } };
  if (transfer_handle >= 0) return conn->WriteVFlush(iov, 2, transfer_handle) == hdr.len ? hdr.len : 0;
  else                      return conn->WriteVFlush(iov, 2)                  == hdr.len ? hdr.len : 0;
}

void ProcessAPIClient::StartServer(const string &server_program) {
  if (!LocalFile(server_program, "r").Opened()) return ERROR("ProcessAPIClient: \"", server_program, "\" doesnt exist");
  INFO("ProcessAPIClient starting server ", server_program);
  Socket conn_socket = -1;
  bool conn_control_messages = 0;

#if defined(LFL_SOCKETPAIR_IPC)
  Socket fd[2];
  CHECK(SystemNetwork::OpenSocketPair(fd, false));
  SystemNetwork::SetSocketCloseOnExec(fd[1], true);
  string arg0 = server_program, arg1 = StrCat("fd://", fd[0]);
#elif defined(LFL_TCP_IPC)
  Socket l = -1;
  IPV4Endpoint listen;
  CHECK_NE(-1, (l = SystemNetwork::Listen(Protocol::TCP, IPV4::Parse("127.0.0.1"), 0, 1, true)))
  CHECK_EQ(0, SystemNetwork::GetSockName(l, &listen.addr, &listen.port));
  SystemNetwork::SetSocketCloseOnExec(l, true);
  string arg0 = server_program, arg1 = StrCat("tcp://", listen.name());
#elif defined(LFL_NAMEDPIPE_IPC)
  string named_pipe = StrCat("\\\\.\\pipe\\", server_program, ".", getpid()), arg0 = server_program, arg1 = StrCat("np://", named_pipe);
  HANDLE hpipe = CreateNamedPipe(named_pipe.c_str(), PIPE_ACCESS_DUPLEX, PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT, 1, 65536, 65536, 0, NULL);
  CHECK_NE(INVALID_HANDLE_VALUE, hpipe);
  if (!ConnectNamedPipe(hpipe, NULL)) CHECK_EQ(ERROR_PIPE_CONNECTED, GetLastError());
#else
#error no_ipc_impl
#endif

#ifdef WIN32
  LUID change_notify_name_luid;
  BYTE WinWorldSid_buf[SECURITY_MAX_SID_SIZE];
  SID *WinWorldSID = reinterpret_cast<SID*>(WinWorldSid_buf);
  DWORD user_tokeninfo_size = sizeof(TOKEN_USER) + SECURITY_MAX_SID_SIZE, group_tokeninfo_size, priv_tokeninfo_size = 0, WinWorldSid_size = sizeof(WinWorldSid_buf);
  HANDLE process_token = INVALID_HANDLE_VALUE, restricted_token = INVALID_HANDLE_VALUE, impersonation_token = INVALID_HANDLE_VALUE;
  CHECK(OpenProcessToken(GetCurrentProcess(), TOKEN_DUPLICATE | TOKEN_ASSIGN_PRIMARY | TOKEN_QUERY, &process_token));
  GetTokenInformation(process_token, TokenGroups,     NULL, 0, &group_tokeninfo_size);
  GetTokenInformation(process_token, TokenPrivileges, NULL, 0, & priv_tokeninfo_size);
  CHECK(user_tokeninfo_size && group_tokeninfo_size && priv_tokeninfo_size);
  unique_ptr<BYTE> group_tokeninfo_buf(new BYTE[group_tokeninfo_size]), user_tokeninfo_buf(new BYTE[user_tokeninfo_size]), priv_tokeninfo_buf(new BYTE[priv_tokeninfo_size]);
  CHECK(GetTokenInformation(process_token, TokenUser,        user_tokeninfo_buf.get(),  user_tokeninfo_size, & user_tokeninfo_size));
  CHECK(GetTokenInformation(process_token, TokenGroups,     group_tokeninfo_buf.get(), group_tokeninfo_size, &group_tokeninfo_size));
  CHECK(GetTokenInformation(process_token, TokenPrivileges,  priv_tokeninfo_buf.get(),  priv_tokeninfo_size, & priv_tokeninfo_size));
  TOKEN_USER       *token_user   = reinterpret_cast<TOKEN_USER      *>( user_tokeninfo_buf.get());
  TOKEN_GROUPS     *token_groups = reinterpret_cast<TOKEN_GROUPS    *>(group_tokeninfo_buf.get());
  TOKEN_PRIVILEGES *token_privs  = reinterpret_cast<TOKEN_PRIVILEGES*>(priv_tokeninfo_buf.get());
  CHECK(LookupPrivilegeValue(NULL, SE_CHANGE_NOTIFY_NAME, &change_notify_name_luid));
  CHECK(CreateWellKnownSid(WinWorldSid, NULL, WinWorldSid_buf, &WinWorldSid_size));
  vector<SID_AND_ATTRIBUTES> deny_sid, restrict_sid;
  vector<LUID_AND_ATTRIBUTES> deny_priv;
  deny_sid.push_back({ token_user->User.Sid, SE_GROUP_USE_FOR_DENY_ONLY });
  for (int i = 0, l = token_groups->GroupCount; i < l; ++i) {
    PSID sid = token_groups->Groups[i].Sid;
    DWORD a = token_groups->Groups[i].Attributes;
    if (EqualSid(sid, WinWorldSID) || (a & SE_GROUP_INTEGRITY) || (a & SE_GROUP_LOGON_ID)) continue;
    deny_sid.push_back({ token_groups->Groups[i].Sid, SE_GROUP_USE_FOR_DENY_ONLY });
  }
  for (int i = 0, l = token_privs->PrivilegeCount; i < l; ++i) {
    const LUID &luid = token_privs->Privileges[i].Luid;
    if (EqualLuid(change_notify_name_luid, luid)) continue;
    deny_priv.push_back({ luid, 0 });
  }
  CHECK(CreateRestrictedToken(process_token, SANDBOX_INERT, deny_sid.size(), deny_sid.data(), deny_priv.size(), deny_priv.data(),
                              restrict_sid.size(), restrict_sid.data(), &restricted_token));
  CHECK(CreateRestrictedToken(process_token, SANDBOX_INERT, 0,               NULL,            deny_priv.size(), deny_priv.data(),
                              restrict_sid.size(), restrict_sid.data(), &impersonation_token));
  CHECK((impersonation_token = MakeImpersonationToken (impersonation_token)));
  CHECK((impersonation_token = MakeUninheritableHandle(impersonation_token)));
  CHECK((   restricted_token = MakeUninheritableHandle(   restricted_token)));
  string av = StrCat(arg0, " ", arg1);
  PROCESS_INFORMATION pi;
  STARTUPINFO si;
  memzero(pi);
  memzero(si);
  si.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
  if (!CreateProcessAsUser(restricted_token, 0, &av[0], 0, 0, 0, CREATE_SUSPENDED | DETACHED_PROCESS, 0, 0, &si, &pi)) FATAL("CreateProcess", av, ": ", GetLastError());
  server_process = pi.hProcess;
  CHECK(SetThreadToken(&pi.hThread, impersonation_token));
  CHECK(ResumeThread(pi.hThread));
  CloseHandle(pi.hThread);
  CloseHandle(impersonation_token);
  CloseHandle(restricted_token);
  CloseHandle(process_token);
#else
  int pid = fork();
  if (!pid) {
    // close(0); close(1); close(2);
    vector<char*> av = { &arg0[0], &arg1[0], 0 };
    CHECK(!execvp(av[0], &av[0]));
  }
  CHECK_GT(pid, 0);
#endif

#if defined(LFL_SOCKETPAIR_IPC)
  close(fd[0]);
  conn_socket = fd[1];
  conn_control_messages = true;
#elif defined(LFL_TCP_IPC)
  CHECK_NE(-1, (conn_socket = SystemNetwork::Accept(l, 0, 0)));
  SystemNetwork::SetSocketBlocking(conn_socket, 0);
  SystemNetwork::CloseSocket(l);
#else
#error no_ipc_impl
#endif
  
  conn = new Connection(Singleton<UnixClient>::Get(), new ConnectionHandler(this));
  CHECK_NE(-1, (conn->socket = conn_socket));
  conn->control_messages = conn_control_messages;
  conn->state = Connection::Connected;
  conn->svc->conn[conn->socket] = conn;
  app->network->active.Add(conn->socket, SocketSet::READABLE, &conn->self_reference);
}

void ProcessAPIClient::Navigate(const string &url) { 
  bool ok = SendRPC(conn, seq++, -1, NavigateRequest, fb.CreateString(url));
  IPCTrace("ProcessAPIClient Navigate url='%s' sent=%d\n", url.c_str(), ok);
}

void ProcessAPIClient::LoadResource(const string &content, const string &fn, const ProcessAPIClient::LoadResourceCompleteCB &cb) { 
  MultiProcessBuffer mpb(server_process);
  if (!conn || !mpb.Create(MultiProcessFileResource(content, fn, ""))) return cb(MultiProcessTextureResource());
  bool ok = SendRPC(conn, seq++, mpb.transfer_handle, LoadResourceRequest, MakeResourceHandle(MultiProcessFileResource::Type, mpb));
  if (ok) reqmap[seq-1] = new LoadResourceQuery(cb);
  else cb(MultiProcessTextureResource());
}

// void ProcessAPIClient::LoadResourceResponse(Connection *c, int seq, const LoadResourceResponse *res) {}

void ProcessAPIClient::LoadResourceSucceeded(LoadResourceQuery *query, const MultiProcessTextureResource &tex) {
  if (reply_success_from_main_thread && !MainThread()) return RunInMainThread(new Callback(bind(&ProcessAPIClient::LoadResourceSucceeded, this, query, tex)));
  query->cb(tex);
  // query->response.Close();
  delete query;
}

void ProcessAPIClient::HandleLoadResourceResponse(Connection *c, int seq, const LoadResourceResponse *res) {
  auto reply = reqmap.find(seq);
  auto mpbi = ipc_buffer.find(res->mpb_id());
  if (reply == reqmap.end()) return ERROR("ProcessAPIClient missing seq ", seq);
  if (mpbi == ipc_buffer.end()) return ERROR("ProcessAPIClient missing mpb ", res->mpb_id());
  bool erase_reply = true, delete_reply = true;

  MultiProcessBuffer *res_mpb = mpbi->second;
  IPCTrace("ProcessAPIClient LoadResourceResponse url='%s'\n", res_mpb->url.c_str());

  MultiProcessTextureResource tex;
  if (res_mpb->buf && MultiProcessResource::Read(*res_mpb, MultiProcessTextureResource::Type, &tex)) {
    IPCTrace("ProcessAPIClient Texture width=%d height=%d ret=1\n", tex.width, tex.height);
    LoadResourceSucceeded(reply->second, tex);
    delete_reply = false;
  } else { ERROR("ProcessAPIClient res_mpb.Open: ", res_mpb->url); reply->second->cb(MultiProcessTextureResource()); }

  if (erase_reply) reqmap.erase(seq);
  if (delete_reply) delete reply->second;
}

void ProcessAPIClient::HandleAllocateBufferRequest(Connection *c, int seq, const AllocateBufferRequest *req) {
  IPCTrace("ProcessAPIClient AllocateBufferRequest bytes=%d\n", req->bytes());

  for (++ipc_buffer_id; !ipc_buffer_id || Contains(ipc_buffer, ipc_buffer_id); ++ipc_buffer_id) {}
  auto mpb = new MultiProcessBuffer(server_process);
  ipc_buffer[ipc_buffer_id] = mpb;

  if (!mpb->Create(req->bytes())) return ERROR("ProcessAPIClient ResposneMPB Create");
  SendRPC(c, seq, mpb->transfer_handle, AllocateBufferResponse, MakeResourceHandle(req->type(), *mpb), ipc_buffer_id);
}

void ProcessAPIClient::HandleNavigateResponse(Connection *c, int seq, const NavigateResponse *req) {}

void ProcessAPIClient::HandleWGetRequest(Connection *c, int seq, const WGetRequest *req) {
  string url = req->url()->str();
  WGetQuery *wget = new WGetQuery(this, seq);
  IPCTrace("ProcessAPIClient WGetRequest url='%s'\n", url.c_str());
  if (app->network_thread) app->network_thread->Write(new Callback([=]{ Singleton<HTTPClient>::Get()->WGet(url, 0, bind(&WGetQuery::WGetResponseCB, wget, _1, _2, _3, _4, _5)); }));
  else                                                                { Singleton<HTTPClient>::Get()->WGet(url, 0, bind(&WGetQuery::WGetResponseCB, wget, _1, _2, _3, _4, _5)); }
}

void ProcessAPIClient::WGetQuery::WGetResponseCB(Connection *c, const char *h, const string &ct, const char *b, int l) {
  int len = h ? strlen(h)+1 : l;
  MultiProcessBuffer res_mpb(parent->server_process);
  if (len && res_mpb.Create(len)) {
    memcpy(res_mpb.buf, h ? h : b, len);
    parent->SendRPC(parent->conn, seq, res_mpb.transfer_handle, WGetResponse, h!=0, MakeResourceHandle(0, res_mpb));
    IPCTrace("ProcessAPIClient WGetResponseCB %d %d %d %d %s\n", h!=0, b!=0, l, len, res_mpb.url.c_str());
    res_mpb.Close();
  } else if (len) ERROR("ProcessAPIClient ResposneMPB Create");
  if (!h && (!b || !l)) delete this;
}

void ProcessAPIServer::Start(const string &socket_name) {
  static string fd_url = "fd://", np_url = "np://", tcp_url = "tcp://";
  if (PrefixMatch(socket_name, fd_url)) {
    conn = new Connection(Singleton<UnixClient>::Get(), static_cast<Connection::Handler*>(nullptr));
    conn->state = Connection::Connected;
    conn->socket = atoi(socket_name.c_str() + fd_url.size());
    conn->control_messages = true;
  } else if (PrefixMatch(socket_name, tcp_url)) {
    conn = new Connection(Singleton<UnixClient>::Get(), static_cast<Connection::Handler*>(nullptr));
    string host, port;
    HTTP::ParseHost(socket_name.c_str() + tcp_url.size(), socket_name.c_str() + socket_name.size(), &host, &port);
    CHECK_NE(-1, (conn->socket = SystemNetwork::OpenSocket(Protocol::TCP)));
    CHECK_EQ(0, SystemNetwork::Connect(conn->socket, IPV4::Parse(host), atoi(port), 0));
    SystemNetwork::SetSocketBlocking(conn->socket, 0);
    conn->state = Connection::Connected;
  } else return;
  INFO("ProcessAPIServer opened ", socket_name);
}

void ProcessAPIServer::HandleLoadResourceRequest(Connection *c, int seq, const LoadResourceRequest *req, const MultiProcessFileResource &mpf) {
  IPCTrace("ProcessAPIServer LoadResourceRequest fn='%s' %p %d\n", mpf.name.buf, mpf.buf.buf, mpf.buf.len);
  Texture *tex=0;
  if ((tex = Asset::LoadTexture(mpf))) {
    IPCTrace("ProcessAPIServer AllocateBufferBufferRequest %d\n", tex->BufferSize());
    if (!SendRPC(conn, seq, -1, AllocateBufferRequest, MultiProcessBuffer::Size(MultiProcessTextureResource(*tex)),
                 MultiProcessTextureResource::Type)) ERROR("ProcessAPIServer write");
    else swap(resmap[seq], tex);
  } else {
    IPCTrace("ProcessAPIServer LoadResourceResponse failed\n");
    if (!SendRPC(conn, seq, -1, LoadResourceResponse, 0)) ERROR("ProcessAPIServer write");
  }
  delete tex;
}

void ProcessAPIServer::HandleAllocateBufferResponse(Connection *c, int seq, const AllocateBufferResponse *res, MultiProcessBuffer &mpb) {
  IPCTrace("ProcessAPIServer AllocateBufferResponse url='%s'\n", res->mpb()->url()->c_str());
  auto reply = resmap.find(seq);
  if (reply != resmap.end()) {
    bool success = mpb.Copy(MultiProcessTextureResource(*reply->second));
    delete reply->second;
    resmap.erase(seq);
    if (!SendRPC(conn, seq, -1, LoadResourceResponse, res->mpb_id())) ERROR("ProcessAPIServer write");
  } else ERROR("missing seq ", seq);
}

void ProcessAPIServer::HandleWGetResponse(Connection *c, int seq, const WGetResponse *res, MultiProcessBuffer &mpb) {
  IPCTrace("ProcessAPIServer WGetResponse '%s'\n", res->mpb()->url()->c_str());
  auto wget = wgetmap.find(seq);
  if (wget != wgetmap.end() || !res->mpb()->url()->str().size()) {
    if (1) { // mpb.Open()) {
      if (res->headers() && mpb.len) wget->second(0, mpb.buf, "", 0, 0);
      else                           wget->second(0, 0, "", mpb.buf, mpb.len);
      if (!mpb.len) wgetmap.erase(seq);
    } else { ERROR("wget mpb read"); wget->second(0, 0, "", 0, 0); wgetmap.erase(seq); }
  } else ERROR("missing seq ", seq);
}

void ProcessAPIServer::HandleNavigateRequest(Connection *c, int seq, const NavigateRequest *req) {
  IPCTrace("ProcessAPIServer NavigateRequest url='%s'\n", req->url()->c_str());
  browser->Open(req->url()->str());
}

void ProcessAPIServer::WGet(const string &url, const HTTPClient::ResponseCB &c) {
  if (!SendRPC(conn, seq++, -1, WGetRequest, fb.CreateString(url))) return ERROR("ProcessAPIServer write");
  else wgetmap[seq-1] = c;
}
#endif // LFL_MOBILE

}; // namespace LFL
