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

#ifndef LFL_LFAPP_RESOLVER_H__
#define LFL_LFAPP_RESOLVER_H__
namespace LFL {
    
struct Resolver {
  typedef function<void(IPV4::Addr, DNS::Response*)> ResponseCB;
  struct Nameserver;
  struct Request {
    typedef unsigned short Type;
    string query;
    Type type;
    Time stamp;
    unsigned retrys=0;
    ResponseCB cb;
    Request() : type(DNS::Type::A), stamp(Now()) {}
    Request(const string &Q, Type T=DNS::Type::A, ResponseCB CB=ResponseCB(), int R=0) :
      query(Q), type(T), stamp(Now()), retrys(R), cb(CB) {}
  };

  struct Nameserver {
    Connection *c=0;
    bool timedout=0;
    unordered_map<unsigned short, Request> request_map;
    void Timedout() { timedout=true; c->SetError(); INFO(c->Name(), ": nameserver timeout"); }
    bool WriteResolveRequest(const Request &req);
  };

  vector<Request> queue;
  vector<IPV4::Addr> conn_available;
  unordered_map<IPV4::Addr, unique_ptr<Nameserver>> conn;
  int max_outstanding_per_ns=10, auto_disconnect_seconds=0;
  bool remove_timedout_servers = false;

  virtual ~Resolver() {}
  virtual bool HandleNoConnections() { return false; }
  bool Connected() const { for (auto &n : conn) if (n.second->c->state == Connection::Connected) return 1; return 0; }
  Nameserver *Connect(const vector<IPV4::Addr> &addrs, bool randomize=false);
  Nameserver *Connect(IPV4::Addr addr);
  void HandleClosed(Nameserver*);
  void HandleResponse(Nameserver*, const DNS::Header *hdr, int len);
  void Heartbeat(Nameserver*);
  void DequeueTo(Nameserver*);

  void NSLookup(const string &host, const ResponseCB &cb);
  bool QueueResolveRequest(const Request &req);
  Nameserver *NextAvailableNameserver(bool skip_first=false);
};

struct SystemResolver : public Resolver {
  vector<IPV4::Addr> nameservers;
  SystemResolver() { remove_timedout_servers = true; }
  bool HandleNoConnections();
  static void GetNameservers(vector<IPV4::Addr> *nameservers);
};

struct RecursiveResolver {
  struct AuthorityTreeNode {
    typedef map<string, unique_ptr<AuthorityTreeNode>> Children;
    typedef map<string, unique_ptr<DNS::Response>> Cache;
    int depth=0;
    Children child;
    Resolver resolver;
    Cache Acache, MXcache;
    DNS::Response authority;
    string authority_domain;
    AuthorityTreeNode() { resolver.auto_disconnect_seconds=10; }
  };

  struct Request {
    typedef unsigned short Type;
    string query;
    Type type;
    Resolver::ResponseCB cb;
    vector<DNS::Response> answer;
    bool missing_answer=0;
    Request *parent_request;
    unordered_set<Request*> child_request, pending_child_request;
    unordered_set<AuthorityTreeNode*> seen_authority;

    Request(const string &Q, Type T=DNS::Type::A, Resolver::ResponseCB C=Resolver::ResponseCB(), Request *P=0) :
      query(Q), type(T), cb(C), parent_request(P) {}
    virtual ~Request() { CHECK_EQ(child_request.size(), 0); }
    int Ancestors() const { return parent_request ? 1 + parent_request->Ancestors() : 0; }
  };

  AuthorityTreeNode root;
  long long queries_requested=0, queries_completed=0;
  RecursiveResolver() { ConnectoToRootServers(); }

  void ConnectoToRootServers();
  bool StartResolveRequest(Request *req);
  bool QueueResolveRequest(AuthorityTreeNode *node, Request *req);
  AuthorityTreeNode *GetAuthorityTreeNode(const string &query, bool create);
  void HandleResponse(Request *req, IPV4::Addr addr, DNS::Response *res, vector<DNS::Response> *subres);
  int ResolveAnyMissingAnswers(Request *req, const vector<DNS::Record> &R, const DNS::AnswerMap *answer);
  void Complete(Request *req, IPV4::Addr addr, DNS::Response *res);
  void HandleChildResponse(Request *req, Request *subreq, DNS::Response *res);
  void StartChildResolve(Request *req, Request *subreq);
};

}; // namespace LFL
#endif // LFL_LFAPP_RESOLVER_H__
