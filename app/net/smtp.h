/*
 * $Id: network.h 1335 2014-12-02 04:13:46Z justin $
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

#ifndef LFL_CORE_APP_NET_SMTP_H__
#define LFL_CORE_APP_NET_SMTP_H__
namespace LFL {
  
struct SMTP {
  struct Message {
    string mail_from, content;
    vector<string> rcpt_to;
    void clear() { mail_from.clear(); content.clear(); rcpt_to.clear(); }
  };
  static void HTMLMessage(const string& from, const string& to, const string& subject, const string& content, string *out);
  static void SystemSendmail(const string &message);
  static string EmailFrom(const string &message);
  static int SuccessCode  (int code) { return code == 250; }
  static int RetryableCode(int code) {
    return !code || code == 221 || code == 421 || code == 450 || code == 451 || code == 500 || code == 501 || code == 502 || code == 503;
  }
};

struct SMTPClient : public SocketService {
  typedef function<bool(Connection*, const string&, SMTP::Message*)> DeliverableCB;
  typedef function<void(Connection*, const SMTP::Message &, int, const string&)> DeliveredCB;

  long long total_connected=0, total_disconnected=0, delivered=0, failed=0;
  map<IPV4::Addr, string> domains; string domain;
  SMTPClient() : SocketService("SMTPClient") {}

  int Connected(SocketConnection *c) override { total_connected++; return 0; }
  string HeloDomain(IPV4::Addr addr) const { return domain.empty() ? FindOrDie(domains, addr) : domain; }
  Connection *DeliverTo(IPV4::Addr mx, IPV4EndpointSource*, DeliverableCB deliverableCB, DeliveredCB deliveredCB);

  static void DeliverDeferred(Connection *c);
};

struct SMTPServer : public SocketService {
  long long total_connected=0;
  map<IPV4::Addr, string> domains;
  string domain;

  SMTPServer(const string &n) : SocketService("SMTPServer"), domain(n) {}
  int Connected(SocketConnection *c) override;
  virtual void ReceiveMail(Connection *c, const SMTP::Message &message);
  string HeloDomain(IPV4::Addr addr) const { return domain.empty() ? FindOrDie(domains, addr) : domain; }
};

}; // namespace LFL
#endif // LFL_CORE_APP_NET_SMTP_H__
