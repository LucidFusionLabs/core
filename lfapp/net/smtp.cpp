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

#include "lfapp/lfapp.h"
#include "lfapp/network.h"
#include "lfapp/net/smtp.h"

namespace LFL {
struct SMTPClientConnection : public Connection::Handler {
  enum { INIT=0, SENT_HELO=1, READY=2, MAIL_FROM=3, RCPT_TO=4, SENT_DATA=5, SENDING=6, RESETING=7, QUITING=8 };
  SMTPClient *server;
  SMTPClient::DeliverableCB deliverable_cb;
  SMTPClient::DeliveredCB delivered_cb;
  int state=0, rcpt_index=0;
  string greeting, helo_domain, ehlo_response, response_lines;
  SMTP::Message mail;
  string RcptTo(int index) const { return StrCat("RCPT TO: <", mail.rcpt_to[index], ">\r\n"); }
  SMTPClientConnection(SMTPClient *S, SMTPClient::DeliverableCB CB1, SMTPClient::DeliveredCB CB2)
    : server(S), deliverable_cb(CB1), delivered_cb(CB2) {}

  int Connected(Connection *c) { helo_domain = server->HeloDomain(c->src_addr); return 0; }

  void Close(Connection *c) {
    server->total_disconnected++;
    if (DeliveringState(state)) delivered_cb(0, mail, 0, "");
    deliverable_cb(c, helo_domain, 0);
  }

  int Read(Connection *c) {
    int processed = 0;
    StringLineIter lines(c->rb.buf, StringLineIter::Flag::BlankLines);
    for (string line = lines.NextString(); !lines.Done(); line = lines.NextString()) {
      processed = lines.next_offset;
      if (!response_lines.empty()) response_lines.append("\r\n");
      response_lines.append(line);

      const char *dash = FindChar(line.c_str(), notnum);
      bool multiline = dash && *dash == '-';
      if (multiline) continue;

      int code = atoi(line), need_code=0; string response;
      if      (state == INIT)        { response=StrCat("EHLO ", helo_domain, "\r\n"); greeting=response_lines; }
      else if (state == SENT_HELO)   { need_code=250; ehlo_response=response_lines; }
      else if (state == READY)       { ERROR("read unexpected line: ", response_lines); return -1; }
      else if (state == MAIL_FROM)   { need_code=250; response=RcptTo(rcpt_index++); }
      else if (state == RCPT_TO)     { need_code=250; response="DATA\r\n";
        if (rcpt_index < mail.rcpt_to.size()) { response=RcptTo(rcpt_index++); state--; }
      }
      else if (state == SENT_DATA)   { need_code=354; response=StrCat(mail.content, "\r\n.\r\n"); }
      else if (state == SENDING)     { delivered_cb(c, mail, code, response_lines); server->delivered++; state=READY-1; }
      else if (state == RESETING)    { need_code=250; state=READY-1; }
      else if (state == QUITING)     { /**/ }
      else { ERROR("unknown state ", state); return -1; }

      if (need_code && code != need_code) {
        ERROR(StateString(state), " failed: ", response_lines);
        if (state == SENT_HELO || state == RESETING || state == QUITING) return -1;

        if (DeliveringState(state)) delivered_cb(c, mail, code, response_lines);
        response="RSET\r\n"; server->failed++; state=RESETING-1;
      }
      if (!response.empty()) if (c->WriteFlush(response) != response.size()) return -1;

      response_lines.clear();
      state++;
    }
    c->ReadFlush(processed);

    if (state == READY) {
      mail.clear();
      rcpt_index = 0;
      if (deliverable_cb(c, helo_domain, &mail)) Deliver(c);
    }
    return 0;
  }

  void Deliver(Connection *c) {
    string response;
    if (!mail.mail_from.size() || !mail.rcpt_to.size() || !mail.content.size()) { response="QUIT\r\n"; state=QUITING; }
    else { response=StrCat("MAIL FROM: <", mail.mail_from, ">\r\n"); state++; }
    if (c->WriteFlush(response) != response.size()) c->SetError();
  }

  static const char *StateString(int n) {
    static const char *s[] = { "INIT", "SENT_HELO", "READY", "MAIL_FROM", "RCPT_TO", "SENT_DATA", "SENDING", "RESETING", "QUITTING" };
    return (n >= 0 && n < sizeofarray(s)) ? s[n] : "";
  }
  static bool DeliveringState(int state) { return (state >= MAIL_FROM && state <= SENDING); }
};

Connection *SMTPClient::DeliverTo(IPV4::Addr ipv4_addr, IPV4EndpointSource *src_pool,
                                  DeliverableCB deliverable_cb, DeliveredCB delivered_cb) {
  static const int tcp_port = 25;
  Connection *c = app->net->tcp_client->Connect(ipv4_addr, tcp_port, src_pool);
  if (!c) return 0;

  c->handler = make_unique<SMTPClientConnection>(this, deliverable_cb, delivered_cb);
  return c;
}

void SMTPClient::DeliverDeferred(Connection *c) { dynamic_cast<SMTPClientConnection*>(c->handler.get())->Deliver(c); }

/* SMTPServer */

struct SMTPServerConnection : public Connection::Handler {
  SMTPServer *server;
  string my_domain, client_domain;
  SMTP::Message message;
  bool in_data;

  SMTPServerConnection(SMTPServer *s) : server(s), in_data(0) {}
  void ClearStateTable() { message.mail_from.clear(); message.rcpt_to.clear(); message.content.clear(); in_data=0; }

  int Connected(Connection *c) {
    my_domain = server->HeloDomain(c->src_addr);
    string greeting = StrCat("220 ", my_domain, " Simple Mail Transfer Service Ready\r\n");
    return (c->Write(greeting) == greeting.size()) ? 0 : -1;
  }
  int Read(Connection *c) {
    int offset = 0, processed;
    while (c->state == Connection::Connected) {
      bool last_in_data = in_data;
      if (in_data) { if ((processed = ReadData    (c, c->rb.begin()+offset, c->rb.size()-offset)) < 0) return -1; }
      else         { if ((processed = ReadCommands(c, c->rb.begin()+offset, c->rb.size()-offset)) < 0) return -1; }
      offset += processed;
      if (last_in_data == in_data) break;
    }
    if (offset) c->ReadFlush(offset);
    return 0;
  }

  int ReadCommands(Connection *c, const char *in, int len) {
    int processed = 0;
    StringLineIter lines(StringPiece(in, len), StringLineIter::Flag::BlankLines);
    for (const char *line = lines.Next(); line && lines.next_offset>=0 && !in_data; line = lines.Next()) {
      processed = lines.next_offset;
      StringWordIter words(line, lines.cur_len, isint3<' ', '\t', ':'>);
      string cmd = toupper(words.NextString());
      string a1_orig = words.NextString();
      string a1 = toupper(a1_orig), response="500 unrecognized command\r\n";

      if (cmd == "MAIL" && a1 == "FROM") {
        ClearStateTable();
        message.mail_from = words.RemainingString();
        response = "250 OK\r\n";
      }
      else if (cmd == "RCPT" && a1 == "TO") {
        message.rcpt_to.push_back(words.RemainingString());
        response="250 OK\r\n"; }
      else if (cmd == "DATA") {
        if      (!message.rcpt_to.size())   response = "503 valid RCPT command must precede DATA\r\n";
        else if (!message.mail_from.size()) response = "503 valid FROM command must precede DATA\r\n";
        else                 { in_data = 1; response = "354 Start mail input; end with <CRLF>.<CRLF>\r\n"; }
      }
      else if (cmd == "EHLO") { response=StrCat("250 ", server->domain, " greets ", a1_orig, "\r\n"); client_domain=a1_orig; }
      else if (cmd == "HELO") { response=StrCat("250 ", server->domain, " greets ", a1_orig, "\r\n"); client_domain=a1_orig; }
      else if (cmd == "RSET") { response="250 OK\r\n"; ClearStateTable(); }
      else if (cmd == "NOOP") { response="250 OK\r\n"; }
      else if (cmd == "VRFY") { response="250 OK\r\n"; }
      else if (cmd == "QUIT") { c->WriteFlush(StrCat("221 ", server->domain, " closing connection\r\n")); c->SetError(); }
      if (!response.empty()) if (c->Write(response) != response.size()) return -1;
    }
    return processed;
  }

  int ReadData(Connection *c, const char *in, int len) {
    int processed = 0;
    StringLineIter lines(StringPiece(in, len), StringLineIter::Flag::BlankLines);
    for (const char *line = lines.Next(); line && lines.next_offset>=0; line = lines.Next()) {
      processed = lines.next_offset;
      if (lines.cur_len == 1 && *line == '.') { in_data=0; break; }
      message.content.append(line, lines.cur_len);
      message.content.append("\r\n");
    }
    if (!in_data) {
      c->Write("250 OK\r\n");
      server->ReceiveMail(c, message); 
      ClearStateTable();
    }
    return processed;
  }
};

int SMTPServer::Connected(Connection *c) { total_connected++; c->handler = make_unique<SMTPServerConnection>(this); return 0; }

void SMTPServer::ReceiveMail(Connection *c, const SMTP::Message &mail) {
  INFO("SMTPServer::ReceiveMail FROM=", mail.mail_from, ", TO=", mail.rcpt_to, ", content=", mail.content);
}

/* SMTP */

void SMTP::HTMLMessage(const string& from, const string& to, const string& subject, const string& content, string *out) {
  static const char seperator[] = "XzYzZy";
  *out = StrCat("From: ", from, "\nTo: ", to, "\nSubject: ", subject,
                "\nMIME-Version: 1.0\nContent-Type: multipart/alternative; boundary=\"", seperator, "\"\n\n");
  StrAppend(out, "--", seperator, "\nContent-type: text/html\n\n", content, "\n--", seperator, "--\n");
}

void SMTP::NativeSendmail(const string &message) {
#ifdef __linux__
  ProcessPipe smtp;
  const char *argv[] = { "/usr/bin/sendmail", "-i", "-t", 0 };
  if (smtp.Open(argv)) return;
  fwrite(message.c_str(), message.size(), 1, smtp.out);
#endif
}

string SMTP::EmailFrom(const string &message) {
  int lt, gt;
  string mail_from = HTTP::GrepHeaders(message.c_str(), 0, "From");
  if ((lt = mail_from.find("<"    )) == mail_from.npos ||
      (gt = mail_from.find(">", lt)) == mail_from.npos) FATAL("parse template from ", mail_from);
  return mail_from.substr(lt+1, gt-lt-1);
}

}; // namespace LFL
