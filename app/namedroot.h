// wget http://www.internic.net/domain/named.root
// cat named.root | grep " A " | awk '{print "XX(\"" $1 "\"); YY(\"" $4 "\");" }' | sed s/\\.\"/\"/g

XX("A.ROOT-SERVERS.NET"); YY("198.41.0.4");
XX("B.ROOT-SERVERS.NET"); YY("192.228.79.201");
XX("C.ROOT-SERVERS.NET"); YY("192.33.4.12");
XX("D.ROOT-SERVERS.NET"); YY("199.7.91.13");
XX("E.ROOT-SERVERS.NET"); YY("192.203.230.10");
XX("F.ROOT-SERVERS.NET"); YY("192.5.5.241");
XX("G.ROOT-SERVERS.NET"); YY("192.112.36.4");
XX("H.ROOT-SERVERS.NET"); YY("128.63.2.53");
XX("I.ROOT-SERVERS.NET"); YY("192.36.148.17");
XX("J.ROOT-SERVERS.NET"); YY("192.58.128.30");
XX("K.ROOT-SERVERS.NET"); YY("193.0.14.129");
XX("L.ROOT-SERVERS.NET"); YY("199.7.83.42");
XX("M.ROOT-SERVERS.NET"); YY("202.12.27.33");
