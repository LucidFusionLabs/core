var PUBNUB = require('pubnub');
var express = require('express');
var http = require('http');
var app = express();

var mtgox_channel = {
    "trade":  "dbf1dee9-4f2e-4a08-8cb7-748919a71b21",
    "depth":  "24e67e0d-1cad-4cc0-9e7a-f8523ef460fe",
    "ticker": "d5f06780-30a8-4a48-a2f8-7ed181b4a13f",
    "lag":    "85174711-be64-4de1-b783-0628995d7914"
};

var pubnub = null;
var CompressEmptyTicks = true;

var Ticks = [], AskTicks = [], BidTicks = [];
var TIME=0, OPEN=1, HIGH=2, LOW=3, CLOSE=4, VOLUME=5;

var MinVolume = 1;
var Bids = [], Asks = [];

var TradeChannel = { name: "trade", init: false, first_time: 0, last_time: 0, buffer: [], handler: HandleTrade };
var DepthChannel = { name: "depth", init: false, first_time: 0, last_time: 0, buffer: [], handler: HandleDepth };

function LessThan   (l, r) { return l < r ? true : false; }
function GreaterThan(l, r) { return r < l ? true : false; }

Array.prototype.last = function() { return this[this.length-1]; }
Array.prototype.binarySearch = function(needle, cmp, field) {
    var low = 0, high = this.length, ind, missing;
    while (low <= high) {
        ind = Math.floor((low + high) / 2);
        missing = this[ind] == undefined;
        if (!missing &&     this[ind][field] == needle)  return ind;
        if (!missing && cmp(this[ind][field],   needle)) low  = ind + 1;
        else                                             high = ind - 1;
    }
    return low;
};

function FetchMtGoxOrderbook() {
    http.get("http://data.mtgox.com/api/2/BTCUSD/money/depth/full", function(res) {
        var body = '';
        res.on('data', function(chunk) { body += chunk; });
        res.on('end',  function() { HandleMtGoxOrderbook(JSON.parse(body)); });
    }).on('error',     function() { HandleMtGoxOrderbook(null); } );
}

function HandleMtGoxOrderbook(msg) {
    if (!msg) {
        console.log("HandleMtGoxOrderbook: Connection error");
        DepthChannel.init = true;
    } else {
        for (var i = 0; i < msg.data.asks.length; i++) {
            var ask = msg.data.asks[i];
            Asks.push({ type: "ask", price: parseInt(ask.price_int), volume: parseInt(ask.amount_int),
                        timeout_price: undefined, timeout_volume: undefined });
        }
        for (var i = 0; i < msg.data.bids.length; i++) {
            var bid = msg.data.bids[i];
            Bids.push({ type: "bid", price: parseInt(bid.price_int), volume: parseInt(bid.amount_int),
                        timeout_price: undefined, timeout_volume: undefined });
        }
        PubNubHistory("depth", msg.data.now, DepthChannel);
    }
}

function FetchPubNubHistoricalTrades() {
    PubNubHistory("trade", (new Date).getTime()*1000 - 5400000000, TradeChannel);
}

function PubNubHistory(channel_name, start_time, chan) {
    var start_time_text = new Date(start_time / 1000).toLocaleString();
    console.log("PubNubHistory: " + channel_name + " " + start_time_text + " (" + start_time + ")");

    pubnub.history({
        count    : 100,
        reverse  : true,
        start    : start_time*10,
        channel  : mtgox_channel[channel_name],
        callback : function(a) {
            var start = a[1]/10, end = a[2]/10, i;
            for (i = 0; i < a[0].length; i++) {
                var msg = a[0][i];
                if (chan.first_time && msg.stamp >= chan.first_time) break;
                chan.handler(msg[chan.name]);
            }
            if (a[0].length == 100 && i == 100) {
                PubNubHistory(channel_name, end, chan);
                return;
            }
            for (var i = 0; i < chan.buffer.length; i++) chan.handler(chan.buffer[i]);
            chan.buffer = [];
            chan.init = true;
            console.log("PubNubHistory: " + channel_name + " complete");
        }
    });
}

function PubNubSubscribe(channel_name, connect_cb) {
    pubnub.subscribe({
        restore    : true,
        channel    : mtgox_channel[channel_name],
        message    : function(m) { HandleMessage(m); },
        error      : function() { console.log("PubNub: Network error " + channel_name); },
        reconnect  : function() { console.log("PubNub: Reconnected "   + channel_name); },
        disconnect : function() { console.log("PubNub: Disconnected "  + channel_name); },
        connect    : function() { console.log("PubNub: Connected "     + channel_name);
                                  if (connect_cb) connect_cb(); },
    });
}

function OpenPubNub() {
    pubnub = PUBNUB.init({ subscribe_key: 'sub-c-50d56e1e-2fd9-11e3-a041-02ee2ddab7fe' });
    PubNubSubscribe("trade", FetchPubNubHistoricalTrades);
    PubNubSubscribe("depth", FetchMtGoxOrderbook);
}

function HandleMessage(msg) {
    if (msg.op != "private") {
        console.log("MtGox: Got Op: " + msg.op);
        return;
    }
    switch(msg["private"]) {
        case "trade":
        HandleChannelMessage(TradeChannel, msg);
        break;

        case "depth":
        HandleChannelMessage(DepthChannel, msg);
        break;

        default:
        console.log("MtGox: Private Op: " + msg["private"]);
        break;
    }
}

function HandleChannelMessage(chan, msg) {
    var timestamp = parseInt(msg.stamp);

    if (!chan.first_time)            chan.first_time = timestamp;
    if (chan.last_time <= timestamp) chan.last_time  = timestamp;
    else console.log("HandleChannelMessage: " + chan.name + " " + chan.last_time + " > " + timestamp);

    if (chan.init) chan.handler(msg[chan.name]);
    else       chan.buffer.push(msg[chan.name]);
}

function HandleTrade(trade) {
    if (trade.price_currency != "USD") return;

    var now = (new Date).getTime();
    var trade_time = parseInt(trade.tid);
    var trade_price = parseInt(trade.price_int);
    var trade_amount = parseInt(trade.amount_int);
    var lag = now - trade_time / 1000;

    var type = trade.trade_type == "bid" ? "ask" : "bid";
    var orders  = type == "bid" ? Bids     : Asks;
    var cmptext = type == "bid" ? " < "    : " > "; 
    var cmp     = type == "bid" ? LessThan : GreaterThan;

    while (TradeChannel.init && DepthChannel.init && orders.length && cmp(trade_price, orders[0].price)) {
        console.log(type + " trade inconsistency: " + trade_price + cmptext + orders[0].price);
        orders.shift();
    }

    if (1)             AddTrade(Ticks,    trade_time, trade_price, trade_amount);
    if (type == "ask") AddTrade(AskTicks, trade_time, trade_price, trade_amount);
    if (type == "bid") AddTrade(BidTicks, trade_time, trade_price, trade_amount);
}

function AddTrade(ticks, trade_time, trade_price, trade_amount) {
    if (ticks.length == 0) ticks.push([ trade_time, 0, 0, 0, 0, 0 ]);

    var last_tick = ticks.last();
    var last_tick_min = Math.floor(last_tick[TIME] / 60000000);
    var cur_tick_min = Math.floor(trade_time / 60000000);
    var mins_since_last_tick = cur_tick_min - last_tick_min;
    var start_i = mins_since_last_tick && CompressEmptyTicks ? mins_since_last_tick-1 : 0;
    for (var i = start_i; i < mins_since_last_tick; i++) {
        var min = last_tick_min + i + 1;
        var v = i != mins_since_last_tick-1 ? last_tick[CLOSE] : 0;
        ticks.push([ min * 60000000, v, v, v, v, 0 ]);
    }
    while (ticks.length > 180) ticks.shift();

    var cur_tick = ticks.last();
    if (cur_tick[OPEN] == 0) {
        cur_tick[OPEN] = trade_price;
        cur_tick[LOW]  = trade_price;
    }

    if (cur_tick[HIGH] < trade_price) cur_tick[HIGH] = trade_price;
    if (cur_tick[LOW]  > trade_price) cur_tick[LOW]  = trade_price;

    cur_tick[CLOSE] = trade_price;
    cur_tick[VOLUME] += trade_amount;
}

function HandleDepth(depth) {
    if (depth.currency != "USD") return;

    var order = { type:           depth.type_str,
                  price:          parseInt(depth.price_int), 
                  volume:         parseInt(depth.total_volume_int),
                  timeout_price:  undefined,
                  timeout_volume: undefined }; 

    var orders  = order.type == "bid" ? Bids        : Asks;
    var cmp     = order.type == "bid" ? GreaterThan : LessThan;

    var orders_ind = orders.binarySearch(order.price, cmp, 'price');

    if (orders[orders_ind] == undefined || orders[orders_ind].price != order.price) {
        if (order.volume >= MinVolume) orders.splice(orders_ind, 0, order);
    } else {
        if (order.volume >= MinVolume) orders[orders_ind].volume = order.volume;
        else orders.splice(orders_ind, 1);
    }
}

app.get('/mtgox_archive/tick/M1', function (req, res) {
    console.log(req.connection.remoteAddress + ": " + req.method + " " + req.url);
    res.writeHead(200, { 'Content-Type': 'application/json', "Access-Control-Allow-Origin":"*" });
    res.write(JSON.stringify({ now: TradeChannel.last_time, ticks: Ticks, asks: AskTicks, bids: BidTicks }));
    res.end();
});

app.get('/mtgox_archive/orderbook', function (req, res) {
    console.log(req.connection.remoteAddress + ": " + req.method + " " + req.url);
    res.writeHead(200, { 'Content-Type': 'application/json', "Access-Control-Allow-Origin":"*" });
    res.write(JSON.stringify({ now: DepthChannel.last_time, asks: Asks.slice(0, 500), bids: Bids.slice(0, 500) }));
    res.end();
});

OpenPubNub();
app.listen(8090);

