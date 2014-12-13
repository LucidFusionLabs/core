var express = require('express');
var http = require('http');
var app = express();

var CompressEmptyTicks = true;

var Ticks = [], AskTicks = [], BidTicks = [];
var TIME=0, OPEN=1, HIGH=2, LOW=3, CLOSE=4, VOLUME=5;

var MinVolume = 0;
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

var Pusher = require('pusher-client');
var pusher = null;

function PusherSubscribe(channel_name, bind_name, connect_cb, data_cb) {
    var channel = pusher.subscribe(channel_name);
    channel.bind(bind_name, data_cb);
    connect_cb();
}

function OpenBitstamp() {
    pusher = new Pusher('de504dc5763aeef9ff52');
    PusherSubscribe('live_trades', 'trade', FetchBitstampHistoricalTrades, HandleTrade);
    PusherSubscribe('diff_order_book', 'data', FetchBitstampOrderbook, HandleDepth);
}

function FetchBitstampHistoricalTrades() {}
function FetchBitstampOrderbook() {
    http.get("http://www.bitstamp.net/api/order_book/", function(res) {
        var body = '';
        res.on('data', function(chunk) { body += chunk; });
        res.on('end',  function() { HandleBitstampOrderbook(JSON.parse(body)); });
    }).on('error',     function() { HandleBitstampOrderbook(null); } );
}

function HandleBitstampOrderbook(msg) {
    if (!msg) {
        console.log("HandleBitstampOrderbook: Connection error");
        DepthChannel.init = true;
    } else {
        for (var i = 0; i < msg.asks.length; i++) {
            var ask = msg.asks[i];
            Asks.push({ type: "ask", price: parseFloat(ask[0]), volume: parseFloat(ask[1]),
                        timeout_price: undefined, timeout_volume: undefined });
        }
        for (var i = 0; i < msg.bids.length; i++) {
            var bid = msg.bids[i];
            Bids.push({ type: "bid", price: parseFloat(bid[0]), volume: parseFloat(bid[1]),
                        timeout_price: undefined, timeout_volume: undefined });
        }
    }
}

function HandleTrade(msg) {
    var trade_time = (new Date).getTime();
    var trade_price = parseFloat(msg.price);
    var trade_amount = parseFloat(msg.amount);
    AddTrade(Ticks, trade_time, trade_price, trade_amount);
}

function AddTrade(ticks, trade_time, trade_price, trade_amount) {
    if (ticks.length == 0) ticks.push([ trade_time, 0, 0, 0, 0, 0 ]);

    var last_tick = ticks.last();
    var last_tick_min = Math.floor(last_tick[TIME] / 60000);
    var cur_tick_min = Math.floor(trade_time / 60000);
    var mins_since_last_tick = cur_tick_min - last_tick_min;
    var start_i = mins_since_last_tick && CompressEmptyTicks ? mins_since_last_tick-1 : 0;
    for (var i = start_i; i < mins_since_last_tick; i++) {
        var min = last_tick_min + i + 1;
        var v = i != mins_since_last_tick-1 ? last_tick[CLOSE] : 0;
        ticks.push([ min * 60000, v, v, v, v, 0 ]);
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

function HandleDepth(msg) {
    for (var i = 0; i < msg.asks.length; i++) {
        var ask = msg.asks[i];
        HandleDepthUpdate({ type: "ask", price: parseFloat(ask[0]), volume: parseFloat(ask[1]),
                          timeout_price: undefined, timeout_volume: undefined });
    }
    for (var i = 0; i < msg.bids.length; i++) {
        var bid = msg.bids[i];
        HandleDepthUpdate({ type: "bid", price: parseFloat(bid[0]), volume: parseFloat(bid[1]),
                          timeout_price: undefined, timeout_volume: undefined });
    }
}

function HandleDepthUpdate(order) {
    var orders  = order.type == "bid" ? Bids        : Asks;
    var cmp     = order.type == "bid" ? GreaterThan : LessThan;

    var orders_ind = orders.binarySearch(order.price, cmp, 'price');

    if (orders[orders_ind] == undefined || orders[orders_ind].price != order.price) {
        if (order.volume > MinVolume) orders.splice(orders_ind, 0, order);
    } else {
        if (order.volume > MinVolume) orders[orders_ind].volume = order.volume;
        else orders.splice(orders_ind, 1);
    }
}

app.get('/bitstamp_archive/tick/M1', function (req, res) {
    console.log(req.connection.remoteAddress + ": " + req.method + " " + req.url);
    res.writeHead(200, { 'Content-Type': 'application/json', "Access-Control-Allow-Origin":"*" });
    res.write(JSON.stringify({ now: TradeChannel.last_time, ticks: Ticks, asks: AskTicks, bids: BidTicks }));
    res.end();
});

app.get('/bitstamp_archive/orderbook', function (req, res) {
    console.log(req.connection.remoteAddress + ": " + req.method + " " + req.url);
    res.writeHead(200, { 'Content-Type': 'application/json', "Access-Control-Allow-Origin":"*" });
    res.write(JSON.stringify({ now: DepthChannel.last_time, asks: Asks.slice(0, 500), bids: Bids.slice(0, 500) }));
    res.end();
});

OpenBitstamp();
app.listen(8090);

